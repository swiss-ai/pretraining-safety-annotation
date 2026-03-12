"""Co-optimization pipeline: generate charter reflections, judge them, iterate.

Usage:
    uv run python -m pipeline.phase2.run
    uv run python -m pipeline.phase2.run phase2.iteration.n_items=10 phase2.scoring.accept_threshold=3
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
import dotenv
dotenv.load_dotenv()

import openai
from tqdm.asyncio import tqdm_asyncio

from pipeline.config import (
    CHARTER_PATH,
    PIPELINE_DATA_DIR,
    AppConfig,
    extract_charter_elements,
    generator_api_name,
    judge_api_name,
    load_config,
    resolve_prompt_path,
)
from pipeline.phase2.storage import (
    load_items_for_iteration,
    load_latest_items,
    load_runs,
    save_item,
    save_run,
)
from pipeline.storage import compute_item_id

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0


def make_api_client(cfg: AppConfig) -> tuple[openai.AsyncOpenAI, asyncio.Semaphore]:
    """Create an OpenAI client and concurrency semaphore from config."""
    load_dotenv()
    api_key = os.environ.get("SWISS_AI_API_KEY")
    assert api_key, "SWISS_AI_API_KEY not set in environment"
    client = openai.AsyncOpenAI(api_key=api_key, base_url=cfg.phase2.endpoint)
    semaphore = asyncio.Semaphore(cfg.phase2.iteration.max_concurrent)
    return client, semaphore

FINEWEB_DATASET = "locuslab/fineweb_annotated"
FINEWEB_SUBSETS = [f"score_{i}" for i in range(6)]


def _gather(*coros, desc: str) -> list:
    """Run async coroutines concurrently with a tqdm progress bar.

    Creates a temporary event loop that doesn't touch SIGINT handling,
    so Ctrl+C raises KeyboardInterrupt normally.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(tqdm_asyncio.gather(*coros, desc=desc))
    finally:
        loop.close()


async def _api_call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Make a single API call with network-error retry.

    Returns (content, reasoning_content). reasoning_content is None if the
    model does not produce reasoning output.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model, messages=messages,
                    extra_body={
                        "separate_reasoning": True,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                )
            msg = response.choices[0].message
            content = msg.content
            assert content is not None, "API returned None content"
            reasoning = getattr(msg, "reasoning_content", None)
            return content.strip(), reasoning
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            print(f"  Retry {attempt + 2}/{MAX_RETRIES} due to: {last_error}")
            await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")


def health_check(client: openai.AsyncOpenAI, model: str) -> None:
    """Ping the API with a lightweight request. Fail fast if model unavailable."""
    async def _check():
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        assert response.choices[0].message.content is not None

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_check())
        print(f"Health check passed: model={model}")
    except Exception as e:
        raise RuntimeError(f"Health check failed for model={model}: {e}") from e
    finally:
        loop.close()


def _parse_generation(raw: str) -> dict:
    """Parse generator JSON output into structured fields.

    Extracts JSON from response, handling optional markdown code fences.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    parsed = json.loads(text)
    required = {"analysis", "preflection", "reflection"}
    missing = required - set(parsed.keys())
    assert not missing, f"Missing fields in generation: {missing}"
    return parsed


def _parse_judgment(raw: str) -> dict:
    """Parse judge JSON output into structured fields.

    Extracts JSON from response, handling optional markdown code fences.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    parsed = json.loads(text)
    required = {"scores", "decision", "reasoning"}
    missing = required - set(parsed.keys())
    assert not missing, f"Missing fields in judgment: {missing}"
    assert isinstance(parsed["scores"], dict), "scores must be a dict"
    assert len(parsed["scores"]) > 0, "scores must not be empty"
    assert parsed["decision"] in ("accept", "reject"), f"Invalid decision: {parsed['decision']}"
    parsed["aggregate"] = sum(parsed["scores"].values()) / len(parsed["scores"])
    return parsed


def _load_gold_items() -> list[dict]:
    """Load gold set items from annotation data (SQLite)."""
    from pipeline.phase1.storage import load_latest_annotations

    annotations = load_latest_annotations()
    seen_ids: set[str] = set()
    records = []
    for (item_id, _), record in annotations.items():
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            records.append({
                "item_id": item_id,
                "subset": record["subset"],
                "text": record["text"],
                "reflection_point": record["reflection_point"],
                "is_gold": True,
            })
    return records


FINEWEB_CACHE_PATH = PIPELINE_DATA_DIR / "fineweb_cache.jsonl"
FINEWEB_CACHE_SIZE = 500


def _load_or_build_cache(seed: int) -> list[dict]:
    """Load cached FineWeb texts, or stream from HF and cache locally.

    Caches FINEWEB_CACHE_SIZE raw items (text + subset) to avoid repeated
    HF downloads. The cache is a simple JSONL file.
    """
    if FINEWEB_CACHE_PATH.exists():
        records = []
        for line in FINEWEB_CACHE_PATH.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
        if records:
            print(f"Loaded {len(records)} items from FineWeb cache")
            return records

    print(f"Building FineWeb cache ({FINEWEB_CACHE_SIZE} items, stratified across {len(FINEWEB_SUBSETS)} subsets)...")
    from datasets import load_dataset

    per_subset = FINEWEB_CACHE_SIZE // len(FINEWEB_SUBSETS)
    records = []
    for subset in FINEWEB_SUBSETS:
        ds = load_dataset(FINEWEB_DATASET, subset, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        count = 0
        for row in ds:
            if count >= per_subset:
                break
            records.append({"text": row["text"], "subset": subset})
            count += 1
        print(f"  {subset}: {count} items")

    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINEWEB_CACHE_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Cached {len(records)} items to {FINEWEB_CACHE_PATH}")
    return records


def _sample_fresh_items(n: int, seed: int, exclude_ids: set[str]) -> list[dict]:
    """Sample fresh FineWeb items, stratified equally across subsets."""
    rng = random.Random(seed)
    cache = _load_or_build_cache(seed)

    # Group by subset and shuffle each group
    by_subset: dict[str, list[dict]] = {}
    for row in cache:
        by_subset.setdefault(row["subset"], []).append(row)
    for rows in by_subset.values():
        rng.shuffle(rows)

    # Round-robin across subsets to get stratified sample
    subsets = sorted(by_subset.keys())
    cursors = {s: 0 for s in subsets}
    items = []
    while len(items) < n:
        made_progress = False
        for subset in subsets:
            if len(items) >= n:
                break
            rows = by_subset.get(subset, [])
            while cursors[subset] < len(rows):
                row = rows[cursors[subset]]
                cursors[subset] += 1
                text = row["text"]
                item_id = compute_item_id(text)
                if item_id in exclude_ids:
                    continue
                min_pos = max(1, int(len(text) * 0.1))
                max_pos = max(min_pos + 1, int(len(text) * 0.9))
                char_pos = rng.randint(min_pos, max_pos)
                space_idx = text.find(" ", char_pos)
                if space_idx != -1 and space_idx - char_pos < 50:
                    char_pos = space_idx
                items.append({
                    "item_id": item_id,
                    "subset": subset,
                    "text": text,
                    "reflection_point": char_pos,
                    "is_gold": False,
                })
                exclude_ids.add(item_id)
                made_progress = True
                break
        if not made_progress:
            break

    assert len(items) >= n, f"Could only sample {len(items)}/{n} fresh items (cache has {len(cache)})"
    return items[:n]


def select_items(n_total: int, n_gold: int, seed: int) -> list[dict]:
    """Select a mix of gold set items and fresh random FineWeb samples.

    Returns up to n_total items: min(n_gold, available_gold) gold items,
    rest filled with fresh samples.
    """
    gold = _load_gold_items()
    rng = random.Random(seed)
    rng.shuffle(gold)
    selected_gold = gold[:n_gold]
    n_fresh = n_total - len(selected_gold)

    exclude_ids = {item["item_id"] for item in selected_gold}
    if n_fresh > 0:
        fresh = _sample_fresh_items(n_fresh, seed, exclude_ids)
    else:
        fresh = []

    items = selected_gold + fresh
    print(f"Selected {len(selected_gold)} gold + {len(fresh)} fresh = {len(items)} items")
    return items


def generate_batch(
    items: list[dict],
    prompt_path: Path,
    charter_text: str,
    model: str,
    iteration: int,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    save: bool = True,
) -> list[dict]:
    """Generate charter reflections for a batch of items.

    Runs API calls concurrently via a temporary event loop.
    When save=True, saves each item to JSONL progressively as it completes.
    Returns the list of completed item records.
    """
    prompt_template = prompt_path.read_text(encoding="utf-8")
    system_prompt = prompt_template.replace("{charter}", charter_text)
    prompt_filename = prompt_path.name

    async def process_one(item: dict) -> dict:
        rp = item["reflection_point"]
        context_before = item["text"][:rp]
        context_after = item["text"][rp:]
        user_content = (
            f"## Full Text\n\n{item['text']}\n\n"
            f"## Reflection Point\n\n"
            f"The reflection point is at character {rp}. "
            f"Text before the reflection point:\n\n{context_before}\n\n"
            f"Text after the reflection point (the reflection must NOT use this):\n\n{context_after}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        t0 = time.monotonic()
        raw, reasoning = await _api_call(client, model, messages, semaphore)
        latency_ms = int((time.monotonic() - t0) * 1000)

        parsed = _parse_generation(raw)
        charter_elements = extract_charter_elements(parsed["reflection"])
        record = {
            "item_id": item["item_id"],
            "iteration": iteration,
            "is_gold": item.get("is_gold", False),
            "subset": item["subset"],
            "text": item["text"],
            "reflection_point": item["reflection_point"],
            "gen_prompt": prompt_filename,
            "model": model,
            "analysis": parsed["analysis"],
            "preflection": parsed["preflection"],
            "reflection": parsed["reflection"],
            "charter_elements": charter_elements,
            "raw_response": raw,
            "reasoning": reasoning,
            "latency_ms": latency_ms,
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "judgment": None,
        }
        if save:
            save_item(record)
        return record

    coros = [process_one(item) for item in items]
    return list(_gather(*coros, desc="Generating"))


async def _judge_one_part(
    item: dict,
    part_type: str,
    prompt_template: str,
    accept_threshold: float,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, str, str | None]:
    """Judge a single part (preflection or reflection) of a generated item.

    For preflection: uses full text as context.
    For reflection: uses only text up to the reflection point.

    Returns (parsed_judgment, raw_response, reasoning_content).
    """
    system_prompt = (
        prompt_template
        .replace("{part_type}", part_type)
        .replace("{accept_threshold}", str(accept_threshold))
    )

    if part_type == "preflection":
        source_text = item["text"]
    else:
        source_text = item["text"][:item["reflection_point"]]

    content = item[part_type]

    user_content = (
        f"## Source Text\n\n{source_text}\n\n"
        f"## {part_type.title()} to Judge\n\n{content}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw, reasoning = await _api_call(client, model, messages, semaphore)
    parsed = _parse_judgment(raw)
    return parsed, raw, reasoning


def judge_batch(
    items: list[dict],
    prompt_path: Path,
    model: str,
    iteration: int,
    accept_threshold: float,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    save: bool = True,
) -> list[dict]:
    """Judge generated reflections. Judges preflection and reflection separately.

    Runs API calls concurrently via a temporary event loop.
    Preflection is judged against the full text.
    Reflection is judged against only the context up to the reflection point.
    When save=True, saves each judged item to JSONL progressively.

    Returns the list of judged item records.
    """
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_filename = prompt_path.name

    async def judge_one(item: dict) -> dict:
        pre_parsed, pre_raw, pre_reasoning = await _judge_one_part(
            item, "preflection", prompt_template, accept_threshold,
            model, client, semaphore,
        )
        ref_parsed, ref_raw, ref_reasoning = await _judge_one_part(
            item, "reflection", prompt_template, accept_threshold,
            model, client, semaphore,
        )

        all_scores = list(pre_parsed["scores"].values()) + list(ref_parsed["scores"].values())
        aggregate = sum(all_scores) / len(all_scores)
        # Floor rule: any dimension <= 2 forces reject, matching judge prompt
        has_floor_violation = any(s <= 2 for s in all_scores)
        decision = "reject" if has_floor_violation or aggregate < accept_threshold else "accept"

        judgment = {
            "preflection": {
                "scores": pre_parsed["scores"],
                "aggregate": pre_parsed["aggregate"],
                "reasoning": pre_parsed["reasoning"],
                "model_reasoning": pre_reasoning,
            },
            "reflection": {
                "scores": ref_parsed["scores"],
                "aggregate": ref_parsed["aggregate"],
                "reasoning": ref_parsed["reasoning"],
                "model_reasoning": ref_reasoning,
            },
            "aggregate": aggregate,
            "decision": decision,
            "judge_prompt": prompt_filename,
            "raw_responses": {"preflection": pre_raw, "reflection": ref_raw},
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }
        judged = {**item, "judgment": judgment}
        if save:
            save_item(judged)
        return judged

    coros = [judge_one(item) for item in items]
    return list(_gather(*coros, desc="Judging"))


def run_iteration(cfg: AppConfig, phase_callback=None) -> dict:
    """Run a single generate->judge iteration. Returns the run summary.

    Orchestrates: select items -> generate -> judge -> save run record.
    phase_callback: optional callable(str) for granular progress reporting.
    """
    runs = load_runs()
    iteration = len(runs) + 1
    seed = 42 + iteration

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")

    client, semaphore = make_api_client(cfg)

    gen_model = generator_api_name(cfg)
    jdg_model = judge_api_name(cfg)

    print("Running health check...")
    health_check(client, gen_model)
    if jdg_model != gen_model:
        health_check(client, jdg_model)

    charter_text = CHARTER_PATH.read_text(encoding="utf-8")

    print("Selecting items...")
    items = select_items(cfg.phase2.iteration.n_items, cfg.phase2.iteration.n_gold, seed)

    gen_prompt = resolve_prompt_path(cfg.phase2.generator.prompt, alias=cfg.phase2.generator.model)
    judge_prompt = resolve_prompt_path(cfg.phase2.judge.prompt, alias=cfg.phase2.judge.model)

    if phase_callback:
        phase_callback("generating")
    print(f"Generating with {gen_prompt.name}...")
    generated = generate_batch(
        items, gen_prompt, charter_text, gen_model, iteration, client, semaphore,
    )

    if phase_callback:
        phase_callback("judging")
    print(f"Judging with {judge_prompt.name}...")
    judged = judge_batch(
        generated, judge_prompt, jdg_model, iteration,
        cfg.phase2.scoring.accept_threshold, client, semaphore,
    )

    n_accepted = sum(1 for item in judged if item["judgment"]["decision"] == "accept")
    n_rejected = len(judged) - n_accepted
    scores = [item["judgment"]["aggregate"] for item in judged]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    # Check reasoning availability
    gen_has_reasoning = any(item.get("reasoning") is not None for item in judged)
    judge_has_reasoning = any(
        item["judgment"]["preflection"].get("model_reasoning") is not None
        or item["judgment"]["reflection"].get("model_reasoning") is not None
        for item in judged
    )
    reasoning_note = (
        f"Generator reasoning: {'available' if gen_has_reasoning else 'NOT available (not a reasoning model)'}. "
        f"Judge reasoning: {'available' if judge_has_reasoning else 'NOT available (not a reasoning model)'}."
    )

    summary = (
        f"Iteration {iteration}: {n_accepted} accepted, {n_rejected} rejected, "
        f"mean score {mean_score:.2f}. {reasoning_note}"
    )
    print(f"\n{summary}")

    save_run(
        iteration=iteration,
        gen_prompt=gen_prompt.name,
        judge_prompt=judge_prompt.name,
        generator_model=cfg.phase2.generator.model,
        judge_model=cfg.phase2.judge.model,
        n_items=len(judged),
        n_gold=sum(1 for item in judged if item.get("is_gold")),
        config={
            "accept_threshold": cfg.phase2.scoring.accept_threshold,
            "max_concurrent": cfg.phase2.iteration.max_concurrent,
        },
        analysis=summary,
    )

    return {
        "iteration": iteration,
        "n_items": len(judged),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "mean_score": mean_score,
        "items": judged,
    }


def rejudge_reviewed_items(cfg: AppConfig) -> list[dict]:
    """Re-judge all items that have human reviews using the current judge prompt.

    This enables tracking judge calibration progression across prompt versions.
    Items are re-judged and saved (overwriting their judgment in the append-only log).
    Returns the list of re-judged items.
    """
    from pipeline.phase2.storage import load_latest_reviews

    reviews = load_latest_reviews()
    if not reviews:
        print("No reviewed items to re-judge.")
        return []

    # Collect unique (item_id, iteration) pairs that have reviews
    reviewed_keys = {(k[0], k[1]) for k in reviews}

    # Load the actual items
    latest_items = load_latest_items()
    items_to_rejudge = [
        latest_items[key] for key in reviewed_keys if key in latest_items
    ]

    if not items_to_rejudge:
        print("No matching items found for reviewed keys.")
        return []

    client, semaphore = make_api_client(cfg)
    judge_prompt = resolve_prompt_path(cfg.phase2.judge.prompt, alias=cfg.phase2.judge.model)
    model = judge_api_name(cfg)

    print(f"Re-judging {len(items_to_rejudge)} reviewed items with {judge_prompt.name}...")

    judged = judge_batch(
        items=items_to_rejudge,
        prompt_path=judge_prompt,
        model=model,
        iteration=0,  # iteration param unused in judge_batch logic
        accept_threshold=cfg.phase2.scoring.accept_threshold,
        client=client,
        semaphore=semaphore,
        save=True,
    )

    print(f"Re-judged {len(judged)} items with {judge_prompt.name}.")
    return judged


def main():
    """CLI entry point. Runs a single iteration with optional config overrides."""
    overrides = sys.argv[1:] if len(sys.argv) > 1 else None
    cfg = load_config(overrides)

    print(f"Generator: {cfg.phase2.generator.model} ({generator_api_name(cfg)})")
    print(f"Judge: {cfg.phase2.judge.model} ({judge_api_name(cfg)})")
    print(f"Endpoint: {cfg.phase2.endpoint}")
    print(f"Items: {cfg.phase2.iteration.n_items} (gold: {cfg.phase2.iteration.n_gold})")
    print(f"Generator prompt: {resolve_prompt_path(cfg.phase2.generator.prompt, cfg.phase2.generator.model).name}")
    print(f"Judge prompt: {resolve_prompt_path(cfg.phase2.judge.prompt, cfg.phase2.judge.model).name}")
    print(f"Threshold: {cfg.phase2.scoring.accept_threshold}")

    result = run_iteration(cfg)

    print(f"\nDone. {result['n_accepted']}/{result['n_items']} accepted.")


if __name__ == "__main__":
    main()
