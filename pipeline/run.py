"""Co-optimization pipeline: generate charter reflections, judge them, iterate.

Usage:
    uv run python -m pipeline.run
    uv run python -m pipeline.run iteration.n_items=10 scoring.accept_threshold=3
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

import openai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from pipeline.config import (
    ANNOTATION_DATA_DIR,
    CHARTER_PATH,
    PipelineConfig,
    load_config,
    resolve_prompt_path,
)
from pipeline.storage import (
    load_items_for_iteration,
    load_runs,
    save_item,
    save_run,
)

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0

FINEWEB_DATASET = "locuslab/fineweb_annotated"
FINEWEB_SUBSETS = [f"score_{i}" for i in range(6)]


async def _api_call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
) -> str:
    """Make a single API call with network-error retry. Returns response text."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model, messages=messages,
                )
            content = response.choices[0].message.content
            assert content is not None, "API returned None content"
            return content.strip()
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


async def health_check(client: openai.AsyncOpenAI, model: str) -> None:
    """Ping the API with a lightweight request. Fail fast if model unavailable."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        assert response.choices[0].message.content is not None
        print(f"Health check passed: model={model}")
    except Exception as e:
        raise RuntimeError(f"Health check failed for model={model}: {e}") from e


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
    required = {"analysis", "preflection", "reflection", "charter_elements"}
    missing = required - set(parsed.keys())
    assert not missing, f"Missing fields in generation: {missing}"
    assert isinstance(parsed["charter_elements"], list), "charter_elements must be a list"
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
    required = {"scores", "aggregate", "decision", "reasoning"}
    missing = required - set(parsed.keys())
    assert not missing, f"Missing fields in judgment: {missing}"
    assert isinstance(parsed["scores"], dict), "scores must be a dict"
    assert parsed["decision"] in ("accept", "reject"), f"Invalid decision: {parsed['decision']}"
    return parsed


def _load_gold_items() -> list[dict]:
    """Load gold set items from annotation data."""
    ann_path = ANNOTATION_DATA_DIR / "annotations.jsonl"
    if not ann_path.exists():
        return []
    records = []
    seen_ids: set[str] = set()
    for line in ann_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        item_id = record["item_id"]
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


def _sample_fresh_items(n: int, seed: int, exclude_ids: set[str]) -> list[dict]:
    """Sample fresh random FineWeb items, excluding already-used IDs."""
    from annotation.storage import compute_item_id

    rng = random.Random(seed)
    items = []
    subset_cycle = itertools.cycle(FINEWEB_SUBSETS)

    from datasets import load_dataset

    for subset in FINEWEB_SUBSETS:
        ds = load_dataset(FINEWEB_DATASET, subset, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for row in ds:
            if len(items) >= n:
                break
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
        if len(items) >= n:
            break

    assert len(items) >= n, f"Could only sample {len(items)}/{n} fresh items"
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


async def generate_batch(
    items: list[dict],
    prompt_path: Path,
    charter_text: str,
    model: str,
    iteration: int,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Generate charter reflections for a batch of items.

    Saves each item to JSONL progressively as it completes.
    Returns the list of completed item records.
    """
    prompt_template = prompt_path.read_text(encoding="utf-8")
    system_prompt = prompt_template.replace("{charter}", charter_text)
    prompt_filename = prompt_path.name

    async def process_one(item: dict) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["text"]},
        ]
        t0 = time.monotonic()
        raw = await _api_call(client, model, messages, semaphore)
        latency_ms = int((time.monotonic() - t0) * 1000)

        parsed = _parse_generation(raw)
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
            "charter_elements": parsed["charter_elements"],
            "raw_response": raw,
            "latency_ms": latency_ms,
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "judgment": None,
        }
        save_item(record)
        return record

    tasks = [process_one(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating")
    return list(results)


async def judge_batch(
    items: list[dict],
    prompt_path: Path,
    model: str,
    iteration: int,
    accept_threshold: float,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Judge generated reflections. Updates items with judgment and saves.

    Returns the list of judged item records.
    """
    prompt_template = prompt_path.read_text(encoding="utf-8")
    system_prompt = prompt_template.replace("{accept_threshold}", str(accept_threshold))
    prompt_filename = prompt_path.name

    async def judge_one(item: dict) -> dict:
        user_content = (
            f"## Original Text\n\n{item['text']}\n\n"
            f"## Generated Reflection\n\n"
            f"**Analysis:** {item['analysis']}\n\n"
            f"**Preflection:** {item['preflection']}\n\n"
            f"**Reflection:** {item['reflection']}\n\n"
            f"**Charter Elements:** {', '.join(item['charter_elements'])}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        raw = await _api_call(client, model, messages, semaphore)
        parsed = _parse_judgment(raw)

        judgment = {
            "scores": parsed["scores"],
            "aggregate": parsed["aggregate"],
            "decision": parsed["decision"],
            "reasoning": parsed["reasoning"],
            "judge_prompt": prompt_filename,
            "raw_response": raw,
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }
        judged = {**item, "judgment": judgment}
        save_item(judged)
        return judged

    tasks = [judge_one(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Judging")
    return list(results)


async def run_iteration(cfg: PipelineConfig) -> dict:
    """Run a single generate→judge iteration. Returns the run summary.

    Orchestrates: select items → generate → judge → save run record.
    """
    load_dotenv()
    api_key = os.environ.get("SWISS_AI_API_KEY")
    assert api_key, "SWISS_AI_API_KEY not set in environment"

    runs = load_runs()
    iteration = len(runs) + 1
    seed = 42 + iteration

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")

    client = openai.AsyncOpenAI(api_key=api_key, base_url=cfg.endpoint)
    semaphore = asyncio.Semaphore(cfg.iteration.max_concurrent)

    print("Running health check...")
    await health_check(client, cfg.model)

    charter_text = CHARTER_PATH.read_text(encoding="utf-8")

    print("Selecting items...")
    items = select_items(cfg.iteration.n_items, cfg.iteration.n_gold, seed)

    gen_prompt = resolve_prompt_path(cfg.prompts.generator)
    judge_prompt = resolve_prompt_path(cfg.prompts.judge)

    print(f"Generating with {gen_prompt.name}...")
    generated = await generate_batch(
        items, gen_prompt, charter_text, cfg.model, iteration, client, semaphore,
    )

    print(f"Judging with {judge_prompt.name}...")
    judged = await judge_batch(
        generated, judge_prompt, cfg.model, iteration,
        cfg.scoring.accept_threshold, client, semaphore,
    )

    n_accepted = sum(1 for item in judged if item["judgment"]["decision"] == "accept")
    n_rejected = len(judged) - n_accepted
    scores = [item["judgment"]["aggregate"] for item in judged]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    summary = (
        f"Iteration {iteration}: {n_accepted} accepted, {n_rejected} rejected, "
        f"mean score {mean_score:.2f}"
    )
    print(f"\n{summary}")

    save_run(
        iteration=iteration,
        gen_prompt=cfg.prompts.generator,
        judge_prompt=cfg.prompts.judge,
        model=cfg.model,
        n_items=len(judged),
        n_gold=sum(1 for item in judged if item.get("is_gold")),
        config={
            "accept_threshold": cfg.scoring.accept_threshold,
            "max_concurrent": cfg.iteration.max_concurrent,
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


def main():
    """CLI entry point. Runs a single iteration with optional config overrides."""
    overrides = sys.argv[1:] if len(sys.argv) > 1 else None
    cfg = load_config(overrides)

    print(f"Model: {cfg.model}")
    print(f"Endpoint: {cfg.endpoint}")
    print(f"Items: {cfg.iteration.n_items} (gold: {cfg.iteration.n_gold})")
    print(f"Generator: {cfg.prompts.generator}")
    print(f"Judge: {cfg.prompts.judge}")
    print(f"Threshold: {cfg.scoring.accept_threshold}")

    result = asyncio.run(run_iteration(cfg))

    print(f"\nDone. {result['n_accepted']}/{result['n_items']} accepted.")


if __name__ == "__main__":
    main()
