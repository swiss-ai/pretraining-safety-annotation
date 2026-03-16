"""Co-optimization pipeline: generate charter reflections, judge them, iterate.

Usage:
    uv run python -m pipeline.phase2.run
    uv run python -m pipeline.phase2.run phase2.iteration.n_items=10 phase2.scoring.accept_threshold=3
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
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
    load_config,
    resolve_generator_model,
    resolve_judge_model,
    resolve_prompt_path,
)
from pipeline.fineweb import load_or_build_fineweb_cache
from pipeline.tokenizer import compute_reflection_point, truncate_to_max_tokens
from pipeline.phase2.storage import (
    load_items_for_iteration,
    load_latest_items,
    load_runs,
    next_iteration,
    save_item,
    save_run,
)
from pipeline.log import logger
from pipeline.storage import compute_item_id

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0


def make_api_client(cfg: AppConfig) -> tuple[openai.AsyncOpenAI, asyncio.Semaphore]:
    """Create an OpenAI client and concurrency semaphore from config."""
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
) -> tuple[str, str | None, dict]:
    """Make a single API call with network-error retry.

    Returns (content, reasoning_content, usage_dict). reasoning_content is None
    if the model does not produce reasoning output. usage_dict contains
    input_tokens, output_tokens, reasoning_tokens.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={
                        "separate_reasoning": True,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                )
            msg = response.choices[0].message
            content = msg.content
            assert content is not None, "API returned None content"
            reasoning = getattr(msg, "reasoning_content", None)
            usage = response.usage
            details = getattr(usage, "completion_tokens_details", None) or {}
            if isinstance(details, dict):
                detail_reasoning = details.get("reasoning_tokens", 0) or 0
            else:
                detail_reasoning = getattr(details, "reasoning_tokens", 0) or 0
            usage_dict = {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "reasoning_tokens": getattr(usage, "reasoning_tokens", 0)
                or detail_reasoning,
            }
            return content.strip(), reasoning, usage_dict
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            AssertionError,
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            logger.warning(
                "Retry {}/{} due to: {}", attempt + 2, MAX_RETRIES, last_error
            )
            await asyncio.sleep(RETRY_BACKOFF_BASE**attempt)
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
        logger.info("Health check passed: model={}", model)
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
    # Some models return string fields as lists — coerce to str
    for field in ("analysis", "preflection", "reflection"):
        if isinstance(parsed[field], list):
            parsed[field] = "\n".join(str(x) for x in parsed[field])
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
    required = {"scores", "reasoning"}
    missing = required - set(parsed.keys())
    assert not missing, f"Missing fields in judgment: {missing}"
    assert isinstance(parsed["scores"], dict), "scores must be a dict"
    assert len(parsed["scores"]) > 0, "scores must not be empty"
    parsed["aggregate"] = sum(parsed["scores"].values()) / len(parsed["scores"])
    return parsed


def _load_gold_items(max_tokens: int) -> list[dict]:
    """Load gold set items from annotation data (SQLite), truncating to max_tokens."""
    from pipeline.phase1.storage import load_latest_annotations

    annotations = load_latest_annotations()
    seen_ids: set[str] = set()
    records = []
    for (item_id, _), record in annotations.items():
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            text = truncate_to_max_tokens(record["text"], max_tokens)
            rp = min(record["reflection_point"], len(text))
            records.append(
                {
                    "item_id": item_id,
                    "subset": record["subset"],
                    "text": text,
                    "reflection_point": rp,
                    "is_gold": True,
                }
            )
    return records


FINEWEB_CACHE_PATH = PIPELINE_DATA_DIR / "fineweb_cache.jsonl"
FINEWEB_CACHE_SIZE = 4096


def _sample_fresh_items(
    n: int, seed: int, exclude_ids: set[str], max_tokens: int
) -> list[dict]:
    """Sample fresh FineWeb items, stratified equally across subsets.

    Each text is truncated to max_tokens before computing the reflection point.
    """
    rng = random.Random(seed)
    cache = load_or_build_fineweb_cache(
        cache_path=FINEWEB_CACHE_PATH,
        dataset=FINEWEB_DATASET,
        subsets=FINEWEB_SUBSETS,
        per_subset=FINEWEB_CACHE_SIZE // len(FINEWEB_SUBSETS),
        seed=seed,
    )

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
                text = truncate_to_max_tokens(row["text"], max_tokens)
                item_id = compute_item_id(text)
                if item_id in exclude_ids:
                    continue
                items.append(
                    {
                        "item_id": item_id,
                        "subset": subset,
                        "text": text,
                        "reflection_point": compute_reflection_point(text, rng),
                        "is_gold": False,
                    }
                )
                exclude_ids.add(item_id)
                made_progress = True
                break
        if not made_progress:
            break

    assert (
        len(items) >= n
    ), f"Could only sample {len(items)}/{n} fresh items (cache has {len(cache)})"
    return items[:n]


def select_items(n_total: int, n_gold: int, seed: int, max_tokens: int) -> list[dict]:
    """Select a mix of gold set items and fresh random FineWeb samples.

    Returns up to n_total items: min(n_gold, available_gold) gold items,
    rest filled with fresh samples. All texts are truncated to max_tokens.
    """
    gold = _load_gold_items(max_tokens)
    rng = random.Random(seed)
    rng.shuffle(gold)
    selected_gold = gold[:n_gold]
    n_fresh = n_total - len(selected_gold)

    exclude_ids = {item["item_id"] for item in selected_gold}
    if n_fresh > 0:
        fresh = _sample_fresh_items(n_fresh, seed, exclude_ids, max_tokens)
    else:
        fresh = []

    items = selected_gold + fresh
    logger.info(
        "Selected {} gold + {} fresh = {} items",
        len(selected_gold),
        len(fresh),
        len(items),
    )
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
        raw, reasoning, usage = await _api_call(client, model, messages, semaphore)
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
            "timestamp": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "judgment": None,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
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
) -> tuple[dict, str, str | None, dict]:
    """Judge a single part (preflection or reflection) of a generated item.

    For preflection: uses full text as context.
    For reflection: uses only text up to the reflection point.

    Returns (parsed_judgment, raw_response, reasoning_content, usage_dict).
    """
    system_prompt = prompt_template.replace("{part_type}", part_type).replace(
        "{accept_threshold}", str(accept_threshold)
    )

    if part_type == "preflection":
        source_text = item["text"]
    else:
        source_text = item["text"][: item["reflection_point"]]

    content = item[part_type]

    user_content = (
        f"## Source Text\n\n{source_text}\n\n"
        f"## {part_type.title()} to Judge\n\n{content}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw, reasoning, usage = await _api_call(client, model, messages, semaphore)
    parsed = _parse_judgment(raw)
    return parsed, raw, reasoning, usage


def judge_batch(
    items: list[dict],
    prompt_path: Path,
    model: str,
    iteration: int,
    accept_threshold: float,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    save: bool = True,
    floor_threshold: int = 2,
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
        t0 = time.monotonic()
        pre_parsed, pre_raw, pre_reasoning, pre_usage = await _judge_one_part(
            item,
            "preflection",
            prompt_template,
            accept_threshold,
            model,
            client,
            semaphore,
        )
        ref_parsed, ref_raw, ref_reasoning, ref_usage = await _judge_one_part(
            item,
            "reflection",
            prompt_template,
            accept_threshold,
            model,
            client,
            semaphore,
        )
        judge_latency_ms = int((time.monotonic() - t0) * 1000)

        all_scores = list(pre_parsed["scores"].values()) + list(
            ref_parsed["scores"].values()
        )
        aggregate = sum(all_scores) / len(all_scores)
        # Floor rule: any dimension ≤ floor_threshold forces reject (documented in judge prompt)
        has_floor_violation = any(s <= floor_threshold for s in all_scores)
        decision = (
            "reject"
            if has_floor_violation or aggregate < accept_threshold
            else "accept"
        )

        judge_usage = {
            "input_tokens": pre_usage["input_tokens"] + ref_usage["input_tokens"],
            "output_tokens": pre_usage["output_tokens"] + ref_usage["output_tokens"],
            "reasoning_tokens": pre_usage["reasoning_tokens"]
            + ref_usage["reasoning_tokens"],
        }

        judgment = {
            "preflection": {
                "scores": pre_parsed["scores"],
                "aggregate": pre_parsed["aggregate"],
                "reasoning": pre_parsed["reasoning"],
                "model_reasoning": pre_reasoning,
                "usage": pre_usage,
            },
            "reflection": {
                "scores": ref_parsed["scores"],
                "aggregate": ref_parsed["aggregate"],
                "reasoning": ref_parsed["reasoning"],
                "model_reasoning": ref_reasoning,
                "usage": ref_usage,
            },
            "aggregate": aggregate,
            "decision": decision,
            "judge_prompt": prompt_filename,
            "raw_responses": {"preflection": pre_raw, "reflection": ref_raw},
            "usage": judge_usage,
            "latency_ms": judge_latency_ms,
            "timestamp": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
        }
        judged = {**item, "judgment": judgment}
        if save:
            save_item(judged)
        return judged

    coros = [judge_one(item) for item in items]
    return list(_gather(*coros, desc="Judging"))


def _make_run_summary(iteration: int, judged: list[dict]) -> str:
    """Build a human-readable summary string from judged items."""
    n_accepted = sum(1 for item in judged if item["judgment"]["decision"] == "accept")
    n_rejected = len(judged) - n_accepted
    scores = [item["judgment"]["aggregate"] for item in judged]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    gen_has_reasoning = any(item.get("reasoning") is not None for item in judged)
    judge_has_reasoning = any(
        item["judgment"]["preflection"].get("model_reasoning") is not None
        or item["judgment"]["reflection"].get("model_reasoning") is not None
        for item in judged
    )
    reasoning_note = (
        f"Generator reasoning: {'available' if gen_has_reasoning else 'NOT available'}. "
        f"Judge reasoning: {'available' if judge_has_reasoning else 'NOT available'}."
    )
    return (
        f"Iteration {iteration}: {n_accepted} accepted, {n_rejected} rejected, "
        f"mean score {mean_score:.2f}. {reasoning_note}"
    )


def _run_one_pair(
    cfg: AppConfig,
    items: list[dict],
    gen_alias: str,
    judge_alias: str,
    source: str,
    group_id: str | None = None,
) -> dict:
    """Run generate->judge for one (generator, judge) pair. Returns run summary dict.

    Installs signal handlers for graceful shutdown during DB writes.
    """
    from pipeline.storage import _get_conn, checkpoint
    from uuid import uuid4

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _graceful_shutdown(signum, frame):
        logger.warning(
            "Received signal {} during iteration — checkpointing DB before exit", signum
        )
        try:
            _get_conn().commit()
            checkpoint()
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    try:
        return _run_one_pair_inner(cfg, items, gen_alias, judge_alias, source, group_id)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


def _run_one_pair_inner(
    cfg: AppConfig,
    items: list[dict],
    gen_alias: str,
    judge_alias: str,
    source: str,
    group_id: str | None,
) -> dict:
    """Inner implementation of _run_one_pair (split out for signal safety)."""
    iteration = next_iteration()
    client, semaphore = make_api_client(cfg)
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")

    gen_model_cfg = resolve_generator_model(cfg, gen_alias)
    judge_model_cfg = resolve_judge_model(cfg, judge_alias)
    gen_prompt = resolve_prompt_path("generator_latest.md", alias=gen_alias)
    judge_prompt = resolve_prompt_path("judge_latest.md", alias=judge_alias)

    logger.info("Iteration {} — gen={} judge={}", iteration, gen_alias, judge_alias)

    generated = generate_batch(
        items,
        gen_prompt,
        charter_text,
        gen_model_cfg.api_name,
        iteration,
        client,
        semaphore,
    )

    judged = judge_batch(
        generated,
        judge_prompt,
        judge_model_cfg.api_name,
        iteration,
        cfg.phase2.scoring.accept_threshold,
        client,
        semaphore,
        floor_threshold=cfg.phase2.scoring.floor_threshold,
    )

    summary = _make_run_summary(iteration, judged)
    logger.info(summary)

    n_accepted = sum(1 for item in judged if item["judgment"]["decision"] == "accept")
    scores = [item["judgment"]["aggregate"] for item in judged]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    save_run(
        iteration=iteration,
        gen_prompt=gen_prompt.name,
        judge_prompt=judge_prompt.name,
        generator_model=gen_alias,
        judge_model=judge_alias,
        n_items=len(judged),
        n_gold=sum(1 for item in judged if item.get("is_gold")),
        config={
            "accept_threshold": cfg.phase2.scoring.accept_threshold,
            "max_concurrent": cfg.phase2.iteration.max_concurrent,
        },
        analysis=summary,
        source=source,
        group_id=group_id,
    )

    return {
        "iteration": iteration,
        "n_items": len(judged),
        "n_accepted": n_accepted,
        "n_rejected": len(judged) - n_accepted,
        "mean_score": mean_score,
        "items": judged,
        "generator_model": gen_alias,
        "judge_model": judge_alias,
        "group_id": group_id,
    }


def _run_cross_iteration(
    cfg: AppConfig,
    role: str,
    target_alias: str,
    source: str,
) -> list[dict]:
    """Run cross-iteration for a given role and target model.

    For judge role: generate with ALL generators, judge with target.
    For generator role: generate with target, judge with ALL judges.

    Items are selected once (fixed seed). All iterations share a group_id.
    Returns list of run summaries.
    """
    from uuid import uuid4

    client, _semaphore = make_api_client(cfg)

    # Determine fixed vs. iterated models
    if role == "judge":
        fixed_alias = target_alias
        resolve_judge_model(cfg, target_alias)  # validate alias
        counterpart_models = cfg.phase2.generator_models
        pairs = [(m.alias, target_alias) for m in counterpart_models]
    else:
        fixed_alias = target_alias
        resolve_generator_model(cfg, target_alias)  # validate alias
        counterpart_models = cfg.phase2.judge_models
        pairs = [(target_alias, m.alias) for m in counterpart_models]

    # Health-check all involved models upfront
    _health_check_models(client, cfg, role, target_alias)

    # Select items once (fixed seed based on current max iteration)
    base_iter = next_iteration()
    seed = 42 + base_iter
    items = select_items(
        cfg.phase2.iteration.n_items,
        cfg.phase2.iteration.n_gold,
        seed,
        cfg.max_tokens,
    )

    group_id = str(uuid4())
    summaries = []
    for gen_alias, judge_alias in pairs:
        logger.info("Cross-iteration: gen={} judge={}", gen_alias, judge_alias)
        result = _run_one_pair(
            cfg,
            items,
            gen_alias,
            judge_alias,
            source=source,
            group_id=group_id,
        )
        summaries.append(result)

    return summaries


def _health_check_models(
    client: openai.AsyncOpenAI,
    cfg: AppConfig,
    role: str,
    target_alias: str,
) -> None:
    """Health-check the target model and all counterpart models for a cross-iteration."""
    checked: set[str] = set()
    if role == "judge":
        target_cfg = resolve_judge_model(cfg, target_alias)
        health_check(client, target_cfg.api_name)
        checked.add(target_cfg.api_name)
        for m in cfg.phase2.generator_models:
            if m.api_name not in checked:
                health_check(client, m.api_name)
                checked.add(m.api_name)
    else:
        target_cfg = resolve_generator_model(cfg, target_alias)
        health_check(client, target_cfg.api_name)
        checked.add(target_cfg.api_name)
        for m in cfg.phase2.judge_models:
            if m.api_name not in checked:
                health_check(client, m.api_name)
                checked.add(m.api_name)


def run_judge_cross_iteration(
    cfg: AppConfig,
    target_judge_alias: str,
    source: str = "improve_judge",
) -> list[dict]:
    """Generate with ALL generators, judge all with target judge."""
    return _run_cross_iteration(cfg, "judge", target_judge_alias, source)


def run_generator_cross_iteration(
    cfg: AppConfig,
    target_gen_alias: str,
    source: str = "improve_generator",
) -> list[dict]:
    """Generate with target generator, judge with ALL judges."""
    return _run_cross_iteration(cfg, "generator", target_gen_alias, source)


def rejudge_all_prompts_and_models(cfg: AppConfig) -> int:
    """Re-judge all human-reviewed items with ALL judge prompts × ALL judge models.

    Discovers all judge_v*.md files in each judge model's prompt directory,
    then for each (judge_prompt, judge_model) combination, re-judges any
    reviewed items that don't already have correlations. Idempotent.

    Returns total count of newly judged items.
    """
    import re as re_mod

    from pipeline.config import PROMPTS_DIR, resolve_judge_model
    from pipeline.phase2.storage import (
        load_judge_correlations,
        load_latest_reviews,
        save_judge_correlation,
    )

    reviews = load_latest_reviews()
    if not reviews:
        logger.info("No reviewed items to re-judge for correlations.")
        return 0

    reviewed_item_keys: set[tuple[str, int]] = set()
    for item_id, rev_iter, _reviewer in reviews:
        reviewed_item_keys.add((item_id, rev_iter))

    existing = load_judge_correlations()
    existing_keys = {
        (c["item_id"], c["iteration"], c["judge_prompt"], c["judge_model"])
        for c in existing
    }

    latest_items = load_latest_items()

    total_new = 0
    for model_cfg in cfg.phase2.judge_models:
        alias = model_cfg.alias
        model_dir = PROMPTS_DIR / alias
        if not model_dir.exists():
            continue

        judge_files = sorted(
            p for p in model_dir.iterdir() if re_mod.match(r"^judge_v\d+\.md$", p.name)
        )

        for judge_file in judge_files:
            prompt_name = judge_file.name
            needs_judging = [
                latest_items[k]
                for k in reviewed_item_keys
                if k in latest_items
                and (k[0], k[1], prompt_name, alias) not in existing_keys
            ]

            if not needs_judging:
                logger.info(
                    "All reviewed items already done for {} / {}.", prompt_name, alias
                )
                continue

            client, semaphore = make_api_client(cfg)
            api_name = model_cfg.api_name

            logger.info(
                "Re-judging {} items with {} ({})...",
                len(needs_judging),
                prompt_name,
                alias,
            )

            judged = judge_batch(
                items=needs_judging,
                prompt_path=judge_file,
                model=api_name,
                iteration=needs_judging[0]["iteration"],
                accept_threshold=cfg.phase2.scoring.accept_threshold,
                client=client,
                semaphore=semaphore,
                save=False,
                floor_threshold=cfg.phase2.scoring.floor_threshold,
            )

            for item in judged:
                save_judge_correlation(
                    item_id=item["item_id"],
                    iteration=item["iteration"],
                    judge_prompt=prompt_name,
                    judge_model=alias,
                    judgment=item["judgment"],
                )

            total_new += len(judged)
            logger.info(
                "Saved {} correlations for {} / {}.", len(judged), prompt_name, alias
            )

    logger.info("Total new correlations: {}", total_new)
    return total_new


def main():
    """CLI entry point. Runs a cross-iteration with optional config overrides."""
    overrides = sys.argv[1:] if len(sys.argv) > 1 else None
    cfg = load_config(overrides)

    logger.info("Endpoint: {}", cfg.phase2.endpoint)
    logger.info("Generator models: {}", [m.alias for m in cfg.phase2.generator_models])
    logger.info("Judge models: {}", [m.alias for m in cfg.phase2.judge_models])
    logger.info(
        "Items: {} (gold: {})",
        cfg.phase2.iteration.n_items,
        cfg.phase2.iteration.n_gold,
    )
    logger.info("Threshold: {}", cfg.phase2.scoring.accept_threshold)

    # Default: run judge cross-iteration with first judge model
    target = cfg.phase2.judge_models[0].alias
    results = run_judge_cross_iteration(cfg, target)
    for r in results:
        logger.info(
            "  gen={} judge={}: {}/{} accepted, mean={:.2f}",
            r["generator_model"],
            r["judge_model"],
            r["n_accepted"],
            r["n_items"],
            r["mean_score"],
        )


if __name__ == "__main__":
    main()
