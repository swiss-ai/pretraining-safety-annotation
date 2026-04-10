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
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable
import dotenv

dotenv.load_dotenv()

import openai

import yaml

from pipeline.api import (
    api_call,
    extract_json,
    health_check,
    make_api_client,
    run_concurrent,
)
from pipeline.config import (
    CHARTER_PATH,
    PIPELINE_DATA_DIR,
    PROJECT_ROOT,
    WRITING_GUIDELINES_PATH,
    AppConfig,
    load_config,
    resolve_generator_model,
    resolve_judge_model,
    resolve_prompt_path,
    union_charter_elements,
)
from pipeline.data import load_dataset_cache
from pipeline.generation import (
    FIELD_ALIASES,
    GEN_TEXT_FIELDS,
    PREFLECTION_TASK,
    REFLECTION_TASK,
    parse_generation,
)
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

CANARY_RATE = 0.10

CANARIES_PATH = PROJECT_ROOT / "resources" / "canaries.yaml"

# Backwards-compatible aliases for the private names used internally.
_REFLECTION_TASK = REFLECTION_TASK
_PREFLECTION_TASK = PREFLECTION_TASK
_FIELD_ALIASES = FIELD_ALIASES
_GEN_TEXT_FIELDS = GEN_TEXT_FIELDS
_parse_generation = parse_generation

_MODE_SECTION_RE = re.compile(r"<!-- mode: (\w+) -->.*?<!-- /mode -->", re.DOTALL)


def _split_generator_prompt(template: str, mode: str) -> str:
    """Strip mode-specific sections not matching *mode* ('reflection' or 'preflection').

    Sections are delimited by ``<!-- mode: X -->`` / ``<!-- /mode -->`` markers.
    If no markers are present the template is returned unchanged (backward compat).
    """

    def _keep(m: re.Match) -> str:
        return m.group(0) if m.group(1) == mode else ""

    return _MODE_SECTION_RE.sub(_keep, template).strip()


def _load_canaries() -> list[dict]:
    """Load canary quirks from resources/canaries.yaml."""
    with open(CANARIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["canaries"]


def _parse_judgment(raw: str) -> dict:
    """Parse judge JSON output into structured fields (single-voice format).

    Extracts JSON from response, handling prose before/after JSON and code fences.
    """
    parsed = extract_json(raw)
    required = {"scores", "reasoning"}
    missing = required - set(parsed.keys())
    assert not missing, (
        f"Missing fields in judgment: {missing}. "
        f"Got keys: {list(parsed.keys())}. Raw preview: {raw[:200]}"
    )
    assert isinstance(parsed["scores"], dict), "scores must be a dict"
    assert len(parsed["scores"]) > 0, "scores must not be empty"
    parsed["aggregate"] = sum(parsed["scores"].values()) / len(parsed["scores"])
    return parsed


_FOUR_VOICES = ("preflection_3p", "preflection_1p", "reflection_1p", "reflection_3p")


def _parse_combined_judgment(raw: str) -> dict:
    """Parse combined judge JSON output with scores for all four voices."""
    parsed = extract_json(raw)
    missing = set(_FOUR_VOICES) - set(parsed.keys())
    assert not missing, (
        f"Missing voices in combined judgment: {missing}. "
        f"Got keys: {list(parsed.keys())}. Raw preview: {raw[:200]}"
    )
    for voice in _FOUR_VOICES:
        vd = parsed[voice]
        assert isinstance(vd, dict), f"{voice} must be a dict"
        assert (
            "scores" in vd and "reasoning" in vd
        ), f"{voice} must have 'scores' and 'reasoning'"
        assert (
            isinstance(vd["scores"], dict) and len(vd["scores"]) > 0
        ), f"{voice} scores must be a non-empty dict"
        vd["aggregate"] = sum(vd["scores"].values()) / len(vd["scores"])
    return parsed


def _load_gold_items(max_tokens: int) -> list[dict]:
    """Load gold set items from annotation data (SQLite), truncating to max_tokens."""
    from pipeline.phase1.storage import load_latest_annotations

    annotations = load_latest_annotations()
    text_max = max_tokens - REFLECTION_TOKEN_BUDGET
    seen_ids: set[str] = set()
    records = []
    for (item_id, _), record in annotations.items():
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            text = truncate_to_max_tokens(record["text"], text_max)
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


# Max tokens reserved for the reflection itself (subtracted from max_tokens).
REFLECTION_TOKEN_BUDGET = 128


def _sample_fresh_items(
    n: int, seed: int, exclude_ids: set[str], max_tokens: int
) -> list[dict]:
    """Sample fresh items randomly from the Dolma3 dataset cache.

    Each text is truncated to (max_tokens - REFLECTION_TOKEN_BUDGET) before
    computing the reflection point.
    """
    rng = random.Random(seed)
    cache = load_dataset_cache(seed)
    rng.shuffle(cache)

    text_max = max_tokens - REFLECTION_TOKEN_BUDGET
    items: list[dict] = []
    for row in cache:
        if len(items) >= n:
            break
        text = truncate_to_max_tokens(row["text"], text_max)
        item_id = compute_item_id(text)
        if item_id in exclude_ids:
            continue
        items.append(
            {
                "item_id": item_id,
                "subset": "dolma3",
                "text": text,
                "reflection_point": compute_reflection_point(text, rng),
                "safety_score": row.get("safety_score"),
                "is_gold": False,
            }
        )
        exclude_ids.add(item_id)

    assert (
        len(items) >= n
    ), f"Could only sample {len(items)}/{n} fresh items (cache has {len(cache)})"
    return items[:n]


def select_items(n_total: int, n_gold: int, seed: int, max_tokens: int) -> list[dict]:
    """Select a mix of gold set items and fresh random Dolma3 samples.

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


class _JudgeParseError(Exception):
    """Wraps a parse failure inside _judge_combined / _judge_one_part so the
    caller can recover the raw model response and reasoning."""

    def __init__(
        self,
        original: BaseException,
        raw: str | None,
        raw_reasoning: str | None,
        stage: str,
    ):
        super().__init__(str(original))
        self.original = original
        self.raw = raw
        self.raw_reasoning = raw_reasoning
        self.stage = stage


def _failure_record(
    item_id: str,
    stage: str,
    category: str,
    reason: str,
    raw: str | None,
    raw_reasoning: str | None,
    exc: BaseException | None = None,
) -> dict:
    """Build a normalized failure record for the on_failure callback.

    The record carries the raw model response (when available) so the user
    can grep through rejected responses and improve the parser. `category`
    splits api vs parse so downstream rate metrics can report each separately.
    """
    return {
        "item_id": item_id,
        "stage": stage,
        "category": category,
        "reason": reason,
        "raw": raw,
        "raw_reasoning": raw_reasoning,
        "error": f"{type(exc).__name__}: {exc}" if exc is not None else None,
        "ts": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
    }


def generate_batch(
    items: list[dict],
    prompt_path: Path,
    charter_text: str,
    model: str,
    iteration: int,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    save: bool = True,
    writing_guidelines_text: str = "",
    thinking: bool = False,
    json_mode: bool = False,
    canary_rng_seed: int | None = None,
    on_failure: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Generate charter reflections for a batch of items.

    Runs API calls concurrently via a temporary event loop.
    When save=True, saves each item to JSONL progressively as it completes.
    Returns the list of completed item records.
    """
    prompt_template = prompt_path.read_text(encoding="utf-8")
    base_prompt = prompt_template.replace("{charter}", charter_text).replace(
        "{writing_guidelines}", writing_guidelines_text
    )
    refl_system_prompt = _split_generator_prompt(base_prompt, "reflection")
    prefl_system_prompt = _split_generator_prompt(base_prompt, "preflection")
    prompt_filename = prompt_path.name
    canaries = _load_canaries()

    async def process_one(item: dict) -> dict | None:
        rp = item["reflection_point"]
        context_before = item["text"][:rp]
        context_after = item["text"][rp:]

        # ---- Call 1: Reflection (text up to RP only) ----
        refl_user = f"## Full Text\n\n{context_before}"

        # Canary injection: 10% chance, reflections only.
        # When canary_rng_seed is provided, the decision is deterministic in
        # (seed, item_id) so all candidate generators within the same eval
        # see the same items canaried with the same canary id.
        canary_id = None
        if canary_rng_seed is not None:
            item_rng = random.Random(f"{canary_rng_seed}_{item['item_id']}_canary_v1")
            inject = item_rng.random() < CANARY_RATE
            canary = item_rng.choice(canaries) if inject else None
        else:
            inject = random.random() < CANARY_RATE
            canary = random.choice(canaries) if inject else None
        if inject:
            canary_id = canary["id"]
            refl_user += (
                f"\n\n## Canary Injection\n\n"
                f"This sample has a canary injection. Apply to BOTH reflections.\n"
                f"- For reflection_1p: {canary['instruction']}\n"
                f"- For reflection_3p: {canary['instruction_3p']}"
            )

        refl_user += _REFLECTION_TASK

        refl_messages = [
            {"role": "system", "content": refl_system_prompt},
            {"role": "user", "content": refl_user},
        ]

        t0 = time.monotonic()
        refl_raw = None
        refl_reasoning = None
        try:
            refl_raw, refl_reasoning, refl_usage = await api_call(
                client,
                model,
                refl_messages,
                semaphore,
                thinking=thinking,
                json_mode=json_mode,
            )
        except RuntimeError as e:
            logger.warning(
                "Skipping item {} — reflection api failed: {}",
                item["item_id"],
                e,
            )
            if on_failure is not None:
                on_failure(
                    _failure_record(
                        item["item_id"],
                        "reflection",
                        "api",
                        "api_runtime",
                        raw=None,
                        raw_reasoning=None,
                        exc=e,
                    )
                )
            return None
        try:
            refl_parsed = _parse_generation(
                refl_raw,
                required_fields={"analysis", "reflection_1p", "reflection_3p"},
            )
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(
                "Skipping item {} — reflection parse failed: {}",
                item["item_id"],
                e,
            )
            if on_failure is not None:
                reason = (
                    "json_parse"
                    if isinstance(e, json.JSONDecodeError)
                    else "missing_field"
                )
                on_failure(
                    _failure_record(
                        item["item_id"],
                        "reflection",
                        "parse",
                        reason,
                        raw=refl_raw,
                        raw_reasoning=refl_reasoning,
                        exc=e,
                    )
                )
            return None

        # ---- Call 2: Preflection (full text) ----
        prefl_user = f"## Full Text\n\n{context_before}{context_after}"
        prefl_user += _PREFLECTION_TASK

        prefl_messages = [
            {"role": "system", "content": prefl_system_prompt},
            {"role": "user", "content": prefl_user},
        ]

        prefl_raw = None
        prefl_reasoning = None
        try:
            prefl_raw, prefl_reasoning, prefl_usage = await api_call(
                client,
                model,
                prefl_messages,
                semaphore,
                thinking=thinking,
                json_mode=json_mode,
            )
        except RuntimeError as e:
            logger.warning(
                "Skipping item {} — preflection api failed: {}",
                item["item_id"],
                e,
            )
            if on_failure is not None:
                on_failure(
                    _failure_record(
                        item["item_id"],
                        "preflection",
                        "api",
                        "api_runtime",
                        raw=None,
                        raw_reasoning=None,
                        exc=e,
                    )
                )
            return None
        try:
            prefl_parsed = _parse_generation(
                prefl_raw,
                required_fields={"analysis", "preflection_3p", "preflection_1p"},
            )
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(
                "Skipping item {} — preflection parse failed: {}",
                item["item_id"],
                e,
            )
            if on_failure is not None:
                reason = (
                    "json_parse"
                    if isinstance(e, json.JSONDecodeError)
                    else "missing_field"
                )
                on_failure(
                    _failure_record(
                        item["item_id"],
                        "preflection",
                        "parse",
                        reason,
                        raw=prefl_raw,
                        raw_reasoning=prefl_reasoning,
                        exc=e,
                    )
                )
            return None

        latency_ms = int((time.monotonic() - t0) * 1000)

        combined_analysis = (
            f"REFLECTION ANALYSIS:\n{refl_parsed['analysis']}\n\n"
            f"PREFLECTION ANALYSIS:\n{prefl_parsed['analysis']}"
        )

        reflection_charter_elements = union_charter_elements(
            refl_parsed.get("reflection_1p") or refl_parsed.get("reflection", ""),
            refl_parsed.get("reflection_3p"),
        )
        preflection_charter_elements = union_charter_elements(
            prefl_parsed.get("preflection_1p"),
            prefl_parsed.get("preflection_3p") or prefl_parsed.get("preflection", ""),
        )
        record = {
            "item_id": item["item_id"],
            "iteration": iteration,
            "is_gold": item.get("is_gold", False),
            "subset": item["subset"],
            "text": item["text"],
            "reflection_point": item["reflection_point"],
            "gen_prompt": prompt_filename,
            "model": model,
            "analysis": combined_analysis,
            # Legacy single-voice columns (kept for backward compat)
            "preflection": (
                prefl_parsed.get("preflection_3p")
                or prefl_parsed.get("preflection", "")
            ),
            "reflection": (
                refl_parsed.get("reflection_1p") or refl_parsed.get("reflection", "")
            ),
            # Explicit per-voice columns
            "preflection_1p": prefl_parsed.get("preflection_1p"),
            "reflection_3p": refl_parsed.get("reflection_3p"),
            "preflection_charter_elements": preflection_charter_elements,
            "reflection_charter_elements": reflection_charter_elements,
            "raw_response": json.dumps(
                {"reflection": refl_raw, "preflection": prefl_raw}
            ),
            "reasoning": refl_reasoning,
            "latency_ms": latency_ms,
            "timestamp": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "judgment": None,
            "input_tokens": refl_usage["input_tokens"] + prefl_usage["input_tokens"],
            "output_tokens": (
                refl_usage["output_tokens"] + prefl_usage["output_tokens"]
            ),
            "reasoning_tokens": (
                refl_usage["reasoning_tokens"] + prefl_usage["reasoning_tokens"]
            ),
            "safety_score": item.get("safety_score"),
            "canary": canary_id,
        }
        if save:
            save_item(record)
        return record

    coros = [process_one(item) for item in items]
    results = run_concurrent(*coros, desc="Generating")
    skipped = sum(1 for r in results if r is None)
    if skipped:
        logger.warning(
            "Generation: {}/{} items skipped due to parse/API errors",
            skipped,
            len(items),
        )
    return [r for r in results if r is not None]


async def _judge_one_part(
    item: dict,
    part_type: str,
    prompt_template: str,
    accept_threshold: float,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    charter_text: str = "",
    writing_guidelines_text: str = "",
    thinking: bool = False,
) -> tuple[str, str | None, dict]:
    """Make the judge API call for a single part and return the raw response.

    For preflection: uses full text as context.
    For reflection: uses only text up to the reflection point.

    Parsing is the caller's responsibility — returning the raw response
    lets judge_one log it when parsing fails.

    Returns (raw_response, reasoning_content, usage_dict).
    """
    system_prompt = (
        prompt_template.replace("{part_type}", part_type)
        .replace("{accept_threshold}", str(accept_threshold))
        .replace("{charter}", charter_text)
        .replace("{writing_guidelines}", writing_guidelines_text)
    )

    # Preflection variants use full text; reflection variants use only text up to reflection point
    if part_type in ("preflection", "preflection_3p", "preflection_1p"):
        source_text = item["text"]
    else:
        source_text = item["text"][: item["reflection_point"]]

    # Resolve the item key: prefer explicit per-voice columns, fall back to legacy columns
    _PART_KEY_FALLBACK = {
        "preflection_3p": "preflection",
        "reflection_1p": "reflection",
    }
    if part_type in item and item[part_type] is not None:
        content = item[part_type]
    elif part_type in _PART_KEY_FALLBACK and _PART_KEY_FALLBACK[part_type] in item:
        content = item[_PART_KEY_FALLBACK[part_type]]
    else:
        content = item[part_type]  # will raise KeyError with a clear message

    user_content = (
        f"## Source Text\n\n{source_text}\n\n"
        f"## {part_type.title()} to Judge\n\n{content}"
    )

    # Inform the judge about canary injections
    canary_id = item.get("canary")
    if part_type in ("reflection", "reflection_1p", "reflection_3p") and canary_id:
        canaries = _load_canaries()
        canary = next((c for c in canaries if c["id"] == canary_id), None)
        if canary:
            user_content += (
                f"\n\n## Canary Notice\n\n"
                f"This reflection has a canary injection (quirk: {canary['quirk']}, "
                f"value: {canary['value']}). The reflection was instructed to mention "
                f"this. Do NOT penalize the reflection for including this canary — "
                f"judge the rest of the reflection on its own merits."
            )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw, reasoning, usage = await api_call(
        client, model, messages, semaphore, thinking=thinking
    )
    try:
        parsed = _parse_judgment(raw)
    except (json.JSONDecodeError, AssertionError) as e:
        raise _JudgeParseError(e, raw, reasoning, f"judge_{part_type}") from e
    return parsed, raw, reasoning, usage


async def _judge_combined(
    item: dict,
    prompt_template: str,
    accept_threshold: float,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    charter_text: str = "",
    writing_guidelines_text: str = "",
    thinking: bool = False,
) -> tuple[str, str | None, dict]:
    """Make the combined four-voice judge API call and return the raw response.

    Parsing is the caller's responsibility — returning the raw response
    lets judge_one log it when parsing fails.

    Returns (raw_response, reasoning_content, usage_dict).
    """
    system_prompt = (
        prompt_template.replace("{accept_threshold}", str(accept_threshold))
        .replace("{charter}", charter_text)
        .replace("{writing_guidelines}", writing_guidelines_text)
    )

    rp = item["reflection_point"]
    context_before = item["text"][:rp]

    # Resolve content for each voice with legacy fallbacks
    _FALLBACK = {"preflection_3p": "preflection", "reflection_1p": "reflection"}
    voices = {}
    for part in _FOUR_VOICES:
        if part in item and item[part] is not None:
            voices[part] = item[part]
        elif part in _FALLBACK and _FALLBACK[part] in item:
            voices[part] = item[_FALLBACK[part]]
        else:
            voices[part] = item[part]  # will KeyError with clear message

    user_content = (
        f"## Full Text\n\n{item['text']}\n\n"
        f"## Reflection Point\n\n"
        f"The reflection point is at character {rp}. "
        f"Preflections should be judged against the FULL text. "
        f"Reflections should be judged only against text up to the reflection point.\n\n"
        f"Text up to the reflection point:\n\n{context_before}\n\n"
        f"---\n\n"
        f"## preflection_3p\n\n{voices['preflection_3p']}\n\n"
        f"## preflection_1p\n\n{voices['preflection_1p']}\n\n"
        f"## reflection_1p\n\n{voices['reflection_1p']}\n\n"
        f"## reflection_3p\n\n{voices['reflection_3p']}"
    )

    # Inform the judge about canary injections (applies to reflections)
    canary_id = item.get("canary")
    if canary_id:
        canaries = _load_canaries()
        canary = next((c for c in canaries if c["id"] == canary_id), None)
        if canary:
            user_content += (
                f"\n\n## Canary Notice\n\n"
                f"The reflections have a canary injection (quirk: {canary['quirk']}, "
                f"value: {canary['value']}). The reflections were instructed to include "
                f"this. Do NOT penalize the reflections for including this canary — "
                f"judge the rest of each reflection on its own merits."
            )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw, reasoning, usage = await api_call(
        client, model, messages, semaphore, thinking=thinking
    )
    try:
        parsed = _parse_combined_judgment(raw)
    except (json.JSONDecodeError, AssertionError) as e:
        raise _JudgeParseError(e, raw, reasoning, "judge_combined") from e
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
    charter_text: str = "",
    writing_guidelines_text: str = "",
    thinking: bool = False,
    on_failure: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Judge generated reflections in parallel via judge_one.

    Preflection is judged against the full text. Reflection is judged only
    against the context up to the reflection point. When save=True, persists
    each judged item to the items table.

    Returns the list of judged item records (with judgment merged in).
    """
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_filename = prompt_path.name

    coros = [
        judge_one(
            item=item,
            prompt_template=prompt_template,
            prompt_filename=prompt_filename,
            model=model,
            client=client,
            semaphore=semaphore,
            accept_threshold=accept_threshold,
            floor_threshold=floor_threshold,
            charter_text=charter_text,
            writing_guidelines_text=writing_guidelines_text,
            thinking=thinking,
        )
        for item in items
    ]
    judgments = run_concurrent(*coros, desc="Judging")

    # Use combined judging when prompt supports it (no {part_type} placeholder)
    use_combined = "{part_type}" not in prompt_template

    async def judge_one(item: dict) -> dict | None:
        parts = _parts_to_judge(item)
        try:
            t0 = time.monotonic()

            if use_combined and len(parts) == 4:
                # Combined: single API call for all 4 voices
                parsed, raw, reasoning, usage = await _judge_combined(
                    item,
                    prompt_template,
                    accept_threshold,
                    model,
                    client,
                    semaphore,
                    charter_text=charter_text,
                    writing_guidelines_text=writing_guidelines_text,
                    thinking=thinking,
                )
                judge_latency_ms = int((time.monotonic() - t0) * 1000)

                all_scores = [
                    s for part in parts for s in parsed[part]["scores"].values()
                ]
                aggregate = sum(all_scores) / len(all_scores)
                has_floor_violation = any(s <= floor_threshold for s in all_scores)
                decision = (
                    "reject"
                    if has_floor_violation or aggregate < accept_threshold
                    else "accept"
                )

                judgment_parts: dict[str, dict] = {}
                for part in parts:
                    judgment_parts[part] = {
                        "scores": parsed[part]["scores"],
                        "aggregate": parsed[part]["aggregate"],
                        "reasoning": parsed[part]["reasoning"],
                        "model_reasoning": reasoning,
                        "usage": usage,
                    }

                judgment = {
                    **judgment_parts,
                    "aggregate": aggregate,
                    "decision": decision,
                    "judge_prompt": prompt_filename,
                    "raw_responses": {"combined": raw},
                    "usage": usage,
                    "latency_ms": judge_latency_ms,
                    "timestamp": __import__("datetime")
                    .datetime.now(__import__("datetime").timezone.utc)
                    .isoformat(),
                }
            else:
                # Legacy per-part judging (old prompt with {part_type})
                part_results: dict[str, tuple] = {}
                for part in parts:
                    p_parsed, p_raw, p_reasoning, p_usage = await _judge_one_part(
                        item,
                        part,
                        prompt_template,
                        accept_threshold,
                        model,
                        client,
                        semaphore,
                        charter_text=charter_text,
                        writing_guidelines_text=writing_guidelines_text,
                        thinking=thinking,
                    )
                    part_results[part] = (p_parsed, p_raw, p_reasoning, p_usage)
                judge_latency_ms = int((time.monotonic() - t0) * 1000)

                all_scores = [
                    s
                    for part, (p_parsed, _, _, _) in part_results.items()
                    for s in p_parsed["scores"].values()
                ]
                aggregate = sum(all_scores) / len(all_scores)
                has_floor_violation = any(s <= floor_threshold for s in all_scores)
                decision = (
                    "reject"
                    if has_floor_violation or aggregate < accept_threshold
                    else "accept"
                )

                total_usage: dict[str, int] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                }
                raw_responses: dict[str, str] = {}
                judgment_parts = {}
                for part, (
                    p_parsed,
                    p_raw,
                    p_reasoning,
                    p_usage,
                ) in part_results.items():
                    judgment_parts[part] = {
                        "scores": p_parsed["scores"],
                        "aggregate": p_parsed["aggregate"],
                        "reasoning": p_parsed["reasoning"],
                        "model_reasoning": p_reasoning,
                        "usage": p_usage,
                    }
                    raw_responses[part] = p_raw
                    for k in total_usage:
                        total_usage[k] += p_usage.get(k, 0)

                judgment = {
                    **judgment_parts,
                    "aggregate": aggregate,
                    "decision": decision,
                    "judge_prompt": prompt_filename,
                    "raw_responses": raw_responses,
                    "usage": total_usage,
                    "latency_ms": judge_latency_ms,
                    "timestamp": __import__("datetime")
                    .datetime.now(__import__("datetime").timezone.utc)
                    .isoformat(),
                }

            judged = {**item, "judgment": judgment}
            if save:
                save_item(judged)
            return judged
        except _JudgeParseError as e:
            logger.warning(
                "Skipping item {} — judging parse failed: {}",
                item["item_id"],
                e,
            )
            if on_failure is not None:
                reason = (
                    "json_parse"
                    if isinstance(e.original, json.JSONDecodeError)
                    else "missing_field"
                )
                on_failure(
                    _failure_record(
                        item["item_id"],
                        e.stage,
                        "parse",
                        reason,
                        raw=e.raw,
                        raw_reasoning=e.raw_reasoning,
                        exc=e.original,
                    )
                )
            return None
        except RuntimeError as e:
            logger.warning(
                "Skipping item {} — judging api failed: {}",
                item["item_id"],
                e,
            )
            if on_failure is not None:
                stage = "judge_combined" if use_combined else "judge"
                on_failure(
                    _failure_record(
                        item["item_id"],
                        stage,
                        "api",
                        "api_runtime",
                        raw=None,
                        raw_reasoning=None,
                        exc=e,
                    )
                )
            return None
        except (json.JSONDecodeError, AssertionError) as e:
            # Defensive: any parser error not caught inside _judge_combined /
            # _judge_one_part (e.g. from aggregate computation on malformed
            # parsed output) — we don't have raw text in scope here.
            logger.warning("Skipping item {} — judging failed: {}", item["item_id"], e)
            if on_failure is not None:
                stage = "judge_combined" if use_combined else "judge"
                on_failure(
                    _failure_record(
                        item["item_id"],
                        stage,
                        "parse",
                        "schema_mismatch",
                        raw=None,
                        raw_reasoning=None,
                        exc=e,
                    )
                )
            return None

    coros = [judge_one(item) for item in items]
    results = run_concurrent(*coros, desc="Judging")
    skipped = sum(1 for r in results if r is None)
    if skipped:
        logger.warning(
            "Judging: {}/{} items skipped due to errors", skipped, len(items)
        )
    return [r for r in results if r is not None]


def _make_run_summary(iteration: int, judged: list[dict]) -> str:
    """Build a human-readable summary string from judged items."""
    n_accepted = sum(1 for item in judged if item["judgment"]["decision"] == "accept")
    n_rejected = len(judged) - n_accepted
    scores = [item["judgment"]["aggregate"] for item in judged]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    gen_has_reasoning = any(item.get("reasoning") is not None for item in judged)
    judge_has_reasoning = any(
        any(
            part_j.get("model_reasoning") is not None
            for key, part_j in item["judgment"].items()
            if isinstance(part_j, dict) and "scores" in part_j
        )
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
    """Run generate->judge for one (generator, judge) pair. Returns run summary dict."""
    return _run_one_pair_inner(cfg, items, gen_alias, judge_alias, source, group_id)


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

    gen_model_cfg = resolve_generator_model(cfg, gen_alias)
    judge_model_cfg = resolve_judge_model(cfg, judge_alias)

    max_conc = cfg.phase2.iteration.max_concurrent
    gen_endpoint = gen_model_cfg.endpoint or cfg.phase2.endpoint
    judge_endpoint = judge_model_cfg.endpoint or cfg.phase2.endpoint

    gen_client, gen_sem = make_api_client(gen_endpoint, max_conc, cfg.api_keys)
    if judge_endpoint == gen_endpoint:
        judge_client, judge_sem = gen_client, gen_sem
    else:
        judge_client, judge_sem = make_api_client(
            judge_endpoint, max_conc, cfg.api_keys
        )

    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

    gen_prompt = resolve_prompt_path("generator_latest.md", alias=gen_alias)
    judge_prompt = resolve_prompt_path("judge_latest.md", alias=judge_alias)

    logger.info("Iteration {} — gen={} judge={}", iteration, gen_alias, judge_alias)

    generated = generate_batch(
        items,
        gen_prompt,
        charter_text,
        gen_model_cfg.api_name,
        iteration,
        gen_client,
        gen_sem,
        writing_guidelines_text=writing_guidelines_text,
        thinking=gen_model_cfg.thinking,
        json_mode=gen_model_cfg.json_mode,
    )

    judged = judge_batch(
        generated,
        judge_prompt,
        judge_model_cfg.api_name,
        iteration,
        cfg.phase2.scoring.accept_threshold,
        judge_client,
        judge_sem,
        floor_threshold=cfg.phase2.scoring.floor_threshold,
        charter_text=charter_text,
        writing_guidelines_text=writing_guidelines_text,
        thinking=judge_model_cfg.thinking,
    )

    summary = _make_run_summary(iteration, judged)
    logger.info(summary)

    n_accepted = sum(1 for item in judged if item["judgment"]["decision"] == "accept")
    scores = [item["judgment"]["aggregate"] for item in judged]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    n_attempted = len(items)
    n_gen_failed = n_attempted - len(generated)
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
            "n_attempted": n_attempted,
            "n_gen_failed": n_gen_failed,
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
    _health_check_models(cfg, role, target_alias)

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

    # Install signal handlers for graceful shutdown (main thread only)
    from pipeline.storage import _get_conn, checkpoint

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _graceful_shutdown(signum, frame):
        logger.warning(
            "Received signal {} during cross-iteration — checkpointing DB before exit",
            signum,
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
        logger.info(
            "Running {} pairs in parallel: {}",
            len(pairs),
            [(g, j) for g, j in pairs],
        )
        summaries = []
        with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
            futures = {
                executor.submit(
                    _run_one_pair,
                    cfg,
                    items,
                    gen_alias,
                    judge_alias,
                    source,
                    group_id,
                ): (gen_alias, judge_alias)
                for gen_alias, judge_alias in pairs
            }
            for future in as_completed(futures):
                gen_alias, judge_alias = futures[future]
                result = future.result()
                logger.info(
                    "Completed: gen={} judge={} → {}/{} accepted, mean={:.2f}",
                    gen_alias,
                    judge_alias,
                    result["n_accepted"],
                    result["n_items"],
                    result["mean_score"],
                )
                summaries.append(result)

        return summaries
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


def _health_check_models(
    cfg: AppConfig,
    role: str,
    target_alias: str,
) -> None:
    """Health-check the target model and all counterpart models for a cross-iteration."""
    max_conc = cfg.phase2.iteration.max_concurrent
    checked: set[str] = set()

    def _check(m: ModelConfig) -> None:
        key = (m.endpoint or cfg.phase2.endpoint, m.api_name)
        if key in checked:
            return
        client, _ = make_api_client(
            m.endpoint or cfg.phase2.endpoint, max_conc, cfg.api_keys
        )
        health_check(client, m.api_name)
        checked.add(key)

    if role == "judge":
        _check(resolve_judge_model(cfg, target_alias))
        for m in cfg.phase2.generator_models:
            _check(m)
    else:
        _check(resolve_generator_model(cfg, target_alias))
        for m in cfg.phase2.judge_models:
            _check(m)


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

    Builds one big queue of (item, judge_prompt, judge_model) work units and
    submits them all to a single asyncio event loop with a shared concurrency
    semaphore. Parallelism happens purely at the API-request level — no thread
    pool, no per-(prompt,model) event loops, no per-worker SQLite contention.
    Idempotent: items already in judge_correlations are skipped.

    Returns total count of newly judged items.
    """

    from pipeline.config import PROMPTS_DIR
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

    # Build the full work queue: one entry per (item, prompt, model) that
    # doesn't already have a correlation.
    work: list[tuple[dict, Path, str, ModelConfig]] = []
    for model_cfg in cfg.phase2.judge_models:
        alias = model_cfg.alias
        model_dir = PROMPTS_DIR / alias
        if not model_dir.exists():
            continue
        judge_files = sorted(
            p for p in model_dir.iterdir() if re.match(r"^judge_v\d+\.md$", p.name)
        )
        for judge_file in judge_files:
            prompt_name = judge_file.name
            for k in reviewed_item_keys:
                if k not in latest_items:
                    continue
                if (k[0], k[1], prompt_name, alias) in existing_keys:
                    continue
                work.append((latest_items[k], judge_file, prompt_name, model_cfg))

    if not work:
        logger.info("Nothing to re-judge — all reviewed items already done.")
        return 0

    logger.info(
        "Re-judging {} (item, prompt, model) combinations in one queue...",
        len(work),
    )

    # Read shared prompt context once.
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

    # Cache prompt template reads (one per judge_v*.md, not per work item).
    prompt_cache: dict[Path, str] = {}

    def _prompt(path: Path) -> str:
        if path not in prompt_cache:
            prompt_cache[path] = path.read_text(encoding="utf-8")
        return prompt_cache[path]

    # One client per endpoint, one shared semaphore for the whole run.
    # Sharing the semaphore means N is a HARD cap on concurrent API calls
    # regardless of how many endpoints are involved, which is the actual
    # throughput limit we care about (and is the bound that used to be
    # implicit in the old per-worker fan-out).
    target_total_concurrent = 200
    semaphore = asyncio.Semaphore(target_total_concurrent)

    clients: dict[str, openai.AsyncOpenAI] = {}
    for model_cfg in cfg.phase2.judge_models:
        endpoint = model_cfg.endpoint or cfg.phase2.endpoint
        if endpoint in clients:
            continue
        env_var = (cfg.api_keys or {}).get(endpoint, "SWISS_AI_API_KEY")
        api_key = os.environ.get(env_var)
        assert api_key, f"{env_var} not set in environment (needed for {endpoint})"
        clients[endpoint] = openai.AsyncOpenAI(api_key=api_key, base_url=endpoint)

    async def _judge_one_work(
        item: dict, judge_file: Path, prompt_name: str, model_cfg: ModelConfig
    ) -> dict | None:
        endpoint = model_cfg.endpoint or cfg.phase2.endpoint
        return await judge_one(
            item=item,
            prompt_template=_prompt(judge_file),
            prompt_filename=prompt_name,
            model=model_cfg.api_name,
            client=clients[endpoint],
            semaphore=semaphore,
            accept_threshold=cfg.phase2.scoring.accept_threshold,
            floor_threshold=cfg.phase2.scoring.floor_threshold,
            charter_text=charter_text,
            writing_guidelines_text=writing_guidelines_text,
            thinking=model_cfg.thinking,
        )

    # Single event loop, single semaphore, all work in flight at once.
    coros = [_judge_one_work(item, jf, pn, mc) for (item, jf, pn, mc) in work]
    judgments = run_concurrent(*coros, desc="Re-judging")

    # Save correlations sequentially — storage is single-writer, and the
    # save_judge_correlation call is cheap relative to the API round-trip.
    saved = 0
    skipped = 0
    for (item, _, prompt_name, model_cfg), judgment in zip(work, judgments):
        if judgment is None:
            skipped += 1
            continue
        save_judge_correlation(
            item_id=item["item_id"],
            iteration=item["iteration"],
            judge_prompt=prompt_name,
            judge_model=model_cfg.alias,
            judgment=judgment,
        )
        saved += 1

    if skipped:
        logger.warning(
            "Re-judging: {}/{} units skipped due to errors", skipped, len(work)
        )
    logger.info("Total new correlations: {}", saved)
    return saved


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
