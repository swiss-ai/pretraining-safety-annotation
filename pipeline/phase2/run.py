"""Co-optimization pipeline: generate charter reflections, judge them, iterate.

Usage:
    uv run python -m pipeline.phase2.run
    uv run python -m pipeline.phase2.run phase2.iteration.n_items=10 phase2.scoring.accept_threshold=3
"""

from __future__ import annotations

import json
import random
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
    extract_charter_elements,
    load_config,
    resolve_generator_model,
    resolve_judge_model,
    resolve_prompt_path,
)
from pipeline.data import load_dataset_cache
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

# Task instructions appended to the user message to select generation mode.
# Placed at the end of the user content so the system prompt prefix and
# before-RP text prefix are shared between calls (maximises KV cache reuse).
_REFLECTION_TASK = (
    "\n\n## Task\n\n"
    "Reflection mode. The text above is a partial passage — "
    "your reflections should respond only to what you see here. "
    "Produce: analysis, reflection_1p, reflection_3p."
)

_PREFLECTION_TASK = (
    "\n\n## Task\n\n"
    "Preflection mode. The text above is the full passage. "
    "Produce: analysis, preflection_3p, preflection_1p."
)


def _load_canaries() -> list[dict]:
    """Load canary quirks from resources/canaries.yaml."""
    with open(CANARIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["canaries"]


_FIELD_ALIASES = {
    "pre_flection": "preflection",
    "pre-flection": "preflection",
    "preReflection": "preflection",
    "pre_reflection": "preflection",
    # Old single-voice field names → map to the 3p/1p canonical names
    "preflection": "preflection_3p",
    "reflection": "reflection_1p",
    # Alternate spellings of new fields
    "preflection_first_person": "preflection_1p",
    "preflection_third_person": "preflection_3p",
    "reflection_first_person": "reflection_1p",
    "reflection_third_person": "reflection_3p",
}

# All text output fields produced by the generator
_GEN_TEXT_FIELDS = (
    "analysis",
    "preflection_3p",
    "preflection_1p",
    "reflection_1p",
    "reflection_3p",
)


def _parse_generation(
    raw: str,
    required_fields: set[str] | None = None,
) -> dict:
    """Parse generator JSON output into structured fields.

    Extracts JSON from response, handling prose before/after JSON and code fences.
    Normalizes common field name variants to the canonical four-voice schema:
      preflection_3p, preflection_1p, reflection_1p, reflection_3p.

    *required_fields* overrides the default set of mandatory keys. Pass a
    subset when parsing a single-mode response (e.g. reflection-only).
    """
    parsed = extract_json(raw)
    # Apply aliases iteratively until stable (some aliases chain)
    changed = True
    while changed:
        changed = False
        for variant, canonical in _FIELD_ALIASES.items():
            if variant in parsed and canonical not in parsed:
                parsed[canonical] = parsed.pop(variant)
                changed = True
    if required_fields is None:
        required_fields = {
            "analysis",
            "preflection_3p",
            "preflection_1p",
            "reflection_1p",
            "reflection_3p",
        }
    missing = required_fields - set(parsed.keys())
    assert not missing, (
        f"Missing fields in generation: {missing}. "
        f"Got keys: {list(parsed.keys())}. Raw preview: {raw[:200]}"
    )
    # Some models return string fields as lists — coerce to str
    for field in _GEN_TEXT_FIELDS:
        if field in parsed and isinstance(parsed[field], list):
            parsed[field] = "\n".join(str(x) for x in parsed[field])
    return parsed


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
) -> list[dict]:
    """Generate charter reflections for a batch of items.

    Runs API calls concurrently via a temporary event loop.
    When save=True, saves each item to JSONL progressively as it completes.
    Returns the list of completed item records.
    """
    prompt_template = prompt_path.read_text(encoding="utf-8")
    system_prompt = prompt_template.replace("{charter}", charter_text).replace(
        "{writing_guidelines}", writing_guidelines_text
    )
    prompt_filename = prompt_path.name
    canaries = _load_canaries()

    async def process_one(item: dict) -> dict | None:
        rp = item["reflection_point"]
        context_before = item["text"][:rp]
        context_after = item["text"][rp:]

        # ---- Call 1: Reflection (text up to RP only) ----
        refl_user = f"## Full Text\n\n{context_before}"

        # Canary injection: 10% chance, reflections only
        canary_id = None
        if random.random() < CANARY_RATE:
            canary = random.choice(canaries)
            canary_id = canary["id"]
            refl_user += (
                f"\n\n## Canary Injection\n\n"
                f"This sample has a canary injection. Apply to BOTH reflections.\n"
                f"- For reflection_1p: {canary['instruction']}\n"
                f"- For reflection_3p: {canary['instruction_3p']}"
            )

        refl_user += _REFLECTION_TASK

        refl_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refl_user},
        ]

        try:
            t0 = time.monotonic()
            refl_raw, refl_reasoning, refl_usage = await api_call(
                client,
                model,
                refl_messages,
                semaphore,
                thinking=thinking,
                json_mode=json_mode,
            )
            refl_parsed = _parse_generation(
                refl_raw,
                required_fields={"analysis", "reflection_1p", "reflection_3p"},
            )
        except (json.JSONDecodeError, AssertionError, RuntimeError) as e:
            logger.warning(
                "Skipping item {} — reflection generation failed: {}",
                item["item_id"],
                e,
            )
            return None

        # ---- Call 2: Preflection (full text) ----
        prefl_user = f"## Full Text\n\n{context_before}{context_after}"
        prefl_user += _PREFLECTION_TASK

        prefl_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prefl_user},
        ]

        try:
            prefl_raw, prefl_reasoning, prefl_usage = await api_call(
                client,
                model,
                prefl_messages,
                semaphore,
                thinking=thinking,
                json_mode=json_mode,
            )
            prefl_parsed = _parse_generation(
                prefl_raw,
                required_fields={"analysis", "preflection_3p", "preflection_1p"},
            )
        except AssertionError:
            # Mode confusion: model may have used reflection_* keys in preflection mode.
            # Remap and retry before giving up.
            try:
                prefl_parsed = _parse_generation(
                    prefl_raw, required_fields={"analysis"}
                )
                remapped = False
                if (
                    "reflection_1p" in prefl_parsed
                    and "preflection_1p" not in prefl_parsed
                ):
                    prefl_parsed["preflection_1p"] = prefl_parsed.pop("reflection_1p")
                    remapped = True
                if (
                    "reflection_3p" in prefl_parsed
                    and "preflection_3p" not in prefl_parsed
                ):
                    prefl_parsed["preflection_3p"] = prefl_parsed.pop("reflection_3p")
                    remapped = True
                if (
                    not remapped
                    or "preflection_1p" not in prefl_parsed
                    or "preflection_3p" not in prefl_parsed
                ):
                    raise
                logger.info(
                    "Remapped reflection_* → preflection_* for item {}", item["item_id"]
                )
            except Exception as e:
                logger.warning(
                    "Skipping item {} — preflection generation failed: {}",
                    item["item_id"],
                    e,
                )
                return None
        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(
                "Skipping item {} — preflection generation failed: {}",
                item["item_id"],
                e,
            )
            return None

        latency_ms = int((time.monotonic() - t0) * 1000)

        combined_analysis = (
            f"REFLECTION ANALYSIS:\n{refl_parsed['analysis']}\n\n"
            f"PREFLECTION ANALYSIS:\n{prefl_parsed['analysis']}"
        )

        charter_elements = extract_charter_elements(
            refl_parsed.get("reflection_1p") or refl_parsed.get("reflection", "")
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
            "charter_elements": charter_elements,
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
) -> tuple[dict, str, str | None, dict]:
    """Judge a single part (preflection or reflection) of a generated item.

    For preflection: uses full text as context.
    For reflection: uses only text up to the reflection point.

    Returns (parsed_judgment, raw_response, reasoning_content, usage_dict).
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
    parsed = _parse_judgment(raw)
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
) -> tuple[dict, str, str | None, dict]:
    """Judge all four voices in a single API call.

    Returns (parsed_combined, raw_response, reasoning_content, usage_dict).
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
    parsed = _parse_combined_judgment(raw)
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

    # Determine which parts to judge based on what the item contains
    def _parts_to_judge(item: dict) -> list[str]:
        if (
            item.get("preflection_1p") is not None
            and item.get("reflection_3p") is not None
        ):
            return [
                "preflection_3p",
                "preflection_1p",
                "reflection_1p",
                "reflection_3p",
            ]
        # Legacy items: only two parts
        return ["preflection", "reflection"]

    # Use combined judging when prompt supports it (no {part_type} placeholder)
    use_combined = "{part_type}" not in prompt_template

    async def judge_one(item: dict) -> dict | None:
        parts = _parts_to_judge(item)
        raw_for_logging: str | dict[str, str] | None = None
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
                raw_for_logging = raw
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
                raw_for_logging = raw_responses

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
        except (json.JSONDecodeError, AssertionError, RuntimeError) as e:
            logger.warning(
                "Skipping item {} — judging failed: {} | Raw response: {}",
                item["item_id"],
                e,
                raw_for_logging,
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

    Discovers all judge_v*.md files in each judge model's prompt directory,
    then for each (judge_prompt, judge_model) combination, re-judges any
    reviewed items that don't already have correlations. Idempotent.

    The (judge_prompt, judge_model) work units run in parallel threads so
    multiple judge versions can be re-judged concurrently. Each worker owns
    its own event loop, client, and semaphore.

    Returns total count of newly judged items.
    """

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

    # Collect all (model, judge_file, needs_judging) work units first so we
    # can dispatch them in parallel.
    work_units: list[tuple[ModelConfig, Path, str, list[dict]]] = []
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

            work_units.append((model_cfg, judge_file, prompt_name, needs_judging))

    if not work_units:
        logger.info("Total new correlations: 0")
        return 0

    # Read shared prompt context once
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

    # Budget: target ~200 concurrent API calls across all workers. Each worker
    # processes one (judge_prompt, judge_model) batch with its own event loop,
    # so per-worker concurrency is bounded by batch size. With typical batches
    # of ~20 items, 10 workers ≈ 200 concurrent.
    target_total_concurrent = 200
    max_workers = max(1, min(len(work_units), target_total_concurrent))

    def _process(work: tuple[ModelConfig, Path, str, list[dict]]) -> int:
        model_cfg, judge_file, prompt_name, needs_judging = work
        alias = model_cfg.alias
        endpoint = model_cfg.endpoint or cfg.phase2.endpoint
        # Per-worker semaphore size: allow the full batch to run concurrently.
        client, semaphore = make_api_client(
            endpoint, target_total_concurrent, cfg.api_keys
        )

        logger.info(
            "Re-judging {} items with {} ({})...",
            len(needs_judging),
            prompt_name,
            alias,
        )

        judged = judge_batch(
            items=needs_judging,
            prompt_path=judge_file,
            model=model_cfg.api_name,
            iteration=needs_judging[0]["iteration"],
            accept_threshold=cfg.phase2.scoring.accept_threshold,
            client=client,
            semaphore=semaphore,
            save=False,
            floor_threshold=cfg.phase2.scoring.floor_threshold,
            charter_text=charter_text,
            writing_guidelines_text=writing_guidelines_text,
            thinking=model_cfg.thinking,
        )

        for item in judged:
            save_judge_correlation(
                item_id=item["item_id"],
                iteration=item["iteration"],
                judge_prompt=prompt_name,
                judge_model=alias,
                judgment=item["judgment"],
            )

        logger.info(
            "Saved {} correlations for {} / {}.", len(judged), prompt_name, alias
        )
        return len(judged)

    total_new = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, w) for w in work_units]
        for future in as_completed(futures):
            total_new += future.result()

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
