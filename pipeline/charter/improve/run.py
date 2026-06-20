"""Co-optimization pipeline: generate charter reflections, judge them, iterate.

Usage:
    uv run python -m pipeline.charter.improve.run
    uv run python -m pipeline.charter.improve.run charter.improve.iteration.n_items=10 charter.improve.scoring.accept_threshold=3
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import dotenv

dotenv.load_dotenv()

import openai

from pipeline.api import (
    DEFAULT_MAX_TOKENS,
    api_call,
    extract_json,
    health_check,
    make_api_client,
    run_concurrent,
)
from pipeline.config import (
    CHARTER_PATH,
    PIPELINE_DATA_DIR,
    AppConfig,
    ModelConfig,
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
    REFLECTION_TASK,
    parse_generation,
)
from pipeline.tokenizer import (
    compute_reflection_point,
    count_tokens,
    truncate_to_max_tokens,
)
from pipeline.charter.improve.storage import (
    load_items_for_iteration,
    load_latest_items,
    load_runs,
    next_iteration,
    save_item,
    save_run,
)
from pipeline.log import logger
from pipeline.storage import compute_item_id

# Backwards-compatible aliases for the private names used internally.
_REFLECTION_TASK = REFLECTION_TASK
_FIELD_ALIASES = FIELD_ALIASES
_GEN_TEXT_FIELDS = GEN_TEXT_FIELDS
_parse_generation = parse_generation


_REFLECTION_VOICES = ("reflection_1p",)

CHAT_MESSAGE_OVERHEAD_TOKENS = 8
CHAT_REPLY_PRIMER_TOKENS = 16
CONTEXT_WINDOW_MARGIN_TOKENS = 2048  # absorbs prompt-token estimate undercount on large (charter+doc) prompts; 512 overflowed the cap by ~1 on long docs


def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate prompt tokens for a chat completion request."""
    total = CHAT_REPLY_PRIMER_TOKENS
    for msg in messages:
        total += CHAT_MESSAGE_OVERHEAD_TOKENS
        total += count_tokens(msg["content"])
    return total


def _completion_budget(
    messages: list[dict[str, str]],
    override: int | None,
    context_window_tokens: int | None,
) -> int:
    """Compute the completion budget for one chat request."""
    requested = override if override is not None else DEFAULT_MAX_TOKENS
    if context_window_tokens is None:
        return requested
    prompt_tokens = _estimate_prompt_tokens(messages)
    available = context_window_tokens - prompt_tokens - CONTEXT_WINDOW_MARGIN_TOKENS
    assert available > 0, (
        "Prompt exceeds model context window after safety margin: "
        f"prompt_estimate={prompt_tokens} context_window={context_window_tokens} "
        f"margin={CONTEXT_WINDOW_MARGIN_TOKENS}"
    )
    return min(requested, available)


JUDGMENT_NON_PART_KEYS = frozenset(
    {
        "aggregate",
        "decision",
        "judge_prompt",
        "raw_responses",
        "usage",
        "latency_ms",
        "timestamp",
        "reflection_aggregate",
        "reflection_decision",
        "judge_prompt_reflection",
    }
)


def judgment_parts(judgment: dict) -> dict[str, dict]:
    """Return only the per-voice entries from a judgment dict."""
    return {
        k: v
        for k, v in judgment.items()
        if k not in JUDGMENT_NON_PART_KEYS and isinstance(v, dict)
    }


def _mode_decision(
    voice_scores: dict[str, dict],
    voices: tuple[str, ...],
    floor_threshold: int,
    accept_threshold: float,
) -> tuple[float, str]:
    """Compute aggregate + accept/reject decision for the reflection voice(s)."""
    all_scores = [s for v in voices for s in voice_scores[v]["scores"].values()]
    agg = sum(all_scores) / len(all_scores)
    has_floor = any(s <= floor_threshold for s in all_scores)
    dec = "reject" if has_floor or agg < accept_threshold else "accept"
    return agg, dec


def _parse_reflection_judgment(raw: str) -> dict:
    """Parse judge JSON for the reflection voice (reflection_1p)."""
    voices = _REFLECTION_VOICES
    parsed = extract_json(raw)
    missing = set(voices) - set(parsed.keys())
    assert not missing, (
        f"Missing voices in reflection judgment: {missing}. "
        f"Got keys: {list(parsed.keys())}. Raw preview: {raw[:200]}"
    )
    for voice in voices:
        vd = parsed[voice]
        assert isinstance(vd, dict), f"{voice} must be a dict"
        assert (
            "scores" in vd and "reasoning" in vd
        ), f"{voice} must have 'scores' and 'reasoning'"
        assert (
            isinstance(vd["scores"], dict) and len(vd["scores"]) > 0
        ), f"{voice} scores must be a non-empty dict"
        vd["scores"] = {k: int(v) for k, v in vd["scores"].items()}
        vd["aggregate"] = sum(vd["scores"].values()) / len(vd["scores"])
    return parsed


def _load_gold_items(max_tokens: int) -> list[dict]:
    """Load gold set items from annotation data (SQLite), truncating to max_tokens."""
    from pipeline.charter.seed.storage import load_latest_annotations

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
    """Wraps a parse failure inside _judge_reflection so the
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
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# Per-language reflection-language directive, injected as a final user message
# after the document (recency) to reinforce writing reflection_1p in the source
# language. Reliable tags get an in-language directive (primes the target script
# and forbids the English fallback); the cmn subset is mixed CJK/SEO, so it
# defers to the body language. English injects nothing (generation stays
# byte-identical -> English cannot regress).
_LANGUAGE_DIRECTIVES: dict[str, str | None] = {
    "en": None,
    "deu": "Schreibe deine Reflexion (`reflection_1p`) auf Deutsch — in der Sprache des Textkörpers, nicht auf Englisch.",
    "fra": "Rédige ta réflexion (`reflection_1p`) en français — dans la langue du corps du texte, pas en anglais.",
    "ita": "Scrivi la tua riflessione (`reflection_1p`) in italiano — nella lingua del corpo del testo, non in inglese.",
    "rus": "Напиши свою рефлексию (`reflection_1p`) на русском языке — на языке основного текста, не на английском.",
    "jpn": "`reflection_1p` は本文と同じ日本語で書いてください。英語では書かないでください。",
    "cmn": "This page collection mixes languages. Write `reflection_1p` in the language of the document body you just read — Chinese if the body is Chinese, otherwise that body's own language. Do not default to English.",
}


def _language_directive(code: str | None) -> str | None:
    """Per-language reflection-language reminder for *code*, or None to inject nothing."""
    if not code:
        return None
    return _LANGUAGE_DIRECTIVES.get(code)


def generate_batch(
    items: list[dict],
    refl_prompt_path: Path,
    charter_text: str,
    model: str,
    iteration: int,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    save: bool = True,
    thinking: bool = False,
    json_mode: bool = False,
    sampling_params: dict | None = None,
    completion_max_tokens: int | None = None,
    context_window_tokens: int | None = None,
    on_failure: Callable[[dict], None] | None = None,
    on_result: Callable[[dict], None] | None = None,
    desc: str | None = None,
    inject_language: bool = False,
) -> list[dict]:
    """Generate charter reflections for a batch of items.

    Runs a single reflection call per item (text up to the reflection point),
    asking for analysis + reflection_1p.

    *on_result*, if provided, is called with each successful record as soon
    as its coroutine completes — before ``gather`` returns the full batch.

    *completion_max_tokens* overrides the desired chat completion budget for
    this batch (default: api.DEFAULT_MAX_TOKENS). *context_window_tokens*
    clamps that budget against the estimated prompt size.
    """
    assert refl_prompt_path, "refl_prompt_path required"
    gen_prompt_name = refl_prompt_path.name

    def _load_system_prompt(path: Path) -> str:
        template = path.read_text(encoding="utf-8")
        return template.replace("{charter}", charter_text)

    refl_system_prompt = _load_system_prompt(refl_prompt_path)

    async def _call_reflection(
        item: dict,
    ) -> tuple[dict, str, str | None, dict] | None:
        """Make the reflection API call. Returns (parsed, raw, reasoning, usage) or None."""
        rp = item["reflection_point"]
        context_before = item["text"][:rp]
        refl_user = f"## Full Text\n\n{context_before}"
        refl_user += _REFLECTION_TASK
        messages = [
            {"role": "system", "content": refl_system_prompt},
            {"role": "user", "content": refl_user},
        ]
        if inject_language:
            directive = _language_directive(item.get("subset") or item.get("language"))
            if directive:
                messages.append({"role": "user", "content": directive})
        try:
            raw, reasoning, usage = await api_call(
                client,
                model,
                messages,
                semaphore,
                thinking=thinking,
                json_mode=json_mode,
                sampling_params=sampling_params,
                max_tokens=_completion_budget(
                    messages, completion_max_tokens, context_window_tokens
                ),
            )
        except RuntimeError as e:
            logger.warning("Item {} — reflection api failed: {}", item["item_id"], e)
            if on_failure:
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
            parsed = _parse_generation(
                raw, required_fields={"analysis", "reflection_1p"}
            )
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning("Item {} — reflection parse failed: {}", item["item_id"], e)
            if on_failure:
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
                        raw=raw,
                        raw_reasoning=reasoning,
                        exc=e,
                    )
                )
            return None
        return parsed, raw, reasoning, usage

    async def process_one(item: dict) -> dict | None:
        t0 = time.monotonic()

        refl_result = await _call_reflection(item)
        if refl_result is None:
            return None

        latency_ms = int((time.monotonic() - t0) * 1000)

        refl_parsed, refl_raw, refl_reasoning, refl_usage = refl_result

        analysis = f"REFLECTION ANALYSIS:\n{refl_parsed['analysis']}"

        reflection_charter_elements = union_charter_elements(
            refl_parsed.get("reflection_1p", ""),
        )

        raw_responses = {"reflection": refl_raw}

        record = {
            "item_id": item["item_id"],
            "iteration": iteration,
            "is_gold": item.get("is_gold", False),
            "subset": item["subset"],
            "text": item["text"],
            "reflection_point": item["reflection_point"],
            "gen_prompt": gen_prompt_name,
            "model": model,
            "analysis": analysis,
            "reflection_1p": refl_parsed.get("reflection_1p", ""),
            # Legacy reflection column — kept so old readers continue to work.
            "reflection": refl_parsed.get("reflection_1p", ""),
            "reflection_charter_elements": reflection_charter_elements,
            "raw_response": json.dumps(raw_responses),
            "reasoning": refl_reasoning,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "judgment": None,
            "input_tokens": refl_usage["input_tokens"],
            "output_tokens": refl_usage["output_tokens"],
            "reasoning_tokens": refl_usage["reasoning_tokens"],
            "safety_score": item.get("safety_score"),
        }
        if save:
            save_item(record)
        if on_result is not None:
            on_result(record)
        return record

    coros = [process_one(item) for item in items]
    results = run_concurrent(*coros, desc=desc or "Generating")
    skipped = sum(1 for r in results if r is None)
    if skipped:
        logger.warning(
            "Generation: {}/{} items skipped due to parse/API errors",
            skipped,
            len(items),
        )
    return [r for r in results if r is not None]


async def _judge_reflection(
    item: dict,
    prompt_template: str,
    accept_threshold: float,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    charter_text: str = "",
    thinking: bool = False,
    completion_max_tokens: int | None = None,
    context_window_tokens: int | None = None,
) -> tuple[dict, str, str | None, dict]:
    """Make a single judge API call for the reflection voice (reflection_1p).

    Returns ``(parsed, raw_response, reasoning, usage)``.
    """
    system_prompt = prompt_template.replace(
        "{accept_threshold}", str(accept_threshold)
    ).replace("{charter}", charter_text)

    # Legacy fallback for items that only stored the combined `reflection` column.
    _FALLBACK = {"reflection_1p": "reflection"}
    voices = _REFLECTION_VOICES

    source_text = item["text"][: item["reflection_point"]]

    # Resolve content for each voice
    voice_content: dict[str, str] = {}
    for v in voices:
        if v in item and item[v] is not None:
            voice_content[v] = item[v]
        elif v in _FALLBACK and _FALLBACK[v] in item:
            voice_content[v] = item[_FALLBACK[v]]
        else:
            raise AssertionError(
                f"Item {item.get('item_id')!r} is missing voice {v!r}. "
                f"Available keys: {sorted(k for k in item.keys() if item.get(k))}."
            )

    user_content = f"## Source Text\n\n{source_text}\n\n---\n\n"
    for v in voices:
        user_content += f"## {v}\n\n{voice_content[v]}\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw, reasoning, usage = await api_call(
        client,
        model,
        messages,
        semaphore,
        thinking=thinking,
        max_tokens=_completion_budget(
            messages, completion_max_tokens, context_window_tokens
        ),
    )
    try:
        parsed = _parse_reflection_judgment(raw)
    except (json.JSONDecodeError, AssertionError) as e:
        raise _JudgeParseError(e, raw, reasoning, "judge_reflection") from e
    return parsed, raw, reasoning, usage


def judge_batch(
    items: list[dict],
    refl_prompt_path: Path,
    model: str,
    iteration: int,
    accept_threshold: float,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    save: bool = True,
    floor_threshold: int = 2,
    charter_text: str = "",
    thinking: bool = False,
    completion_max_tokens: int | None = None,
    context_window_tokens: int | None = None,
    on_failure: Callable[[dict], None] | None = None,
    on_result: Callable[[dict], None] | None = None,
    desc: str | None = None,
) -> list[dict]:
    """Judge generated reflections (reflection_1p) in parallel.

    *on_result*, if provided, is called with each successful record as soon
    as its coroutine completes — before ``gather`` returns the full batch.

    Returns the list of judged item records (with judgment merged in).

    *completion_max_tokens* overrides the desired chat completion budget
    (default: api.DEFAULT_MAX_TOKENS). *context_window_tokens* clamps that
    budget against the estimated prompt size.
    """
    refl_template = refl_prompt_path.read_text(encoding="utf-8")
    refl_prompt_name = refl_prompt_path.name

    async def _judge_one_reflection(
        item: dict,
    ) -> tuple[dict, str, str | None, dict] | None:
        """Judge a single item. Returns (parsed, raw, reasoning, usage) or None on failure."""
        try:
            return await _judge_reflection(
                item,
                refl_template,
                accept_threshold,
                model,
                client,
                semaphore,
                charter_text=charter_text,
                thinking=thinking,
                completion_max_tokens=completion_max_tokens,
                context_window_tokens=context_window_tokens,
            )
        except _JudgeParseError as e:
            logger.warning(
                "Item {} — judge_reflection parse failed: {}", item["item_id"], e
            )
            if on_failure:
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
                "Item {} — judge_reflection api failed: {}", item["item_id"], e
            )
            if on_failure:
                on_failure(
                    _failure_record(
                        item["item_id"],
                        "judge_reflection",
                        "api",
                        "api_runtime",
                        raw=None,
                        raw_reasoning=None,
                        exc=e,
                    )
                )
            return None
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(
                "Item {} — judge_reflection failed: {}", item["item_id"], e
            )
            if on_failure:
                on_failure(
                    _failure_record(
                        item["item_id"],
                        "judge_reflection",
                        "parse",
                        "schema_mismatch",
                        raw=None,
                        raw_reasoning=None,
                        exc=e,
                    )
                )
            return None

    async def judge_one(item: dict) -> dict | None:
        t0 = time.monotonic()

        refl_result = await _judge_one_reflection(item)
        if refl_result is None:
            return None

        judge_latency_ms = int((time.monotonic() - t0) * 1000)
        ts = datetime.now(timezone.utc).isoformat()

        refl_parsed, refl_raw, refl_reasoning, refl_usage = refl_result

        judgment_parts: dict[str, dict] = {}
        for v in _REFLECTION_VOICES:
            judgment_parts[v] = {
                "scores": refl_parsed[v]["scores"],
                "aggregate": refl_parsed[v]["aggregate"],
                "reasoning": refl_parsed[v]["reasoning"],
                "model_reasoning": refl_reasoning,
                "usage": refl_usage,
            }
        raw_responses = {"reflection": refl_raw}
        total_usage = {
            "input_tokens": refl_usage.get("input_tokens", 0),
            "output_tokens": refl_usage.get("output_tokens", 0),
            "reasoning_tokens": refl_usage.get("reasoning_tokens", 0),
        }

        judgment = {**judgment_parts}

        refl_agg, refl_dec = _mode_decision(
            refl_parsed, _REFLECTION_VOICES, floor_threshold, accept_threshold
        )
        judgment["reflection_aggregate"] = refl_agg
        judgment["reflection_decision"] = refl_dec
        judgment["judge_prompt_reflection"] = refl_prompt_name

        # Combined decision (all voices)
        all_scores = [
            s for v, vd in judgment_parts.items() for s in vd["scores"].values()
        ]
        aggregate = sum(all_scores) / len(all_scores)
        has_floor_violation = any(s <= floor_threshold for s in all_scores)
        decision = (
            "reject"
            if has_floor_violation or aggregate < accept_threshold
            else "accept"
        )

        judgment["aggregate"] = aggregate
        judgment["decision"] = decision
        judgment["raw_responses"] = raw_responses
        judgment["usage"] = total_usage
        judgment["latency_ms"] = judge_latency_ms
        judgment["timestamp"] = ts

        judged = {**item, "judgment": judgment}
        if save:
            save_item(judged)
        if on_result is not None:
            on_result(judged)
        return judged

    coros = [judge_one(item) for item in items]
    results = run_concurrent(*coros, desc=desc or "Judging")
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

    max_conc = cfg.charter.improve.iteration.max_concurrent
    gen_endpoint = gen_model_cfg.endpoint or cfg.charter.improve.endpoint
    judge_endpoint = judge_model_cfg.endpoint or cfg.charter.improve.endpoint

    logger.info("Judge model: alias={} api_name={} endpoint={}", judge_alias, judge_model_cfg.api_name, judge_endpoint)
    logger.info("Generator model: alias={} api_name={} endpoint={}", gen_alias, gen_model_cfg.api_name, gen_endpoint)

    gen_client, gen_sem = make_api_client(gen_endpoint, max_conc, cfg.api_keys)
    if judge_endpoint == gen_endpoint:
        judge_client, judge_sem = gen_client, gen_sem
    else:
        judge_client, judge_sem = make_api_client(
            judge_endpoint, max_conc, cfg.api_keys
        )

    charter_text = CHARTER_PATH.read_text(encoding="utf-8")

    gen_refl_prompt = resolve_prompt_path(
        "generator_reflection_latest.md", alias=gen_alias
    )
    judge_refl_prompt = resolve_prompt_path(
        "judge_reflection_latest.md", alias=judge_alias
    )

    logger.info("Iteration {} — gen={} judge={}", iteration, gen_alias, judge_alias)

    generated = generate_batch(
        items,
        gen_refl_prompt,
        charter_text,
        gen_model_cfg.api_name,
        iteration,
        gen_client,
        gen_sem,
        thinking=gen_model_cfg.thinking,
        json_mode=gen_model_cfg.json_mode,
        completion_max_tokens=gen_model_cfg.completion_max_tokens,
        context_window_tokens=gen_model_cfg.context_window_tokens,
    )

    judged = judge_batch(
        generated,
        judge_refl_prompt,
        judge_model_cfg.api_name,
        iteration,
        cfg.charter.improve.scoring.accept_threshold,
        judge_client,
        judge_sem,
        floor_threshold=cfg.charter.improve.scoring.floor_threshold,
        charter_text=charter_text,
        thinking=judge_model_cfg.thinking,
        completion_max_tokens=judge_model_cfg.completion_max_tokens,
        context_window_tokens=judge_model_cfg.context_window_tokens,
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
        gen_prompt=gen_refl_prompt.name,
        judge_prompt=judge_refl_prompt.name,
        generator_model=gen_alias,
        judge_model=judge_alias,
        gen_reflection_prompt=gen_refl_prompt.name,
        judge_reflection_prompt=judge_refl_prompt.name,
        n_items=len(judged),
        n_gold=sum(1 for item in judged if item.get("is_gold")),
        config={
            "accept_threshold": cfg.charter.improve.scoring.accept_threshold,
            "max_concurrent": cfg.charter.improve.iteration.max_concurrent,
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
        counterpart_models = cfg.charter.improve.generator_models
        pairs = [(m.alias, target_alias) for m in counterpart_models]
    else:
        fixed_alias = target_alias
        resolve_generator_model(cfg, target_alias)  # validate alias
        counterpart_models = cfg.charter.improve.judge_models
        pairs = [(target_alias, m.alias) for m in counterpart_models]

    # Health-check all involved models upfront
    _health_check_models(cfg, role, target_alias)

    # Select items once (fixed seed based on current max iteration)
    base_iter = next_iteration()
    seed = 42 + base_iter
    items = select_items(
        cfg.charter.improve.iteration.n_items,
        cfg.charter.improve.iteration.n_gold,
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
    max_conc = cfg.charter.improve.iteration.max_concurrent
    checked: set[str] = set()

    def _check(m: ModelConfig) -> None:
        key = (m.endpoint or cfg.charter.improve.endpoint, m.api_name)
        if key in checked:
            return
        client, _ = make_api_client(
            m.endpoint or cfg.charter.improve.endpoint, max_conc, cfg.api_keys
        )
        health_check(client, m.api_name)
        checked.add(key)

    if role == "judge":
        _check(resolve_judge_model(cfg, target_alias))
        for m in cfg.charter.improve.generator_models:
            _check(m)
    else:
        _check(resolve_generator_model(cfg, target_alias))
        for m in cfg.charter.improve.judge_models:
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


def rejudge_all_prompts_and_models(cfg: AppConfig, mode: str | None = None) -> int:
    """Re-judge all human-reviewed items with ALL judge prompts × ALL judge models.

    Builds one big queue of (item, judge_prompt, judge_model) work units and
    submits them all to a single asyncio event loop with a shared concurrency
    semaphore. Parallelism happens purely at the API-request level — no thread
    pool, no per-(prompt,model) event loops, no per-worker SQLite contention.
    Idempotent: items already in judge_correlations are skipped.

    mode is accepted for backward compatibility but ignored — only the
    reflection pipeline exists.

    Returns total count of newly judged items.
    """

    from pipeline.config import PROMPTS_DIR
    from pipeline.charter.improve.storage import (
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
    for model_cfg in cfg.charter.improve.judge_models:
        alias = model_cfg.alias
        model_dir = PROMPTS_DIR / alias
        if not model_dir.exists():
            continue
        # Find judge reflection prompt files: judge_reflection_vN.md
        refl_files = sorted(
            p
            for p in model_dir.iterdir()
            if re.match(r"^judge_reflection_v\d+\.md$", p.name)
        )
        for refl_file in refl_files:
            prompt_name = refl_file.name
            for k in reviewed_item_keys:
                if k not in latest_items:
                    continue
                if (k[0], k[1], prompt_name, alias) in existing_keys:
                    continue
                work.append((latest_items[k], refl_file, prompt_name, model_cfg))

    if not work:
        logger.info("Nothing to re-judge — all reviewed items already done.")
        return 0

    logger.info(
        "Re-judging {} (item, prompt, model) combinations in one queue...",
        len(work),
    )

    # Read shared prompt context once.
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")

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
    for model_cfg in cfg.charter.improve.judge_models:
        endpoint = model_cfg.endpoint or cfg.charter.improve.endpoint
        if endpoint in clients:
            continue
        env_var = (cfg.api_keys or {}).get(endpoint, "SWISS_AI_API_KEY")
        api_key = os.environ.get(env_var)
        assert api_key, f"{env_var} not set in environment (needed for {endpoint})"
        clients[endpoint] = openai.AsyncOpenAI(api_key=api_key, base_url=endpoint)

    async def _judge_one_work(
        item: dict,
        refl_file: Path,
        prompt_name: str,
        model_cfg: ModelConfig,
    ) -> dict | None:
        endpoint = model_cfg.endpoint or cfg.charter.improve.endpoint
        at = cfg.charter.improve.scoring.accept_threshold
        ft = cfg.charter.improve.scoring.floor_threshold
        try:
            t0 = time.monotonic()

            refl_parsed, refl_raw, refl_reasoning, refl_usage = await _judge_reflection(
                item,
                _prompt(refl_file),
                at,
                model_cfg.api_name,
                clients[endpoint],
                semaphore,
                charter_text=charter_text,
                thinking=model_cfg.thinking,
                completion_max_tokens=model_cfg.completion_max_tokens,
                context_window_tokens=model_cfg.context_window_tokens,
            )
            judge_latency_ms = int((time.monotonic() - t0) * 1000)

            judgment_parts: dict = {}
            for v in _REFLECTION_VOICES:
                judgment_parts[v] = {
                    "scores": refl_parsed[v]["scores"],
                    "aggregate": refl_parsed[v]["aggregate"],
                    "reasoning": refl_parsed[v]["reasoning"],
                    "model_reasoning": refl_reasoning,
                    "usage": refl_usage,
                }
            raw_responses = {"reflection": refl_raw}
            total_usage = dict(refl_usage)

            all_scores = [
                s for vd in judgment_parts.values() for s in vd["scores"].values()
            ]
            aggregate = sum(all_scores) / len(all_scores) if all_scores else 0
            has_floor = any(s <= ft for s in all_scores)
            decision = "reject" if has_floor or aggregate < at else "accept"

            refl_agg, refl_dec = _mode_decision(
                judgment_parts, _REFLECTION_VOICES, ft, at
            )

            return {
                **judgment_parts,
                "aggregate": aggregate,
                "decision": decision,
                "judge_prompt": prompt_name,
                "raw_responses": raw_responses,
                "usage": total_usage,
                "latency_ms": judge_latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reflection_aggregate": refl_agg,
                "reflection_decision": refl_dec,
                "judge_prompt_reflection": refl_file.name,
            }
        except Exception:
            logger.warning(
                "Judge failed for item {} with {}",
                item.get("item_id", "?"),
                prompt_name,
            )
            return None

    # Single event loop, single semaphore, all work in flight at once.
    coros = [_judge_one_work(item, rf, pn, mc) for (item, rf, pn, mc) in work]
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

    logger.info("Endpoint: {}", cfg.charter.improve.endpoint)
    logger.info("Generator models: {}", [m.alias for m in cfg.charter.improve.generator_models])
    logger.info("Judge models: {}", [m.alias for m in cfg.charter.improve.judge_models])
    logger.info(
        "Items: {} (gold: {})",
        cfg.charter.improve.iteration.n_items,
        cfg.charter.improve.iteration.n_gold,
    )
    logger.info("Threshold: {}", cfg.charter.improve.scoring.accept_threshold)

    # Default: run judge cross-iteration with first judge model
    target = cfg.charter.improve.judge_models[0].alias
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
