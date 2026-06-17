"""Estimate throughput for reflection/preflection generation on annotation samples.

Queries an already-running model via the API, measures throughput on a small
subset, and extrapolates to the full annotation dataset (~102M samples).

The ``--mode`` flag controls which pipeline(s) to benchmark:
  - ``reflection``: partial text (up to reflection point), separate prompt
  - ``preflection``: full text, separate prompt
  - ``both`` (default): both calls per item, matching the production pipeline

Usage:
    # Benchmark reflection only
    uv run python -m throughput_estimations.estimate \
        --model-alias kimi-k2.5 --role generator --mode reflection \
        --n-samples 200 --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated

    # Benchmark preflection only
    uv run python -m throughput_estimations.estimate \
        --model-alias kimi-k2.5 --role generator --mode preflection \
        --n-samples 200 --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated

    # Benchmark both (default, 2 API calls per sample)
    uv run python -m throughput_estimations.estimate \
        --model-alias kimi-k2.5 --role generator --n-samples 200 \
        --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated --n-nodes 4

    # Judge
    uv run python -m throughput_estimations.estimate \
        --model-alias kimi-k2.5 --role judge --mode reflection --n-nodes 4 \
        --generations-path throughput_estimations/results/generator_reflection_kimi-k2.5_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openai
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm_asyncio

from pipeline.config import (
    CHARTER_PATH,
    load_config,
    resolve_generator_model,
    resolve_judge_model,
    resolve_prompt_path,
    union_charter_elements,
)
from pipeline.api import MAX_RETRIES, RETRY_BACKOFF_BASE, resolve_sampling_params
from pipeline.generation import REFLECTION_TASK, PREFLECTION_TASK
from pipeline.charter.improve.run import _parse_generation, _parse_mode_judgment
from pipeline.tokenizer import compute_reflection_point, truncate_to_max_tokens

MAX_TOKENS = 1920  # annotation samples: 2048 seq - 128 reflection budget


async def _api_call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
    thinking: bool = False,
    max_tokens: int | None = None,
    sampling_params: dict[str, float | int] | None = None,
) -> tuple[str, str | None, dict]:
    """API call with retry. Tolerates content=None when reasoning is present."""
    extra_body = None
    if thinking:
        extra_body = {
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    # Sampling params: temperature, top_p, presence_penalty are native OpenAI
    # API kwargs; top_k goes into extra_body (sglang/vllm extension).
    sp = sampling_params or {}
    api_kwargs: dict = {}
    for k in ("temperature", "top_p", "presence_penalty"):
        if k in sp:
            api_kwargs[k] = sp[k]
    if "top_k" in sp:
        extra_body = extra_body or {}
        extra_body["top_k"] = sp["top_k"]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    **api_kwargs,
                )
            msg = response.choices[0].message
            content = msg.content or ""
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
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_BACKOFF_BASE**attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_samples(data_path: str, n_samples: int) -> list[str]:
    """Load first *n_samples* texts from parquet dir or single parquet file.

    For directories: reads sorted files one at a time until enough rows.
    Only reads the ``text`` column to avoid schema issues.
    """
    path = Path(data_path)
    if path.is_file() and path.suffix == ".parquet":
        table = pq.read_table(str(path), columns=["text"])
        texts = table.column("text").to_pylist()[:n_samples]
    elif path.is_dir():
        files = sorted(path.glob("*.parquet"))
        assert files, f"No parquet files in {path}"
        texts: list[str] = []
        for f in files:
            if len(texts) >= n_samples:
                break
            table = pq.read_table(str(f), columns=["text"])
            texts.extend(table.column("text").to_pylist())
        texts = texts[:n_samples]
    else:
        raise ValueError(f"data-path must be a .parquet file or directory: {data_path}")

    assert len(texts) >= n_samples, f"Only found {len(texts)} samples, need {n_samples}"
    return texts


def prepare_items(
    texts: list[str], seed: int, max_tokens: int = MAX_TOKENS
) -> list[dict]:
    """Truncate texts and compute reflection points.

    Pre-initialises the tokenizer so that load time is not included in
    measurement.
    """
    # Warm up tokenizer singleton before measurement
    truncate_to_max_tokens("warmup", max_tokens)

    rng = random.Random(seed)
    items = []
    for text in texts:
        text = truncate_to_max_tokens(text, max_tokens)
        rp = compute_reflection_point(text, rng)
        items.append({"text": text, "reflection_point": rp})
    return items


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def make_client(
    endpoint: str | None,
    api_key: str | None,
    max_concurrent: int,
    cfg_endpoint: str,
) -> tuple[openai.AsyncOpenAI, asyncio.Semaphore]:
    """Build an async OpenAI client and concurrency semaphore.

    If *endpoint* is provided, uses it directly (with optional *api_key*).
    Otherwise falls back to the config endpoint + SWISS_AI_API_KEY env var.
    """
    if endpoint:
        key = api_key or "none"
        if key == "none":
            key = "placeholder"
        client = openai.AsyncOpenAI(api_key=key, base_url=endpoint)
    else:
        key = api_key or os.environ.get("SWISS_AI_API_KEY")
        assert key, "SWISS_AI_API_KEY not set and no --api-key provided"
        client = openai.AsyncOpenAI(api_key=key, base_url=cfg_endpoint)
    return client, asyncio.Semaphore(max_concurrent)


# ---------------------------------------------------------------------------
# Generator estimation
# ---------------------------------------------------------------------------


def _build_refl_messages(
    item: dict,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build messages for a reflection call (text up to RP)."""
    rp = item["reflection_point"]
    user_content = f"## Full Text\n\n{item['text'][:rp]}"
    user_content += REFLECTION_TASK
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_prefl_messages(
    item: dict,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build messages for a preflection call (full text)."""
    user_content = f"## Full Text\n\n{item['text']}"
    user_content += PREFLECTION_TASK
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


_REFL_FIELDS = {"analysis", "reflection_1p", "reflection_3p"}
_PREFL_FIELDS = {"analysis", "preflection_3p", "preflection_1p"}


def run_generator_estimation(
    items: list[dict],
    refl_system_prompt: str | None,
    prefl_system_prompt: str | None,
    mode: str,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    warmup: int,
    cooldown: int = 0,
    thinking: bool = False,
    max_tokens: int | None = None,
    sampling_params: dict[str, float | int] | None = None,
) -> list[dict]:
    """Run generator API calls on *items*, returning per-request metrics.

    *mode* controls which pipeline(s) to run:
      - ``"reflection"``: only reflection call (text up to RP).
      - ``"preflection"``: only preflection call (full text).
      - ``"both"``: both calls per item (2 API calls, tokens summed).

    The first *warmup* and last *cooldown* results are tagged and excluded
    from summary statistics.
    """
    n_total = len(items)
    run_refl = mode in ("reflection", "both")
    run_prefl = mode in ("preflection", "both")

    async def _do_call(messages):
        return await _api_call(
            client, model, messages, semaphore,
            thinking=thinking, max_tokens=max_tokens,
            sampling_params=sampling_params,
        )

    async def process_one(idx: int, item: dict) -> dict:
        rp = item["reflection_point"]
        t0 = time.monotonic()
        try:
            total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
            refl_parsed = {}
            prefl_parsed = {}
            raw_parts = {}

            if run_refl:
                msgs = _build_refl_messages(item, refl_system_prompt)
                raw, reasoning, usage = await _do_call(msgs)
                for k in total_usage:
                    total_usage[k] += usage[k]
                raw_parts["reflection_raw"] = raw
                raw_parts["reflection_reasoning"] = reasoning
                try:
                    refl_parsed = _parse_generation(raw, required_fields=_REFL_FIELDS)
                except Exception:
                    refl_parsed = {"analysis": raw, "reflection_1p": "", "reflection_3p": ""}

            if run_prefl:
                msgs = _build_prefl_messages(item, prefl_system_prompt)
                raw, reasoning, usage = await _do_call(msgs)
                for k in total_usage:
                    total_usage[k] += usage[k]
                raw_parts["preflection_raw"] = raw
                raw_parts["preflection_reasoning"] = reasoning
                try:
                    prefl_parsed = _parse_generation(raw, required_fields=_PREFL_FIELDS)
                except Exception:
                    prefl_parsed = {"analysis": raw, "preflection_3p": "", "preflection_1p": ""}

            latency_ms = int((time.monotonic() - t0) * 1000)

            # Merge parsed fields
            analysis = refl_parsed.get("analysis") or prefl_parsed.get("analysis", "")
            reflection_1p = refl_parsed.get("reflection_1p", "")
            reflection_3p = refl_parsed.get("reflection_3p", "")
            preflection_3p = prefl_parsed.get("preflection_3p", "")
            preflection_1p = prefl_parsed.get("preflection_1p", "")

            try:
                reflection_charter_elements = union_charter_elements(
                    reflection_1p, reflection_3p
                ) if run_refl else []
            except Exception:
                reflection_charter_elements = []
            try:
                preflection_charter_elements = union_charter_elements(
                    preflection_1p, preflection_3p
                ) if run_prefl else []
            except Exception:
                preflection_charter_elements = []

            is_excluded = idx < warmup or idx >= n_total - cooldown
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "is_cooldown": idx >= n_total - cooldown,
                "is_excluded": is_excluded,
                "success": True,
                "latency_ms": latency_ms,
                **total_usage,
                "text": item["text"],
                "reflection_point": rp,
                "analysis": analysis,
                "preflection_3p": preflection_3p,
                "preflection_1p": preflection_1p,
                "reflection_1p": reflection_1p,
                "reflection_3p": reflection_3p,
                "preflection_charter_elements": preflection_charter_elements,
                "reflection_charter_elements": reflection_charter_elements,
                **raw_parts,
            }
        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            is_excluded = idx < warmup or idx >= n_total - cooldown
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "is_cooldown": idx >= n_total - cooldown,
                "is_excluded": is_excluded,
                "success": False,
                "latency_ms": latency_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "error": str(e),
                "text": item["text"],
                "reflection_point": rp,
            }

    coros = [process_one(i, item) for i, item in enumerate(items)]
    loop = asyncio.new_event_loop()
    t_wall_start = time.monotonic()
    try:
        results = loop.run_until_complete(tqdm_asyncio.gather(*coros, desc="Generator"))
    finally:
        loop.close()
    wall_time_s = time.monotonic() - t_wall_start

    return list(results), wall_time_s


# ---------------------------------------------------------------------------
# Judge estimation
# ---------------------------------------------------------------------------


def run_judge_estimation(
    generations: list[dict],
    refl_judge_template: str | None,
    prefl_judge_template: str | None,
    mode: str,
    accept_threshold: float,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    warmup: int,
    cooldown: int = 0,
    thinking: bool = False,
    max_tokens: int | None = None,
    sampling_params: dict[str, float | int] | None = None,
) -> list[dict]:
    """Run judge API calls on annotation voices per generation.

    *mode* controls which mode(s) to judge:
      - ``"reflection"``: judge reflection voices only.
      - ``"preflection"``: judge preflection voices only.
      - ``"both"``: judge both modes (2 API calls per item).
    """
    n_total = len(generations)

    _MODES_TO_RUN = []
    if mode in ("reflection", "both"):
        _MODES_TO_RUN.append(("reflection", ("reflection_1p", "reflection_3p"), refl_judge_template))
    if mode in ("preflection", "both"):
        _MODES_TO_RUN.append(("preflection", ("preflection_3p", "preflection_1p"), prefl_judge_template))

    _PART_KEY_FALLBACK = {
        "preflection_3p": "preflection",
        "reflection_1p": "reflection",
    }

    async def judge_one(idx: int, item: dict) -> dict:
        t0 = time.monotonic()
        try:
            total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
            n_calls = 0

            for judge_mode, voices, template in _MODES_TO_RUN:
                source_text = (
                    item["text"][: item["reflection_point"]]
                    if judge_mode == "reflection"
                    else item["text"]
                )
                user_content = f"## Source Text\n\n{source_text}\n\n---\n\n"
                for v in voices:
                    if v in item and item[v] is not None:
                        content = item[v]
                    elif v in _PART_KEY_FALLBACK and _PART_KEY_FALLBACK[v] in item:
                        content = item[_PART_KEY_FALLBACK[v]]
                    else:
                        content = item[v]
                    user_content += f"## {v}\n\n{content}\n\n"

                system = template.replace(
                    "{accept_threshold}", str(accept_threshold)
                )
                raw, reasoning, usage = await _api_call(
                    client,
                    model,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    semaphore,
                    thinking=thinking,
                    max_tokens=max_tokens,
                    sampling_params=sampling_params,
                )
                _parse_mode_judgment(raw, judge_mode)

                for k in total_usage:
                    total_usage[k] += usage[k]
                n_calls += 1

            latency_ms = int((time.monotonic() - t0) * 1000)
            is_excluded = idx < warmup or idx >= n_total - cooldown
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "is_cooldown": idx >= n_total - cooldown,
                "is_excluded": is_excluded,
                "success": True,
                "latency_ms": latency_ms,
                "n_calls": n_calls,
                **total_usage,
            }
        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            is_excluded = idx < warmup or idx >= n_total - cooldown
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "is_cooldown": idx >= n_total - cooldown,
                "is_excluded": is_excluded,
                "success": False,
                "latency_ms": latency_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "error": str(e),
            }

    coros = [judge_one(i, gen) for i, gen in enumerate(generations)]
    loop = asyncio.new_event_loop()
    t_wall_start = time.monotonic()
    try:
        results = loop.run_until_complete(tqdm_asyncio.gather(*coros, desc="Judge"))
    finally:
        loop.close()
    wall_time_s = time.monotonic() - t_wall_start

    return list(results), wall_time_s


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(
    results: list[dict],
    wall_time_s: float,
    total_samples: int,
    n_nodes: int,
    gpus_per_node: int,
    max_concurrent: int,
    tp_size: int = 1,
    dp_size: int = 1,
) -> dict:
    """Compute summary statistics and extrapolation from measured results.

    *wall_time_s* is the actual elapsed wall-clock time for the entire batch
    (measured externally), which is the correct basis for throughput since
    per-request latency includes semaphore wait time.
    """
    measured = [r for r in results if not r.get("is_excluded") and r["success"]]
    failed = [r for r in results if not r.get("is_excluded") and not r["success"]]
    warmup_count = sum(1 for r in results if r.get("is_warmup"))
    cooldown_count = sum(1 for r in results if r.get("is_cooldown"))

    if not measured:
        return {"error": "No successful measured requests"}

    input_toks = np.array([r["input_tokens"] for r in measured])
    output_toks = np.array([r["output_tokens"] for r in measured])
    reasoning_toks = np.array([r["reasoning_tokens"] for r in measured])

    n_measured = len(measured)
    n_gpus = n_nodes * gpus_per_node

    # Throughput from actual wall time
    samples_per_sec = len(results) / wall_time_s if wall_time_s > 0 else 0

    # Point estimate
    extrap_wall_s = (
        total_samples / samples_per_sec if samples_per_sec > 0 else float("inf")
    )
    extrap_gpu_h = extrap_wall_s * n_gpus / 3600

    # Confidence range: scale by output token variance (p25/p75)
    # More output tokens → slower, so use token distribution for range
    p25_out = float(np.percentile(output_toks, 25))
    p75_out = float(np.percentile(output_toks, 75))
    mean_out = float(np.mean(output_toks))
    if mean_out > 0:
        opt_gpu_h = extrap_gpu_h * (p25_out / mean_out)
        pes_gpu_h = extrap_gpu_h * (p75_out / mean_out)
    else:
        opt_gpu_h = extrap_gpu_h
        pes_gpu_h = extrap_gpu_h

    return {
        "n_measured": n_measured,
        "n_failed": len(failed),
        "n_warmup": warmup_count,
        "n_cooldown": cooldown_count,
        "max_concurrent": max_concurrent,
        "n_gpus": n_gpus,
        "tp_size": tp_size,
        "dp_size": dp_size,
        "wall_time_s": wall_time_s,
        "input_tokens": {
            "mean": float(np.mean(input_toks)),
            "median": float(np.median(input_toks)),
        },
        "output_tokens": {
            "mean": float(np.mean(output_toks)),
            "median": float(np.median(output_toks)),
        },
        "reasoning_tokens": {
            "mean": float(np.mean(reasoning_toks)),
            "median": float(np.median(reasoning_toks)),
        },
        "throughput": {
            "samples_per_sec": samples_per_sec,
            "input_tok_per_sec": samples_per_sec * float(np.mean(input_toks)),
            "output_tok_per_sec": samples_per_sec * float(np.mean(output_toks)),
        },
        "extrapolation": {
            "total_samples": total_samples,
            "wall_time_s": extrap_wall_s,
            "wall_time_days": extrap_wall_s / 86400,
            "gpu_hours": extrap_gpu_h,
            "gpu_hours_optimistic": min(opt_gpu_h, pes_gpu_h),
            "gpu_hours_pessimistic": max(opt_gpu_h, pes_gpu_h),
        },
    }


def print_summary(stats: dict, model_name: str, model_alias: str, role: str) -> None:
    """Print a formatted summary table to stdout."""
    if "error" in stats:
        print(f"\nERROR: {stats['error']}")
        return

    n_total = (
        stats["n_measured"]
        + stats["n_failed"]
        + stats["n_warmup"]
        + stats.get("n_cooldown", 0)
    )
    ext = stats["extrapolation"]
    tp = stats["throughput"]
    inp = stats["input_tokens"]
    out = stats["output_tokens"]
    rea = stats["reasoning_tokens"]

    print(f"\n{'=' * 60}")
    print(f"Throughput Estimation: {model_alias} ({role})")
    print(f"{'=' * 60}")
    print(
        f"\nSamples: {stats['n_measured']} / {n_total} successful "
        f"({stats['n_failed']} failed, {stats['n_warmup']} warmup + {stats.get('n_cooldown', 0)} cooldown excluded)"
    )
    print(
        f"Model: {model_name} on {stats['n_gpus']} GPUs (TP={stats['tp_size']}, DP={stats['dp_size']})"
    )
    print(f"Wall time: {stats['wall_time_s']:.1f}s")

    print(f"\nPer-request token stats:")
    print(f"  Input tokens:     mean={inp['mean']:.0f}  median={inp['median']:.0f}")
    print(f"  Output tokens:    mean={out['mean']:.0f}  median={out['median']:.0f}")
    print(f"  Reasoning tokens: mean={rea['mean']:.0f}  median={rea['median']:.0f}")

    print(f"\nThroughput (at max_concurrent={stats['max_concurrent']}):")
    print(f"  Samples/sec:      {tp['samples_per_sec']:.2f}")
    print(f"  Input tok/sec:    {tp['input_tok_per_sec']:,.0f}")
    print(f"  Output tok/sec:   {tp['output_tok_per_sec']:,.0f}")

    print(f"\nExtrapolation to {ext['total_samples']:,} samples:")
    if ext["wall_time_days"] < 1:
        print(f"  Wall time:        ~{ext['wall_time_s'] / 3600:.1f} hours")
    else:
        print(f"  Wall time:        ~{ext['wall_time_days']:.1f} days")
    print(f"  GPU-hours:        ~{ext['gpu_hours']:,.0f}")
    print(
        f"  Estimate range:   ~{ext['gpu_hours_optimistic']:,.0f}"
        f" - ~{ext['gpu_hours_pessimistic']:,.0f} GPU-h (p25-p75 output tokens)"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate throughput for reflection/preflection generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-alias",
        help="Model alias from config.yaml (e.g. kimi-k2.5). "
        "Required unless --api-name is set.",
    )
    p.add_argument(
        "--role",
        choices=["generator", "judge"],
        default="generator",
    )
    p.add_argument(
        "--mode",
        choices=["reflection", "preflection", "both"],
        default="both",
        help="Which pipeline(s) to benchmark: reflection (partial text), "
        "preflection (full text), or both (default).",
    )
    p.add_argument(
        "--data-path", help="Path to annotated parquet dir or sidecar.parquet."
    )
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument(
        "--max-text-tokens",
        type=int,
        default=None,
        help="Max tokens per text sample (overrides MAX_TOKENS=1920).",
    )
    p.add_argument("--max-concurrent", type=int, default=50)
    p.add_argument("--total-samples", type=int, default=102_772_028)
    p.add_argument("--n-nodes", type=int, default=4)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    p.add_argument("--dp-size", type=int, default=1, help="Data parallel size.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--cooldown", type=int, default=10)
    p.add_argument(
        "--generations-path",
        help="For judge role: path to saved generator results JSON.",
    )
    p.add_argument(
        "--output-dir",
        default="throughput_estimations/results",
    )
    p.add_argument(
        "--endpoint",
        help="API endpoint override (e.g. http://172.28.53.207:5000/v1).",
    )
    p.add_argument(
        "--api-name",
        help="API model name override (skip config lookup).",
    )
    p.add_argument(
        "--api-key",
        help='API key override. Use "none" for local endpoints without auth.',
    )
    p.add_argument(
        "--max-tokens", type=int, default=6144,
        help="Max output tokens per request. Pass 0 for no cap (server default).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode (auto-detected from config if --model-alias is set).",
    )
    p.add_argument(
        "--temperature", type=float, default=None, help="Override sampling temperature."
    )
    p.add_argument(
        "--top-p", type=float, default=None, help="Override top-p (nucleus sampling)."
    )
    p.add_argument("--top-k", type=int, default=None, help="Override top-k sampling.")
    p.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Override presence penalty.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    # Resolve model name, alias, and thinking mode
    if args.api_name:
        model_name = args.api_name
        model_alias = args.api_name.split("/")[-1][:20]
        thinking = args.thinking or False
    elif args.model_alias:
        model_cfg = resolve_generator_model(cfg, args.model_alias)
        model_name = model_cfg.api_name
        model_alias = args.model_alias
        thinking = args.thinking if args.thinking is not None else model_cfg.thinking
    else:
        raise SystemExit("Either --model-alias or --api-name is required.")

    # Resolve per-model sampling params, then apply CLI overrides
    sampling_params = resolve_sampling_params(model_name, model_alias)
    if args.temperature is not None:
        sampling_params["temperature"] = args.temperature
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    if args.presence_penalty is not None:
        sampling_params["presence_penalty"] = args.presence_penalty
    if sampling_params:
        print(f"Sampling params: {sampling_params}")

    # Build client
    client, semaphore = make_client(
        endpoint=args.endpoint,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        cfg_endpoint=cfg.charter.improve.endpoint,
    )

    mode = args.mode

    def _load_system_prompt(prompt_path: Path) -> str:
        charter_text = CHARTER_PATH.read_text(encoding="utf-8")
        return (
            prompt_path.read_text(encoding="utf-8")
            .replace("{charter}", charter_text)
        )

    def _resolve_gen_prompt(kind: str) -> Path:
        """Resolve generator prompt path for 'reflection' or 'preflection'."""
        if args.model_alias:
            return resolve_prompt_path(
                f"generator_{kind}_latest.md", alias=args.model_alias
            )
        from pipeline.config import _INIT_PROMPTS_DIR
        return _INIT_PROMPTS_DIR / f"init_generator_{kind}.md"

    def _resolve_judge_prompt(kind: str) -> Path:
        """Resolve judge prompt path for 'reflection' or 'preflection'."""
        if args.model_alias:
            return resolve_prompt_path(
                f"judge_{kind}_latest.md", alias=args.model_alias
            )
        from pipeline.config import _INIT_PROMPTS_DIR
        return _INIT_PROMPTS_DIR / f"init_judge_{kind}.md"

    # ---- Load data & prompts BEFORE waiting for API ----
    if args.role == "generator":
        assert args.data_path, "--data-path is required for generator role"
        total_samples = args.n_samples + args.warmup + args.cooldown
        max_text_tokens = args.max_text_tokens or MAX_TOKENS

        print(f"Loading {total_samples} samples from {args.data_path} ...")
        texts = load_samples(args.data_path, total_samples)
        items = prepare_items(texts, args.seed, max_tokens=max_text_tokens)
        print(f"Prepared {len(items)} items (max {max_text_tokens} tokens each)")

        # Build system prompts per mode
        refl_system_prompt = (
            _load_system_prompt(_resolve_gen_prompt("reflection"))
            if mode in ("reflection", "both")
            else None
        )
        prefl_system_prompt = (
            _load_system_prompt(_resolve_gen_prompt("preflection"))
            if mode in ("preflection", "both")
            else None
        )
        print(f"Mode: {mode}")

    elif args.role == "judge":
        assert args.generations_path, "--generations-path is required for judge role"
        gen_path = Path(args.generations_path)
        assert gen_path.exists(), f"Generations file not found: {gen_path}"

        with open(gen_path) as f:
            gen_data = json.load(f)
        generations = [
            r
            for r in gen_data["results"]
            if r.get("success") and not r.get("is_warmup")
        ]
        n_judge = min(len(generations), args.n_samples + args.warmup + args.cooldown)
        generations = generations[:n_judge]
        print(f"Loaded {len(generations)} generations from {gen_path}")

        refl_judge_template = (
            _resolve_judge_prompt("reflection").read_text(encoding="utf-8")
            if mode in ("reflection", "both")
            else None
        )
        prefl_judge_template = (
            _resolve_judge_prompt("preflection").read_text(encoding="utf-8")
            if mode in ("preflection", "both")
            else None
        )
        print(f"Mode: {mode}")

    # ---- Wait for API to become ready (poll with backoff) ----
    print(f"Waiting for API: {model_name} ...", flush=True)
    for attempt in range(120):  # up to ~30 min
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
            )
            print(f"  API ready (attempt {attempt + 1})")
            break
        except Exception:
            if attempt == 0:
                print(f"  Not ready, polling every 15s ...", flush=True)
            elif (attempt + 1) % 10 == 0:
                print(f"  Still waiting (attempt {attempt + 1}) ...", flush=True)
            time.sleep(15)
        finally:
            loop.close()
    else:
        raise SystemExit("API did not become ready after 30 minutes.")

    # ---- Run estimation ----
    role_mode_label = f"{args.role} ({mode})"

    if args.role == "generator":
        print(
            f"\nRunning generator estimation ({mode}): {len(items)} items, "
            f"max_concurrent={args.max_concurrent}, warmup={args.warmup}, "
            f"cooldown={args.cooldown}, thinking={thinking}"
        )
        results, wall_time_s = run_generator_estimation(
            items,
            refl_system_prompt,
            prefl_system_prompt,
            mode,
            model_name,
            client,
            semaphore,
            args.warmup,
            cooldown=args.cooldown,
            thinking=thinking,
            max_tokens=(args.max_tokens if args.max_tokens > 0 else None),
            sampling_params=sampling_params,
        )

        stats = compute_stats(
            results,
            wall_time_s,
            args.total_samples,
            args.n_nodes,
            args.gpus_per_node,
            args.max_concurrent,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
        )
        print_summary(stats, model_name, model_alias, role_mode_label)

        # Save results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"generator_{mode}_{model_alias}_{ts}.json"
        output = {
            "meta": {
                "model_name": model_name,
                "model_alias": model_alias,
                "role": "generator",
                "mode": mode,
                "n_samples": args.n_samples,
                "warmup": args.warmup,
                "cooldown": args.cooldown,
                "max_concurrent": args.max_concurrent,
                "n_nodes": args.n_nodes,
                "gpus_per_node": args.gpus_per_node,
                "tp_size": args.tp_size,
                "dp_size": args.dp_size,
                "seed": args.seed,
                "endpoint": args.endpoint or cfg.charter.improve.endpoint,
                "sampling_params": sampling_params or None,
                "timestamp": ts,
            },
            "stats": stats,
            "results": results,
        }
        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"Results saved to: {out_path}")

    elif args.role == "judge":
        print(
            f"\nRunning judge estimation ({mode}): {len(generations)} items, "
            f"max_concurrent={args.max_concurrent}, warmup={args.warmup}, "
            f"cooldown={args.cooldown}, thinking={thinking}"
        )
        results, wall_time_s = run_judge_estimation(
            generations,
            refl_judge_template,
            prefl_judge_template,
            mode,
            cfg.charter.improve.scoring.accept_threshold,
            model_name,
            client,
            semaphore,
            args.warmup,
            cooldown=args.cooldown,
            thinking=thinking,
            max_tokens=(args.max_tokens if args.max_tokens > 0 else None),
            sampling_params=sampling_params,
        )

        stats = compute_stats(
            results,
            wall_time_s,
            args.total_samples,
            args.n_nodes,
            args.gpus_per_node,
            args.max_concurrent,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
        )
        print_summary(stats, model_name, model_alias, role_mode_label)

        # Save results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"judge_{mode}_{model_alias}_{ts}.json"
        output = {
            "meta": {
                "model_name": model_name,
                "model_alias": model_alias,
                "role": "judge",
                "mode": mode,
                "n_samples": len(generations),
                "warmup": args.warmup,
                "cooldown": args.cooldown,
                "max_concurrent": args.max_concurrent,
                "n_nodes": args.n_nodes,
                "gpus_per_node": args.gpus_per_node,
                "tp_size": args.tp_size,
                "dp_size": args.dp_size,
                "endpoint": args.endpoint or cfg.charter.improve.endpoint,
                "sampling_params": sampling_params or None,
                "timestamp": ts,
                "generations_source": str(gen_path),
            },
            "stats": stats,
            "results": results,
        }
        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
