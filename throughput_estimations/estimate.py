"""Estimate throughput for reflection/preflection generation on annotation samples.

Queries an already-running model via the API, measures throughput on a small
subset, and extrapolates to the full annotation dataset (~102M samples).

Usage:
    uv run python -m throughput_estimations.estimate \
        --model-alias kimi-k2.5 --role generator --n-samples 200 \
        --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated --n-nodes 4

    uv run python -m throughput_estimations.estimate \
        --model-alias kimi-k2.5 --role judge --n-nodes 4 \
        --generations-path throughput_estimations/results/generator_kimi-k2.5_*.json
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
    WRITING_GUIDELINES_PATH,
    extract_charter_elements,
    load_config,
    resolve_generator_model,
    resolve_judge_model,
    resolve_prompt_path,
)
from pipeline.api import MAX_RETRIES, RETRY_BACKOFF_BASE
from pipeline.phase2.run import _parse_generation, _parse_judgment
from pipeline.tokenizer import compute_reflection_point, truncate_to_max_tokens

MAX_TOKENS = 1920  # annotation samples: 2048 seq - 128 reflection budget


async def _api_call(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
    thinking: bool = False,
) -> tuple[str, str | None, dict]:
    """API call with retry. Tolerates content=None when reasoning is present."""
    extra_body = None
    if thinking:
        extra_body = {
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body=extra_body,
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


def run_generator_estimation(
    items: list[dict],
    system_prompt: str,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    warmup: int,
    cooldown: int = 0,
    thinking: bool = False,
) -> list[dict]:
    """Run generator API calls on *items*, returning per-request metrics.

    The first *warmup* and last *cooldown* results are tagged and excluded
    from summary statistics.
    """
    n_total = len(items)
    results: list[dict] = []

    async def process_one(idx: int, item: dict) -> dict:
        rp = item["reflection_point"]
        context_before = item["text"][:rp]
        context_after = item["text"][rp:]
        user_content = (
            f"## Full Text\n\n"
            f"{context_before}"
            f"\n\n--- REFLECTION POINT (character {rp}) ---\n\n"
            f"{context_after}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        t0 = time.monotonic()
        try:
            raw, reasoning, usage = await _api_call(
                client, model, messages, semaphore, thinking=thinking
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            try:
                parsed = _parse_generation(raw)
                charter_elements = extract_charter_elements(parsed["reflection_1p"])
            except Exception:
                parsed = {
                    "analysis": raw,
                    "preflection_3p": "",
                    "preflection_1p": "",
                    "reflection_1p": "",
                    "reflection_3p": "",
                }
                charter_elements = []
            is_excluded = idx < warmup or idx >= n_total - cooldown
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "is_cooldown": idx >= n_total - cooldown,
                "is_excluded": is_excluded,
                "success": True,
                "latency_ms": latency_ms,
                **usage,
                "text": item["text"],
                "reflection_point": rp,
                "analysis": parsed["analysis"],
                "preflection_3p": parsed["preflection_3p"],
                "preflection_1p": parsed["preflection_1p"],
                "reflection_1p": parsed["reflection_1p"],
                "reflection_3p": parsed["reflection_3p"],
                "charter_elements": charter_elements,
                "raw_response": raw,
                "reasoning": reasoning,
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
    judge_prompt_template: str,
    accept_threshold: float,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    warmup: int,
    cooldown: int = 0,
    thinking: bool = False,
) -> list[dict]:
    """Run judge API calls on all 4 annotation voices per generation."""
    n_total = len(generations)

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

    # Map part types to the correct source text and item key
    _PREFLECTION_PARTS = {"preflection", "preflection_3p", "preflection_1p"}
    _PART_KEY_FALLBACK = {
        "preflection_3p": "preflection",
        "reflection_1p": "reflection",
    }

    async def judge_one(idx: int, item: dict) -> dict:
        t0 = time.monotonic()
        try:
            parts = _parts_to_judge(item)
            total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

            for part_type in parts:
                system = judge_prompt_template.replace(
                    "{part_type}", part_type
                ).replace("{accept_threshold}", str(accept_threshold))

                # Preflection variants use full text; reflection variants use text up to RP
                if part_type in _PREFLECTION_PARTS:
                    source_text = item["text"]
                else:
                    source_text = item["text"][: item["reflection_point"]]

                # Resolve item key with fallback for legacy data
                if part_type in item and item[part_type] is not None:
                    content = item[part_type]
                elif (
                    part_type in _PART_KEY_FALLBACK
                    and _PART_KEY_FALLBACK[part_type] in item
                ):
                    content = item[_PART_KEY_FALLBACK[part_type]]
                else:
                    content = item[part_type]

                user_content = (
                    f"## Source Text\n\n{source_text}\n\n"
                    f"## {part_type.title()} to Judge\n\n{content}"
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
                )
                _parse_judgment(raw)

                for k in total_usage:
                    total_usage[k] += usage[k]

            latency_ms = int((time.monotonic() - t0) * 1000)
            is_excluded = idx < warmup or idx >= n_total - cooldown
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "is_cooldown": idx >= n_total - cooldown,
                "is_excluded": is_excluded,
                "success": True,
                "latency_ms": latency_ms,
                "n_parts_judged": len(parts),
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
    print(f"Model: {model_name} on {stats['n_gpus']} GPUs")
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode (auto-detected from config if --model-alias is set).",
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

    # Build client
    client, semaphore = make_client(
        endpoint=args.endpoint,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        cfg_endpoint=cfg.phase2.endpoint,
    )

    # ---- Load data & prompts BEFORE waiting for API ----
    if args.role == "generator":
        assert args.data_path, "--data-path is required for generator role"
        total_samples = args.n_samples + args.warmup + args.cooldown
        max_text_tokens = args.max_text_tokens or MAX_TOKENS

        print(f"Loading {total_samples} samples from {args.data_path} ...")
        texts = load_samples(args.data_path, total_samples)
        items = prepare_items(texts, args.seed, max_tokens=max_text_tokens)
        print(f"Prepared {len(items)} items (max {max_text_tokens} tokens each)")

        # Build system prompt
        charter_text = CHARTER_PATH.read_text(encoding="utf-8")
        writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")
        if args.model_alias:
            prompt_path = resolve_prompt_path(
                "generator_latest.md", alias=args.model_alias
            )
        else:
            from pipeline.config import _INIT_PROMPTS_DIR

            prompt_path = _INIT_PROMPTS_DIR / "init_generator.md"
        prompt_template = prompt_path.read_text(encoding="utf-8")
        system_prompt = prompt_template.replace("{charter}", charter_text).replace(
            "{writing_guidelines}", writing_guidelines_text
        )

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

        if args.model_alias:
            judge_prompt_path = resolve_prompt_path(
                "judge_latest.md", alias=args.model_alias
            )
        else:
            from pipeline.config import _INIT_PROMPTS_DIR

            judge_prompt_path = _INIT_PROMPTS_DIR / "init_judge.md"
        judge_template = judge_prompt_path.read_text(encoding="utf-8")

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
    if args.role == "generator":
        print(
            f"\nRunning generator estimation: {len(items)} items, "
            f"max_concurrent={args.max_concurrent}, warmup={args.warmup}, "
            f"cooldown={args.cooldown}, thinking={thinking}"
        )
        results, wall_time_s = run_generator_estimation(
            items,
            system_prompt,
            model_name,
            client,
            semaphore,
            args.warmup,
            cooldown=args.cooldown,
            thinking=thinking,
        )

        stats = compute_stats(
            results,
            wall_time_s,
            args.total_samples,
            args.n_nodes,
            args.gpus_per_node,
            args.max_concurrent,
        )
        print_summary(stats, model_name, model_alias, "generator")

        # Save results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"generator_{model_alias}_{ts}.json"
        output = {
            "meta": {
                "model_name": model_name,
                "model_alias": model_alias,
                "role": "generator",
                "n_samples": args.n_samples,
                "warmup": args.warmup,
                "cooldown": args.cooldown,
                "max_concurrent": args.max_concurrent,
                "n_nodes": args.n_nodes,
                "gpus_per_node": args.gpus_per_node,
                "seed": args.seed,
                "endpoint": args.endpoint or cfg.phase2.endpoint,
                "timestamp": ts,
            },
            "stats": stats,
            "results": results,
        }
        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"Results saved to: {out_path}")

    elif args.role == "judge":
        print(
            f"\nRunning judge estimation: {len(generations)} items, "
            f"max_concurrent={args.max_concurrent}, warmup={args.warmup}, "
            f"cooldown={args.cooldown}, thinking={thinking}"
        )
        results, wall_time_s = run_judge_estimation(
            generations,
            judge_template,
            cfg.phase2.scoring.accept_threshold,
            model_name,
            client,
            semaphore,
            args.warmup,
            cooldown=args.cooldown,
            thinking=thinking,
        )

        stats = compute_stats(
            results,
            wall_time_s,
            args.total_samples,
            args.n_nodes,
            args.gpus_per_node,
            args.max_concurrent,
        )
        print_summary(stats, model_name, model_alias, "judge")

        # Save results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"judge_{model_alias}_{ts}.json"
        output = {
            "meta": {
                "model_name": model_name,
                "model_alias": model_alias,
                "role": "judge",
                "n_samples": len(generations),
                "warmup": args.warmup,
                "cooldown": args.cooldown,
                "max_concurrent": args.max_concurrent,
                "n_nodes": args.n_nodes,
                "gpus_per_node": args.gpus_per_node,
                "endpoint": args.endpoint or cfg.phase2.endpoint,
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
