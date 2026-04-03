"""Estimate throughput for neutral text summarization.

Queries an already-running model via the API, measures throughput on a small
subset, and extrapolates to the full dataset (~102M samples).

Unlike the annotation estimator, this generates a short neutral summary
(max ~128 tokens) with no charter, no reflection point, no moral framing.

Usage:
    uv run python -m throughput_estimations.estimate_summary \
        --api-name HuggingFaceTB/SmolLM3-3B \
        --endpoint http://nid007123:8080/v1 \
        --api-key none \
        --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated \
        --n-nodes 1 --gpus-per-node 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openai
from tqdm.asyncio import tqdm_asyncio

from pipeline.api import api_call, resolve_sampling_params
from pipeline.tokenizer import truncate_to_max_tokens

MAX_TOKENS = 1920  # text budget: 2048 seq - 128 summary budget

SYSTEM_PROMPT = """\
You are a neutral text summarizer. Given a passage, write a concise factual \
summary of what the text is about. Be objective — do not add opinions, moral \
judgments, or evaluations. Just describe the content in 1-3 short sentences. \
Keep your summary under 128 tokens."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(data_path: str, n_samples: int) -> list[str]:
    """Load first *n_samples* texts from parquet dir or single parquet file."""
    import pyarrow.parquet as pq

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

    assert len(texts) >= n_samples, (
        f"Only found {len(texts)} samples, need {n_samples}"
    )
    return texts


def prepare_items(texts: list[str]) -> list[str]:
    """Truncate texts to max tokens. Pre-initialises the tokenizer."""
    truncate_to_max_tokens("warmup", MAX_TOKENS)
    return [truncate_to_max_tokens(t, MAX_TOKENS) for t in texts]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def run_summary_estimation(
    items: list[str],
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    warmup: int,
    thinking: bool = False,
    sampling_params: dict[str, float | int] | None = None,
) -> tuple[list[dict], float]:
    """Run summary API calls on *items*, returning per-request metrics."""

    async def process_one(idx: int, text: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        t0 = time.monotonic()
        try:
            raw, reasoning, usage = await api_call(
                client, model, messages, semaphore,
                thinking=thinking,
                sampling_params=sampling_params,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "success": True,
                "latency_ms": latency_ms,
                **usage,
                "summary": raw,
                "reasoning": reasoning,
            }
        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                "idx": idx,
                "is_warmup": idx < warmup,
                "success": False,
                "latency_ms": latency_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "error": str(e),
            }

    coros = [process_one(i, text) for i, text in enumerate(items)]
    loop = asyncio.new_event_loop()
    t_wall_start = time.monotonic()
    try:
        results = loop.run_until_complete(
            tqdm_asyncio.gather(*coros, desc="Summary")
        )
    finally:
        loop.close()
    wall_time_s = time.monotonic() - t_wall_start

    return list(results), wall_time_s


# ---------------------------------------------------------------------------
# Content token analysis
# ---------------------------------------------------------------------------

CONTENT_TOKENIZER_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def _strip_content(text: str) -> str:
    """Strip model-specific overhead to get the actual summary content.

    Handles:
    - <think>...</think> blocks (SmolLM3, GLM with separate_reasoning off)
    - <|channel|>analysis...<|channel|>final<|message|> (gpt-oss)
    """
    # gpt-oss channel format: take content after last final channel marker
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>")[-1]
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def compute_content_tokens(results: list[dict]) -> dict:
    """Tokenize actual summary content with SmolLM2 tokenizer.

    Returns stats dict with mean, median, p5, p95 of content token counts.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CONTENT_TOKENIZER_ID)
    measured = [r for r in results if not r["is_warmup"] and r["success"]]
    if not measured:
        return {"error": "No successful measured requests"}

    counts = np.array([
        len(tok.encode(_strip_content(r.get("summary", ""))))
        for r in measured
    ])
    api_toks = np.array([r["output_tokens"] for r in measured])

    return {
        "tokenizer": CONTENT_TOKENIZER_ID,
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "p5": float(np.percentile(counts, 5)),
        "p95": float(np.percentile(counts, 95)),
        "overhead_ratio": float(np.mean(api_toks) / np.mean(counts))
        if np.mean(counts) > 0
        else float("inf"),
    }


# ---------------------------------------------------------------------------
# Statistics (reused from estimate.py)
# ---------------------------------------------------------------------------

def compute_stats(
    results: list[dict],
    wall_time_s: float,
    total_samples: int,
    n_nodes: int,
    gpus_per_node: int,
    max_concurrent: int,
) -> dict:
    """Compute summary statistics and extrapolation from measured results."""
    measured = [r for r in results if not r["is_warmup"] and r["success"]]
    failed = [r for r in results if not r["is_warmup"] and not r["success"]]
    warmup_count = sum(1 for r in results if r["is_warmup"])

    if not measured:
        return {"error": "No successful measured requests"}

    input_toks = np.array([r["input_tokens"] for r in measured])
    output_toks = np.array([r["output_tokens"] for r in measured])
    reasoning_toks = np.array([r["reasoning_tokens"] for r in measured])

    n_gpus = n_nodes * gpus_per_node
    samples_per_sec = len(results) / wall_time_s if wall_time_s > 0 else 0

    extrap_wall_s = total_samples / samples_per_sec if samples_per_sec > 0 else float("inf")
    extrap_gpu_h = extrap_wall_s * n_gpus / 3600

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
        "n_measured": len(measured),
        "n_failed": len(failed),
        "n_warmup": warmup_count,
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


def print_summary(stats: dict, model_name: str, model_alias: str) -> None:
    """Print a formatted summary table to stdout."""
    if "error" in stats:
        print(f"\nERROR: {stats['error']}")
        return

    n_total = stats["n_measured"] + stats["n_failed"] + stats["n_warmup"]
    ext = stats["extrapolation"]
    tp = stats["throughput"]
    inp = stats["input_tokens"]
    out = stats["output_tokens"]
    rea = stats["reasoning_tokens"]

    print(f"\n{'=' * 60}")
    print(f"Summary Throughput: {model_alias}")
    print(f"{'=' * 60}")
    print(
        f"\nSamples: {stats['n_measured']} / {n_total} successful "
        f"({stats['n_failed']} failed, {stats['n_warmup']} warmup discarded)"
    )
    print(f"Model: {model_name} on {stats['n_gpus']} GPUs")
    print(f"Wall time: {stats['wall_time_s']:.1f}s")

    print(f"\nPer-request token stats:")
    print(f"  Input tokens:     mean={inp['mean']:.0f}  median={inp['median']:.0f}")
    print(f"  Output tokens:    mean={out['mean']:.0f}  median={out['median']:.0f}")
    print(f"  Reasoning tokens: mean={rea['mean']:.0f}  median={rea['median']:.0f}")

    ct = stats.get("content_tokens")
    if ct and "error" not in ct:
        print(f"\nContent tokens ({ct['tokenizer'].split('/')[-1]}):")
        print(f"  Mean={ct['mean']:.0f}  median={ct['median']:.0f}  p5={ct['p5']:.0f}  p95={ct['p95']:.0f}")
        print(f"  API overhead:     {ct['overhead_ratio']:.2f}x (thinking/channel tokens)")

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
        description="Estimate throughput for neutral text summarization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--api-name", required=True, help="API model name.")
    p.add_argument("--data-path", required=True, help="Path to parquet dir or file.")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--max-concurrent", type=int, default=50)
    p.add_argument("--total-samples", type=int, default=102_772_028)
    p.add_argument("--n-nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output-dir", default="throughput_estimations/results")
    p.add_argument("--endpoint", required=True, help="API endpoint (e.g. http://host:8080/v1).")
    p.add_argument("--api-key", default="none", help='API key. Use "none" for local endpoints.')
    p.add_argument("--thinking", action="store_true", help="Enable thinking mode.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    p.add_argument("--top-p", type=float, default=None, help="Override top-p (nucleus sampling).")
    p.add_argument("--top-k", type=int, default=None, help="Override top-k sampling.")
    p.add_argument("--presence-penalty", type=float, default=None, help="Override presence penalty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.api_name
    model_alias = model_name.split("/")[-1][:30]

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
    key = args.api_key
    if key == "none":
        key = "placeholder"
    client = openai.AsyncOpenAI(api_key=key, base_url=args.endpoint)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Health check
    print(f"Health check: {model_name} ...", end=" ", flush=True)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        )
    except Exception as e:
        raise SystemExit(f"FAILED\n  {e}") from e
    finally:
        loop.close()
    print("OK")

    total_samples = args.n_samples + args.warmup
    print(f"Loading {total_samples} samples from {args.data_path} ...")
    texts = load_samples(args.data_path, total_samples)
    items = prepare_items(texts)
    print(f"Prepared {len(items)} items (max {MAX_TOKENS} tokens each)")

    print(
        f"\nRunning summary estimation: {len(items)} items, "
        f"max_concurrent={args.max_concurrent}, warmup={args.warmup}, "
        f"thinking={args.thinking}"
    )
    results, wall_time_s = run_summary_estimation(
        items, model_name, client, semaphore, args.warmup,
        thinking=args.thinking,
        sampling_params=sampling_params,
    )

    stats = compute_stats(
        results, wall_time_s, args.total_samples,
        args.n_nodes, args.gpus_per_node, args.max_concurrent,
    )
    print("Computing content tokens with SmolLM2 tokenizer...")
    stats["content_tokens"] = compute_content_tokens(results)
    print_summary(stats, model_name, model_alias)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"summary_{model_alias}_{ts}.json"
    output = {
        "meta": {
            "model_name": model_name,
            "model_alias": model_alias,
            "task": "summary",
            "n_samples": args.n_samples,
            "warmup": args.warmup,
            "max_concurrent": args.max_concurrent,
            "n_nodes": args.n_nodes,
            "gpus_per_node": args.gpus_per_node,
            "seed": args.seed,
            "endpoint": args.endpoint,
            "thinking": args.thinking,
            "sampling_params": sampling_params or None,
            "timestamp": ts,
        },
        "stats": stats,
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
