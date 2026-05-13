"""Benchmark summary generation: generate + judge for a model.

Usage:
    uv run python -m pipeline.summaries.benchmark --model glm-4.5-air --n 200
    uv run python -m pipeline.summaries.benchmark --model glm-4.5-air --n 200 --judge kimi-k2.5
"""

from __future__ import annotations

import argparse

from pipeline.config import load_config
from pipeline.summaries.tools import run_summary_batch


def main():
    parser = argparse.ArgumentParser(description="Benchmark summary generation")
    parser.add_argument("--model", type=str, required=True, help="Generator model alias")
    parser.add_argument("--n", type=int, default=200, help="Number of items")
    parser.add_argument("--judge", type=str, default=None, help="Judge model alias")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    judged, run_id = run_summary_batch(
        gen_alias=args.model,
        judge_alias=args.judge,
        n=args.n,
        seed=args.seed,
        source="benchmark",
    )

    cfg = load_config()
    threshold = cfg.charter.improve.scoring.accept_threshold
    mean_score = sum(j["aggregate"] for j in judged) / len(judged) if judged else 0.0
    n_accepted = sum(1 for j in judged if j["aggregate"] >= threshold)

    print(f"\n{'=' * 50}")
    print(f"Benchmark: {args.model}")
    print(f"{'=' * 50}")
    print(f"  Run ID:    {run_id}")
    print(f"  Items:     {len(judged)}")
    print(f"  Accepted:  {n_accepted}/{len(judged)} ({100 * n_accepted / max(len(judged), 1):.1f}%)")
    print(f"  Mean:      {mean_score:.2f}")

    if judged:
        dims = list(judged[0]["scores"].keys())
        print(f"\n  Per-dimension means:")
        for dim in dims:
            dim_mean = sum(j["scores"][dim] for j in judged) / len(judged)
            print(f"    {dim:15s}: {dim_mean:.2f}")


if __name__ == "__main__":
    main()
