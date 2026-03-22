"""Analyze safety annotation results: class distribution and basic stats."""

import sys
from pathlib import Path

import pyarrow.parquet as pq


def analyze(data_dir: str) -> None:
    path = Path(data_dir)
    files = sorted(path.glob("shard_*.parquet"))
    if not files:
        print(f"No parquet files found in {path}")
        sys.exit(1)

    table = pq.read_table(files, columns=["safety_score"])
    scores = table["safety_score"].to_pylist()
    n = len(scores)

    print(f"Dataset: {path}")
    print(f"Files:   {len(files)}")
    print(f"Samples: {n:,}\n")

    labels = {
        0: "safe",
        1: "minimal concern",
        2: "mild",
        3: "moderate",
        4: "significant",
        5: "severe",
    }

    counts = [0] * 6
    for s in scores:
        counts[s] += 1

    print(f"{'Score':<6} {'Label':<18} {'Count':>10} {'Pct':>7}")
    print("-" * 45)
    for i in range(6):
        pct = 100 * counts[i] / n if n else 0
        print(f"  {i:<4} {labels[i]:<18} {counts[i]:>10,} {pct:>6.2f}%")

    safe = counts[0] + counts[1]
    unsafe = n - safe
    print(f"\nSafe (0-1):   {safe:>10,}  ({100*safe/n:.2f}%)")
    print(f"Unsafe (2-5): {unsafe:>10,}  ({100*unsafe/n:.2f}%)")


if __name__ == "__main__":
    import os
    scratch = os.environ.get(
        "SCRATCH",
        f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
    )
    analyze(sys.argv[1] if len(sys.argv) > 1 else f"{scratch}/safety_annotations/all")
