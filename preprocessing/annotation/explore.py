"""Explore high-severity safety annotations by joining back to source texts.

Usage::

    # Show 10 random severe (5) samples
    uv run python preprocessing/annotation/explore.py --source-dir $SCRATCH/finephrase/all --min-score 5 -n 10

    # Show all significant+ (4-5) samples, truncate text to 500 chars
    uv run python preprocessing/annotation/explore.py --source-dir $SCRATCH/dolma3_mix-6T/default --min-score 4 --max-chars 500

    # Custom column names
    uv run python preprocessing/annotation/explore.py --source-dir $SCRATCH/dolma3_mix-6T/default --id-column doc_id
"""

import argparse
import random
from glob import glob as _glob
from pathlib import Path

import pyarrow.parquet as pq


LABELS = {
    0: "safe",
    1: "minimal concern",
    2: "mild",
    3: "moderate",
    4: "significant",
    5: "severe",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Explore high-severity safety annotations.")
    p.add_argument("--annotations-dir", default="data/safety_annotations/all", help="Dir with annotation shard parquet files")
    p.add_argument("--min-score", type=int, default=4, help="Minimum score to show (default: 4)")
    p.add_argument("-n", type=int, default=None, help="Sample N items randomly (default: show all)")
    p.add_argument("--max-chars", type=int, default=1000, help="Truncate text display (0=no limit)")
    p.add_argument("--id-column", default="id", help="ID column name in source dataset (default: id)")
    p.add_argument("--text-column", default="text", help="Text column name in source dataset (default: text)")
    p.add_argument("--source-dir", required=True, help="Local parquet dir with source texts")
    args = p.parse_args()

    # Load annotations
    path = Path(args.annotations_dir)
    files = sorted(path.glob("shard_*.parquet"))
    assert files, f"No shard_*.parquet files found in {path}"
    table = pq.read_table(files, columns=["id", "safety_score", "safety_probs"])

    mask = table["safety_score"].to_pylist()
    ids_scores = {
        table["id"][i].as_py(): (mask[i], table["safety_probs"][i].as_py())
        for i in range(len(table))
        if mask[i] >= args.min_score
    }
    print(f"Found {len(ids_scores):,} samples with score >= {args.min_score}")

    if args.n and args.n < len(ids_scores):
        sampled_ids = set(random.sample(list(ids_scores.keys()), args.n))
        ids_scores = {k: v for k, v in ids_scores.items() if k in sampled_ids}
        print(f"Randomly sampled {args.n}")

    target_ids = set(ids_scores.keys())

    # Load source texts
    data_files = sorted(_glob(f"{args.source_dir}/*.parquet"))
    assert data_files, f"No parquet files found in {args.source_dir}"
    print(f"Reading source texts from {args.source_dir} ({len(data_files)} files)\n")

    found = 0
    id_col, text_col = args.id_column, args.text_column
    for f in data_files:
        src = pq.read_table(f, columns=[id_col, text_col])
        for i in range(len(src)):
            sample_id = str(src[id_col][i].as_py())
            if sample_id not in target_ids:
                continue
            score, probs = ids_scores[sample_id]
            text = src[text_col][i].as_py()
            if args.max_chars and len(text) > args.max_chars:
                text = text[: args.max_chars] + f"... [{len(text) - args.max_chars} more chars]"

            print(f"{'='*80}")
            print(f"ID:    {sample_id}")
            print(f"Score: {score} ({LABELS[score]})")
            print(f"Probs: {['%.3f' % p for p in probs]}")
            print(f"{'─'*80}")
            print(text)
            print()

            target_ids.discard(sample_id)
            found += 1
            if not target_ids:
                break
        if not target_ids:
            break

    print(f"{'='*80}")
    print(f"Showed {found}/{len(ids_scores)} samples")


if __name__ == "__main__":
    main()
