"""Explore high-severity safety annotations by joining back to source texts.

Usage::

    # Show 10 random severe (5) samples (source texts from local parquet)
    uv run python preprocessing/annotation/explore.py --source-dir $SCRATCH/finephrase/all --min-score 5 -n 10

    # Stream from HF instead of local files
    uv run python preprocessing/annotation/explore.py --dataset HuggingFaceFW/finephrase --subset all -n 10

    # Show all significant+ (4-5) samples, truncate text to 500 chars
    uv run python preprocessing/annotation/explore.py --source-dir $SCRATCH/dolma3_mix-6T/default --min-score 4 --max-chars 500

    # Custom column names
    uv run python preprocessing/annotation/explore.py --source-dir $SCRATCH/dolma3_mix-6T/default --id-column doc_id
"""

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset


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

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", help="Local parquet dir with source texts (from safety_annotation.download)")
    source.add_argument("--dataset", help="HuggingFace dataset ID to stream source texts from")
    p.add_argument("--subset", default=None, help="Dataset subset (only used with --dataset)")
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
    if args.source_dir is not None:
        from glob import glob as _glob

        data_files = sorted(_glob(f"{args.source_dir}/*.parquet"))
        assert data_files, f"No parquet files found in {args.source_dir}"
        print(f"Reading source texts from {args.source_dir} ({len(data_files)} files)\n")
        ds = load_dataset("parquet", data_files=data_files, split="train")
    else:
        print(f"Streaming {args.dataset}[{args.subset}] to fetch texts...\n")
        ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)

    found = 0
    for sample in ds:
        sample_id = str(sample[args.id_column])
        if sample_id in target_ids:
            score, probs = ids_scores[sample_id]
            text = sample[args.text_column]
            if args.max_chars and len(text) > args.max_chars:
                text = text[: args.max_chars] + f"... [{len(text) - args.max_chars} more chars]"

            print(f"{'='*80}")
            print(f"ID:    {sample_id}")
            print(f"Score: {score} ({LABELS[score]})")
            print(f"Probs: {['%.3f' % p for p in probs]}")
            print(f"{'─'*80}")
            print(text)
            print()

            found += 1
            if found >= len(ids_scores):
                break

    print(f"{'='*80}")
    print(f"Showed {found}/{len(ids_scores)} samples")


if __name__ == "__main__":
    main()
