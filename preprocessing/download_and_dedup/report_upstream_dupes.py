"""Report upstream duplicates in a HuggingFace dataset shard.

Downloads a specific file from the dataset repo, decompresses it,
and reports duplication statistics.

Usage::

    uv run python preprocessing/report_upstream_dupes.py \
        --dataset allenai/dolma3_mix-6T \
        --file data/common_crawl-crime_and_law-0019/shard_00000079.jsonl.zst
"""

import argparse
import io
import json
from collections import Counter

import zstandard
from huggingface_hub import hf_hub_download


def main() -> None:
    p = argparse.ArgumentParser(description="Report upstream duplicates in a HuggingFace dataset shard.")
    p.add_argument("--dataset", required=True, help="HuggingFace dataset ID")
    p.add_argument("--file", required=True, help="Path to file within the dataset repo")
    p.add_argument("--id-column", default="id", help="ID column name (default: id)")
    args = p.parse_args()

    print(f"Downloading {args.dataset}/{args.file} ...")
    local_path = hf_hub_download(args.dataset, args.file, repo_type="dataset")

    print(f"Reading {local_path} ...")
    dctx = zstandard.ZstdDecompressor()
    rows_by_id: dict[str, list[int]] = {}
    n_rows = 0
    with open(local_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            for i, line in enumerate(io.TextIOWrapper(reader, encoding="utf-8")):
                record = json.loads(line)
                rid = str(record[args.id_column])
                rows_by_id.setdefault(rid, []).append(i)
                n_rows += 1

    n_unique = len(rows_by_id)
    occurrence_counts = Counter(len(lines) for lines in rows_by_id.values())

    print(f"\nFile: {args.file}")
    print(f"Total rows:  {n_rows:,}")
    print(f"Unique IDs:  {n_unique:,}")
    print(f"Duplicates:  {n_rows - n_unique:,} ({100 * (1 - n_unique / n_rows):.1f}% of rows)")
    print(f"\nOccurrence distribution:")
    for count, n_ids in sorted(occurrence_counts.items()):
        label = "unique" if count == 1 else "duplicated"
        print(f"  {count}x: {n_ids:,} IDs ({label})")

    # Check whether duplicates are consecutive and identical
    duped_ids = [(rid, lines) for rid, lines in rows_by_id.items() if len(lines) > 1]
    if duped_ids:
        sample_id, sample_lines = duped_ids[0]
        consecutive = all(sample_lines[i + 1] - sample_lines[i] == 1 for i in range(len(sample_lines) - 1))
        print(f"\nSample duplicated ID: {sample_id}")
        print(f"  Appears at lines: {sample_lines}")
        print(f"  Consecutive: {consecutive}")


if __name__ == "__main__":
    main()
