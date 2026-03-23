"""Consolidate and upload deduplicated safety annotations to HuggingFace Hub.

Reads annotation shards from all tasks, deduplicates by ID (keeping one score
per unique ID), and uploads as a clean parquet dataset. This is the pure
annotation data without text or upsampling — just (id, safety_score, safety_probs).

Usage::

    # Consolidate and upload
    python -m preprocessing.annotation.upload_annotations \
        --annotation-dir $SCRATCH/safety_annotations/dolma3 \
        --output-dir $SCRATCH/safety_annotations/dolma3_consolidated \
        --repo-id jkminder/dolma3-safety-annotations \
        --workers 16

    # Just consolidate, don't upload
    python -m preprocessing.annotation.upload_annotations \
        --annotation-dir $SCRATCH/safety_annotations/dolma3 \
        --output-dir $SCRATCH/safety_annotations/dolma3_consolidated \
        --no-upload
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidate and upload safety annotations.")
    p.add_argument(
        "--annotation-dir", type=str, required=True,
        help="Directory containing task_XXXX/ subdirectories from array job",
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write consolidated parquet files",
    )
    p.add_argument(
        "--repo-id", type=str, default="jkminder/dolma3-safety-annotations",
        help="HuggingFace repo ID",
    )
    p.add_argument("--private", action="store_true", help="Create a private dataset")
    p.add_argument("--no-upload", action="store_true", help="Skip HF upload, just consolidate")
    p.add_argument(
        "--rows-per-file", type=int, default=5_000_000,
        help="Max rows per output parquet file (default: 5M)",
    )
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


SCHEMA = pa.schema([
    ("id", pa.string()),
    ("safety_score", pa.int8()),
    ("safety_probs", pa.list_(pa.float32())),
])


def main() -> None:
    args = parse_args()
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(annotation_dir.glob("task_*"))
    assert task_dirs, f"No task_XXXX directories found in {annotation_dir}"

    missing_done = [d.name for d in task_dirs if not (d / "DONE").exists()]
    assert not missing_done, f"Incomplete tasks: {missing_done}"

    print(f"Found {len(task_dirs)} tasks, all DONE.")

    # Build global id -> (score, probs) dict
    id_to_annotation: dict[str, tuple[int, list[float]]] = {}
    total_shard_rows = 0
    score_counts: Counter = Counter()

    for task_dir in task_dirs:
        meta = json.loads((task_dir / "task_meta.json").read_text())
        world_size = meta["world_size"]
        shard_files = []
        for rank in range(world_size):
            shard_files.extend(sorted(task_dir.glob(f"shard_{rank:04d}_part*.parquet")))

        task_rows = 0
        for sf in shard_files:
            table = pq.read_table(str(sf))
            if len(table) == 0:
                continue
            ids = table.column("id").to_pylist()
            scores = table.column("safety_score").to_pylist()
            probs = table.column("safety_probs").to_pylist()
            for i, doc_id in enumerate(ids):
                id_to_annotation[doc_id] = (scores[i], probs[i])
                score_counts[scores[i]] += 1
            task_rows += len(table)
        total_shard_rows += task_rows
        print(f"  {task_dir.name}: {task_rows:,} shard rows")

    n_unique = len(id_to_annotation)
    n_cross_dupes = total_shard_rows - n_unique
    print(f"\nTotal shard rows: {total_shard_rows:,}")
    print(f"Unique annotations: {n_unique:,}")
    print(f"Cross-file duplicates removed: {n_cross_dupes:,}")

    # Write consolidated parquet files
    print(f"\nWriting consolidated files ({args.rows_per_file:,} rows each)...")
    items = list(id_to_annotation.items())
    file_idx = 0
    pos = 0
    while pos < len(items):
        chunk = items[pos:pos + args.rows_per_file]
        table = pa.table({
            "id": [item[0] for item in chunk],
            "safety_score": [item[1][0] for item in chunk],
            "safety_probs": [item[1][1] for item in chunk],
        }, schema=SCHEMA)
        out_path = output_dir / f"part_{file_idx:05d}.parquet"
        pq.write_table(table, str(out_path))
        print(f"  {out_path.name}: {len(table):,} rows")
        file_idx += 1
        pos += args.rows_per_file

    # Write metadata
    labels = ["safe", "minimal", "mild", "moderate", "significant", "severe"]
    metadata = {
        "total_unique_annotations": n_unique,
        "total_shard_rows": total_shard_rows,
        "cross_file_duplicates_removed": n_cross_dupes,
        "n_tasks": len(task_dirs),
        "n_output_files": file_idx,
        "score_distribution": {f"{i}_{labels[i]}": score_counts[i] for i in range(6)},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nMetadata written to {output_dir / 'metadata.json'}")

    # Write HF dataset card
    score_table = "\n".join(
        f"| {i} | {labels[i]} | {score_counts[i]:,} | {100*score_counts[i]/n_unique:.2f}% |"
        for i in range(6)
    )
    readme = f"""---
license: apache-2.0
task_categories:
  - text-classification
tags:
  - safety
  - content-moderation
  - annotations
size_categories:
  - 100M<n<1B
---

# Safety Annotations for dolma3_mix

Safety score annotations for a 20K-shard subset of [allenai/dolma3](https://huggingface.co/datasets/allenai/dolma-v1_7) using
[locuslab/safety-classifier_gte-large-en-v1.5](https://huggingface.co/locuslab/safety-classifier_gte-large-en-v1.5).

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Row identifier (matches source dataset) |
| `safety_score` | int8 | Argmax safety class (0-5) |
| `safety_probs` | list[float32] | Full 6-class probability distribution |

## Safety scale

| Score | Label | Count | Percentage |
|-------|-------|------:|------------|
{score_table}

## How this was produced

1. **Annotation** (`annotate.py`): 33 SLURM array tasks on 4×GH200 nodes, each processing ~607 parquet files.
   Within-file deduplication reduces work by ~66% (quality-aware upsampling in source data).
   Uses `torchrun` for multi-GPU inference with length-sorted batching and torch.compile.

2. **Consolidation** (`upload_annotations.py`): All annotation shards deduplicated by ID into
   a single clean dataset. Cross-file duplicate IDs (same ID in multiple source files) are
   collapsed to one entry.

3. **This dataset contains only annotations** — no text. Join on `id` with the source dataset
   to get text + safety scores.

## Resource usage

- ~600 GPU-hours on NVIDIA GH200 120GB (4 GPUs per node)
- Throughput: ~666 unique samples/sec per node (mean across 33 tasks)
- Total unique annotations: **{n_unique:,}**
"""
    (output_dir / "README.md").write_text(readme)
    print("README.md written.")

    if args.no_upload:
        print("\nSkipping upload (--no-upload).")
        return

    # Upload to HuggingFace
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"\nUploading to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["part_*.parquet", "metadata.json", "README.md"],
    )
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
