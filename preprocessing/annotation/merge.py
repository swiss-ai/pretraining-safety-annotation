"""Merge safety annotations back into original parquet files.

Reads per-task annotation shards produced by the array job and writes the
``safety_score`` column (int8, argmax only) into a separate output directory,
preserving original filenames.

Positional merge: each task's annotations preserve input row order.
``split_dataset_by_node`` with contiguous sharding gives rank 0 rows [0, N/W),
rank 1 rows [N/W, 2N/W), etc. Concatenating annotation shards in rank order
reconstructs the full annotation sequence matching the original row order.

Usage::

    python -m preprocessing.annotation.merge \\
        --data-dir $SCRATCH/dolma3_mix-1T \\
        --annotation-dir data/safety_annotations/dolma3 \\
        --output-dir $SCRATCH/dolma3_mix-1T_annotated \\
        --workers 8
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge safety annotations into parquet files.")
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with original part_*.parquet files",
    )
    p.add_argument(
        "--annotation-dir",
        type=str,
        required=True,
        help="Directory containing task_XXXX/ subdirectories from array job",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write annotated parquet files (same filenames as input)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for writing output files (default: 8)",
    )
    return p.parse_args()


def _read_task_annotations(task_dir: Path) -> tuple[dict, np.ndarray]:
    """Read task metadata and concatenate all annotation shards into a single score array.

    Returns (task_meta dict, int8 score array of length n_input_rows).
    """
    meta_path = task_dir / "task_meta.json"
    assert meta_path.exists(), f"Missing task_meta.json in {task_dir}"
    meta = json.loads(meta_path.read_text())

    world_size = meta["world_size"]
    shard_files = []
    for rank in range(world_size):
        rank_shards = sorted(task_dir.glob(f"shard_{rank:04d}_part*.parquet"))
        shard_files.extend(rank_shards)

    assert shard_files, f"No annotation shards found in {task_dir}"

    scores = []
    for sf in shard_files:
        table = pq.read_table(str(sf), columns=["safety_score"])
        scores.append(table.column("safety_score").to_numpy())

    scores_arr = np.concatenate(scores).astype(np.int8)
    assert len(scores_arr) == meta["n_input_rows"], (
        f"Task {task_dir.name}: annotation count {len(scores_arr)} != "
        f"expected {meta['n_input_rows']} input rows"
    )
    return meta, scores_arr


def _write_annotated_file(
    input_path: Path,
    output_path: Path,
    scores_slice: np.ndarray,
) -> int:
    """Read an input parquet file, append safety_score column, write to output."""
    table = pq.read_table(str(input_path))
    assert len(table) == len(scores_slice), (
        f"{input_path.name}: table rows {len(table)} != scores slice {len(scores_slice)}"
    )
    table = table.append_column(
        "safety_score",
        pa.array(scores_slice, type=pa.int8()),
    )
    pq.write_table(table, str(output_path))
    return len(table)


def main() -> None:
    args = parse_args()
    annotation_dir = Path(args.annotation_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── discover and validate tasks ─────────────────────────────────
    task_dirs = sorted(annotation_dir.glob("task_*"))
    assert task_dirs, f"No task_XXXX directories found in {annotation_dir}"

    missing_done = [d.name for d in task_dirs if not (d / "DONE").exists()]
    assert not missing_done, (
        f"{len(missing_done)} task(s) missing DONE marker — "
        f"incomplete or failed: {missing_done}"
    )

    missing_meta = [d.name for d in task_dirs if not (d / "task_meta.json").exists()]
    assert not missing_meta, (
        f"{len(missing_meta)} task(s) missing task_meta.json: {missing_meta}"
    )

    print(f"Found {len(task_dirs)} tasks, all with DONE markers.")

    # ── process each task ───────────────────────────────────────────
    total_files_written = 0
    total_rows_written = 0

    for task_dir in task_dirs:
        meta, scores_arr = _read_task_annotations(task_dir)
        input_files = [data_dir / f for f in meta["files"]]

        # get row counts per input file via metadata (fast, no data load)
        row_counts = []
        for f in input_files:
            assert f.exists(), f"Input file not found: {f}"
            row_counts.append(pq.read_metadata(str(f)).num_rows)

        total_input_rows = sum(row_counts)
        assert total_input_rows == meta["n_input_rows"], (
            f"Task {task_dir.name}: sum of input file rows {total_input_rows} != "
            f"metadata n_input_rows {meta['n_input_rows']}"
        )

        # compute cumulative offsets to slice the score array per file
        offsets = np.cumsum([0] + row_counts)

        # prepare work items for parallel writing
        work_items = []
        for i, input_file in enumerate(input_files):
            output_path = output_dir / input_file.name
            score_slice = scores_arr[offsets[i] : offsets[i + 1]]
            work_items.append((input_file, output_path, score_slice))

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_write_annotated_file, *item) for item in work_items]
            for fut in futures:
                total_rows_written += fut.result()

        total_files_written += len(input_files)
        print(f"  {task_dir.name}: merged {len(input_files)} files ({total_input_rows:,} rows)")

    print(f"\nDone. {total_files_written} files, {total_rows_written:,} rows written to {output_dir}")


if __name__ == "__main__":
    main()
