"""Merge safety annotations back into original parquet files.

Reads per-task annotation shards produced by the array job and writes the
``safety_score`` column (int8, argmax only) into a separate output directory,
preserving original filenames.

Id-based merge: annotation shards contain (id, safety_score) pairs for
deduplicated rows. The merge reads each original input file, looks up
every row's id in the annotation dict, and writes the score for every row
(including duplicates that share the same id and therefore the same score).

Usage::

    python -m preprocessing.annotation.merge \\
        --data-dir $SCRATCH/dolma3_mix-1T \\
        --annotation-dir $SCRATCH/safety_annotations/dolma3 \\
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
    p.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Column name for row ID used in id-based merge (default: id)",
    )
    return p.parse_args()


def _read_task_annotations(task_dir: Path, id_column: str = "id") -> tuple[dict, dict[str, int]]:
    """Read task metadata and build id -> safety_score mapping from annotation shards.

    Returns (task_meta dict, {id_string: safety_score_int}).
    """
    meta_path = task_dir / "task_meta.json"
    assert meta_path.exists(), f"Missing task_meta.json in {task_dir}"
    meta = json.loads(meta_path.read_text())

    world_size = meta["world_size"]
    shard_files = []
    for rank in range(world_size):
        rank_shards = sorted(task_dir.glob(f"shard_{rank:04d}_part*.parquet"))
        assert rank_shards, (
            f"Task {task_dir.name}: no shards for rank {rank} "
            f"(expected {world_size} ranks)"
        )
        shard_files.extend(rank_shards)

    id_to_score: dict[str, int] = {}
    for sf in shard_files:
        table = pq.read_table(str(sf), columns=[id_column, "safety_score"])
        id_to_score.update(zip(
            table.column(id_column).to_pylist(),
            table.column("safety_score").to_pylist(),
        ))

    assert len(id_to_score) == meta["n_input_rows"], (
        f"Task {task_dir.name}: {len(id_to_score)} unique annotations != "
        f"{meta['n_input_rows']} expected (deduped count)"
    )
    return meta, id_to_score


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
        meta, id_to_score = _read_task_annotations(task_dir, id_column=args.id_column)
        input_files = [data_dir / f for f in meta["files"]]

        for f in input_files:
            assert f.exists(), f"Input file not found: {f}"

        row_counts = [pq.read_metadata(str(f)).num_rows for f in input_files]
        total_input_rows = sum(row_counts)
        assert total_input_rows == meta["n_original_rows"], (
            f"Task {task_dir.name}: sum of input file rows {total_input_rows} != "
            f"metadata n_original_rows {meta['n_original_rows']}"
        )

        work_items = []
        for input_file in input_files:
            ids = pq.read_table(str(input_file), columns=[args.id_column]).column(args.id_column).to_pylist()
            scores = np.array([id_to_score[str(doc_id)] for doc_id in ids], dtype=np.int8)
            work_items.append((input_file, output_dir / input_file.name, scores))

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_write_annotated_file, *item) for item in work_items]
            for fut in futures:
                total_rows_written += fut.result()

        total_files_written += len(input_files)
        dedup_pct = 100 * (1 - len(id_to_score) / total_input_rows) if total_input_rows > 0 else 0
        print(f"  {task_dir.name}: merged {len(input_files)} files "
              f"({total_input_rows:,} rows, {len(id_to_score):,} unique, {dedup_pct:.1f}% dedup)")

    print(f"\nDone. {total_files_written} files, {total_rows_written:,} rows written to {output_dir}")


if __name__ == "__main__":
    main()
