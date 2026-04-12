"""Streaming additive merge: add a run's columns to the sidecar parquet.

Memory usage is O(row_group_size) — one row group at a time. Results
are sorted into a temporary file, then consumed in lock-step with the
sidecar row groups. Each run's columns are added alongside existing
ones; previous runs' columns are preserved.

Column renames on first reflections merge:
- ``reflection`` (empty placeholder) -> dropped, replaced by ``reflection_3p``
- ``preflection`` (empty placeholder) -> dropped, replaced by ``preflection_3p``
"""

from __future__ import annotations

import heapq
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.phase4.runs import get_run


# Columns that get renamed on first merge (empty placeholders -> real data)
_RENAME_MAP = {
    "reflection": "reflection_3p",
    "preflection": "preflection_3p",
}


def merge_shards(
    output_dir: str,
    run_name: str,
    sidecar_path: str,
    out_path: str | None = None,
    allow_missing: bool = False,
) -> str:
    """Streaming merge-join: add a run's columns to the sidecar parquet.

    Args:
        output_dir: Phase 4 output directory (contains ``{run_name}/NNNNN/results.jsonl``).
        run_name: Name of the run to merge (e.g. "reflections").
        sidecar_path: Path to the existing sidecar parquet.
        out_path: Output path. If None, writes to ``{sidecar_path}.merged``.
        allow_missing: If True, fill missing rows with empty strings instead of failing.

    Returns:
        Path to the merged parquet file.
    """
    run_def = get_run(run_name)
    if out_path is None:
        out_path = sidecar_path + ".merged"

    # Step 1: Sort all results by global_row_idx into a temp file
    logger.info("Sorting results from shards...")
    sorted_path, n_results = _sort_results(output_dir, run_name)

    try:
        # Step 2: Merge with sidecar row-group by row-group
        pf = pq.ParquetFile(sidecar_path)
        total_rows = pf.metadata.num_rows

        logger.info("Sorted {} result rows, sidecar has {} rows", n_results, total_rows)

        missing_count = total_rows - n_results
        if missing_count > 0:
            if not allow_missing:
                assert False, (
                    f"Missing {missing_count} rows in results. "
                    f"Expected {total_rows}, got {n_results}. "
                    f"Use allow_missing=True to fill with empty strings."
                )
            logger.warning(
                "{} rows have no results — filling with empty values", missing_count
            )

        output_columns = run_def.output_columns
        writer = None
        row_offset = 0

        # Open sorted results as a streaming cursor
        cursor = _ResultCursor(sorted_path)

        try:
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx)
                rg_num_rows = table.num_rows

                # Handle column renames
                col_names = set(table.column_names)
                for old_name, new_name in _RENAME_MAP.items():
                    if old_name in col_names and new_name in output_columns:
                        table = table.drop(old_name)

                # Build new columns from sorted results cursor
                new_col_data: dict[str, list] = {col: [] for col in output_columns}
                for i in range(rg_num_rows):
                    global_idx = row_offset + i
                    result = cursor.get(global_idx)
                    if result is not None:
                        for col in output_columns:
                            new_col_data[col].append(result.get(col))
                    else:
                        for col in output_columns:
                            new_col_data[col].append(_default_value(col))

                # Append new columns to the table
                for col in output_columns:
                    if col in table.column_names:
                        idx = table.column_names.index(col)
                        table = table.set_column(
                            idx, col, pa.array(new_col_data[col], type=_infer_arrow_type(col))
                        )
                    else:
                        table = table.append_column(
                            col, pa.array(new_col_data[col], type=_infer_arrow_type(col))
                        )

                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)

                row_offset += rg_num_rows
                logger.info(
                    "Merged row group {}/{} ({} rows)",
                    rg_idx + 1, pf.metadata.num_row_groups, row_offset,
                )

        finally:
            if writer is not None:
                writer.close()
            cursor.close()

    finally:
        # Clean up temp file
        sorted_path.unlink(missing_ok=True)

    logger.info("Merge complete: {}", out_path)
    return out_path


class _ResultCursor:
    """Streams through a sorted JSONL file, yielding results by global_row_idx.

    Maintains at most one record in memory. Call ``get(idx)`` with
    monotonically increasing idx values.
    """

    def __init__(self, sorted_path: Path):
        self._f = open(sorted_path, encoding="utf-8")
        self._current: dict | None = None
        self._current_idx: int = -1
        self._advance()

    def _advance(self):
        """Read the next valid record from the file."""
        for line in self._f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                self._current_idx = record["global_row_idx"]
                self._current = record
                return
            except (json.JSONDecodeError, KeyError):
                continue
        self._current = None
        self._current_idx = -1

    def get(self, global_idx: int) -> dict | None:
        """Return the result for global_idx, or None if not present."""
        if self._current is not None and self._current_idx == global_idx:
            result = self._current
            self._advance()
            return result
        return None

    def close(self):
        self._f.close()


def _infer_arrow_type(col_name: str) -> pa.DataType:
    """Infer the Arrow type for an output column."""
    if col_name == "reflection_position":
        return pa.int32()
    if col_name == "canary_type":
        return pa.string()  # nullable
    return pa.large_string()


def _default_value(col_name: str):
    """Default value for a missing row in a given column."""
    if col_name == "reflection_position":
        return 0
    if col_name == "canary_type":
        return None
    return ""


def _sort_results(output_dir: str, run_name: str) -> tuple[Path, int]:
    """Sort all JSONL results by global_row_idx into a temporary file.

    Uses heapq.merge for an O(N log K) merge of K sorted shards.
    Returns (path_to_sorted_file, total_count).
    """
    run_dir = Path(output_dir) / run_name
    shard_files: list[Path] = []

    if run_dir.exists():
        for rank_dir in sorted(run_dir.iterdir()):
            if not rank_dir.is_dir():
                continue
            results_file = rank_dir / "results.jsonl"
            if results_file.exists():
                shard_files.append(results_file)

    # Write sorted output to a temp file
    sorted_path = Path(tempfile.mktemp(suffix=".sorted.jsonl", dir=str(run_dir)))
    count = 0

    if not shard_files:
        sorted_path.touch()
        return sorted_path, 0

    # Results within each shard are in completion order (async), not in
    # global_row_idx order. Load each shard, sort in memory (at most
    # rows_per_task entries ≈ 100K), then merge-sort across shards.
    def _load_sorted_shard(path: Path) -> list[tuple[int, str]]:
        entries: list[tuple[int, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    entries.append((record["global_row_idx"], line))
                except (json.JSONDecodeError, KeyError):
                    continue
        entries.sort(key=lambda x: x[0])
        return entries

    sorted_shards = [_load_sorted_shard(p) for p in shard_files]

    with open(sorted_path, "w", encoding="utf-8") as out:
        for idx, line in heapq.merge(*sorted_shards, key=lambda x: x[0]):
            out.write(line + "\n")
            count += 1

    return sorted_path, count
