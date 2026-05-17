"""Streaming additive merge: add a run's columns to the sidecar parquet.

Memory usage is O(row_group_size) — one row group at a time. Results
are sorted into a temporary file, then consumed in lock-step with the
sidecar row groups. Each run's columns are added alongside existing
ones; previous runs' columns are preserved.

Placeholder columns in the sidecar are dropped on first merge of a run
that adds any of their real replacements — see ``_RENAME_MAP``.
"""

from __future__ import annotations

import heapq
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.charter.scale.runs import get_run

# Placeholder columns in the sidecar that must be dropped when a run adds
# its real, correctly-named columns. Map: placeholder_name -> set of
# output columns that justify dropping the placeholder. The placeholder is
# dropped when any of the mapped columns is in this run's output_columns.
#
# - `reflection` placeholder is dropped by any reflections run (which writes
#   reflection_1p).
# - `preflection` placeholder is dropped by any preflections run — the
#   legacy 2-voice run wrote preflection_3p; the current 4-field run writes
#   charter_summary / neutral / judgemental / idealisation.
_RENAME_MAP: dict[str, set[str]] = {
    "reflection": {"reflection_1p"},
    "preflection": {
        "preflection_3p",
        "charter_summary",
        "neutral",
        "judgemental",
        "idealisation",
    },
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
        output_dir: charter.scale output directory (contains ``{run_name}/NNNNN/results.jsonl``).
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
        output_columns_set = set(output_columns)
        placeholders_to_drop = {
            name
            for name, justifiers in _RENAME_MAP.items()
            if justifiers & output_columns_set
        }
        writer = None
        row_offset = 0

        # Open sorted results as a streaming cursor
        cursor = _ResultCursor(sorted_path)

        try:
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx)
                rg_num_rows = table.num_rows

                col_names = set(table.column_names)
                for old_name in placeholders_to_drop & col_names:
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
                    col_values = new_col_data[col]
                    col_type = _infer_arrow_type(col)
                    # Sanitize lone surrogates — LLM output can contain
                    # them but PyArrow requires valid UTF-8.
                    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
                        col_values = [
                            v.encode("utf-8", errors="surrogatepass")
                             .decode("utf-8", errors="replace")
                            if isinstance(v, str) else v
                            for v in col_values
                        ]
                    arr = pa.array(col_values, type=col_type)
                    if col in table.column_names:
                        idx = table.column_names.index(col)
                        table = table.set_column(idx, col, arr)
                    else:
                        table = table.append_column(col, arr)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)

                row_offset += rg_num_rows
                logger.info(
                    "Merged row group {}/{} ({} rows)",
                    rg_idx + 1,
                    pf.metadata.num_row_groups,
                    row_offset,
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


_COLUMN_META: dict[str, tuple[pa.DataType, object]] = {
    "reflection_position": (pa.int32(), 0),
    "reflection_end_position": (pa.int32(), 0),
    # -1 sentinel: "not backfilled".  0 is a legitimate tok_idx for
    # degenerate single-token docs, so we can't use it as the default.
    "reflection_token_index": (pa.int32(), -1),
    "reflection_end_token_index": (pa.int32(), -1),
    "canary_type": (pa.string(), None),
    "canary_type_end": (pa.string(), None),
    "summary_token_count": (pa.int32(), 0),
}
_DEFAULT_COLUMN_META: tuple[pa.DataType, object] = (pa.large_string(), "")


def _infer_arrow_type(col_name: str) -> pa.DataType:
    """Infer the Arrow type for an output column."""
    return _COLUMN_META.get(col_name, _DEFAULT_COLUMN_META)[0]


def _default_value(col_name: str):
    """Default value for a missing row in a given column."""
    return _COLUMN_META.get(col_name, _DEFAULT_COLUMN_META)[1]


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
