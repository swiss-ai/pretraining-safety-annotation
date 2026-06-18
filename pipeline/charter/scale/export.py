"""Export per-rank JSONL results into the doc_id-keyed annotation dataset.

Replaces the legacy merge-into-sidecar step. Each rank's ``results.jsonl`` is
transcoded into one ``dataset/{rank}.parquet`` shard, deduped by ``doc_id``
(keep last). There is no source read and no global ordering — the produced
dataset is keyed and joined downstream by ``doc_id``. The collection of shards
is the single annotation dataset.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.charter.scale.runs import get_run
from pipeline.log import logger

# Arrow types for the run's output columns. Columns not listed fall back to
# large_string. Provenance/usage columns are typed explicitly below and must
# NOT go through this map (it would coerce ints to large_string).
_OUTPUT_COLUMN_TYPES: dict[str, pa.DataType] = {
    "reflection_position": pa.int32(),
}
_DEFAULT_OUTPUT_TYPE = pa.large_string()

# Provenance + token-usage columns added to every row, independent of the run.
_PROVENANCE_TYPES: dict[str, pa.DataType] = {
    "doc_id": pa.string(),
    "corpus": pa.string(),
    "source_shard": pa.large_string(),
    "language": pa.string(),
    "safety_score": pa.int64(),
}
_USAGE_TYPES: dict[str, pa.DataType] = {
    "input_tokens": pa.int64(),
    "output_tokens": pa.int64(),
    "reasoning_tokens": pa.int64(),
}


def _output_type(col: str) -> pa.DataType:
    return _OUTPUT_COLUMN_TYPES.get(col, _DEFAULT_OUTPUT_TYPE)


def _sanitize(values: list) -> list:
    """Replace lone UTF-16 surrogates from LLM output — PyArrow needs valid UTF-8."""
    return [
        v.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        if isinstance(v, str)
        else v
        for v in values
    ]


def _load_dedup(results_path: Path) -> list[dict]:
    """Read a rank's results.jsonl, dedup by doc_id (keep last complete record).

    Torn last lines (the save loop does not fsync) and records missing
    ``doc_id`` are skipped — mirrors the resume tolerance in ``_load_done_set``.
    """
    by_id: dict[str, dict] = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                doc_id = record["doc_id"]
            except (json.JSONDecodeError, KeyError):
                continue
            by_id[doc_id] = record
    return list(by_id.values())


def _build_table(records: list[dict], output_columns: list[str], corpus: str) -> pa.Table:
    """Build the dataset Arrow table: provenance + run output columns + usage."""
    n = len(records)
    cols: dict[str, pa.Array] = {}

    cols["doc_id"] = pa.array([r.get("doc_id") for r in records], type=_PROVENANCE_TYPES["doc_id"])
    cols["corpus"] = pa.array([corpus] * n, type=_PROVENANCE_TYPES["corpus"])
    cols["source_shard"] = pa.array(
        _sanitize([r.get("source_shard") for r in records]), type=_PROVENANCE_TYPES["source_shard"]
    )
    cols["language"] = pa.array([r.get("language") for r in records], type=_PROVENANCE_TYPES["language"])
    cols["safety_score"] = pa.array(
        [r.get("safety_score") for r in records], type=_PROVENANCE_TYPES["safety_score"]
    )

    for col in output_columns:
        col_type = _output_type(col)
        values = [r.get(col) for r in records]
        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
            values = _sanitize(values)
        cols[col] = pa.array(values, type=col_type)

    for col, col_type in _USAGE_TYPES.items():
        cols[col] = pa.array([r.get(col) for r in records], type=col_type)

    return pa.table(cols)


def _write_atomic(table: pa.Table, out_path: Path) -> None:
    """Write *table* to *out_path* via a same-dir temp file + atomic rename."""
    fd, tmp_name = tempfile.mkstemp(suffix=".parquet.tmp", dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def export_run(output_dir: str, run_name: str, corpus: str) -> str:
    """Transcode all per-rank results.jsonl into ``dataset/{rank}.parquet``.

    Idempotent: re-running overwrites each shard. Completeness is validated by
    the datatrove completion markers / ``status`` (not here); this only warns
    about ranks with failures or no completion marker.

    Returns the dataset directory path.
    """
    run_def = get_run(run_name)
    run_dir = Path(output_dir) / run_name
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    completions_dir = run_dir / "completions"

    n_shards = 0
    n_rows = 0
    n_warn = 0
    for rank_dir in sorted(run_dir.iterdir()):
        if not rank_dir.is_dir() or not rank_dir.name.isdigit():
            continue
        results_path = rank_dir / "results.jsonl"
        if not results_path.exists():
            continue
        rank = rank_dir.name

        if not (completions_dir / rank).exists():
            logger.warning("export: rank {} has no completion marker", rank)
            n_warn += 1
        failures = rank_dir / "failures.jsonl"
        if failures.exists() and failures.stat().st_size > 0:
            logger.warning("export: rank {} has a non-empty failures.jsonl", rank)
            n_warn += 1

        records = _load_dedup(results_path)
        if not records:
            continue
        table = _build_table(records, run_def.output_columns, corpus)
        _write_atomic(table, dataset_dir / f"{rank}.parquet")
        n_shards += 1
        n_rows += len(records)

    logger.info(
        "export: wrote {} shards / {} rows to {} ({} warnings)",
        n_shards, n_rows, dataset_dir, n_warn,
    )
    return str(dataset_dir)
