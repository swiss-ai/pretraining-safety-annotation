"""Shared helpers for reading and identifying the charter.scale sidecar parquet.

Collects utilities used by the submit command, backfill, and verify
scripts so we don't keep reimplementing row-group walks and fingerprints.
"""

from __future__ import annotations

import hashlib
import os

import pyarrow.parquet as pq


# The 10M run's generate.py pre-sliced text before tokenizing with
# ``char_limit = max_text_tokens * 10``.  Backfill / verify scripts must
# replicate that exactly to reproduce the generator's sampling decision,
# so the multiplier lives here and is shared.
LEGACY_CHAR_MULTIPLIER = 10


def apply_legacy_pre_slice(text: str, max_tokens_cap: int) -> str:
    """Reproduce the pre-slice the generator used during the EXP-001 run.

    ``max_tokens_cap`` is the run-level cap (``cfg.max_tokens``, typically
    1920) — NOT the per-doc ``token_length``.  The generator used the
    constant at the time, not the per-doc value, so RNG replay must match.
    """
    char_limit = max_tokens_cap * LEGACY_CHAR_MULTIPLIER
    return text if len(text) <= char_limit else text[:char_limit]


def load_rank_docs(
    sidecar_path: str, rank: int, rows_per_task: int
) -> dict[int, tuple[str, str, int]]:
    """Return ``{global_row_idx: (doc_id, text, token_length)}`` for one rank.

    Mirrors ``SidecarReader.run``'s row-group walk but returns a dict so
    callers can look up rows by their ``global_row_idx`` from results
    JSONL.
    """
    start = rank * rows_per_task
    end = start + rows_per_task
    pf = pq.ParquetFile(sidecar_path)
    end = min(end, pf.metadata.num_rows)

    out: dict[int, tuple[str, str, int]] = {}
    row_offset = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_num_rows = pf.metadata.row_group(rg_idx).num_rows
        rg_start = row_offset
        rg_end = row_offset + rg_num_rows
        if rg_end <= start or rg_start >= end:
            row_offset = rg_end
            continue
        table = pf.read_row_group(
            rg_idx, columns=["doc_id", "text", "token_length"]
        )
        slice_start = max(0, start - rg_start)
        slice_end = min(rg_num_rows, end - rg_start)
        table = table.slice(slice_start, slice_end - slice_start)
        col = table.to_pydict()
        for i in range(table.num_rows):
            gidx = rg_start + slice_start + i
            out[gidx] = (col["doc_id"][i], col["text"][i], col["token_length"][i])
        row_offset = rg_end
    return out


def sidecar_fingerprint(sidecar_path: str) -> dict:
    """Cheap, deterministic identity for a sidecar parquet.

    Full sha256 of a 500GB file would take ~20 min; instead we combine
    file size, row count, row-group count, and a hash of the Arrow
    schema.  Any swap, resort, or column change shows up in at least one
    field.  The backfill script asserts this matches the fingerprint
    recorded at submit time before it rewrites any results.
    """
    pf = pq.ParquetFile(sidecar_path)
    meta = pf.metadata
    schema_bytes = meta.schema.to_arrow_schema().serialize().to_pybytes()
    return {
        "file_size": os.path.getsize(sidecar_path),
        "num_rows": meta.num_rows,
        "num_row_groups": meta.num_row_groups,
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
    }
