"""Materialize a deterministic prompts list to a parquet file.

This runs ONCE on the login node before SLURM submit. Compute nodes
typically don't have HF auth/internet — they read the materialised
prompts.parquet from $SCRATCH instead.

Each row in the parquet has: ``global_row_idx`` (int64), ``source``
(string), ``source_id`` (string), ``user`` (large_string), ``meta``
(string, JSON-encoded). Order is the deterministic order from
``sample_mix(n, seed)`` so rank R always handles rows
``[R*rows_per_task, (R+1)*rows_per_task)``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.phase5.data import sample_mix


def materialize_prompts(
    out_path: Path,
    n: int,
    seed: int,
) -> dict:
    """Write a deterministic prompts.parquet. Returns a fingerprint dict.

    If `out_path` exists, the function asserts that its fingerprint
    matches the one for ``(n, seed)`` and returns the existing fingerprint
    without rewriting (so submit/rerun is idempotent).
    """
    assert n >= 3 and n % 3 == 0, (
        f"n must be divisible by 3 and >= 3 for the three-way source split, got {n}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = _read_fingerprint(out_path)
        if existing.get("n") != n or existing.get("seed") != seed:
            raise AssertionError(
                f"Existing prompts.parquet fingerprint mismatch: "
                f"expected n={n}, seed={seed}; "
                f"file has n={existing.get('n')}, seed={existing.get('seed')}. "
                f"Delete {out_path} to regenerate."
            )
        # Re-read prompts and re-hash to detect upstream HF dataset drift
        # (HarmfulQA repo update, WildChat shard reshuffle).
        existing_hash = existing.get("content_sha256")
        if existing_hash is not None:
            current_hash = _content_sha256_from_parquet(out_path)
            if current_hash != existing_hash:
                raise AssertionError(
                    f"prompts.parquet content drift: stored sha256 "
                    f"{existing_hash[:16]}... != recomputed {current_hash[:16]}... "
                    f"Delete {out_path} to regenerate (will pick up new upstream rows)."
                )
        logger.info("prompts.parquet already exists with matching fingerprint: {}", existing)
        return existing

    logger.info("sampling {} prompts (seed={})...", n, seed)
    picks = sample_mix(n=n, seed=seed)
    rows = []
    for i, sp in enumerate(picks):
        rows.append({
            "global_row_idx": i,
            "source": sp.source,
            "source_id": sp.source_id,
            "user": sp.user,
            "meta": json.dumps(sp.meta or {}, ensure_ascii=False),
        })

    schema = pa.schema([
        ("global_row_idx", pa.int64()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("user", pa.large_string()),
        ("meta", pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)

    fingerprint = {
        "n": n,
        "seed": seed,
        "n_harmfulqa": sum(1 for r in rows if r["source"] == "harmfulqa"),
        "n_wildchat": sum(1 for r in rows if r["source"] == "wildchat"),
        "n_wildguardmix": sum(1 for r in rows if r["source"] == "wildguardmix"),
        "content_sha256": _content_sha256(rows),
    }
    (out_path.parent / "prompts_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2)
    )
    logger.info("wrote {} ({} rows). fingerprint: {}", out_path, len(rows), fingerprint)
    return fingerprint


def _read_fingerprint(prompts_path: Path) -> dict:
    """Read the sidecar fingerprint json next to prompts.parquet."""
    fp_path = prompts_path.parent / "prompts_fingerprint.json"
    if not fp_path.exists():
        # Reconstruct minimal fingerprint from the file itself
        pf = pq.ParquetFile(prompts_path)
        return {"n": pf.metadata.num_rows, "seed": None}
    return json.loads(fp_path.read_text())


def _content_sha256(rows: list[dict]) -> str:
    """sha256 over (source_id, user) for content drift detection."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r["source_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(r["user"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _content_sha256_from_parquet(parquet_path: Path) -> str:
    """Recompute content_sha256 by streaming row-groups (avoids full load)."""
    h = hashlib.sha256()
    pf = pq.ParquetFile(parquet_path)
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["source_id", "user"])
        sids = table.column("source_id").to_pylist()
        users = table.column("user").to_pylist()
        for sid, user in zip(sids, users):
            h.update(sid.encode("utf-8"))
            h.update(b"\x00")
            h.update(user.encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()
