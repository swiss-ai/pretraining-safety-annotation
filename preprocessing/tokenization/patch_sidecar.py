"""Patch an existing sidecar.parquet to add safety_score and is_bad columns.

Replicates the MegatronAnnotatedShuffler's positional ordering contract:
reads source annotated parquets in ``sorted(rglob("*.parquet"))`` order,
applies the same ``default_rng(seed).permutation(n_docs)`` shuffle, and
assigns each sidecar row its source safety_score.  is_bad is derived as
safety_score >= threshold.

This avoids a doc_id join (which fails when duplicate IDs exist across
files with different scores).

Usage::

    python -m preprocessing.tokenization.patch_sidecar \\
        --sidecar $SCRATCH/tokenized/annotated/sidecar.parquet \\
        --annotated-data-dir $SCRATCH/dolma3_mix-1T_subsampled/annotated

    # Custom threshold / seed
    python -m preprocessing.tokenization.patch_sidecar \\
        --sidecar $SCRATCH/tokenized/annotated/sidecar.parquet \\
        --annotated-data-dir $SCRATCH/dolma3_mix-1T_subsampled/annotated \\
        --threshold 4 --seed 42

    # Dry run (prints stats, doesn't write)
    python -m preprocessing.tokenization.patch_sidecar \\
        --sidecar $SCRATCH/tokenized/annotated/sidecar.parquet \\
        --annotated-data-dir $SCRATCH/dolma3_mix-1T_subsampled/annotated \\
        --dry-run
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.random import default_rng
from tqdm import tqdm

ANNOTATION_THRESHOLD = 3


def parse_args() -> argparse.Namespace:
    scratch = os.environ.get(
        "SCRATCH",
        f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
    )
    p = argparse.ArgumentParser(
        description="Patch sidecar.parquet with safety_score and is_bad columns."
    )
    p.add_argument(
        "--sidecar",
        type=str,
        default=f"{scratch}/tokenized/annotated/sidecar.parquet",
        help="Path to existing sidecar.parquet",
    )
    p.add_argument(
        "--annotated-data-dir",
        type=str,
        default=f"{scratch}/dolma3_mix-1T_subsampled/annotated",
        help="Directory with source annotated parquets (same as tokenize --annotated-data-dir)",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=ANNOTATION_THRESHOLD,
        help=f"is_bad threshold: safety_score >= this (default: {ANNOTATION_THRESHOLD})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (must match tokenize --seed, default: 42)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print stats only, don't write")
    return p.parse_args()


def build_shuffled_scores(annotated_data_dir: str, seed: int) -> np.ndarray:
    """Build safety_score array in shuffled sidecar output order.

    Replicates the MegatronAnnotatedShuffler ordering contract:
    1. sorted(rglob("*.parquet")) gives file order
    2. Within each file, rows are in parquet row order
    3. default_rng(seed).permutation(n_docs) shuffles to output order
    """
    sorted_parquets = sorted(Path(annotated_data_dir).rglob("*.parquet"))
    assert sorted_parquets, f"No parquets found in {annotated_data_dir}"

    # Pass 1: count rows per file
    file_row_counts = []
    for pf in tqdm(sorted_parquets, desc="Counting rows", unit="file"):
        file_row_counts.append(pq.ParquetFile(str(pf)).metadata.num_rows)
    n_docs = sum(file_row_counts)
    print(f"Total docs: {n_docs:,} across {len(sorted_parquets):,} files")

    # Pass 2: read safety_score in file order → flat array
    scores_flat = np.empty(n_docs, dtype=np.int8)
    offset = 0
    for fi, pf in enumerate(
        tqdm(sorted_parquets, desc="Reading safety_scores", unit="file")
    ):
        table = pq.read_table(str(pf), columns=["safety_score"])
        n = len(table)
        scores_flat[offset : offset + n] = table.column("safety_score").to_numpy()
        offset += n
    assert offset == n_docs

    # Apply same permutation as the shuffler
    rng = default_rng(seed)
    perm = rng.permutation(n_docs)
    shuffled_scores = scores_flat[perm]
    print(f"Shuffled {n_docs:,} scores (seed={seed})")
    return shuffled_scores


def main() -> None:
    args = parse_args()
    sidecar_path = Path(args.sidecar)
    assert sidecar_path.exists(), f"Sidecar not found: {sidecar_path}"

    # Check existing schema
    pf = pq.ParquetFile(str(sidecar_path))
    existing_cols = set(pf.schema_arrow.names)
    n_rows = pf.metadata.num_rows
    n_rg = pf.metadata.num_row_groups

    if "safety_score" in existing_cols and "is_bad" in existing_cols:
        print(f"Sidecar already has safety_score and is_bad columns. Nothing to do.")
        return

    print(f"Sidecar: {n_rows:,} rows, {n_rg} row groups")
    print(f"Existing columns: {sorted(existing_cols)}")

    # Build shuffled score array
    t0 = time.time()
    shuffled_scores = build_shuffled_scores(args.annotated_data_dir, args.seed)
    print(f"Scores built in {time.time() - t0:.1f}s")

    assert len(shuffled_scores) == n_rows, (
        f"Score count ({len(shuffled_scores):,}) != sidecar rows ({n_rows:,}). "
        f"Wrong --annotated-data-dir or --seed?"
    )

    if args.dry_run:
        print(f"Dry run: would write {n_rows:,} rows with safety_score and is_bad")
        print(f"  Score distribution: {np.bincount(shuffled_scores.astype(np.uint8), minlength=6)}")
        return

    # Spot-check: verify doc_ids match between sidecar and source for first row group
    rg0 = pf.read_row_group(0, columns=["doc_id"])
    rg0_ids = rg0.column("doc_id").to_pylist()
    # Read first few source parquets to build expected id order after shuffle
    sorted_parquets = sorted(Path(args.annotated_data_dir).rglob("*.parquet"))
    file_row_counts = [pq.ParquetFile(str(f)).metadata.num_rows for f in sorted_parquets]
    n_docs = sum(file_row_counts)
    # Build id_flat for the first N ids we need
    rng = default_rng(args.seed)
    perm = rng.permutation(n_docs)
    n_check = min(100, len(rg0_ids))
    # For checking, we need the actual IDs at the first n_check perm positions
    # Build (file_idx, row_in_file) for those positions
    cum = np.cumsum([0] + file_row_counts)
    check_positions = perm[:n_check]
    file_for_pos = np.searchsorted(cum[1:], check_positions, side="right")
    row_for_pos = check_positions - cum[file_for_pos]
    n_id_match = 0
    for i in range(n_check):
        fi = int(file_for_pos[i])
        ri = int(row_for_pos[i])
        t = pq.read_table(str(sorted_parquets[fi]), columns=["id"])
        expected_id = t.column("id")[ri].as_py()  # keep as native (may be None)
        if rg0_ids[i] == expected_id:
            n_id_match += 1
    print(f"Spot-check: {n_id_match}/{n_check} doc_ids match (positional)")
    assert n_id_match == n_check, (
        f"Positional doc_id mismatch! Only {n_id_match}/{n_check} matched. "
        f"Wrong --annotated-data-dir or --seed?"
    )

    # Write patched sidecar
    new_schema = pa.schema([
        (field.name, field.type) for field in pf.schema_arrow
    ] + [
        ("safety_score", pa.int8()),
        ("is_bad", pa.bool_()),
    ])

    tmp_path = sidecar_path.with_suffix(".parquet.tmp")
    backup_path = sidecar_path.with_suffix(".parquet.bak")

    row_offset = 0
    with pq.ParquetWriter(str(tmp_path), new_schema) as writer:
        for rg_i in tqdm(range(n_rg), desc="Patching row groups", unit="rg"):
            rg = pf.read_row_group(rg_i)
            rg_len = len(rg)

            scores_slice = shuffled_scores[row_offset : row_offset + rg_len]
            scores_arr = pa.array(scores_slice.tolist(), type=pa.int8())
            is_bad_arr = pa.array(
                (scores_slice >= args.threshold).tolist(),
                type=pa.bool_(),
            )

            new_rg = rg.append_column("safety_score", scores_arr)
            new_rg = new_rg.append_column("is_bad", is_bad_arr)
            writer.write_table(new_rg)
            row_offset += rg_len

    assert row_offset == n_rows

    # Atomic replace: backup original, move temp to final
    if backup_path.exists():
        backup_path.unlink()
    shutil.move(str(sidecar_path), str(backup_path))
    shutil.move(str(tmp_path), str(sidecar_path))
    print(f"Patched sidecar written to {sidecar_path}")
    print(f"Original backed up to {backup_path}")

    # Verify
    check = pq.ParquetFile(str(sidecar_path))
    assert check.metadata.num_rows == n_rows
    assert "safety_score" in set(check.schema_arrow.names)
    assert "is_bad" in set(check.schema_arrow.names)
    print(f"Verified: {check.metadata.num_rows:,} rows, schema OK")


if __name__ == "__main__":
    main()
