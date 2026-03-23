"""Annotation-based subsampling with two output datasets.

Produces two subsets from annotated source parquets:

* ``annotated/`` — rows marked for annotation (safety_score >= threshold,
  plus a matched random sample from lower scores).
* ``unannotated/`` — the remaining rows.

Token budgets are split proportionally so that the annotation ratio in the
output matches the full dataset.  A global per-row priority (seeded RNG)
ensures monotonic subset inclusion: sampling at budget X is always a subset
of sampling at budget X+Y with identical ``has_annotation`` flags.

Usage::

    python -m preprocessing.subsample_and_stratify.subsample \
        --source-dir $SCRATCH/dolma3_mix-1T_annotated \
        --target-tokens 500_000_000_000

    # Smaller test run
    python -m preprocessing.subsample_and_stratify.subsample \
        --source-dir $SCRATCH/dolma3_mix-10m_annotated \
        --target-tokens 1_000_000 --output-dir $SCRATCH/subsampled_test

    # Legacy stratified mode (three independent budgets)
    python -m preprocessing.subsample_and_stratify.subsample \
        --source-dir $SCRATCH/dolma3_mix-1T_annotated \
        --bad-fraction 0.025 --good-fraction 0.025
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


def scan_source(
    source_dir: Path,
    id_column: str,
    text_column: str,
    chars_per_token: float,
) -> pa.Table:
    """Scan source parquet files and build a lightweight index table.

    Source files must contain ``id``, ``text``, and ``safety_score`` columns
    (produced by the annotation merge step).

    Returns a PyArrow table with columns:
    ``(id: string, est_tokens: float32, safety_score: int8, file_idx: int32)``.
    """
    source_files = sorted(source_dir.glob("part_*.parquet"))
    assert source_files, f"No part_*.parquet files found in {source_dir}"
    print(f"\nScanning {len(source_files)} source files...")

    all_ids: list[pa.Array] = []
    all_tokens: list[pa.Array] = []
    all_scores: list[pa.Array] = []
    all_file_idx: list[pa.Array] = []

    for file_idx, fpath in enumerate(tqdm(source_files, desc="Source files")):
        table = pq.read_table(fpath, columns=[id_column, text_column, "safety_score"])
        ids = table.column(id_column)
        text = table.column(text_column)
        scores = table.column("safety_score").cast(pa.int8())

        lengths = pc.utf8_length(text).cast(pa.float32())
        est_tokens = pc.divide(lengths, pa.scalar(chars_per_token, pa.float32()))

        all_ids.append(ids)
        all_tokens.append(est_tokens)
        all_scores.append(scores)
        all_file_idx.append(pa.array([file_idx] * len(table), type=pa.int32()))

    # Column arrays from pq.read_table are ChunkedArrays; flatten before concat
    def _concat(arrays: list[pa.Array | pa.ChunkedArray]) -> pa.Array:
        chunks = []
        for a in arrays:
            if isinstance(a, pa.ChunkedArray):
                chunks.extend(a.chunks)
            else:
                chunks.append(a)
        return pa.concat_arrays(chunks)

    index = pa.table({
        "id": _concat(all_ids),
        "est_tokens": _concat(all_tokens),
        "safety_score": _concat(all_scores),
        "file_idx": _concat(all_file_idx),
    })

    total_tokens = pc.sum(index.column("est_tokens")).as_py()
    print(f"\nIndex: {len(index):,} rows, {_fmt_tokens(total_tokens)} total tokens")

    # Score distribution
    scores_np = index.column("safety_score").to_pylist()
    counts = [0] * 6
    for s in scores_np:
        counts[s] += 1
    print("\nScore distribution:")
    for i in range(6):
        pct = 100 * counts[i] / len(index)
        print(f"  {i}: {counts[i]:>12,} ({pct:5.2f}%)")

    return index


# ---------------------------------------------------------------------------
# New annotation-based sampling (default)
# ---------------------------------------------------------------------------


def mark_and_sample(
    index: pa.Table,
    target_tokens: float,
    seed: int,
    annotation_threshold: int = 3,
) -> tuple[pa.Table, pa.Table, dict]:
    """Mark annotations and sample two pools preserving the annotation ratio.

    Phase 1 (budget-independent): all rows with safety_score >= *annotation_threshold*
    are annotated.  An equal token budget of lower-score rows (sorted by priority)
    is also annotated.  This marking is deterministic and independent of
    *target_tokens*.

    Phase 2 (budget-dependent): each pool (annotated / unannotated) is filled to
    ``target_tokens * R`` and ``target_tokens * (1-R)`` respectively, where R is the
    annotation ratio from Phase 1.

    Returns ``(annotated_table, unannotated_table, stats_dict)`` where each table
    has the index columns plus ``has_annotation`` (bool).
    """
    total_available = pc.sum(index.column("est_tokens")).as_py()
    est_tokens = index.column("est_tokens").to_numpy(zero_copy_only=False)

    # Global per-row priority — ensures monotonic subset inclusion.
    rng = np.random.default_rng(seed)
    priorities = rng.random(len(index))

    # ── Phase 1: annotation marking (budget-independent) ─────────────
    scores = index.column("safety_score")
    high_mask = pc.greater_equal(scores, annotation_threshold)
    high_indices = pc.indices_nonzero(high_mask).to_numpy().copy()
    low_indices = pc.indices_nonzero(pc.invert(high_mask)).to_numpy().copy()

    high_score_tokens = float(est_tokens[high_indices].sum())

    # Sample low-score rows to match high-score token count
    low_sorted = low_indices[np.argsort(priorities[low_indices])]
    sampled_low, sampled_low_tokens = _fill_budget(
        low_sorted, est_tokens, high_score_tokens,
    )
    if sampled_low_tokens < high_score_tokens:
        print(f"  WARNING: low-score pool shortfall — got {_fmt_tokens(sampled_low_tokens)} "
              f"of {_fmt_tokens(high_score_tokens)} target")

    # Build annotation mask (fixed per row, independent of target_tokens)
    is_annotated = np.zeros(len(index), dtype=bool)
    is_annotated[high_indices] = True
    is_annotated[sampled_low] = True

    annotated_indices = np.where(is_annotated)[0]
    unannotated_indices = np.where(~is_annotated)[0]

    total_annotated_tokens = float(est_tokens[annotated_indices].sum())
    annotation_ratio = total_annotated_tokens / total_available

    print(f"\nAnnotation marking (threshold={annotation_threshold}):")
    print(f"  High-score (>={annotation_threshold}): {len(high_indices):,} rows, "
          f"{_fmt_tokens(high_score_tokens)} tokens")
    print(f"  Sampled low-score:  {len(sampled_low):,} rows, "
          f"{_fmt_tokens(sampled_low_tokens)} tokens")
    print(f"  Total annotated:    {len(annotated_indices):,} rows, "
          f"{_fmt_tokens(total_annotated_tokens)} tokens "
          f"({100 * annotation_ratio:.2f}%)")
    print(f"  Total unannotated:  {len(unannotated_indices):,} rows")

    # ── Phase 2: budget filling (preserving annotation ratio) ────────
    if total_available < target_tokens:
        scale = total_available / target_tokens
        print(f"\nWARNING: available tokens ({_fmt_tokens(total_available)}) < "
              f"target ({_fmt_tokens(target_tokens)}). "
              f"Scaling budgets by {scale:.4f}.")
        target_tokens = total_available

    ann_budget = target_tokens * annotation_ratio
    unann_budget = target_tokens * (1.0 - annotation_ratio)

    print(f"\nSampling budgets:")
    print(f"  Annotated:   {_fmt_tokens(ann_budget)} ({100 * annotation_ratio:.2f}%)")
    print(f"  Unannotated: {_fmt_tokens(unann_budget)} ({100 * (1 - annotation_ratio):.2f}%)")
    print(f"  Total:       {_fmt_tokens(target_tokens)}")

    ann_sorted = annotated_indices[np.argsort(priorities[annotated_indices])]
    unann_sorted = unannotated_indices[np.argsort(priorities[unannotated_indices])]

    ann_selected, ann_actual = _fill_budget(ann_sorted, est_tokens, ann_budget)
    unann_selected, unann_actual = _fill_budget(unann_sorted, est_tokens, unann_budget)

    # Build output tables
    ann_table = index.take(pa.array(ann_selected, type=pa.int64()))
    ann_table = ann_table.append_column(
        "has_annotation", pa.array([True] * len(ann_selected), type=pa.bool_()),
    )
    unann_table = index.take(pa.array(unann_selected, type=pa.int64()))
    unann_table = unann_table.append_column(
        "has_annotation", pa.array([False] * len(unann_selected), type=pa.bool_()),
    )

    total_selected = len(ann_selected) + len(unann_selected)
    total_selected_tokens = ann_actual + unann_actual

    print(f"\nSampling result:")
    print(f"  Annotated:   {len(ann_selected):,} rows, {_fmt_tokens(ann_actual)} tokens")
    print(f"  Unannotated: {len(unann_selected):,} rows, {_fmt_tokens(unann_actual)} tokens")
    print(f"  Total:       {total_selected:,} rows, {_fmt_tokens(total_selected_tokens)} tokens")

    stats = {
        "target_tokens": target_tokens,
        "total_available_tokens": total_available,
        "annotation_threshold": annotation_threshold,
        "annotation_ratio": annotation_ratio,
        "high_score_rows": len(high_indices),
        "high_score_tokens": high_score_tokens,
        "sampled_good_rows": len(sampled_low),
        "sampled_good_tokens": sampled_low_tokens,
        "annotated": {
            "budget": ann_budget,
            "selected_rows": len(ann_selected),
            "selected_tokens": ann_actual,
        },
        "unannotated": {
            "budget": unann_budget,
            "selected_rows": len(unann_selected),
            "selected_tokens": unann_actual,
        },
        "selected_rows": total_selected,
        "selected_tokens": total_selected_tokens,
    }

    return ann_table, unann_table, stats


# ---------------------------------------------------------------------------
# Legacy stratified sampling (activated by --bad-fraction + --good-fraction)
# ---------------------------------------------------------------------------


def sample_strata(
    index: pa.Table,
    target_tokens: float,
    bad_fraction: float,
    good_fraction: float,
    seed: int,
) -> tuple[pa.Table, dict]:
    """Legacy: sample three independent strata and assign has_annotation flags.

    Kept for backward compatibility.  Activated when both --bad-fraction and
    --good-fraction are explicitly provided on the command line.

    Returns (selected_table, stats_dict) where selected_table is the index
    filtered to selected rows with an added ``has_annotation`` boolean column.
    """
    total_available = pc.sum(index.column("est_tokens")).as_py()
    unmarked_fraction = 1.0 - bad_fraction - good_fraction

    if total_available < target_tokens:
        scale = total_available / target_tokens
        print(f"\nWARNING: available tokens ({_fmt_tokens(total_available)}) < "
              f"target ({_fmt_tokens(target_tokens)}). "
              f"Scaling budgets by {scale:.4f} to maintain proportions.")
        target_tokens = total_available

    bad_budget = target_tokens * bad_fraction
    good_budget = target_tokens * good_fraction
    unmarked_budget = target_tokens * unmarked_fraction

    print(f"\nSampling budgets (legacy mode):")
    print(f"  Bad (4-5) annotated:  {_fmt_tokens(bad_budget)}")
    print(f"  Good (0-3) annotated: {_fmt_tokens(good_budget)}")
    print(f"  Unmarked:             {_fmt_tokens(unmarked_budget)}")
    print(f"  Total target:         {_fmt_tokens(target_tokens)}")

    scores = index.column("safety_score")
    bad_mask = pc.or_(pc.equal(scores, 4), pc.equal(scores, 5))
    good_mask = pc.invert(bad_mask)

    bad_indices = pc.indices_nonzero(bad_mask).to_numpy().copy()
    good_indices = pc.indices_nonzero(good_mask).to_numpy().copy()

    rng = np.random.default_rng(seed)
    priorities = rng.random(len(index))

    bad_indices = bad_indices[np.argsort(priorities[bad_indices])]
    good_indices = good_indices[np.argsort(priorities[good_indices])]

    est_tokens = index.column("est_tokens").to_numpy(zero_copy_only=False)

    bad_selected, bad_actual = _fill_budget(bad_indices, est_tokens, bad_budget)
    if bad_actual < bad_budget:
        print(f"  WARNING: bad pool shortfall — got {_fmt_tokens(bad_actual)} "
              f"of {_fmt_tokens(bad_budget)} budget")

    good_selected, good_actual = _fill_budget(good_indices, est_tokens, good_budget)
    if good_actual < good_budget:
        print(f"  WARNING: good pool shortfall — got {_fmt_tokens(good_actual)} "
              f"of {_fmt_tokens(good_budget)} budget")

    annotated_mask = np.zeros(len(index), dtype=bool)
    annotated_mask[bad_selected] = True
    annotated_mask[good_selected] = True
    remainder_indices = np.where(~annotated_mask)[0]
    remainder_indices = remainder_indices[np.argsort(priorities[remainder_indices])]

    unmarked_selected, unmarked_actual = _fill_budget(
        remainder_indices, est_tokens, unmarked_budget,
    )
    if unmarked_actual < unmarked_budget:
        print(f"  WARNING: unmarked pool shortfall — got {_fmt_tokens(unmarked_actual)} "
              f"of {_fmt_tokens(unmarked_budget)} budget")

    all_selected = np.concatenate([bad_selected, good_selected, unmarked_selected])
    all_annotated = np.concatenate([
        np.ones(len(bad_selected), dtype=bool),
        np.ones(len(good_selected), dtype=bool),
        np.zeros(len(unmarked_selected), dtype=bool),
    ])

    sort_order = np.argsort(all_selected)
    selected_indices = all_selected[sort_order]
    annotation_flags = all_annotated[sort_order]

    selected_table = index.take(pa.array(selected_indices, type=pa.int64()))
    selected_table = selected_table.append_column(
        "has_annotation", pa.array(annotation_flags.tolist(), type=pa.bool_()),
    )

    total_selected_tokens = pc.sum(selected_table.column("est_tokens")).as_py()
    n_annotated = int(annotation_flags.sum())
    annotated_tokens = float(est_tokens[bad_selected].sum() + est_tokens[good_selected].sum())

    print(f"\nSampling result:")
    print(f"  Selected rows:     {len(selected_table):,}")
    print(f"  Selected tokens:   {_fmt_tokens(total_selected_tokens)}")
    print(f"  Annotated rows:    {n_annotated:,} ({100 * n_annotated / len(selected_table):.2f}%)")
    print(f"  Annotated tokens:  {_fmt_tokens(annotated_tokens)} "
          f"({100 * annotated_tokens / total_selected_tokens:.2f}%)")
    print(f"    Bad (4-5):       {len(bad_selected):,} rows, {_fmt_tokens(bad_actual)} tokens")
    print(f"    Good (0-3):      {len(good_selected):,} rows, {_fmt_tokens(good_actual)} tokens")
    print(f"  Unmarked rows:     {len(unmarked_selected):,}")

    stats = {
        "target_tokens": target_tokens,
        "total_available_tokens": total_available,
        "selected_rows": len(selected_table),
        "selected_tokens": total_selected_tokens,
        "bad_annotated_rows": len(bad_selected),
        "bad_annotated_tokens": bad_actual,
        "good_annotated_rows": len(good_selected),
        "good_annotated_tokens": good_actual,
        "unmarked_rows": len(unmarked_selected),
        "unmarked_tokens": unmarked_actual,
    }

    return selected_table, stats


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _fill_budget(
    indices: np.ndarray,
    est_tokens: np.ndarray,
    budget: float,
) -> tuple[np.ndarray, float]:
    """Greedily select indices until token budget is met.

    Returns (selected_indices as numpy array, actual_tokens).
    """
    if len(indices) == 0:
        return indices, 0.0
    tokens_for_indices = est_tokens[indices]
    cumsum = np.cumsum(tokens_for_indices)
    n_take = np.searchsorted(cumsum, budget, side="left") + 1
    n_take = min(n_take, len(indices))
    selected = indices[:n_take]
    actual = float(cumsum[n_take - 1]) if n_take > 0 else 0.0
    return selected, actual


def _write_partition(
    source_dir: Path,
    selected: pa.Table,
    output_subdir: Path,
    id_column: str,
    rows_per_file: int,
    has_annotation_value: bool,
) -> None:
    """Write a partition of selected rows to an output directory.

    Re-reads source files for selected rows, adds ``has_annotation`` column
    with the given constant value, and writes buffered parquet output.
    """
    source_files = sorted(source_dir.glob("part_*.parquet"))

    id_col = selected.column("id").to_pylist()
    file_idx_col = selected.column("file_idx").to_pylist()

    files_needed: dict[int, set[str]] = {}
    for doc_id, fidx in zip(id_col, file_idx_col):
        files_needed.setdefault(fidx, set()).add(doc_id)

    output_subdir.mkdir(parents=True, exist_ok=True)

    buffer_rows: list[pa.Table] = []
    buffer_count = 0
    part_idx = 0

    def _flush():
        nonlocal buffer_rows, buffer_count, part_idx
        if not buffer_rows:
            return
        combined = pa.concat_tables(buffer_rows)
        out_path = output_subdir / f"part_{part_idx:05d}.parquet"
        pq.write_table(combined, str(out_path))
        part_idx += 1
        buffer_rows = []
        buffer_count = 0

    selected_id_set = set(id_col)
    label = "annotated" if has_annotation_value else "unannotated"
    print(f"\n  Writing {label} to {output_subdir}...")
    print(f"    Source files to read: {len(files_needed)} of {len(source_files)}")

    for fidx in tqdm(sorted(files_needed.keys()), desc=f"  {label}"):
        target_ids = files_needed[fidx]
        fpath = source_files[fidx]
        table = pq.read_table(fpath)

        ids = table.column(id_column)
        mask = pc.is_in(ids, value_set=pa.array(list(target_ids), type=ids.type))
        filtered = table.filter(mask)

        filtered = filtered.append_column(
            "has_annotation",
            pa.array([has_annotation_value] * len(filtered), type=pa.bool_()),
        )

        buffer_rows.append(filtered)
        buffer_count += len(filtered)

        if buffer_count >= rows_per_file:
            _flush()

    _flush()
    print(f"    Wrote {part_idx} files")


def write_output(
    source_dir: Path,
    selected: pa.Table,
    output_dir: Path,
    id_column: str,
    rows_per_file: int,
) -> None:
    """Legacy: write single output directory with mixed annotated/unannotated rows."""
    source_files = sorted(source_dir.glob("part_*.parquet"))

    id_col = selected.column("id").to_pylist()
    file_idx_col = selected.column("file_idx").to_pylist()
    annotation_col = selected.column("has_annotation").to_pylist()

    id_to_annotation: dict[str, bool] = dict(zip(id_col, annotation_col))

    files_needed: dict[int, set[str]] = {}
    for doc_id, fidx in zip(id_col, file_idx_col):
        files_needed.setdefault(fidx, set()).add(doc_id)

    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_rows: list[pa.Table] = []
    buffer_count = 0
    part_idx = 0

    def _flush():
        nonlocal buffer_rows, buffer_count, part_idx
        if not buffer_rows:
            return
        combined = pa.concat_tables(buffer_rows)
        out_path = output_dir / f"part_{part_idx:05d}.parquet"
        pq.write_table(combined, str(out_path))
        part_idx += 1
        buffer_rows = []
        buffer_count = 0

    print(f"\nWriting output to {output_dir}...")
    print(f"  Source files to read: {len(files_needed)} of {len(source_files)}")

    for fidx in tqdm(sorted(files_needed.keys()), desc="Writing output"):
        target_ids = files_needed[fidx]
        fpath = source_files[fidx]
        table = pq.read_table(fpath)

        ids = table.column(id_column)
        mask = pc.is_in(ids, value_set=pa.array(list(target_ids), type=ids.type))
        filtered = table.filter(mask)

        filtered_ids = filtered.column(id_column).to_pylist()
        filtered = filtered.append_column(
            "has_annotation",
            pa.array([id_to_annotation[doc_id] for doc_id in filtered_ids], type=pa.bool_()),
        )

        buffer_rows.append(filtered)
        buffer_count += len(filtered)

        if buffer_count >= rows_per_file:
            _flush()

    _flush()
    print(f"  Wrote {part_idx} output files")


def _fmt_tokens(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,.0f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    scratch = os.environ.get(
        "SCRATCH",
        f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
    )

    p = argparse.ArgumentParser(
        description="Annotation-based subsampling with two output datasets.",
    )
    p.add_argument("--source-dir", type=str, required=True,
                   help="Dir with part_*.parquet source files (must include safety_score column)")
    p.add_argument("--output-dir", type=str, default=f"{scratch}/subsampled",
                   help="Output directory (default: $SCRATCH/subsampled)")
    p.add_argument("--target-tokens", type=float, default=500_000_000_000,
                   help="Total token budget (default: 500B)")
    p.add_argument("--annotation-threshold", type=int, default=3,
                   help="Safety scores >= this are unconditionally annotated (default: 3)")
    p.add_argument("--bad-fraction", type=float, default=None,
                   help="Legacy: fraction for bad (4-5) stratum. Must set with --good-fraction.")
    p.add_argument("--good-fraction", type=float, default=None,
                   help="Legacy: fraction for good (0-3) stratum. Must set with --bad-fraction.")
    p.add_argument("--chars-per-token", type=float, default=4.068,
                   help="Chars-per-token ratio (default: 4.068)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--id-column", type=str, default="id",
                   help="ID column name (default: id)")
    p.add_argument("--text-column", type=str, default="text",
                   help="Text column name (default: text)")
    p.add_argument("--rows-per-file", type=int, default=500_000,
                   help="Max rows per output parquet file (default: 500000)")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    assert source_dir.exists(), f"Source dir not found: {source_dir}"

    # Legacy mode: both fractions must be set together
    legacy_mode = args.bad_fraction is not None or args.good_fraction is not None
    if legacy_mode:
        assert args.bad_fraction is not None and args.good_fraction is not None, (
            "--bad-fraction and --good-fraction must both be set for legacy mode"
        )

    if args.overwrite and output_dir.exists():
        print(f"--overwrite: removing {output_dir}")
        shutil.rmtree(output_dir)
    assert not output_dir.exists(), (
        f"Output dir already exists: {output_dir} (use --overwrite to replace)"
    )

    t_start = time.time()

    # Pass 1: scan source files and build index
    index = scan_source(
        source_dir, args.id_column, args.text_column, args.chars_per_token,
    )

    if legacy_mode:
        # Legacy: three independent strata, single output directory
        print("\n*** Legacy stratified mode ***")
        selected, stats = sample_strata(
            index, args.target_tokens,
            args.bad_fraction, args.good_fraction, args.seed,
        )
        write_output(
            source_dir, selected, output_dir, args.id_column, args.rows_per_file,
        )
        metadata = {
            "source_dir": str(source_dir),
            "target_tokens": args.target_tokens,
            "bad_fraction": args.bad_fraction,
            "good_fraction": args.good_fraction,
            "chars_per_token": args.chars_per_token,
            "seed": args.seed,
            "id_column": args.id_column,
            "text_column": args.text_column,
            "rows_per_file": args.rows_per_file,
            **stats,
        }
    else:
        # Default: annotation-based split, two output directories
        ann_table, unann_table, stats = mark_and_sample(
            index, args.target_tokens, args.seed, args.annotation_threshold,
        )
        print(f"\nWriting output...")
        _write_partition(
            source_dir, ann_table, output_dir / "annotated",
            args.id_column, args.rows_per_file, has_annotation_value=True,
        )
        _write_partition(
            source_dir, unann_table, output_dir / "unannotated",
            args.id_column, args.rows_per_file, has_annotation_value=False,
        )
        metadata = {
            "source_dir": str(source_dir),
            "target_tokens": args.target_tokens,
            "annotation_threshold": args.annotation_threshold,
            "chars_per_token": args.chars_per_token,
            "seed": args.seed,
            "id_column": args.id_column,
            "text_column": args.text_column,
            "rows_per_file": args.rows_per_file,
            **stats,
        }

    elapsed = time.time() - t_start
    metadata["elapsed_s"] = round(elapsed, 1)
    metadata["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"  Output:    {output_dir}")
    print(f"  Tokens:    {_fmt_tokens(stats['selected_tokens'])}")
    print(f"  Rows:      {stats['selected_rows']:,}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
