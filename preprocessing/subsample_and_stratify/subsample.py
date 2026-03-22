"""Stratified subsampling: select a token budget with annotation marking.

Produces a subset of the source dataset where a controlled fraction of tokens
is marked for human annotation (``has_annotation=True``), split between "bad"
(safety_score 4-5) and "good" (safety_score 0-3) strata.  The remaining tokens
fill out the dataset unmarked.

Two-pass algorithm:
  1. Scan source files & build a lightweight in-memory index.
  2. Sample strata, re-read selected source rows, and write output.

Usage::

    python -m preprocessing.subsample_and_stratify.subsample \
        --source-dir $SCRATCH/dolma3_mix-1T \
        --annotations-dir $SCRATCH/safety_annotations/all \
        --target-tokens 500_000_000_000

    # Smaller test run
    python -m preprocessing.subsample_and_stratify.subsample \
        --source-dir $SCRATCH/dolma3_mix-10m \
        --annotations-dir $SCRATCH/safety_annotations/test \
        --target-tokens 1_000_000 --output-dir $SCRATCH/subsampled_test
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


def load_annotations(annotations_dir: Path) -> dict[str, int]:
    """Load all annotation shards into a dict mapping id → safety_score."""
    files = sorted(annotations_dir.glob("shard_*.parquet"))
    assert files, f"No shard_*.parquet files found in {annotations_dir}"
    print(f"Loading annotations from {len(files)} shard files...")
    annotations: dict[str, int] = {}
    for f in tqdm(files, desc="Annotation shards"):
        table = pq.read_table(f, columns=["id", "safety_score"])
        ids = table.column("id").to_pylist()
        scores = table.column("safety_score").to_pylist()
        for doc_id, score in zip(ids, scores):
            annotations[doc_id] = score
    print(f"Loaded {len(annotations):,} annotations")
    return annotations


def scan_source(
    source_dir: Path,
    annotations: dict[str, int],
    id_column: str,
    text_column: str,
    chars_per_token: float,
) -> pa.Table:
    """Scan source parquet files and build a lightweight index table.

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

    n_missing = 0
    for file_idx, fpath in enumerate(tqdm(source_files, desc="Source files")):
        table = pq.read_table(fpath, columns=[id_column, text_column])
        ids = table.column(id_column)
        text = table.column(text_column)

        # est_tokens = utf8_length(text) / chars_per_token
        lengths = pc.utf8_length(text).cast(pa.float32())
        est_tokens = pc.divide(lengths, pa.scalar(chars_per_token, pa.float32()))

        # Look up safety scores (batch convert to avoid per-row .as_py())
        id_list = ids.to_pylist()
        scores = []
        for doc_id in id_list:
            score = annotations.get(doc_id)
            if score is None:
                n_missing += 1
            scores.append(score if score is not None else -1)

        all_ids.append(ids)
        all_tokens.append(est_tokens)
        all_scores.append(pa.array(scores, type=pa.int8()))
        all_file_idx.append(pa.array([file_idx] * len(table), type=pa.int32()))

    assert n_missing == 0, (
        f"{n_missing:,} source rows have no annotation — all rows must be annotated"
    )

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


def sample_strata(
    index: pa.Table,
    target_tokens: float,
    bad_fraction: float,
    good_fraction: float,
    seed: int,
) -> tuple[pa.Table, dict]:
    """Sample strata from the index and assign has_annotation flags.

    Returns (selected_table, stats_dict) where selected_table is the index
    filtered to selected rows with an added ``has_annotation`` boolean column.
    """
    total_available = pc.sum(index.column("est_tokens")).as_py()
    unmarked_fraction = 1.0 - bad_fraction - good_fraction

    # Scale down if we don't have enough tokens
    if total_available < target_tokens:
        scale = total_available / target_tokens
        print(f"\nWARNING: available tokens ({_fmt_tokens(total_available)}) < "
              f"target ({_fmt_tokens(target_tokens)}). "
              f"Scaling budgets by {scale:.4f} to maintain proportions.")
        target_tokens = total_available

    bad_budget = target_tokens * bad_fraction
    good_budget = target_tokens * good_fraction
    unmarked_budget = target_tokens * unmarked_fraction

    print(f"\nSampling budgets:")
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
    rng.shuffle(bad_indices)
    rng.shuffle(good_indices)

    est_tokens = index.column("est_tokens").to_numpy(zero_copy_only=False)

    # Sample bad stratum
    bad_selected, bad_actual = _fill_budget(bad_indices, est_tokens, bad_budget)
    if bad_actual < bad_budget:
        print(f"  WARNING: bad pool shortfall — got {_fmt_tokens(bad_actual)} "
              f"of {_fmt_tokens(bad_budget)} budget")

    # Sample good stratum
    good_selected, good_actual = _fill_budget(good_indices, est_tokens, good_budget)
    if good_actual < good_budget:
        print(f"  WARNING: good pool shortfall — got {_fmt_tokens(good_actual)} "
              f"of {_fmt_tokens(good_budget)} budget")

    # Remaining rows for unmarked stratum (numpy mask avoids Python iteration over all indices)
    annotated_mask = np.zeros(len(index), dtype=bool)
    annotated_mask[bad_selected] = True
    annotated_mask[good_selected] = True
    remainder_indices = np.where(~annotated_mask)[0]
    rng.shuffle(remainder_indices)

    unmarked_selected, unmarked_actual = _fill_budget(
        remainder_indices, est_tokens, unmarked_budget,
    )
    if unmarked_actual < unmarked_budget:
        print(f"  WARNING: unmarked pool shortfall — got {_fmt_tokens(unmarked_actual)} "
              f"of {_fmt_tokens(unmarked_budget)} budget")

    # Build result using numpy for efficiency
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


def _fill_budget(
    indices: np.ndarray,
    est_tokens: np.ndarray,
    budget: float,
) -> tuple[np.ndarray, float]:
    """Greedily select indices until token budget is met.

    Returns (selected_indices as numpy array, actual_tokens).
    """
    tokens_for_indices = est_tokens[indices]
    cumsum = np.cumsum(tokens_for_indices)
    n_take = np.searchsorted(cumsum, budget, side="left") + 1
    n_take = min(n_take, len(indices))
    selected = indices[:n_take]
    actual = float(cumsum[n_take - 1]) if n_take > 0 else 0.0
    return selected, actual


def write_output(
    source_dir: Path,
    selected: pa.Table,
    output_dir: Path,
    id_column: str,
    rows_per_file: int,
) -> None:
    """Re-read source files for selected rows and write output parquet files.

    Groups selected rows by file_idx to minimize source file reads.
    Adds ``safety_score`` (int8) and ``has_annotation`` (bool) columns
    from the selected table's metadata.
    """
    source_files = sorted(source_dir.glob("part_*.parquet"))

    # Build lookups from the selected table (already carries scores + annotations)
    id_col = selected.column("id").to_pylist()
    file_idx_col = selected.column("file_idx").to_pylist()
    annotation_col = selected.column("has_annotation").to_pylist()
    score_col = selected.column("safety_score").to_pylist()

    id_to_meta: dict[str, tuple[int, bool]] = {}
    for doc_id, score, has_ann in zip(id_col, score_col, annotation_col):
        id_to_meta[doc_id] = (score, has_ann)

    # Group by file_idx
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

        # Filter to selected IDs
        ids = table.column(id_column)
        mask = pc.is_in(ids, value_set=pa.array(list(target_ids), type=ids.type))
        filtered = table.filter(mask)

        # Add safety_score and has_annotation columns from selected table metadata
        filtered_ids = filtered.column(id_column).to_pylist()
        meta = [id_to_meta[doc_id] for doc_id in filtered_ids]
        filtered = filtered.append_column(
            "safety_score", pa.array([m[0] for m in meta], type=pa.int8()),
        )
        filtered = filtered.append_column(
            "has_annotation", pa.array([m[1] for m in meta], type=pa.bool_()),
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stratified subsampling."""
    scratch = os.environ.get(
        "SCRATCH",
        f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
    )

    p = argparse.ArgumentParser(
        description="Stratified subsampling with annotation marking.",
    )
    p.add_argument("--source-dir", type=str, required=True,
                   help="Dir with part_*.parquet source files")
    p.add_argument("--annotations-dir", type=str, required=True,
                   help="Dir with shard_*.parquet annotation files")
    p.add_argument("--output-dir", type=str, default=f"{scratch}/subsampled",
                   help="Output directory (default: $SCRATCH/subsampled)")
    p.add_argument("--target-tokens", type=float, default=500_000_000_000,
                   help="Total token budget (default: 500B)")
    p.add_argument("--bad-fraction", type=float, default=0.025,
                   help="Fraction of tokens for bad (4-5) annotated samples (default: 0.025)")
    p.add_argument("--good-fraction", type=float, default=0.025,
                   help="Fraction of tokens for good (0-3) annotated samples (default: 0.025)")
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
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)

    assert source_dir.exists(), f"Source dir not found: {source_dir}"
    assert annotations_dir.exists(), f"Annotations dir not found: {annotations_dir}"

    if args.overwrite and output_dir.exists():
        print(f"--overwrite: removing {output_dir}")
        shutil.rmtree(output_dir)
    assert not output_dir.exists(), (
        f"Output dir already exists: {output_dir} (use --overwrite to replace)"
    )

    t_start = time.time()

    # Pass 1: load annotations and scan source
    annotations = load_annotations(annotations_dir)
    index = scan_source(
        source_dir, annotations, args.id_column, args.text_column, args.chars_per_token,
    )

    # Sampling
    selected, stats = sample_strata(
        index, args.target_tokens, args.bad_fraction, args.good_fraction, args.seed,
    )

    # Pass 2: write output
    write_output(
        source_dir, selected, output_dir, args.id_column, args.rows_per_file,
    )

    # Write metadata
    elapsed = time.time() - t_start
    metadata = {
        "source_dir": str(source_dir),
        "annotations_dir": str(annotations_dir),
        "target_tokens": args.target_tokens,
        "bad_fraction": args.bad_fraction,
        "good_fraction": args.good_fraction,
        "chars_per_token": args.chars_per_token,
        "seed": args.seed,
        "id_column": args.id_column,
        "text_column": args.text_column,
        "rows_per_file": args.rows_per_file,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **stats,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"  Output:    {output_dir}")
    print(f"  Tokens:    {_fmt_tokens(stats['selected_tokens'])}")
    print(f"  Rows:      {stats['selected_rows']:,}")
    print(f"  Annotated: {stats['bad_annotated_rows'] + stats['good_annotated_rows']:,} rows "
          f"({_fmt_tokens(stats['bad_annotated_tokens'] + stats['good_annotated_tokens'])} tokens)")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
