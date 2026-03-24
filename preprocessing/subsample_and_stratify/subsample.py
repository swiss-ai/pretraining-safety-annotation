"""Annotation-based subsampling with two output datasets.

Two-pass, per-file-independent design that uses ~200MB of RAM regardless
of dataset size.

Pass 1 (scan): collect per-file statistics (token counts by score bucket).
Pass 2 (write): for each file independently, select rows using a per-file
    deterministic RNG and per-file token budget, write to two output dirs.

Usage::

    python -m preprocessing.subsample_and_stratify.subsample \
        --source-dir $SCRATCH/dolma3_mix-1T_annotated \
        --target-tokens 1_000_000_000_000

    # Custom threshold (annotate scores >= 4 only)
    python -m preprocessing.subsample_and_stratify.subsample \
        --source-dir $SCRATCH/dolma3_mix-1T_annotated \
        --annotation-threshold 4
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

MAX_EST_TOKENS = 2048
_DEFAULT_WORKERS = 16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileStats:
    file_index: int
    file_path: str
    total_rows: int
    total_tokens: float
    above_rows: int
    above_tokens: float
    below_rows: int
    below_tokens: float


# ---------------------------------------------------------------------------
# Pass 1: per-file statistics
# ---------------------------------------------------------------------------


def _collect_one_file(
    file_index: int,
    file_path: str,
    text_column: str,
    chars_per_token: float,
    threshold: int,
) -> FileStats:
    """Read one parquet file and return lightweight statistics."""
    table = pq.read_table(file_path, columns=[text_column, "safety_score"])
    lengths = pc.utf8_length(table.column(text_column)).cast(pa.float64())
    est_tokens = pc.divide(lengths, pa.scalar(chars_per_token, pa.float64()))
    est_tokens = pc.min_element_wise(est_tokens, pa.scalar(float(MAX_EST_TOKENS), pa.float64()))

    scores = table.column("safety_score")
    above_mask = pc.greater_equal(scores, threshold)

    above_tokens_arr = pc.filter(est_tokens, above_mask)
    below_tokens_arr = pc.filter(est_tokens, pc.invert(above_mask))

    return FileStats(
        file_index=file_index,
        file_path=file_path,
        total_rows=len(table),
        total_tokens=pc.sum(est_tokens).as_py(),
        above_rows=int(pc.sum(above_mask.cast(pa.int64())).as_py()),
        above_tokens=pc.sum(above_tokens_arr).as_py() or 0.0,
        below_rows=len(table) - int(pc.sum(above_mask.cast(pa.int64())).as_py()),
        below_tokens=pc.sum(below_tokens_arr).as_py() or 0.0,
    )


def scan_files(
    source_dir: Path,
    text_column: str,
    chars_per_token: float,
    threshold: int,
    workers: int,
) -> list[FileStats]:
    """Pass 1: scan all files in parallel, return per-file statistics."""
    files = sorted(source_dir.glob("part_*.parquet"))
    assert files, f"No part_*.parquet files found in {source_dir}"
    print(f"\nPass 1: scanning {len(files)} files ({workers} workers)...")

    stats: list[FileStats] = [None] * len(files)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _collect_one_file, i, str(f),
                text_column, chars_per_token, threshold,
            ): i
            for i, f in enumerate(files)
        }
        with tqdm(total=len(files), desc="Scan") as pbar:
            for future in as_completed(futures):
                s = future.result()
                stats[s.file_index] = s
                pbar.update(1)

    return stats


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------


def compute_budgets(
    stats: list[FileStats],
    target_tokens: float | None,
) -> tuple[dict[int, float], dict[int, float], dict]:
    """Compute per-file token budgets for annotated and unannotated splits.

    Returns (ann_budgets, unann_budgets, summary) where each budget dict
    maps file_index → token budget for that split.
    """
    global_above = sum(s.above_tokens for s in stats)
    global_below = sum(s.below_tokens for s in stats)
    global_total = global_above + global_below
    global_rows = sum(s.total_rows for s in stats)

    # Annotation marking: above-threshold tokens + matched below-threshold tokens
    target_below_sample = min(global_above, global_below)
    total_annotated = global_above + target_below_sample
    total_unannotated = global_total - total_annotated
    annotation_ratio = total_annotated / global_total if global_total > 0 else 0

    # Apply token budget
    if target_tokens is not None and target_tokens < global_total:
        scale = target_tokens / global_total
    else:
        scale = 1.0
        target_tokens = global_total

    # Per-file below-threshold sampling budget (proportional to file's below tokens)
    below_budgets: dict[int, float] = {}
    for s in stats:
        if global_below > 0:
            below_budgets[s.file_index] = target_below_sample * scale * (s.below_tokens / global_below)
        else:
            below_budgets[s.file_index] = 0.0

    # Per-file annotated budget = file's above tokens * scale + file's below sample budget
    ann_budgets: dict[int, float] = {}
    unann_budgets: dict[int, float] = {}
    for s in stats:
        ann_budgets[s.file_index] = s.above_tokens * scale + below_budgets[s.file_index]
        unann_budgets[s.file_index] = (s.below_tokens * scale) - below_budgets[s.file_index]

    summary = {
        "global_rows": global_rows,
        "global_tokens": global_total,
        "global_above_tokens": global_above,
        "global_below_tokens": global_below,
        "target_below_sample": target_below_sample * scale,
        "annotation_ratio": annotation_ratio,
        "target_tokens": target_tokens,
        "scale": scale,
    }

    print(f"\n  Total rows:        {global_rows:,}")
    print(f"  Total tokens:      {_fmt(global_total)}")
    print(f"  Above threshold:   {_fmt(global_above)} ({100*global_above/global_total:.2f}%)")
    print(f"  Annotation ratio:  {100*annotation_ratio:.2f}%")
    print(f"  Target tokens:     {_fmt(target_tokens)}")
    print(f"  Scale:             {scale:.4f}")

    return ann_budgets, unann_budgets, summary


# ---------------------------------------------------------------------------
# Pass 2: per-file selection and writing
# ---------------------------------------------------------------------------


def _process_one_file(
    file_path: str,
    file_index: int,
    ann_budget: float,
    unann_budget: float,
    text_column: str,
    chars_per_token: float,
    threshold: int,
    seed: int,
    ann_dir: str,
    unann_dir: str,
) -> dict:
    """Read one file, select rows, write to both output directories."""
    table = pq.read_table(file_path)
    n = len(table)

    # Recompute est_tokens (cheap, avoids storing globally)
    lengths = pc.utf8_length(table.column(text_column)).cast(pa.float64())
    est_tokens = pc.divide(lengths, pa.scalar(chars_per_token, pa.float64()))
    est_tokens = pc.min_element_wise(est_tokens, pa.scalar(float(MAX_EST_TOKENS), pa.float64()))
    est_np = est_tokens.to_numpy(zero_copy_only=False)

    scores = table.column("safety_score")
    above_mask = pc.greater_equal(scores, threshold).to_numpy(zero_copy_only=False)

    # Per-file deterministic RNG
    rng = np.random.default_rng(seed=(seed, file_index))

    # Select above-threshold rows (all annotated, but budget-limited)
    above_indices = np.where(above_mask)[0]
    below_indices = np.where(~above_mask)[0]

    # Shuffle both pools
    rng.shuffle(above_indices)
    rng.shuffle(below_indices)

    # Fill annotated budget: first above-threshold rows, then sampled below
    ann_selected_above = _fill_budget(above_indices, est_np, ann_budget)
    remaining_ann_budget = ann_budget - est_np[ann_selected_above].sum() if len(ann_selected_above) > 0 else ann_budget

    ann_selected_below = _fill_budget(below_indices, est_np, remaining_ann_budget) if remaining_ann_budget > 0 else np.array([], dtype=np.intp)

    # Remaining below-threshold rows go to unannotated pool
    ann_below_set = set(ann_selected_below.tolist())
    unann_candidates = np.array([i for i in below_indices if i not in ann_below_set], dtype=np.intp)
    # unann_candidates are already in shuffled order (from below_indices shuffle)
    unann_selected = _fill_budget(unann_candidates, est_np, unann_budget)

    # Build output masks
    is_annotated = np.zeros(n, dtype=bool)
    is_annotated[ann_selected_above] = True
    is_annotated[ann_selected_below] = True

    is_selected = np.zeros(n, dtype=bool)
    is_selected[ann_selected_above] = True
    is_selected[ann_selected_below] = True
    is_selected[unann_selected] = True

    is_bad = above_mask

    # Add columns
    table = table.append_column("has_annotation", pa.array(is_annotated))
    table = table.append_column("is_bad", pa.array(is_bad))

    # Write annotated split
    ann_mask = is_annotated & is_selected
    ann_rows = int(ann_mask.sum())
    if ann_rows > 0:
        ann_table = table.filter(pa.array(ann_mask))
        pq.write_table(ann_table, f"{ann_dir}/{Path(file_path).name}")

    # Write unannotated split
    unann_mask = ~is_annotated & is_selected
    unann_rows = int(unann_mask.sum())
    if unann_rows > 0:
        unann_table = table.filter(pa.array(unann_mask))
        pq.write_table(unann_table, f"{unann_dir}/{Path(file_path).name}")

    return {
        "file_index": file_index,
        "ann_rows": ann_rows,
        "unann_rows": unann_rows,
        "ann_tokens": float(est_np[ann_mask].sum()),
        "unann_tokens": float(est_np[unann_mask].sum()),
    }


def _fill_budget(indices: np.ndarray, est_tokens: np.ndarray, budget: float) -> np.ndarray:
    """Greedily select indices until token budget is met."""
    if len(indices) == 0 or budget <= 0:
        return np.array([], dtype=np.intp)
    tokens = est_tokens[indices]
    cumsum = np.cumsum(tokens)
    n_take = np.searchsorted(cumsum, budget, side="left") + 1
    n_take = min(n_take, len(indices))
    return indices[:n_take]


def write_files(
    stats: list[FileStats],
    ann_budgets: dict[int, float],
    unann_budgets: dict[int, float],
    text_column: str,
    chars_per_token: float,
    threshold: int,
    seed: int,
    output_dir: Path,
    workers: int,
) -> list[dict]:
    """Pass 2: process all files in parallel, write two output directories."""
    ann_dir = output_dir / "annotated"
    unann_dir = output_dir / "unannotated"
    ann_dir.mkdir(parents=True, exist_ok=True)
    unann_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPass 2: writing to {output_dir} ({workers} workers)...")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_one_file,
                s.file_path, s.file_index,
                ann_budgets[s.file_index], unann_budgets[s.file_index],
                text_column, chars_per_token, threshold, seed,
                str(ann_dir), str(unann_dir),
            ): s.file_index
            for s in stats
        }
        with tqdm(total=len(stats), desc="Write") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fmt(n: float) -> str:
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
    scratch = os.environ.get(
        "SCRATCH", f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}"
    )
    p = argparse.ArgumentParser(
        description="Annotation-based subsampling with two output datasets.",
    )
    p.add_argument("--source-dir", type=str, required=True,
                   help="Dir with part_*.parquet source files")
    p.add_argument("--output-dir", type=str, default=f"{scratch}/subsampled",
                   help="Output directory")
    p.add_argument("--target-tokens", type=float, default=None,
                   help="Total token budget (default: use all)")
    p.add_argument("--annotation-threshold", type=int, default=3,
                   help="Scores >= this are unconditionally annotated (default: 3)")
    p.add_argument("--chars-per-token", type=float, default=4.068,
                   help="Chars-per-token ratio (default: 4.068)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--text-column", type=str, default="text",
                   help="Text column name (default: text)")
    p.add_argument("--workers", type=int, default=_DEFAULT_WORKERS,
                   help=f"Parallel workers (default: {_DEFAULT_WORKERS})")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    if args.overwrite and output_dir.exists():
        import shutil
        print(f"--overwrite: removing {output_dir}")
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output dir not empty: {output_dir} (use --overwrite to replace)"
        )

    t_start = time.time()

    # Pass 1: scan
    stats = scan_files(
        source_dir, args.text_column, args.chars_per_token,
        args.annotation_threshold, args.workers,
    )

    # Compute budgets
    ann_budgets, unann_budgets, summary = compute_budgets(stats, args.target_tokens)

    # Pass 2: write
    output_dir.mkdir(parents=True, exist_ok=True)
    results = write_files(
        stats, ann_budgets, unann_budgets,
        args.text_column, args.chars_per_token,
        args.annotation_threshold, args.seed,
        output_dir, args.workers,
    )

    # Summary
    elapsed = time.time() - t_start
    total_ann_rows = sum(r["ann_rows"] for r in results)
    total_unann_rows = sum(r["unann_rows"] for r in results)
    total_ann_tokens = sum(r["ann_tokens"] for r in results)
    total_unann_tokens = sum(r["unann_tokens"] for r in results)

    metadata = {
        "source_dir": str(source_dir),
        "annotation_threshold": args.annotation_threshold,
        "chars_per_token": args.chars_per_token,
        "seed": args.seed,
        "text_column": args.text_column,
        "workers": args.workers,
        **summary,
        "annotated": {
            "selected_rows": total_ann_rows,
            "selected_tokens": total_ann_tokens,
        },
        "unannotated": {
            "selected_rows": total_unann_rows,
            "selected_tokens": total_unann_tokens,
        },
        "selected_rows": total_ann_rows + total_unann_rows,
        "selected_tokens": total_ann_tokens + total_unann_tokens,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"  Output:      {output_dir}")
    print(f"  Annotated:   {total_ann_rows:,} rows, {_fmt(total_ann_tokens)} tokens")
    print(f"  Unannotated: {total_unann_rows:,} rows, {_fmt(total_unann_tokens)} tokens")
    print(f"  Total:       {total_ann_rows + total_unann_rows:,} rows, "
          f"{_fmt(total_ann_tokens + total_unann_tokens)} tokens")
    print(f"  Elapsed:     {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
