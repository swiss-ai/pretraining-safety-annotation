"""Download upstream shards from a HuggingFace dataset to local parquet files.

Selects N upstream shards (optionally shuffled for cross-source diversity),
and writes one local parquet file per shard. Uses multiprocessing to bypass
the GIL (JSON/zstd parsing is CPU-bound).

Supports incremental resume: the shuffled shard plan is saved to a manifest
file on first run. On restart, already-downloaded shards are skipped.

Use estimate_chars_per_token.py to compute how many shards you need for a
given token budget.

Usage::

    # Download 5000 shuffled shards from dolma3 (8 workers)
    python -m preprocessing.download --dataset allenai/dolma3_mix-6T \
        --n-shards 5000 --shuffle --seed 42 --columns text id source \
        --ignore-errors --workers 8

    # Small test download
    python -m preprocessing.download --dataset allenai/dolma3_mix-6T \
        --n-shards 10 --columns text id source --ignore-errors

    # Overwrite existing data (resets manifest)
    python -m preprocessing.download --n-shards 5000 --overwrite
"""

import argparse
import json
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

DEFAULT_DATASET = "HuggingFaceFW/finephrase"


_worker_ds = None  # per-process cached dataset object


def _patch_lenient_column_cast():
    """Monkey-patch datasets to drop extra columns instead of raising CastError.

    HF's schema casting rejects tables whose columns don't exactly match the
    target schema. This is common with heterogeneous datasets (e.g. dolma3)
    where some shards have extra metadata columns.  The patch drops extra
    columns before delegating to the original cast, so the data is preserved.
    """
    import datasets.table as dt

    if getattr(dt.cast_table_to_schema, "_lenient_patched", False):
        return

    _orig_schema = dt.cast_table_to_schema

    def _lenient_schema(table, schema):
        extra = set(table.column_names) - set(schema.names)
        if extra:
            table = table.drop(list(extra))
        return _orig_schema(table, schema)

    _lenient_schema._lenient_patched = True
    dt.cast_table_to_schema = _lenient_schema

    _orig_features = dt.cast_table_to_features

    def _lenient_features(table, features):
        extra = set(table.column_names) - set(features)
        if extra:
            table = table.drop(list(extra))
        return _orig_features(table, features)

    dt.cast_table_to_features = _lenient_features


def _worker_init(dataset: str, subset: str | None, ignore_errors: bool):
    """Per-process initializer: load dataset once and patch column cast if needed."""
    global _worker_ds

    if ignore_errors:
        _patch_lenient_column_cast()

    from datasets import load_dataset

    _worker_ds = load_dataset(dataset, subset, split="train", streaming=True)


def _download_one(
    shard_idx: int,
    out_path: str,
    columns: list[str] | None,
) -> tuple[int, int | None, str | None]:
    """Download a single upstream shard to a parquet file.

    Uses the per-process cached dataset (set up in _worker_init).
    Returns (shard_idx, n_rows, error_msg). n_rows is None on failure.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        shard = _worker_ds.shard(num_shards=_worker_ds.n_shards, index=shard_idx)
        if columns is not None:
            shard = shard.select_columns(columns)
        rows = [dict(sample) for sample in shard]
    except Exception as e:
        return shard_idx, None, f"{type(e).__name__}: {e}"

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    return shard_idx, len(rows), None


def _load_or_create_manifest(output_dir: Path, args, n_total: int) -> list[int]:
    """Load existing shard manifest or create a new one.

    The manifest stores the deterministic shard order so that restarts
    after a 12h SLURM timeout resume from where they left off.
    """
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"Resuming from manifest ({len(manifest['shard_order']):,} planned shards)")
        return manifest["shard_order"]

    shard_order = list(range(n_total))
    if args.shuffle:
        random.Random(args.seed).shuffle(shard_order)
    shard_order = shard_order[: args.n_shards]

    manifest = {
        "dataset": args.dataset,
        "subset": args.subset,
        "columns": args.columns,
        "n_shards": args.n_shards,
        "seed": args.seed,
        "shard_order": shard_order,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created manifest with {len(shard_order):,} shards")
    return shard_order


def _find_completed(output_dir: Path) -> set[int]:
    """Return set of upstream shard indices already downloaded (from done markers)."""
    done_dir = output_dir / ".done"
    if not done_dir.exists():
        return set()
    return {int(f.stem) for f in done_dir.glob("*.done")}


def _mark_done(done_dir: Path, upstream_idx: int):
    """Write a marker file so we know this shard is complete on resume."""
    (done_dir / f"{upstream_idx}.done").touch()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset download."""
    scratch = os.environ.get("SCRATCH", f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}")

    p = argparse.ArgumentParser(description="Download upstream shards from a HuggingFace dataset.")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"HuggingFace dataset ID (default: {DEFAULT_DATASET})")
    p.add_argument("--subset", default=None, help="Dataset configuration/subset name")
    p.add_argument("--columns", nargs="+", default=None, help="Columns to keep (default: all)")
    p.add_argument("--n-shards", type=int, required=True, help="Number of upstream shards to download")
    p.add_argument("--output-dir", type=str, default=None, help=f"Output directory (default: $SCRATCH/<dataset_name>)")
    p.add_argument("--overwrite", action="store_true", help="Remove existing data before downloading")
    p.add_argument("--shuffle", action="store_true", help="Shuffle upstream shard order for cross-source diversity")
    p.add_argument("--seed", type=int, default=None, help="Random seed for --shuffle")
    p.add_argument("--ignore-errors", action="store_true", help="Skip bad shards instead of crashing")
    p.add_argument("--workers", type=int, default=4, help="Parallel download workers (default: 4)")
    args = p.parse_args()

    if args.output_dir is None:
        dataset_name = args.dataset.split("/")[-1]
        args.output_dir = f"{scratch}/{dataset_name}"

    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.subset is not None:
        output_dir = output_dir / args.subset

    if args.overwrite and output_dir.exists():
        print(f"--overwrite: removing existing data in {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ignore_errors:
        _patch_lenient_column_cast()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)
    n_total = ds.n_shards
    del ds

    # Load or create the shard plan (deterministic across restarts)
    shard_order = _load_or_create_manifest(output_dir, args, n_total)

    # Skip already-completed shards
    completed = _find_completed(output_dir)
    remaining = [i for i in shard_order if i not in completed]
    print(f"Dataset has {n_total:,} upstream shards | plan: {len(shard_order)} | done: {len(completed)} | remaining: {len(remaining)} | workers: {args.workers}")

    if not remaining:
        print("All shards already downloaded.")
        return

    done_dir = output_dir / ".done"
    done_dir.mkdir(exist_ok=True)

    # Count existing parquet files for sequential naming
    next_part_idx = len(list(output_dir.glob("part_*.parquet")))

    errors: list[tuple[int, str]] = []
    n_written_this_run = 0
    t_start = time.time()
    pbar = tqdm(total=len(shard_order), initial=len(completed), desc="Shards", unit="shard")

    candidates = iter(remaining)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.dataset, args.subset, args.ignore_errors),
    ) as pool:
        futures = {}

        def _submit_next():
            for i in candidates:
                tmp_path = str(output_dir / f"_tmp_{i}.parquet")
                fut = pool.submit(_download_one, i, tmp_path, args.columns)
                futures[fut] = (i, tmp_path)
                return True
            return False

        for _ in range(min(args.workers, len(remaining))):
            if not _submit_next():
                break

        while futures:
            done = next(as_completed(futures))
            upstream_idx, tmp_path = futures.pop(done)
            _, rows, err_msg = done.result()

            if err_msg is not None:
                errors.append((upstream_idx, err_msg))
                Path(tmp_path).unlink(missing_ok=True)
                if args.ignore_errors:
                    tqdm.write(f"WARNING: shard {upstream_idx} — {err_msg}, skipping")
                    _mark_done(done_dir, upstream_idx)
                    pbar.update(1)
                    _submit_next()
                    continue
                raise RuntimeError(f"Shard {upstream_idx}: {err_msg}")

            Path(tmp_path).rename(output_dir / f"part_{next_part_idx:05d}.parquet")
            _mark_done(done_dir, upstream_idx)
            n_written_this_run += rows
            next_part_idx += 1
            pbar.update(1)
            _submit_next()

    pbar.close()
    elapsed = time.time() - t_start

    n_downloaded = len(completed) + len(remaining) - len(errors)
    metadata = {
        "source_dataset": args.dataset,
        "subset": args.subset,
        "n_shards_downloaded": next_part_idx,
        "n_shards_skipped": len(errors),
        "total_upstream_shards": n_total,
        "total_rows": n_written_this_run,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
    }
    if args.columns is not None:
        metadata["columns"] = args.columns
    if args.shuffle:
        metadata["shuffled"] = True
        metadata["shuffle_seed"] = args.seed
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    error_msg = f", {len(errors)} skipped" if errors else ""
    print(f"Done this run: {n_written_this_run:,} rows ({elapsed:.1f}s{error_msg})")
    print(f"Total: {next_part_idx} shards in {output_dir}")


if __name__ == "__main__":
    main()
