"""Tokenize dolma3 parquets into training-ready format.

Two pipelines controlled by ``--pipeline {compact,split,both}``:

* **compact** — tokenize ``has_annotation=False`` samples into packed
  2048-token windows via datatrove (ParquetReader → AnnotationFilter →
  TruncatingDocumentTokenizer → Merge+Shuffle → ContextShuffle).
* **split** — extract ``has_annotation=True`` samples, truncate text to
  ``seq_length - reflection_budget`` tokens, and write as parquet for the
  downstream reflection pipeline.

Usage::

    # Both pipelines (default)
    python -m preprocessing.tokenization.tokenize \\
        --data-dir $SCRATCH/dolma3_mix-1T --output-dir $SCRATCH/tokenized

    # Compact only
    python -m preprocessing.tokenization.tokenize --pipeline compact

    # Split only, 4 workers (quick test)
    python -m preprocessing.tokenization.tokenize --pipeline split --workers 4
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def _count_parquets(data_dir: str) -> int:
    """Count parquet files in *data_dir* (recursive)."""
    n = len(list(Path(data_dir).rglob("*.parquet")))
    assert n > 0, f"No parquet files found in {data_dir}"
    return n


# ---------------------------------------------------------------------------
# Compact path (datatrove)
# ---------------------------------------------------------------------------


def run_compact(args: argparse.Namespace) -> None:
    """Tokenize has_annotation=False samples into packed token windows."""
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens.context_shuffler import (
        DocumentTokenizerContextShuffler,
    )
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger

    from preprocessing.tokenization.steps import (
        AnnotationFilter,
        TruncatingDocumentTokenizer,
    )

    out = Path(args.output_dir) / "compact"
    logs = str(Path(args.output_dir) / "logs")
    n_tasks = _count_parquets(args.data_dir)
    print(f"Compact: {n_tasks} input files, {args.workers} workers, seq_length={args.seq_length}")

    # Stage 1: tokenize (parallel across input files)
    stage1 = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=args.data_dir,
                text_key="text",
                id_key="id",
                read_metadata=True,
            ),
            AnnotationFilter(keep_annotated=False),
            TruncatingDocumentTokenizer(
                max_doc_tokens=args.seq_length,
                output_folder=str(out / "tokenized"),
                tokenizer_name_or_path=args.tokenizer,
                eos_token="<|endoftext|>",
                shuffle_documents=True,
                batch_size=10000,
            ),
        ],
        tasks=n_tasks,
        workers=args.workers,
        logging_dir=f"{logs}/compact_stage1",
        skip_completed=True,
    )

    # Stage 2: merge + global shuffle (single task)
    stage2 = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=str(out / "tokenized"),
                output_folder=str(out / "merged"),
                save_filename="dolma3",
                shuffle=True,
            ),
        ],
        tasks=1,
        logging_dir=f"{logs}/compact_stage2",
        depends=stage1,
    )

    # Stage 3: context shuffle into (seq_length + 1)-token windows
    stage3 = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerContextShuffler(
                input_folder=str(out / "merged"),
                output_folder=str(out / "final"),
                window_size=args.seq_length + 1,
            ),
        ],
        tasks=1,
        logging_dir=f"{logs}/compact_stage3",
        depends=stage2,
    )

    stage3.run()
    print(f"Compact done → {out / 'final'}")


# ---------------------------------------------------------------------------
# Split path (pyarrow)
# ---------------------------------------------------------------------------


def _split_one(input_path: str, output_path: str, max_tokens: int) -> dict:
    """Filter has_annotation=True rows, truncate text, write parquet."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from pipeline.tokenizer import truncate_to_max_tokens

    table = pq.read_table(input_path)
    n_in = len(table)

    mask = pc.equal(table.column("has_annotation"), True)
    filtered = table.filter(mask)
    n_out = len(filtered)

    if n_out == 0:
        return {"input": input_path, "n_in": n_in, "n_out": 0}

    texts = filtered.column("text").to_pylist()
    truncated = [truncate_to_max_tokens(t, max_tokens) for t in texts]
    filtered = filtered.set_column(
        filtered.schema.get_field_index("text"),
        "text",
        pa.array(truncated),
    )
    pq.write_table(filtered, output_path)
    return {"input": input_path, "n_in": n_in, "n_out": n_out}


def _find_completed(output_dir: Path) -> set[str]:
    """Return stems of input files already processed (from .done markers)."""
    done_dir = output_dir / ".done"
    if not done_dir.exists():
        return set()
    return {f.stem for f in done_dir.glob("*.done")}


def _mark_done(done_dir: Path, stem: str):
    """Write a marker so this file is skipped on resume."""
    (done_dir / f"{stem}.done").touch()


def run_split(args: argparse.Namespace) -> None:
    """Extract has_annotation=True samples, truncate text, write parquet."""
    out = Path(args.output_dir) / "annotated"
    out.mkdir(parents=True, exist_ok=True)
    done_dir = out / ".done"
    done_dir.mkdir(exist_ok=True)

    max_tokens = args.seq_length - args.reflection_budget
    input_files = sorted(Path(args.data_dir).rglob("*.parquet"))
    assert len(input_files) > 0, f"No parquet files found in {args.data_dir}"

    completed = _find_completed(out)
    remaining = [(f, f.stem) for f in input_files if f.stem not in completed]
    print(
        f"Split: {len(input_files)} input files, {len(completed)} done, "
        f"{len(remaining)} remaining, max_tokens={max_tokens}"
    )

    if not remaining:
        print("All files already processed.")
        return

    total_in = 0
    total_out = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for input_file, stem in remaining:
            output_file = str(out / f"{stem}.parquet")
            fut = pool.submit(_split_one, str(input_file), output_file, max_tokens)
            futures[fut] = stem

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Split", unit="file"):
            stem = futures[fut]
            result = fut.result()
            total_in += result["n_in"]
            total_out += result["n_out"]
            _mark_done(done_dir, stem)

    print(f"Split done → {out} ({total_out:,} annotated rows from {total_in:,} total)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    scratch = os.environ.get(
        "SCRATCH", f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}"
    )

    p = argparse.ArgumentParser(
        description="Tokenize dolma3 parquets into training-ready format."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=f"{scratch}/dolma3_mix-1T",
        help="Input parquets directory",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=f"{scratch}/tokenized",
        help="Base output directory",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="HF tokenizer name or path",
    )
    p.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length for compact path (default: 2048)",
    )
    p.add_argument(
        "--reflection-budget",
        type=int,
        default=128,
        help="Tokens reserved for reflections in split path (default: 128)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Parallel workers (default: 64)",
    )
    p.add_argument(
        "--pipeline",
        choices=["compact", "split", "both"],
        default="both",
        help="Which pipeline(s) to run (default: both)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.pipeline in ("compact", "both"):
        run_compact(args)
    if args.pipeline in ("split", "both"):
        run_split(args)


if __name__ == "__main__":
    main()
