"""Tokenize pre-split parquets into two Megatron-format training streams.

Expects two separate input directories (produced by subsample_and_stratify):

* ``--compact-data-dir`` — parquets with non-annotated samples only.
* ``--annotated-data-dir`` — parquets with annotated samples only.

Two pipelines controlled by ``--pipeline {compact,split,both}``:

* **compact** — pack non-annotated samples into dense 2048-token windows,
  output as Megatron ``.bin`` + ``.idx``.
* **split** — tokenize annotated samples, one per padded 2049-token window,
  output as Megatron ``.bin`` + ``.idx`` plus a sidecar parquet with original
  text and token lengths for downstream reflection.

Usage::

    # Both pipelines (default)
    python -m preprocessing.tokenization.tokenize \\
        --compact-data-dir $SCRATCH/dolma3_non_annotated \\
        --annotated-data-dir $SCRATCH/dolma3_annotated \\
        --output-dir $SCRATCH/tokenized

    # Compact only
    python -m preprocessing.tokenization.tokenize \\
        --compact-data-dir $SCRATCH/dolma3_non_annotated --pipeline compact

    # Split only
    python -m preprocessing.tokenization.tokenize \\
        --annotated-data-dir $SCRATCH/dolma3_annotated --pipeline split
"""

import argparse
import os
from pathlib import Path


def _count_parquets(data_dir: str) -> int:
    """Count parquet files in *data_dir* (recursive)."""
    n = len(list(Path(data_dir).rglob("*.parquet")))
    assert n > 0, f"No parquet files found in {data_dir}"
    return n


def _local_partition(total: int, n_nodes: int, node_id: int) -> tuple[int, int]:
    """Return (local_rank_offset, local_tasks) for *node_id* out of *n_nodes*."""
    per_node = (total + n_nodes - 1) // n_nodes
    offset = node_id * per_node
    local = min(per_node, total - offset)
    return offset, local


# ---------------------------------------------------------------------------
# Compact path (datatrove)
# ---------------------------------------------------------------------------


def run_compact(args: argparse.Namespace) -> None:
    """Tokenize non-annotated samples into packed Megatron .bin + .idx."""
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader

    from preprocessing.tokenization.steps import (
        FastDSConcatenator,
        MegatronContextShuffler,
        TruncatingDocumentTokenizer,
    )

    stage = args.stage
    out = Path(args.output_dir) / "compact"
    logs = str(Path(args.output_dir) / "logs")
    n_tasks = _count_parquets(args.compact_data_dir)

    if stage in ("tokenize", "all"):
        offset, local = _local_partition(n_tasks, args.n_nodes, args.node_id)
        print(
            f"Compact tokenize: {n_tasks} files, node {args.node_id}/{args.n_nodes}, "
            f"tasks [{offset}:{offset + local}], {args.workers} workers"
        )
        stage1 = LocalPipelineExecutor(
            pipeline=[
                ParquetReader(
                    data_folder=args.compact_data_dir,
                    text_key="text",
                    id_key="id",
                ),
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
            local_tasks=local,
            local_rank_offset=offset,
            logging_dir=f"{logs}/compact_stage1",
            skip_completed=True,
        )
        stage1.run()
        print(f"Compact tokenize done (node {args.node_id})")

    if stage in ("merge", "all"):
        print(f"Compact merge: {n_tasks} files → megatron .bin/.idx")
        stage2 = LocalPipelineExecutor(
            pipeline=[
                FastDSConcatenator(
                    input_folder=str(out / "tokenized"),
                    output_folder=str(out / "merged"),
                    save_filename="dolma3",
                ),
            ],
            tasks=1,
            logging_dir=f"{logs}/compact_stage2",
        )
        stage3 = LocalPipelineExecutor(
            pipeline=[
                MegatronContextShuffler(
                    input_folder=str(out / "merged"),
                    output_folder=str(out / "megatron"),
                    window_size=args.seq_length + 1,
                    save_filename="compact",
                    seed=args.seed,
                ),
            ],
            tasks=1,
            logging_dir=f"{logs}/compact_stage3",
            depends=stage2,
        )
        stage3.run()
        print(f"Compact done → {out / 'megatron'}")


# ---------------------------------------------------------------------------
# Split path (datatrove stages 1-2 for parallel tokenization, custom stage 3)
# ---------------------------------------------------------------------------


def run_split(args: argparse.Namespace) -> None:
    """Tokenize annotated samples into padded Megatron .bin + .idx + sidecar.

    Three-stage datatrove pipeline:
      1. Parallel tokenization (ParquetReader → TruncatingDocumentTokenizer)
      2. Merge per-task .ds files into a single .ds
      3. Shuffle, pad to fixed windows, write Megatron .bin/.idx + sidecar
    """
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader

    from preprocessing.tokenization.steps import (
        FastDSConcatenator,
        MegatronAnnotatedShuffler,
        TruncatingDocumentTokenizer,
    )

    stage = args.stage
    out = Path(args.output_dir) / "annotated"
    logs = str(Path(args.output_dir) / "logs")
    max_tokens = args.seq_length - args.reflection_budget
    n_tasks = _count_parquets(args.annotated_data_dir)

    if stage in ("tokenize", "all"):
        offset, local = _local_partition(n_tasks, args.n_nodes, args.node_id)
        print(
            f"Split tokenize: {n_tasks} files, node {args.node_id}/{args.n_nodes}, "
            f"tasks [{offset}:{offset + local}], {args.workers} workers"
        )
        stage1 = LocalPipelineExecutor(
            pipeline=[
                ParquetReader(
                    data_folder=args.annotated_data_dir,
                    text_key="text",
                    id_key="id",
                ),
                TruncatingDocumentTokenizer(
                    max_doc_tokens=max_tokens,
                    output_folder=str(out / "tokenized"),
                    tokenizer_name_or_path=args.tokenizer,
                    eos_token="<|endoftext|>",
                    shuffle_documents=False,
                    batch_size=10000,
                ),
            ],
            tasks=n_tasks,
            workers=args.workers,
            local_tasks=local,
            local_rank_offset=offset,
            logging_dir=f"{logs}/split_stage1",
            skip_completed=True,
        )
        stage1.run()
        print(f"Split tokenize done (node {args.node_id})")

    if stage in ("merge", "all"):
        print(f"Split merge: {n_tasks} files → megatron .bin/.idx + sidecar")
        stage2 = LocalPipelineExecutor(
            pipeline=[
                FastDSConcatenator(
                    input_folder=str(out / "tokenized"),
                    output_folder=str(out / "merged"),
                    save_filename="annotated",
                ),
            ],
            tasks=1,
            logging_dir=f"{logs}/split_stage2",
        )
        stage3 = LocalPipelineExecutor(
            pipeline=[
                MegatronAnnotatedShuffler(
                    input_folder=str(out / "merged"),
                    output_folder=str(out),
                    annotated_data_dir=args.annotated_data_dir,
                    window_size=args.seq_length + 1,
                    save_filename="annotated",
                    seed=args.seed,
                ),
            ],
            tasks=1,
            logging_dir=f"{logs}/split_stage3",
            depends=stage2,
        )
        stage3.run()
        print(f"Split done → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    scratch = os.environ.get(
        "SCRATCH", f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}"
    )

    p = argparse.ArgumentParser(
        description="Tokenize pre-split parquets into two Megatron training streams."
    )
    p.add_argument(
        "--compact-data-dir",
        type=str,
        default=f"{scratch}/dolma3_non_annotated",
        help="Input parquets for compact path (non-annotated samples only)",
    )
    p.add_argument(
        "--annotated-data-dir",
        type=str,
        default=f"{scratch}/dolma3_annotated",
        help="Input parquets for split path (annotated samples only)",
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
        default=244,
        help="Parallel workers (default: 244)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for annotated shuffle (default: 42)",
    )
    p.add_argument(
        "--pipeline",
        choices=["compact", "split", "both"],
        default="both",
        help="Which pipeline(s) to run (default: both)",
    )
    p.add_argument(
        "--stage",
        choices=["tokenize", "merge", "all"],
        default="all",
        help="Which stage to run: tokenize (stage 1, multi-node), "
        "merge (stages 2-3, single node), all (default)",
    )
    p.add_argument(
        "--n-nodes",
        type=int,
        default=1,
        help="Total nodes for multi-node tokenization (default: 1)",
    )
    p.add_argument(
        "--node-id",
        type=int,
        default=0,
        help="This node's ID (0-indexed, default: 0). "
        "Set automatically from SLURM_ARRAY_TASK_ID if available.",
    )
    return p.parse_args()


def _patch_pool_maxtasksperchild(max_tasks: int = 5) -> None:
    """Monkey-patch multiprocess.Pool to recycle workers after *max_tasks*.

    Prevents memory accumulation from Python's inability to release arena
    memory back to the OS between tasks.
    """
    import multiprocess.context as _mp_ctx

    _orig_pool = _mp_ctx.BaseContext.Pool

    def _patched_pool(self, *args, **kwargs):
        kwargs.setdefault("maxtasksperchild", max_tasks)
        return _orig_pool(self, *args, **kwargs)

    _mp_ctx.BaseContext.Pool = _patched_pool


def main() -> None:
    _patch_pool_maxtasksperchild(max_tasks=5)
    args = parse_args()

    # Auto-detect node_id from SLURM array task ID
    if "SLURM_ARRAY_TASK_ID" in os.environ and args.node_id == 0:
        args.node_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if args.pipeline in ("compact", "both"):
        run_compact(args)
    if args.pipeline in ("split", "both"):
        run_split(args)


if __name__ == "__main__":
    main()
