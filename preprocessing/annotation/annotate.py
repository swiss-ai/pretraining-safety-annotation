"""Annotate a text dataset with safety scores.

Uses locuslab/safety-classifier_gte-large-en-v1.5 (0–5 safety scale):
    0 = safe, 1 = minimal concern, 2 = mild, 3 = moderate, 4 = significant, 5 = severe.

Designed for multi-GPU execution via torchrun on Clariden GH200 nodes (4 GPUs).
Also works single-GPU for testing (just run with ``python -m``).

Supports **resume**: on restart, each rank counts rows already written to its
shard files and skips that many samples from the stream before continuing.

Usage::

    # Full run on 4 GPUs (streaming from HF)
    torchrun --nproc_per_node=4 -m preprocessing.annotation.annotate

    # From local parquet (download first with preprocessing.download)
    torchrun --nproc_per_node=4 -m preprocessing.annotation.annotate --data-dir $SCRATCH/finephrase/all

    # Different dataset with custom column names
    torchrun --nproc_per_node=4 -m preprocessing.annotation.annotate \
        --dataset allenai/dolma3_mix-6T --text-column text --id-column id

    # Quick test on single GPU
    python -m preprocessing.annotation.annotate --max-samples 1000

    # Process a slice of local parquet files (used by array_job.sh)
    torchrun --nproc_per_node=4 -m preprocessing.annotation.annotate \
        --data-dir $SCRATCH/dolma3_mix-1T --file-start 0 --file-count 471

    # Monitor progress from login node
    cat data/safety_annotations/all/progress.json
"""

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "locuslab/safety-classifier_gte-large-en-v1.5"

PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("safety_score", pa.int8()),
        ("safety_probs", pa.list_(pa.float32())),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate a text dataset with safety scores.")
    p.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/finephrase",
        help="HuggingFace dataset ID (default: HuggingFaceFW/finephrase)",
    )
    p.add_argument(
        "--subset",
        default=None,
        help="Dataset configuration/subset name (default: None)",
    )
    p.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Column name to use as sample ID (default: id)",
    )
    p.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text to classify (default: text)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples across all GPUs (default: full dataset)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--output-dir",
        type=str,
        default="data/safety_annotations",
    )
    p.add_argument("--model-name", type=str, default=MODEL_ID)
    p.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max token length for truncation (model supports up to 8192)",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=100_000,
        help="Flush results to parquet every N samples per GPU",
    )
    p.add_argument(
        "--total",
        type=int,
        default=None,
        help="Expected total samples for progress ETA (finephrase 'all' ≈ 1.35B rows)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output and start from scratch",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Load from local parquet files instead of streaming from HF (e.g. $SCRATCH/finephrase/all)",
    )
    p.add_argument(
        "--file-start",
        type=int,
        default=0,
        help="Index of first parquet file to process (for array job slicing, default: 0)",
    )
    p.add_argument(
        "--file-count",
        type=int,
        default=None,
        help="Number of parquet files to process (default: all remaining from --file-start)",
    )
    return p.parse_args()


# ── distributed helpers ──────────────────────────────────────────────


def setup_distributed() -> tuple[int, int, torch.device]:
    """Init torch.distributed if launched via torchrun, else single-GPU/CPU.

    Returns (rank, world_size, device).
    """
    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def teardown_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


# ── resume helpers ───────────────────────────────────────────────────


def _shard_pattern(rank: int) -> str:
    return f"shard_{rank:04d}_part*.parquet"


def count_existing_rows(output_dir: Path, rank: int) -> int:
    """Count rows already written for this rank across all part files.

    Corrupt files (e.g. missing footer from a killed job) are deleted so that
    resume starts cleanly from the last complete flush.
    """
    total = 0
    for f in sorted(output_dir.glob(_shard_pattern(rank))):
        try:
            total += pq.read_metadata(str(f)).num_rows
        except Exception:
            print(f"WARNING: corrupt parquet {f.name}, deleting for clean resume")
            f.unlink()
    return total


def next_part_index(output_dir: Path, rank: int) -> int:
    """Find the next available part index for this rank.

    Must be called *after* count_existing_rows (which cleans up corrupt files).
    """
    existing = sorted(output_dir.glob(_shard_pattern(rank)))
    if not existing:
        return 0
    last = existing[-1].stem  # e.g. "shard_0000_part0003"
    return int(last.split("part")[-1]) + 1


def _compute_dedup_indices(data_files: list[str], id_column: str) -> tuple[list[int], int]:
    """Compute global row indices of first-occurrence rows, deduplicating per file.

    Reads only the id column from each file (lightweight). Duplicates are
    within-file only (upstream quality-aware upsampling), so the seen-set
    resets per file.

    Returns (dedup_indices, n_original_rows).
    """
    indices = []
    offset = 0
    for f in data_files:
        ids = pq.read_table(f, columns=[id_column]).column(id_column).to_pylist()
        seen: set[str] = set()
        for j, doc_id in enumerate(ids):
            key = str(doc_id)
            if key not in seen:
                seen.add(key)
                indices.append(offset + j)
        offset += len(ids)
    return indices, offset


# ── inference ────────────────────────────────────────────────────────

SEQ_LEN_BUCKETS = list(range(32, 2049, 32))  # 32, 64, 96, ..., 2048 (64 buckets)


def _next_bucket(seq_len: int) -> int:
    """Round seq_len up to the next fixed bucket for torch.compile cache hits."""
    for b in SEQ_LEN_BUCKETS:
        if b >= seq_len:
            return b
    return seq_len


@torch.no_grad()
def classify_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model: AutoModelForSequenceClassification,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass on pre-tokenized, pre-padded inputs.

    Returns (predicted_classes [B], probabilities [B, 6]).
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(logits, dim=-1).cpu().float()
    scores = probs.argmax(dim=-1)
    return scores, probs


def _prepare_pool(
    pool_ids: list[str],
    pool_texts: list[str],
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    token_budget: int,
) -> list[tuple[list[str], torch.Tensor, torch.Tensor, int, int]]:
    """Tokenize a pool of texts, sort by actual token length, and build padded batches.

    Returns list of (batch_ids, input_ids, attention_mask, actual_tokens, padded_tokens).
    """
    encodings = tokenizer(
        pool_texts,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    token_lengths = [len(ids) for ids in encodings["input_ids"]]
    order = sorted(range(len(pool_texts)), key=lambda i: token_lengths[i])

    batches: list[tuple[list[str], torch.Tensor, torch.Tensor, int, int]] = []
    pos = 0
    while pos < len(order):
        end = min(pos + batch_size, len(order))
        longest_tokens = token_lengths[order[end - 1]]
        bucket = _next_bucket(longest_tokens)
        bs = max(1, min(end - pos, token_budget // bucket))
        # Recompute bucket from the actual batch boundary (shorter sequences)
        bucket = _next_bucket(token_lengths[order[pos + bs - 1]])
        bs = max(1, min(end - pos, token_budget // bucket))
        idx = order[pos : pos + bs]
        pos += bs

        actual_tokens = sum(token_lengths[i] for i in idx)
        padded_tokens = bucket * len(idx)

        batch_features = [{"input_ids": encodings["input_ids"][i]} for i in idx]
        padded = tokenizer.pad(
            batch_features,
            padding="max_length",
            max_length=bucket,
            return_tensors="pt",
        )
        batches.append(
            (
                [pool_ids[i] for i in idx],
                padded["input_ids"].to(device),
                padded["attention_mask"].to(device),
                actual_tokens,
                padded_tokens,
            )
        )
    return batches


def _prefetch_worker(
    ds_iter,
    batch_queue: queue.Queue,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    token_budget: int,
    pool_size: int,
    id_column: str = "id",
    text_column: str = "text",
) -> None:
    """Background thread: read dataset, tokenize pools, enqueue ready batches.

    HF fast tokenizers are Rust-based and release the GIL, so threading gives
    real parallelism for tokenization without multiprocessing overhead.
    """
    pool_ids: list[str] = []
    pool_texts: list[str] = []

    for sample in ds_iter:
        pool_ids.append(str(sample[id_column]))
        pool_texts.append(sample[text_column])

        if len(pool_texts) >= pool_size:
            batches = _prepare_pool(
                pool_ids, pool_texts, tokenizer, device, max_length, batch_size, token_budget
            )
            for batch in batches:
                batch_queue.put(batch)
            pool_ids, pool_texts = [], []

    if pool_texts:
        batches = _prepare_pool(
            pool_ids, pool_texts, tokenizer, device, max_length, batch_size, token_budget
        )
        for batch in batches:
            batch_queue.put(batch)

    batch_queue.put(None)


# ── I/O ──────────────────────────────────────────────────────────────


def flush_to_parquet(writer: pq.ParquetWriter, buffer: list[dict]) -> None:
    """Write buffered results as a row-group to the open parquet writer."""
    table = pa.table(
        {
            "id": [r["id"] for r in buffer],
            "safety_score": [r["safety_score"] for r in buffer],
            "safety_probs": [r["safety_probs"] for r in buffer],
        },
        schema=PARQUET_SCHEMA,
    )
    writer.write_table(table)


def write_progress(
    path: Path,
    n_this_run: int,
    n_previously: int,
    total_per_gpu: int | None,
    t_start: float,
    world_size: int,
) -> None:
    """Write a JSON progress file (rank 0 only) for easy monitoring.

    Check with: cat data/safety_annotations/<subset>/progress.json
    """
    elapsed = time.time() - t_start
    rate = n_this_run / elapsed if elapsed > 0 else 0.0
    n_gpu = n_previously + n_this_run
    n_global = n_gpu * world_size
    global_rate = rate * world_size

    info: dict = {
        "samples_written_per_gpu": n_gpu,
        "samples_written_total": n_global,
        "this_run_per_gpu": n_this_run,
        "resumed_from_per_gpu": n_previously,
        "elapsed_s": round(elapsed, 1),
        "samples_per_sec_per_gpu": round(rate, 1),
        "samples_per_sec_total": round(global_rate, 1),
    }
    if total_per_gpu is not None:
        total = total_per_gpu * world_size
        remaining = max(0, total - n_global)
        eta_s = remaining / global_rate if global_rate > 0 else float("inf")
        info["total"] = total
        info["pct"] = round(100 * n_global / total, 2)
        info["eta_s"] = round(eta_s, 0)
        info["eta_human"] = f"{eta_s / 3600:.1f}h"

    path.write_text(json.dumps(info, indent=2) + "\n")


# ── main loop ────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"world_size={world_size}  device={device}")
        if args.data_dir is not None:
            print(f"data_dir={args.data_dir}")
        else:
            print(f"dataset={args.dataset}[{args.subset}]")
        print(f"model={args.model_name}")
        if args.max_samples:
            print(f"max_samples={args.max_samples}")

    # ── model ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            num_labels=6,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    model = torch.compile(model)
    if is_main:
        print("Model loaded + compiled.")

    # ── dataset ──────────────────────────────────────────────────────
    if args.data_dir is not None:
        from glob import glob as _glob

        data_files = sorted(_glob(f"{args.data_dir}/part_*.parquet"))
        assert data_files, f"No part_*.parquet files found in {args.data_dir}"
        if args.file_count is not None:
            data_files = data_files[args.file_start : args.file_start + args.file_count]
        elif args.file_start > 0:
            data_files = data_files[args.file_start :]
        assert data_files, f"No files in slice [file_start={args.file_start}, file_count={args.file_count}]"
        if is_main:
            print(f"Loading from local parquet: {args.data_dir} ({len(data_files)} files)")
        ds = load_dataset("parquet", data_files=data_files, split="train")
        dedup_indices, n_original = _compute_dedup_indices(data_files, args.id_column)
        ds = ds.select(dedup_indices)
        if is_main and n_original > 0:
            print(f"Dedup: {n_original:,} -> {len(ds):,} rows ({100*(1-len(ds)/n_original):.1f}% removed)")
    else:
        ds = load_dataset(
            args.dataset,
            args.subset,
            split="train",
            streaming=True,
        )

    n_input_rows = len(ds) if args.data_dir is not None else None

    per_gpu = None
    if args.max_samples is not None:
        ds = ds.take(args.max_samples)
        per_gpu = args.max_samples // world_size
        n_input_rows = args.max_samples
    elif args.data_dir is not None:
        per_gpu = len(ds) // world_size

    if world_size > 1:
        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

    total_per_gpu = per_gpu if per_gpu else (args.total // world_size if args.total else None)

    # ── resume ───────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if args.subset is not None:
        output_dir = output_dir / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── task metadata (for merge.py) ────────────────────────────────
    if is_main and args.data_dir is not None:
        task_meta = {
            "data_dir": args.data_dir,
            "file_start": args.file_start,
            "file_count": len(data_files),
            "n_input_rows": n_input_rows,
            "n_original_rows": n_original,
            "world_size": world_size,
            "files": [Path(f).name for f in data_files],
        }
        (output_dir / "task_meta.json").write_text(json.dumps(task_meta, indent=2) + "\n")

    n_skip = 0
    if not args.no_resume:
        n_skip = count_existing_rows(output_dir, rank)

    if n_skip > 0:
        if is_main:
            print(f"Resuming: skipping {n_skip:,} already-processed samples per GPU")
        ds = ds.skip(n_skip)
        if per_gpu is not None and is_main:
            remaining = max(0, per_gpu - n_skip)
            print(f"Will process {remaining:,} more samples per GPU (of {per_gpu:,} target)")

    # ── output file ──────────────────────────────────────────────────
    part_idx = next_part_index(output_dir, rank)
    output_path = output_dir / f"shard_{rank:04d}_part{part_idx:04d}.parquet"
    writer = pq.ParquetWriter(str(output_path), PARQUET_SCHEMA)
    progress_path = output_dir / "progress.json"

    # ── GPU monitor ─────────────────────────────────────────────────
    from gpu_monitor import GPUMonitor

    gpu_monitor = GPUMonitor(
        output_dir=output_dir, device=device, world_size=world_size, rank=rank,
    )
    gpu_monitor.__enter__()

    # ── processing loop ──────────────────────────────────────────────
    pool_size = args.batch_size * 128
    token_budget = args.batch_size * 512
    buffer: list[dict] = []
    n_written = 0
    t_start = time.time()

    tqdm_total = None
    if per_gpu is not None:
        tqdm_total = max(0, per_gpu - n_skip)
    elif args.total is not None:
        tqdm_total = max(0, args.total // world_size - n_skip)

    batch_queue: queue.Queue = queue.Queue(maxsize=2)
    prefetch = threading.Thread(
        target=_prefetch_worker,
        args=(
            ds, batch_queue, tokenizer, device,
            args.max_length, args.batch_size, token_budget, pool_size,
            args.id_column, args.text_column,
        ),
        daemon=True,
    )
    prefetch.start()

    total_actual_tokens = 0
    total_padded_tokens = 0

    pbar = tqdm(desc=f"[GPU {rank}]", total=tqdm_total, disable=not is_main)
    while True:
        item = batch_queue.get()
        if item is None:
            break
        batch_ids, input_ids, attention_mask, actual_tok, padded_tok = item
        total_actual_tokens += actual_tok
        total_padded_tokens += padded_tok
        scores, probs = classify_batch(input_ids, attention_mask, model)
        for j, sample_id in enumerate(batch_ids):
            buffer.append(
                {
                    "id": sample_id,
                    "safety_score": scores[j].item(),
                    "safety_probs": probs[j].tolist(),
                }
            )
        pbar.update(len(batch_ids))

        if len(buffer) >= args.flush_every:
            flush_to_parquet(writer, buffer)
            n_written += len(buffer)
            buffer = []
            if is_main:
                overhead = 100 * (1 - total_actual_tokens / total_padded_tokens) if total_padded_tokens else 0
                print(f"Padding overhead: {overhead:.1f}% ({total_actual_tokens:,} actual / {total_padded_tokens:,} padded tokens)")
                write_progress(
                    progress_path, n_written, n_skip, total_per_gpu, t_start, world_size
                )
    pbar.close()

    if buffer:
        flush_to_parquet(writer, buffer)
        n_written += len(buffer)

    writer.close()

    if is_main:
        write_progress(
            progress_path, n_written, n_skip, total_per_gpu, t_start, world_size
        )
        total_gpu = n_skip + n_written
        overhead = 100 * (1 - total_actual_tokens / total_padded_tokens) if total_padded_tokens else 0
        print(f"Rank 0: wrote {n_written:,} new rows (total for this GPU: {total_gpu:,})")
        print(f"Padding overhead: {overhead:.1f}% ({total_actual_tokens:,} actual / {total_padded_tokens:,} padded tokens)")
        print(f"Output: {output_path}")

    gpu_monitor.__exit__(None, None, None)

    teardown_distributed()

    if is_main:
        (output_dir / "DONE").touch()
        print(f"All done. Shards in {output_dir}/")


if __name__ == "__main__":
    main()
