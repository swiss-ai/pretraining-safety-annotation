"""Estimate tokens-per-shard for a tokenizer on local parquet data.

Reads downloaded parquet shards, tokenizes a sample of documents, and reports
aggregate statistics including estimated tokens per upstream shard.  Use the
result with ``python -m preprocessing.download --n-shards N``.

Usage::

    # Estimate on a small sample (default 10k docs)
    python -m preprocessing.estimate_chars_per_token \
        --data-dir $SCRATCH/dolma3_mix-6T \
        --tokenizer meta-llama/Llama-3.1-8B

    # With a context-length cap (e.g. training truncates to 2048)
    python -m preprocessing.estimate_chars_per_token \
        --data-dir $SCRATCH/dolma3_mix-6T \
        --tokenizer meta-llama/Llama-3.1-8B \
        --max-tokens-per-sample 2048
"""

import argparse
import math
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chars-per-token estimation."""
    p = argparse.ArgumentParser(description="Estimate tokens/shard from local parquet data.")
    p.add_argument("--data-dir", type=str, required=True, help="Directory with part_*.parquet files")
    p.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name or path")
    p.add_argument("--n", type=int, default=10_000, help="Number of documents to sample (default: 10,000)")
    p.add_argument("--text-column", type=str, default="text", help="Column containing text (default: text)")
    p.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=None,
        help="Cap per-sample token count (e.g. 2048 if training truncates)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HF dataset ID — used to count upstream shards (default: auto-detect from metadata.json)",
    )
    p.add_argument("--subset", default=None, help="Dataset subset (for upstream shard count)")
    p.add_argument("--target-tokens", type=int, default=None, help="Token budget to compute --n-shards for")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("part_*.parquet"))
    assert files, f"No part_*.parquet files found in {data_dir}"

    # Read texts
    texts: list[str] = []
    for f in files:
        table = pq.read_table(str(f), columns=[args.text_column])
        texts.extend(table[args.text_column].to_pylist())
        if len(texts) >= args.n:
            break
    texts = texts[: args.n]
    print(f"Loaded {len(texts):,} documents from {data_dir}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    total_chars = 0
    total_tokens = 0
    total_tokens_capped = 0

    for text in tqdm(texts, desc="Tokenizing"):
        n_chars = len(text)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        total_chars += n_chars
        total_tokens += n_tokens
        if args.max_tokens_per_sample is not None:
            total_tokens_capped += min(n_tokens, args.max_tokens_per_sample)
        else:
            total_tokens_capped += n_tokens

    assert total_tokens > 0, "All documents tokenized to 0 tokens — check your data"
    ratio = total_chars / total_tokens
    tokens_per_doc = total_tokens_capped / len(texts)

    print(f"\nTokenizer:               {args.tokenizer}")
    print(f"Documents sampled:       {len(texts):,}")
    print(f"Chars/token (aggregate): {ratio:.3f}")
    print(f"Tokens/doc (avg):        {tokens_per_doc:,.0f}")
    if args.max_tokens_per_sample is not None:
        uncapped = total_tokens / len(texts)
        print(f"Tokens/doc (uncapped):   {uncapped:,.0f}")
        print(f"Max tokens/sample:       {args.max_tokens_per_sample:,}")

    # Count upstream shards
    dataset_id = args.dataset
    if dataset_id is None:
        meta_path = data_dir / "metadata.json"
        if meta_path.exists():
            import json

            dataset_id = json.loads(meta_path.read_text()).get("source_dataset")
    if dataset_id is not None:
        print(f"\nCounting upstream shards for {dataset_id}...")
        ds = load_dataset(dataset_id, args.subset, split="train", streaming=True)
        n_upstream_shards = ds.n_shards
        print(f"Upstream shards:         {n_upstream_shards:,}")

        # Probe an upstream shard to count its rows (try a few in case some are bad)
        print("Probing upstream shard to count rows...")
        docs_per_upstream_shard = None
        for probe_idx in range(min(10, n_upstream_shards)):
            try:
                probe = ds.shard(num_shards=n_upstream_shards, index=probe_idx)
                docs_per_upstream_shard = sum(1 for _ in probe)
                break
            except Exception as e:
                print(f"  shard {probe_idx} failed ({type(e).__name__}), trying next...")
        assert docs_per_upstream_shard is not None, "All probed shards failed"
        tokens_per_upstream_shard = int(docs_per_upstream_shard * tokens_per_doc)

        print(f"Docs/upstream shard:     {docs_per_upstream_shard:,}")
        print(f"Est. tokens/shard:       {tokens_per_upstream_shard:,}")

        if args.target_tokens is not None:
            n_shards_needed = math.ceil(args.target_tokens / tokens_per_upstream_shard)
            print(f"\nFor {args.target_tokens:,} tokens: ~{n_shards_needed:,} shards")
            print(f"\nUsage:")
            print(f"  python -m preprocessing.download --n-shards {n_shards_needed}")
    else:
        print("\nPass --dataset to estimate tokens/shard and compute --n-shards")


if __name__ == "__main__":
    main()
