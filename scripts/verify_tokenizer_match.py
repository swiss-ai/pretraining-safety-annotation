"""Verify that transformers AutoTokenizer matches annotated.bin's tokens.

Training data in ``annotated.bin`` was tokenized with the Rust
``tokenizers`` library via datatrove's ``DocumentTokenizer``.  Charter scale
and the backfill script use ``transformers.AutoTokenizer`` (fast backend)
with ``add_special_tokens=False``.  Both paths load the same SmolLM2
``tokenizer.json``, so content-token IDs should match exactly — but
there are edge cases (unicode normalization, config overrides) where
they could diverge.

This script picks N random docs, tokenizes their raw text with
transformers, and compares the first ``token_length`` tokens against
the corresponding window in ``annotated.bin``.  100% match is the bar.

Layout of ``annotated.bin``:
  - shuffled post-tokenization, each doc padded to a 2049-token window
  - token dtype = uint16 (SmolLM2 vocab < 65536)
  - sidecar row ``i`` corresponds 1:1 to window ``i``
  - within each window: ``content_tokens[0:token_length]``,
    ``EOS (id=0) at [token_length]``, zero-padded afterward

Samples a contiguous block of rows (not scattered) so we only touch
one row group of the sidecar — reading a full row group for hundreds of
random samples on Lustre costs several GB of read per row group and is
prohibitively slow.  A contiguous block representative enough to catch
systematic tokenizer-config drift.

Usage:
    uv run python scripts/verify_tokenizer_match.py \\
        --annotated-bin /iopsstor/.../annotated/annotated.bin \\
        --sidecar /iopsstor/.../annotated/sidecar.parquet \\
        [--n 500] [--start 0] [--window-size 2049]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.tokenizer import _get_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotated-bin", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--n", type=int, default=500,
                        help="How many contiguous rows to check")
    parser.add_argument("--start", type=int, default=0,
                        help="First row index to check")
    parser.add_argument("--window-size", type=int, default=2049)
    args = parser.parse_args()

    pf = pq.ParquetFile(args.sidecar)
    n_rows = pf.metadata.num_rows
    end = min(args.start + args.n, n_rows)
    logger.info("Sidecar: {} rows; checking contiguous block [{}, {})",
                n_rows, args.start, end)

    # Find which row group contains args.start, and read only what we need
    row_offset = 0
    rg_idx = None
    rg_start = 0
    for i in range(pf.metadata.num_row_groups):
        rg_n = pf.metadata.row_group(i).num_rows
        if row_offset <= args.start < row_offset + rg_n:
            rg_idx = i
            rg_start = row_offset
            break
        row_offset += rg_n
    if rg_idx is None:
        logger.error("start index {} beyond sidecar rows {}", args.start, n_rows)
        sys.exit(2)

    logger.info("Reading row group {} (starts at row {})...", rg_idx, rg_start)
    table = pf.read_row_group(rg_idx, columns=["doc_id", "text", "token_length"])
    col = table.to_pydict()

    # Memory-map annotated.bin as uint16 (SmolLM2 vocab < 65536)
    bin_mmap = np.memmap(args.annotated_bin, dtype=np.uint16, mode="r")
    expected_tokens = n_rows * args.window_size
    if bin_mmap.shape[0] != expected_tokens:
        logger.warning(
            "annotated.bin length ({}) != n_rows * window_size ({} * {} = {}). "
            "Proceeding but results may be misaligned.",
            bin_mmap.shape[0], n_rows, args.window_size, expected_tokens,
        )

    tokenizer = _get_tokenizer()

    n_match = 0
    n_mismatch = 0
    n_length_mismatch = 0
    mismatch_examples: list[dict] = []

    for global_idx in range(args.start, end):
        local = global_idx - rg_start
        if local >= table.num_rows:
            logger.warning(
                "global_idx {} crosses row group boundary — stopping early "
                "(add --start within a single row group for larger samples)",
                global_idx,
            )
            break
        doc_id = col["doc_id"][local]
        text = col["text"][local]
        token_length = col["token_length"][local]

        tt_ids = tokenizer.encode(text, add_special_tokens=False).ids

        win_start = global_idx * args.window_size
        bin_content = bin_mmap[win_start : win_start + token_length].tolist()

        if len(tt_ids) < token_length:
            n_length_mismatch += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append({
                    "type": "length",
                    "global_idx": global_idx,
                    "doc_id": doc_id,
                    "transformers_tokens": len(tt_ids),
                    "sidecar_token_length": token_length,
                })
            continue

        tt_content = tt_ids[:token_length]
        if tt_content == bin_content:
            n_match += 1
        else:
            n_mismatch += 1
            if len(mismatch_examples) < 5:
                for p, (a, b) in enumerate(zip(tt_content, bin_content)):
                    if a != b:
                        break
                mismatch_examples.append({
                    "type": "token",
                    "global_idx": global_idx,
                    "doc_id": doc_id,
                    "token_length": token_length,
                    "first_diff_pos": p,
                    "transformers": tt_content[p : p + 5],
                    "bin": bin_content[p : p + 5],
                })

    print("\n=== pipeline.tokenizer vs annotated.bin match ===")
    print(f"  checked:          {n_match + n_mismatch + n_length_mismatch}")
    print(f"  exact match:      {n_match}")
    print(f"  token mismatch:   {n_mismatch}")
    print(f"  length mismatch:  {n_length_mismatch}")
    if mismatch_examples:
        print("\nExamples:")
        for ex in mismatch_examples:
            print(f"  {ex}")

    sys.exit(0 if (n_mismatch == 0 and n_length_mismatch == 0) else 2)


if __name__ == "__main__":
    main()
