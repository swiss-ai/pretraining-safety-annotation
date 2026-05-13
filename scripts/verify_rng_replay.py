"""Verify that RNG replay reproduces the stored reflection_position.

The backfill script (scripts/backfill_reflection_token_index.py) adds a
``reflection_token_index`` column to existing results.jsonl files by
re-running ``compute_reflection_point_tokens`` with the same
``random.Random(f"{reflection_seed}_{doc_id}")`` seed the generator used.
That strategy is only safe if replay reproduces the exact same sampling
decision — which in turn requires the tokenizer state and the text
tokenized at generation time to be identical to what we see now.

This script samples N rows from a completed rank, replays the RNG, and
compares the computed char offset against the stored
``reflection_position``.  Any mismatch is a hard fail: investigate before
running the backfill.

The 10M run (EXP-001) was generated while ``generate.py`` still pre-sliced
text to ``max_text_tokens * 10`` chars before tokenizing.  Pass
``--legacy-pre-slice`` to reproduce that behaviour when verifying legacy
rows.  The backfill script has the same flag and must use the same value
as this verifier.

Usage:
    uv run python scripts/verify_rng_replay.py \\
        --run reflections \\
        --rank 0 \\
        [--sample 1000] \\
        [--legacy-pre-slice] \\
        [--reflection-seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from pipeline.config import load_config
from pipeline.log import logger
from pipeline.charter.scale.sidecar import apply_legacy_pre_slice, load_rank_docs
from pipeline.tokenizer import compute_reflection_point_tokens

_LENGTH_BUCKETS: list[tuple[int, str]] = [
    (2_000, "<2K"),
    (10_000, "2K-10K"),
    (19_200, "10K-19.2K"),
    (float("inf"), ">19.2K"),
]


def _length_bucket(text_len: int) -> str:
    for threshold, label in _LENGTH_BUCKETS:
        if text_len < threshold:
            return label
    return _LENGTH_BUCKETS[-1][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument(
        "--legacy-pre-slice",
        action="store_true",
        help="Apply the pre-slice that generate.py used to do (char_limit = max_tokens * 10)",
    )
    parser.add_argument("--reflection-seed", type=int, default=None,
                        help="Override; defaults to config value")
    args, overrides = parser.parse_known_args()

    cfg = load_config(overrides or None)
    refl_seed = args.reflection_seed if args.reflection_seed is not None else cfg.charter.scale.reflection_seed

    run_dir = Path(cfg.charter.scale.output_dir) / args.run
    results_path = run_dir / f"{args.rank:05d}" / "results.jsonl"
    if not results_path.exists():
        logger.error("No results.jsonl for rank {} at {}", args.rank, results_path)
        sys.exit(1)

    logger.info("Loading rank {} texts from sidecar...", args.rank)
    rank_docs = load_rank_docs(
        cfg.charter.scale.sidecar_path, args.rank, cfg.charter.scale.rows_per_task
    )

    # Read results and sample
    logger.info("Reading results.jsonl...")
    rows: list[dict] = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if args.sample < len(rows):
        rng = random.Random(0xCAFE)
        sample_rows = rng.sample(rows, args.sample)
    else:
        sample_rows = rows

    logger.info(
        "Checking {} rows (seed={}, legacy_pre_slice={})",
        len(sample_rows), refl_seed, args.legacy_pre_slice,
    )

    n_match = 0
    n_mismatch = 0
    mismatch_examples: list[dict] = []
    per_bucket_total: Counter = Counter()
    per_bucket_mismatch: Counter = Counter()

    for row in sample_rows:
        gidx = row["global_row_idx"]
        doc_id = row["doc_id"]
        stored_rp = row["reflection_position"]

        doc = rank_docs.get(gidx)
        if doc is None:
            logger.error("No sidecar text for global_row_idx={} (doc_id={})", gidx, doc_id)
            n_mismatch += 1
            continue
        sidecar_doc_id, text, token_length = doc  # token_length used in mismatch examples below
        if sidecar_doc_id != doc_id:
            logger.error(
                "doc_id mismatch at gidx={}: sidecar={!r}, results={!r}",
                gidx, sidecar_doc_id, doc_id,
            )
            n_mismatch += 1
            continue

        bucket = _length_bucket(len(text))
        per_bucket_total[bucket] += 1

        tok_text = apply_legacy_pre_slice(text, cfg.max_tokens) if args.legacy_pre_slice else text

        rp_rng = random.Random(f"{refl_seed}_{doc_id}")
        # The 10M run passed cfg.max_tokens (1920) as the run-level cap,
        # not the per-doc token_length — mirror that.
        char_offset, _tok_idx = compute_reflection_point_tokens(
            tok_text, rp_rng, max_tokens=cfg.max_tokens
        )

        if char_offset == stored_rp:
            n_match += 1
        else:
            n_mismatch += 1
            per_bucket_mismatch[bucket] += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append({
                    "gidx": gidx,
                    "doc_id": doc_id,
                    "text_len": len(text),
                    "token_length": token_length,
                    "stored_rp": stored_rp,
                    "replay_rp": char_offset,
                    "bucket": bucket,
                })

    print(f"\n=== RNG replay verification ===")
    print(f"  checked: {n_match + n_mismatch}")
    print(f"  match:    {n_match}")
    print(f"  mismatch: {n_mismatch}")
    print(f"\nPer-length-bucket mismatch rate:")
    for _, bucket in _LENGTH_BUCKETS:
        total = per_bucket_total[bucket]
        mism = per_bucket_mismatch[bucket]
        if total:
            print(f"  {bucket:10s}  {mism:5d} / {total:5d}  ({100*mism/total:5.2f}%)")
    if mismatch_examples:
        print("\nMismatch examples:")
        for ex in mismatch_examples:
            print(f"  {ex}")

    sys.exit(0 if n_mismatch == 0 else 2)


if __name__ == "__main__":
    main()
