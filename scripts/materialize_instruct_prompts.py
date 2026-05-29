"""Materialize a WildChat-only prompts.parquet for the instruct top-up run.

Reaching the 300k WildChat instruct target requires annotating ~225k MORE
WildChat prompts beyond the 74,810 already in jkminder/model-raising-pb-300k-3c-sft.
This samples fresh WildChat first-user prompts from shards NOT used by the
original run (3-13), deduped against the conversation_hashes already annotated,
and writes a prompts.parquet identical in schema to
`pipeline.sft.single_turn.prompts_writer` so the existing SLURM `submit`
flow can consume it directly.

Filters mirror the original sampler exactly (via `load_wildchat_shard` +
the <=4000-char cap applied in `sample_mix`):
  English, non-toxic, non-redacted, first turn is user, non-empty, len <= 4000.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.sft.single_turn.data import load_wildchat_shard

PARENT_SFT = Path(
    "/iopsstor/scratch/cscs/jminder/.cache/huggingface/hub/"
    "datasets--jkminder--model-raising-pb-300k-3c-sft/snapshots/"
    "55b53c69e98a06db221873bf51fb6ffbefb11898/data/train-00000-of-00001.parquet"
)
EVAL_REPO = "jkminder/model-raising-pbsft-eval"
MAX_PROMPT_CHARS = 4000


def load_annotated_hashes() -> set[str]:
    """conversation_hashes (source_id) of WildChat rows already annotated."""
    rows = pq.read_table(PARENT_SFT, columns=["source", "source_id"]).to_pylist()
    return {r["source_id"] for r in rows if r["source"] == "wildchat"}


def load_eval_wildchat_hashes() -> set[str]:
    """WildChat source_ids reserved for the eval set — MUST NOT leak into training."""
    from datasets import load_dataset

    ev = load_dataset(EVAL_REPO, split="train")
    return {i for s, i in zip(ev["source"], ev["source_id"]) if s == "wildchat"}


def _content_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(r["source_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(r["user"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="prompts.parquet path")
    ap.add_argument("--n", type=int, default=230_000, help="target new prompts")
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--shards", default="3-13", help="inclusive shard range, e.g. 3-13")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.shards.split("-"))
    shards = list(range(lo, hi + 1))

    annotated = load_annotated_hashes()
    eval_hashes = load_eval_wildchat_hashes()
    exclude = annotated | eval_hashes
    logger.info(
        "excluding {} hashes ({} already-annotated + {} eval-reserved)",
        len(exclude), len(annotated), len(eval_hashes),
    )

    pool: list = []
    seen: set[str] = set()
    n_dup_annotated = 0
    n_dup_within = 0
    n_too_long = 0
    for s in shards:
        sps = load_wildchat_shard(s)
        for sp in sps:
            if len(sp.user) > MAX_PROMPT_CHARS:
                n_too_long += 1
                continue
            if sp.source_id in exclude:
                n_dup_annotated += 1
                continue
            if sp.source_id in seen:
                n_dup_within += 1
                continue
            seen.add(sp.source_id)
            pool.append(sp)
        logger.info("shard {}: pool now {}", s, len(pool))

    logger.info(
        "eligible new pool: {} (skipped {} annotated, {} within-dup, {} too-long)",
        len(pool), n_dup_annotated, n_dup_within, n_too_long,
    )
    assert len(pool) >= args.n, f"pool {len(pool)} < target {args.n}; widen --shards"

    rng = random.Random(args.seed)
    picks = rng.sample(pool, args.n)

    rows = []
    for i, sp in enumerate(picks):
        rows.append({
            "global_row_idx": i,
            "source": sp.source,
            "source_id": sp.source_id,
            "user": sp.user,
            "meta": json.dumps(sp.meta or {}, ensure_ascii=False),
            "harm_category": sp.harm_category,
        })

    schema = pa.schema([
        ("global_row_idx", pa.int64()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("user", pa.large_string()),
        ("meta", pa.string()),
        ("harm_category", pa.string()),
    ])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out)

    fingerprint = {
        "n": len(rows),
        "seed": args.seed,
        "shards": shards,
        "sources": {"wildchat": len(rows)},
        "harm_categories": {"unknown": len(rows)},
        "excluded_annotated": len(annotated),
        "excluded_eval": len(eval_hashes),
        "content_sha256": _content_sha256(rows),
    }
    (out.parent / "prompts_fingerprint.json").write_text(json.dumps(fingerprint, indent=2))
    logger.info("wrote {} ({} rows). fingerprint: {}", out, len(rows), fingerprint)
    print(json.dumps(fingerprint, indent=2))


if __name__ == "__main__":
    main()
