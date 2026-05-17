"""Dataset loading and sampling utilities.

Provides cached access to the Dolma3 annotation sample dataset
from HuggingFace and text sampling for evaluation pipelines.
"""

from __future__ import annotations

import json
import math
import random

from pipeline.config import PIPELINE_DATA_DIR
from pipeline.log import logger
from pipeline.storage import compute_item_id
from pipeline.tokenizer import truncate_to_max_tokens

DATASET = "jkminder/Dolma3_mix_annotation_sample"
DATASET_CACHE_PATH = PIPELINE_DATA_DIR / "dolma3_cache.jsonl"
DATASET_CACHE_SIZE = 4096


def load_dataset_cache(seed: int) -> list[dict]:
    """Load cached Dolma3 texts, or stream from HF and cache locally.

    Returns a flat list of {text, safety_score} dicts.
    """
    if DATASET_CACHE_PATH.exists():
        records = []
        with open(DATASET_CACHE_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            logger.info("Loaded {} items from dataset cache", len(records))
            return records

    logger.info(
        "Building dataset cache ({} items from {})...", DATASET_CACHE_SIZE, DATASET
    )
    import itertools
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset(DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    rows = list(itertools.islice(ds, DATASET_CACHE_SIZE))

    records = [
        {"text": r["text"], "safety_score": int(r["safety_score"])} for r in rows
    ]
    DATASET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_CACHE_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Cached {} items to {}", len(records), DATASET_CACHE_PATH)
    return records


def sample_texts(
    n: int,
    seed: int,
    max_tokens: int,
    exclude_ids: set[str] | None = None,
    min_safety_score: int | None = None,
    max_safety_score: int | None = None,
) -> list[dict]:
    """Sample texts from the Dolma3 dataset cache.

    Returns a list of {item_id, text, safety_score} dicts, each truncated
    to max_tokens. No gold set, no canaries, no reflection point.

    If *min_safety_score* / *max_safety_score* are set, only rows whose
    ``safety_score`` falls in the inclusive range are eligible (used by the
    summaries iterate CLI to separate harmful from harmless evals).
    """
    if exclude_ids is None:
        exclude_ids = set()

    rng = random.Random(seed)
    cache = load_dataset_cache(seed)
    if min_safety_score is not None:
        cache = [r for r in cache if (r.get("safety_score") or 0) >= min_safety_score]
    if max_safety_score is not None:
        cache = [r for r in cache if (r.get("safety_score") or 0) <= max_safety_score]
    rng.shuffle(cache)

    items: list[dict] = []
    for row in cache:
        if len(items) >= n:
            break
        text = truncate_to_max_tokens(row["text"], max_tokens)
        item_id = compute_item_id(text)
        if item_id in exclude_ids:
            continue
        items.append(
            {
                "item_id": item_id,
                "text": text,
                "safety_score": row.get("safety_score"),
            }
        )
        exclude_ids.add(item_id)

    assert (
        len(items) >= n
    ), f"Could only sample {len(items)}/{n} items (cache has {len(cache)})"
    return items[:n]


def _list_dataset_shards() -> list[str]:
    """List the data shard files in the Dolma3 dataset repo on HuggingFace.

    Returns shard paths within the repo (e.g. "data/shard-00000.parquet").
    Filters to .parquet / .jsonl files under the data/ or train/ subtree.
    """
    import huggingface_hub

    files = huggingface_hub.list_repo_files(DATASET, repo_type="dataset")
    shards = [
        f
        for f in files
        if (f.endswith(".parquet") or f.endswith(".jsonl"))
        and (f.startswith("data/") or f.startswith("train/") or "/" not in f)
    ]
    if not shards:
        # Fallback: any parquet/jsonl file in the repo
        shards = [f for f in files if f.endswith(".parquet") or f.endswith(".jsonl")]
    shards.sort()
    return shards


def _reservoir_sample(rows, k: int, rng: random.Random) -> list:
    """Reservoir-sample k items from an iterable in a single pass.

    O(k) memory, O(len(rows)) time, uniform sampling within the iterable.
    """
    reservoir: list = []
    for i, row in enumerate(rows):
        if i < k:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = row
    return reservoir


def sample_diverse(n: int, seed: int, max_tokens: int) -> dict:
    """Sample n items diversely from the full Dolma3 dataset.

    Bypasses the dolma3_cache.jsonl entirely. Loads each shard one at a time
    (memory bounded) and reservoir-samples from each, using an RNG seeded
    with (seed, shard_path) so every shard contributes a stable sample.

    Returns:
        {"items": [{item_id, text, safety_score}, ...],
         "dataset_revision": "<sha>"}
    """
    import huggingface_hub
    import datasets as _datasets_mod

    shard_paths = _list_dataset_shards()
    assert shard_paths, f"No shards found in dataset {DATASET}"

    # Pull a few extra per shard to give the final shuffle headroom and to
    # offset shards that have fewer than per_shard rows.
    per_shard = max(1, math.ceil(n / len(shard_paths)) + 4)

    pooled: list[dict] = []
    for shard in shard_paths:
        # Stable per-shard RNG: depends on both seed AND shard path so the
        # sample for a given shard is reproducible but different across shards.
        shard_rng = random.Random(f"sample_diverse_v1::{seed}::{shard}")
        ds = _datasets_mod.load_dataset(
            DATASET,
            data_files=[shard],
            split="train",
            verification_mode="no_checks",
        )
        sampled = _reservoir_sample(iter(ds), per_shard, shard_rng)
        for row in sampled:
            pooled.append(
                {
                    "text": row["text"],
                    "safety_score": (
                        int(row.get("safety_score", 0))
                        if row.get("safety_score") is not None
                        else None
                    ),
                }
            )

    final_rng = random.Random(f"sample_diverse_final_v1::{seed}")
    final_rng.shuffle(pooled)

    if len(pooled) < n:
        raise RuntimeError(
            f"sample_diverse: only pooled {len(pooled)} rows from "
            f"{len(shard_paths)} shards, but {n} requested. "
            f"Increase per_shard buffer or check shard sizes."
        )

    items: list[dict] = []
    for row in pooled[:n]:
        text = truncate_to_max_tokens(row["text"], max_tokens)
        items.append(
            {
                "item_id": compute_item_id(text),
                "text": text,
                "safety_score": row.get("safety_score"),
            }
        )

    info = huggingface_hub.dataset_info(DATASET)
    dataset_revision = getattr(info, "sha", None) or ""

    return {"items": items, "dataset_revision": dataset_revision}
