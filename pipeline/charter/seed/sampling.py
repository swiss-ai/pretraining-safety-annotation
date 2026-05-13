"""Stratified sampling from locuslab/fineweb_annotated score subsets."""

import random

from pipeline.config import PIPELINE_DATA_DIR, CharterSeedConfig
from pipeline.fineweb import load_or_build_fineweb_cache
from pipeline.storage import compute_item_id
from pipeline.tokenizer import compute_reflection_point, truncate_to_max_tokens

CHARTER_SEED_CACHE_PATH = PIPELINE_DATA_DIR / "charter_seed_fineweb_cache.jsonl"
CHARTER_SEED_CACHE_PER_SUBSET = 100


def _load_or_build_cache(seed_cfg: CharterSeedConfig, seed: int) -> dict[str, list[dict]]:
    """Load cached FineWeb texts by subset, or stream from HF and cache locally.

    Returns {subset: [{text, subset}, ...]} with CHARTER_SEED_CACHE_PER_SUBSET items per subset.
    """
    flat = load_or_build_fineweb_cache(
        cache_path=CHARTER_SEED_CACHE_PATH,
        dataset=seed_cfg.dataset,
        subsets=seed_cfg.subsets,
        per_subset=CHARTER_SEED_CACHE_PER_SUBSET,
        seed=seed,
    )
    by_subset: dict[str, list[dict]] = {}
    for rec in flat:
        by_subset.setdefault(rec["subset"], []).append(rec)
    return by_subset


def sample_items(
    n_per_subset: int,
    seed: int = 42,
    seed_cfg: CharterSeedConfig | None = None,
    max_tokens: int = 3840,
) -> list[dict]:
    """Sample n_per_subset items from each fineweb_annotated score subset.

    Uses a local JSONL cache to avoid repeated HF downloads. On first call,
    streams from HF and builds the cache. Each text is truncated to max_tokens
    before computing the reflection point. Returns items with item_id, subset,
    text, and reflection_point.
    """
    if seed_cfg is None:
        from pipeline.config import load_config
        seed_cfg = load_config().charter.seed

    cache = _load_or_build_cache(seed_cfg, seed)
    rng = random.Random(seed)
    items = []

    for subset in seed_cfg.subsets:
        cached_rows = cache.get(subset, [])
        assert len(cached_rows) >= n_per_subset, (
            f"Cache has {len(cached_rows)} items for {subset}, need {n_per_subset}. "
            f"Delete {CHARTER_SEED_CACHE_PATH} to rebuild."
        )
        selected = cached_rows[:n_per_subset]

        for row in selected:
            text = truncate_to_max_tokens(row["text"], max_tokens)
            assert isinstance(text, str) and len(text) > 0, f"Empty text in {subset}"
            items.append({
                "item_id": compute_item_id(text),
                "subset": subset,
                "text": text,
                "reflection_point": compute_reflection_point(text, rng),
            })

    assert len(items) > 0, "No fineweb items loaded"
    return items
