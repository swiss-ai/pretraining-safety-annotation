"""Source datasets for charter-aware SFT generation.

Loads user prompts from five dataset sources across eight subcategories:

- ``declare-lab/HarmfulQA``: 1960 hand-written harmful prompts (all used).
- ``allenai/WildChat-1M``: 1M+ in-the-wild user turns from GPT-3.5/4
  conversations. Parquet, 14 shards. We take the first user turn from
  English conversations only.
- ``allenai/wildguardmix``: ~87K prompts split into harmful (~46K) and
  benign (~41K) subcategories.
- ``allenai/wildjailbreak``: ~262K prompts across four subcategories:
  adversarial_harmful, adversarial_benign, vanilla_harmful, vanilla_benign.

The mix is deterministic given a seed: all of HarmfulQA (1960) plus
equal draws from the other 7 subcategories (no duplication). Total
target is 201,960.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from pipeline.log import logger

HARMFULQA_REPO = "declare-lab/HarmfulQA"
HARMFULQA_FILE = "data_for_hub.json"

WILDCHAT_REPO = "allenai/WildChat-1M"
WILDCHAT_NUM_SHARDS = 14
WILDCHAT_SHARD_FMT = "data/train-{i:05d}-of-{n:05d}.parquet"

WILDGUARD_REPO = "allenai/wildguardmix"
WILDGUARD_SPLIT = "wildguardtrain"

WILDJAILBREAK_REPO = "allenai/wildjailbreak"
WILDJAILBREAK_CONFIG = "train"

HARM_CATEGORIES = frozenset({
    "harmful", "benign", "adversarial_harmful", "adversarial_benign", "unknown",
})


@dataclass(frozen=True)
class SourcedPrompt:
    """A user prompt with source attribution."""
    source: str          # "harmfulqa", "wildchat", "wildguardmix", or "wildjailbreak"
    source_id: str       # original row id / hash
    user: str            # the user message text
    meta: dict           # source-specific metadata (topic, language, etc.)
    harm_category: str   # one of HARM_CATEGORIES — hint for the generator


def _hf_token() -> str:
    """Return the HF token from env. Crash if not set."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert token, "HF_TOKEN not set in environment"
    return token


def load_harmfulqa() -> list[SourcedPrompt]:
    """Load all 1960 HarmfulQA prompts. Returns one SourcedPrompt per row."""
    token = _hf_token()
    path = hf_hub_download(
        HARMFULQA_REPO, HARMFULQA_FILE, repo_type="dataset", token=token
    )
    rows = json.loads(Path(path).read_text())
    out = []
    for row in rows:
        out.append(SourcedPrompt(
            source="harmfulqa",
            source_id=str(row["id"]),
            user=row["question"],
            meta={"topic": row.get("topic"), "subtopic": row.get("subtopic")},
            harm_category="harmful",
        ))
    return out


def load_wildchat_shard(shard_idx: int) -> list[SourcedPrompt]:
    """Load one WildChat parquet shard, English-only, first user turn.

    Skips toxic-flagged and redacted conversations (signal noise we don't
    want in SFT seed prompts). Multi-turn conversations contribute only
    their first user turn.
    """
    assert 0 <= shard_idx < WILDCHAT_NUM_SHARDS, f"shard_idx out of range: {shard_idx}"
    token = _hf_token()
    path = hf_hub_download(
        WILDCHAT_REPO,
        WILDCHAT_SHARD_FMT.format(i=shard_idx, n=WILDCHAT_NUM_SHARDS),
        repo_type="dataset",
        token=token,
    )
    table = pq.read_table(
        path, columns=["conversation_hash", "conversation", "language", "toxic", "redacted"]
    )
    out = []
    for row in table.to_pylist():
        if row["language"] != "English":
            continue
        if row["toxic"] or row["redacted"]:
            continue
        msgs = row["conversation"] or []
        if not msgs:
            continue
        first = msgs[0]
        # WildChat rows are dicts with 'role' and 'content' keys.
        if first.get("role") != "user":
            continue
        content = (first.get("content") or "").strip()
        if not content:
            continue
        out.append(SourcedPrompt(
            source="wildchat",
            source_id=row["conversation_hash"],
            user=content,
            meta={"language": "English", "shard": shard_idx},
            harm_category="unknown",
        ))
    return out


def load_wildguardmix_all() -> dict[str, list[SourcedPrompt]]:
    """Load WildGuardMix train split, returning harmful and benign pools."""
    from datasets import load_dataset

    ds = load_dataset(WILDGUARD_REPO, WILDGUARD_SPLIT, split="train")
    out: dict[str, list[SourcedPrompt]] = {"harmful": [], "benign": []}
    for i, row in enumerate(ds):
        label = row.get("prompt_harm_label")
        if label == "harmful":
            harm_category = "harmful"
        elif label == "unharmful":
            harm_category = "benign"
        else:
            continue
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        out[harm_category].append(SourcedPrompt(
            source="wildguardmix",
            source_id=str(i),
            user=prompt,
            meta={
                "adversarial": row.get("adversarial"),
                "prompt_harm_label": label,
                "subcategory": row.get("subcategory"),
            },
            harm_category=harm_category,
        ))
    return out


_WILDJAILBREAK_DATA_TYPES = (
    "adversarial_harmful", "adversarial_benign",
    "vanilla_harmful", "vanilla_benign",
)

_WJB_HARM_CATEGORY = {
    "adversarial_harmful": "adversarial_harmful",
    "adversarial_benign": "adversarial_benign",
    "vanilla_harmful": "harmful",
    "vanilla_benign": "benign",
}


def load_wildjailbreak_all() -> dict[str, list[SourcedPrompt]]:
    """Load WildJailbreak train split, returning a dict keyed by data_type.

    Uses streaming to avoid PyArrow schema inference issues (the
    ``adversarial`` column has mixed None/string types that trip up
    Arrow's type guesser on full download).
    """
    from datasets import load_dataset

    ds = load_dataset(WILDJAILBREAK_REPO, WILDJAILBREAK_CONFIG, split="train", streaming=True)
    out: dict[str, list[SourcedPrompt]] = {dt: [] for dt in _WILDJAILBREAK_DATA_TYPES}
    for i, row in enumerate(ds):
        data_type = row.get("data_type")
        if data_type not in out:
            continue
        prompt = (row.get("vanilla") or "").strip()
        if not prompt:
            continue
        out[data_type].append(SourcedPrompt(
            source="wildjailbreak",
            source_id=f"wjb-{data_type}-{i}",
            user=row["adversarial"] if row.get("adversarial") else prompt,
            meta={
                "data_type": data_type,
                "vanilla_prompt": prompt,
            },
            harm_category=_WJB_HARM_CATEGORY[data_type],
        ))
    return out


HARMFULQA_SIZE = 1_960
N_SLOTS = 8  # WildChat counts as 2 slots, 6 other subcategories count as 1 each
DEFAULT_TOTAL = N_SLOTS * 37_500 + HARMFULQA_SIZE  # 301,960


def sample_mix(
    n: int = DEFAULT_TOTAL,
    seed: int = 42,
    wildchat_shards: tuple[int, ...] | None = None,
    max_prompt_chars: int = 4000,
    exclude_sources: frozenset[str] = frozenset(),
) -> list[SourcedPrompt]:
    """Sample prompts across 8 subcategories (no duplication).

    Layout (8 slots of ``per_sub`` each, plus all of HarmfulQA):
      - HarmfulQA: all available (harmful)
      - WildChat: 2 * per_sub (benign) — double weight
      - WildGuardMix harmful: per_sub
      - WildGuardMix benign: per_sub
      - WildJailbreak adversarial_harmful: per_sub
      - WildJailbreak adversarial_benign: per_sub
      - WildJailbreak vanilla_harmful: per_sub
      - WildJailbreak vanilla_benign: per_sub

    ``per_sub`` is computed dynamically as ``(n - HARMFULQA_SIZE) // N_SLOTS``.
    Draws are capped at pool size (no replacement / duplication).
    ``n`` controls the target; the actual count may be slightly less if
    a subcategory pool is smaller than ``per_sub``.

    For small ``n`` (e.g. iterate testing), HarmfulQA is skipped and
    ``per_sub`` is derived from ``n // N_SLOTS``.

    Prompts longer than ``max_prompt_chars`` are dropped.
    """
    rng = random.Random(seed)
    small = n <= HARMFULQA_SIZE * N_SLOTS
    if small:
        per_sub = max(1, n // N_SLOTS)
    else:
        per_sub = (n - HARMFULQA_SIZE) // N_SLOTS

    def _draw(pool: list[SourcedPrompt], k: int) -> list[SourcedPrompt]:
        pool = [p for p in pool if len(p.user) <= max_prompt_chars]
        draw_n = min(k, len(pool))
        return rng.sample(pool, draw_n)

    # --- HarmfulQA (all at scale, skip for small n) ---
    if small or "harmfulqa" in exclude_sources:
        harmfulqa_picks: list[SourcedPrompt] = []
    else:
        harmfulqa_pool = load_harmfulqa()
        harmfulqa_picks = _draw(harmfulqa_pool, len(harmfulqa_pool))

    # --- WildChat (2x weight) ---
    wildchat_target = per_sub * 2
    if "wildchat" in exclude_sources:
        wildchat_picks: list[SourcedPrompt] = []
    else:
        if wildchat_shards is None:
            n_shards_needed = max(1, (wildchat_target + 24_999) // 25_000)
            wildchat_shards = tuple(range(min(n_shards_needed, WILDCHAT_NUM_SHARDS)))
        wild_pool: list[SourcedPrompt] = []
        for s in wildchat_shards:
            wild_pool.extend(load_wildchat_shard(s))
        wildchat_picks = _draw(wild_pool, wildchat_target)

    # --- WildGuardMix (single load) ---
    if "wildguardmix" in exclude_sources:
        wgm_harmful_picks: list[SourcedPrompt] = []
        wgm_benign_picks: list[SourcedPrompt] = []
    else:
        wgm_all = load_wildguardmix_all()
        wgm_harmful_picks = _draw(wgm_all["harmful"], per_sub)
        wgm_benign_picks = _draw(wgm_all["benign"], per_sub)

    # --- WildJailbreak (4 subcategories, single load) ---
    if "wildjailbreak" in exclude_sources:
        wjb_adv_harmful_picks: list[SourcedPrompt] = []
        wjb_adv_benign_picks: list[SourcedPrompt] = []
        wjb_van_harmful_picks: list[SourcedPrompt] = []
        wjb_van_benign_picks: list[SourcedPrompt] = []
    else:
        wjb_all = load_wildjailbreak_all()
        wjb_adv_harmful_picks = _draw(wjb_all["adversarial_harmful"], per_sub)
        wjb_adv_benign_picks = _draw(wjb_all["adversarial_benign"], per_sub)
        wjb_van_harmful_picks = _draw(wjb_all["vanilla_harmful"], per_sub)
        wjb_van_benign_picks = _draw(wjb_all["vanilla_benign"], per_sub)

    picks = (
        harmfulqa_picks
        + wildchat_picks
        + wgm_harmful_picks
        + wgm_benign_picks
        + wjb_adv_harmful_picks
        + wjb_adv_benign_picks
        + wjb_van_harmful_picks
        + wjb_van_benign_picks
    )
    rng.shuffle(picks)
    logger.info(
        "sample_mix: {} total (harmfulqa={}, wildchat={}, wgm_harmful={}, "
        "wgm_benign={}, wjb_adv_harmful={}, wjb_adv_benign={}, "
        "wjb_van_harmful={}, wjb_van_benign={})",
        len(picks), len(harmfulqa_picks), len(wildchat_picks),
        len(wgm_harmful_picks), len(wgm_benign_picks),
        len(wjb_adv_harmful_picks), len(wjb_adv_benign_picks),
        len(wjb_van_harmful_picks), len(wjb_van_benign_picks),
    )
    return picks
