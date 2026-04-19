"""Source datasets for charter-aware SFT generation.

Loads user prompts from three sources and mixes them in equal thirds:

- ``declare-lab/HarmfulQA``: 1960 hand-written harmful prompts. Single
  JSON file, `question` field.
- ``allenai/WildChat-1M``: 1M+ in-the-wild user turns from GPT-3.5/4
  conversations. Parquet, 14 shards. We take the first user turn from
  English conversations only.
- ``allenai/wildguardmix``: ~87K prompts with harm/adversarial labels.
  We use the ``wildguardtrain`` split, prompt-only items.

The mix is deterministic given a seed: we draw N/3 from each source
independently, then concatenate and shuffle.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HARMFULQA_REPO = "declare-lab/HarmfulQA"
HARMFULQA_FILE = "data_for_hub.json"

WILDCHAT_REPO = "allenai/WildChat-1M"
WILDCHAT_NUM_SHARDS = 14
WILDCHAT_SHARD_FMT = "data/train-{i:05d}-of-{n:05d}.parquet"

WILDGUARD_REPO = "allenai/wildguardmix"
WILDGUARD_SPLIT = "wildguardtrain"


@dataclass(frozen=True)
class SourcedPrompt:
    """A user prompt with source attribution."""
    source: str          # "harmfulqa", "wildchat", or "wildguardmix"
    source_id: str       # original row id / hash
    user: str            # the user message text
    meta: dict           # source-specific metadata (topic, language, etc.)


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
        ))
    return out


def load_wildguardmix() -> list[SourcedPrompt]:
    """Load WildGuardMix train split prompts."""
    from datasets import load_dataset

    ds = load_dataset(WILDGUARD_REPO, WILDGUARD_SPLIT, split="train")
    out = []
    for i, row in enumerate(ds):
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        out.append(SourcedPrompt(
            source="wildguardmix",
            source_id=str(i),
            user=prompt,
            meta={
                "adversarial": row.get("adversarial"),
                "prompt_harm_label": row.get("prompt_harm_label"),
                "subcategory": row.get("subcategory"),
            },
        ))
    return out


def _sample_with_replacement(
    pool: list[SourcedPrompt], k: int, rng: random.Random,
) -> list[SourcedPrompt]:
    """Draw k items from pool, with replacement if k > len(pool)."""
    if k <= len(pool):
        return rng.sample(pool, k)
    picks = []
    counts: dict[str, int] = {}
    for _ in range(k):
        base = rng.choice(pool)
        c = counts.get(base.source_id, 0)
        counts[base.source_id] = c + 1
        sid = base.source_id if c == 0 else f"{base.source_id}#{c}"
        picks.append(SourcedPrompt(
            source=base.source, source_id=sid, user=base.user, meta=base.meta,
        ))
    return picks


def sample_mix(
    n: int,
    seed: int = 42,
    wildchat_shards: tuple[int, ...] | None = None,
    max_prompt_chars: int = 4000,
) -> list[SourcedPrompt]:
    """Sample N prompts, equal thirds from HarmfulQA, WildChat, WildGuardMix.

    HarmfulQA has only 1960 questions. When the per-source draw exceeds
    pool size, samples with replacement (qwen at t=1.0 produces diverse
    outputs across repeated calls, so this is acceptable for the SFT corpus).

    WildChat shards are loaded lazily based on need: we iterate shards
    until the English non-toxic pool is large enough. Each shard has
    ~30-40K usable rows after filtering. Pass an explicit tuple via
    `wildchat_shards` to override.

    Prompts longer than `max_prompt_chars` are dropped (long WildChat
    prompts are usually pasted essays, not real questions).

    Returns a shuffled list of SourcedPrompts; len == n exactly. To make
    sample_with_replacement deterministic across reruns, source_id of a
    duplicate is suffixed with the draw index (e.g. ``42#3`` for the 4th
    draw of question id 42).
    """
    assert n >= 3 and n % 3 == 0, f"n must be divisible by 3 and >= 3, got {n}"
    rng = random.Random(seed)
    third = n // 3

    # --- HarmfulQA ---
    harmful_pool = [p for p in load_harmfulqa() if len(p.user) <= max_prompt_chars]
    harmful_picks = _sample_with_replacement(harmful_pool, third, rng)

    # --- WildChat ---
    if wildchat_shards is None:
        n_shards_needed = max(1, (third + 24999) // 25000)
        wildchat_shards = tuple(range(min(n_shards_needed, WILDCHAT_NUM_SHARDS)))

    wild_pool = []
    for s in wildchat_shards:
        wild_pool.extend(load_wildchat_shard(s))
    wild_pool = [p for p in wild_pool if len(p.user) <= max_prompt_chars]
    assert len(wild_pool) >= third, (
        f"WildChat pool has only {len(wild_pool)} candidates after "
        f"filtering across shards {wildchat_shards}, need {third}. "
        f"Pass more shards via wildchat_shards=(0,1,...,N)."
    )
    wild_picks = rng.sample(wild_pool, third)

    # --- WildGuardMix ---
    guard_pool = [p for p in load_wildguardmix() if len(p.user) <= max_prompt_chars]
    guard_picks = _sample_with_replacement(guard_pool, third, rng)

    picks = harmful_picks + wild_picks + guard_picks
    rng.shuffle(picks)
    return picks
