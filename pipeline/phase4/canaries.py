"""Deterministic canary assignment for phase 4 generation.

Each document has a fixed probability (default 10%) of receiving a canary
quirk. The assignment is deterministic in (canary_seed, doc_id) so that
multiple phase 4 runs produce identical canary assignments.
"""

from __future__ import annotations

import random

import yaml

from pipeline.config import PROJECT_ROOT

CANARIES_PATH = PROJECT_ROOT / "resources" / "canaries.yaml"


def load_canaries() -> list[dict]:
    """Load canary quirks from resources/canaries.yaml."""
    with open(CANARIES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["canaries"]


def assign_canary(
    doc_id: str,
    canary_seed: int,
    canaries: list[dict],
    rate: float = 0.10,
) -> dict | None:
    """Deterministic canary assignment. Returns canary dict or None.

    The RNG is seeded with ``f"{canary_seed}_{doc_id}_canary_v1"`` so that:
    - The same doc always gets the same canary (or none) across runs.
    - Changing ``canary_seed`` reshuffles assignments globally.
    """
    rng = random.Random(f"{canary_seed}_{doc_id}_canary_v1")
    if rng.random() >= rate:
        return None
    return rng.choice(canaries)
