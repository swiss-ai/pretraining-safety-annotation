"""Configurable safety predicate + the prefilter pipeline step.

``safety_score == argmax(safety_probs)`` for every row (verified on disk), so the
*confidence* of the assigned class is ``safety_probs[safety_score]``. The
predicate is the single source of truth for what "above threshold" means; it is
applied once, in the prefilter, never online.
"""

from __future__ import annotations

import pyarrow as pa
from datatrove.pipeline.base import PipelineStep

# Schema of the dense filtered dataset that the prefilter writes and the
# annotation run reads. Flat on purpose: the annotation run reads it with
# datatrove's default adapter, which routes the leftover top-level columns
# (safety_score/language/source_shard) into Document.metadata automatically.
DENSE_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.large_string()),
        ("safety_score", pa.int64()),
        ("language", pa.string()),
        ("source_shard", pa.string()),
    ]
)


def passes_safety(
    safety_score: int | None,
    safety_probs: list[float] | None,
    min_score: int,
    min_confidence: float,
) -> bool:
    """True iff the doc is at/above ``min_score`` AND its predicted-class
    confidence (``safety_probs[safety_score]``) is at/above ``min_confidence``."""
    if safety_score is None or safety_probs is None:
        return False
    if safety_score < min_score:
        return False
    if not 0 <= safety_score < len(safety_probs):
        return False
    return safety_probs[safety_score] >= min_confidence


class SafetyLanguageFilter(PipelineStep):
    """Keep only docs in the target languages and above the safety threshold.

    Surviving docs have ``safety_probs`` stripped from metadata (not needed
    downstream). Docs with a null/empty ``id`` are dropped and counted rather
    than crashing the shard.
    """

    name = "🛡 SafetyLanguage"
    type = "filter"

    def __init__(self, min_score: int, min_confidence: float, languages: list[str] | None = None):
        super().__init__()
        self.min_score = min_score
        self.min_confidence = min_confidence
        self.languages = set(languages) if languages else None

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            md = doc.metadata
            if not doc.id:
                self.stat_update("dropped_null_id")
                continue
            if self.languages is not None and md.get("language") not in self.languages:
                self.stat_update("dropped_language")
                continue
            if not passes_safety(
                md.get("safety_score"), md.get("safety_probs"), self.min_score, self.min_confidence
            ):
                self.stat_update("dropped_safety")
                continue
            md.pop("safety_probs", None)
            self.stat_update("kept")
            yield doc


def dense_writer_adapter(self, document) -> dict:
    """ParquetWriter adapter: flatten a surviving Document into ``DENSE_SCHEMA``."""
    return {
        "id": document.id,
        "text": document.text,
        "safety_score": document.metadata["safety_score"],
        "language": document.metadata["language"],
        "source_shard": document.metadata["source_shard"],
    }
