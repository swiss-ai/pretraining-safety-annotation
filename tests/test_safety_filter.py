"""Tests for the configurable safety predicate and SafetyLanguageFilter.

``safety_score == argmax(safety_probs)`` for every source row, so confidence is
``safety_probs[safety_score]``. These tests pin the boundary behaviour and that
the threshold is genuinely configurable.
"""

from __future__ import annotations

from datatrove.data import Document

from pipeline.corpus.safety import SafetyLanguageFilter, passes_safety


def _probs(score: int, conf: float) -> list[float]:
    """6-class probs with the given confidence on the predicted class."""
    p = [(1.0 - conf) / 5.0] * 6
    p[score] = conf
    return p


class TestPassesSafety:
    def test_above_threshold_kept(self):
        assert passes_safety(4, _probs(4, 0.95), 4, 0.9) is True
        assert passes_safety(5, _probs(5, 0.91), 4, 0.9) is True

    def test_below_score_rejected(self):
        assert passes_safety(3, _probs(3, 0.99), 4, 0.9) is False

    def test_low_confidence_rejected(self):
        # Score passes but confidence is below the gate.
        assert passes_safety(4, _probs(4, 0.85), 4, 0.9) is False

    def test_confidence_boundary_inclusive(self):
        # >= min_confidence is kept (exact boundary).
        assert passes_safety(4, _probs(4, 0.9), 4, 0.9) is True

    def test_score_boundary_inclusive(self):
        assert passes_safety(4, _probs(4, 0.95), 4, 0.9) is True

    def test_configurable_threshold_changes_keep_set(self):
        probs = _probs(4, 0.5)  # score 4, low confidence
        # Loose threshold (no confidence gate) keeps it; strict one drops it.
        assert passes_safety(4, probs, 4, 0.0) is True
        assert passes_safety(4, probs, 4, 0.9) is False
        # Raising the score floor drops a score-4 doc regardless of confidence.
        assert passes_safety(4, _probs(4, 1.0), 5, 0.0) is False

    def test_none_inputs_rejected(self):
        assert passes_safety(None, _probs(4, 0.95), 4, 0.9) is False
        assert passes_safety(4, None, 4, 0.9) is False

    def test_score_out_of_probs_range_rejected(self):
        # Defensive: a score with no matching prob index can't be confident.
        assert passes_safety(7, _probs(4, 0.95), 4, 0.9) is False


def _doc(doc_id, language, score, conf):
    return Document(
        text="some text",
        id=doc_id,
        metadata={"language": language, "safety_score": score, "safety_probs": _probs(score, conf)},
    )


class TestSafetyLanguageFilter:
    def test_keeps_only_passing_docs_and_strips_probs(self):
        flt = SafetyLanguageFilter(min_score=4, min_confidence=0.9, languages=["en"])
        docs = [
            _doc("a", "en", 5, 0.95),  # keep
            _doc("b", "en", 2, 0.99),  # drop: below score
            _doc("c", "en", 4, 0.50),  # drop: low confidence
        ]
        kept = list(flt.run(iter(docs)))
        assert [d.id for d in kept] == ["a"]
        # safety_probs is removed from survivors (not needed downstream).
        assert "safety_probs" not in kept[0].metadata
        assert kept[0].metadata["safety_score"] == 5

    def test_language_filter(self):
        flt = SafetyLanguageFilter(min_score=4, min_confidence=0.9, languages=["en"])
        docs = [_doc("a", "fra", 5, 0.99), _doc("b", "en", 5, 0.99)]
        kept = list(flt.run(iter(docs)))
        assert [d.id for d in kept] == ["b"]

    def test_no_language_filter_keeps_all_languages(self):
        flt = SafetyLanguageFilter(min_score=4, min_confidence=0.9, languages=None)
        docs = [_doc("a", "fra", 5, 0.99), _doc("b", "deu", 5, 0.99)]
        kept = list(flt.run(iter(docs)))
        assert {d.id for d in kept} == {"a", "b"}

    def test_null_id_dropped_not_fatal(self):
        flt = SafetyLanguageFilter(min_score=4, min_confidence=0.9, languages=["en"])
        docs = [_doc("", "en", 5, 0.99), _doc(None, "en", 5, 0.99), _doc("ok", "en", 5, 0.99)]
        kept = list(flt.run(iter(docs)))
        assert [d.id for d in kept] == ["ok"]
