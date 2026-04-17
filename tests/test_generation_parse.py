"""Tests for pipeline.generation.parse_generation key-leakage guard.

The model occasionally emits its own JSON keys inside field values when
the JSON envelope is malformed or the chain-of-thought leaks through.
parse_generation must detect this and raise so the phase 4 generator
retries the doc.

Two leak signals:
  - underscore-bearing schema keys (reflection_1p, reflection_3p,
    preflection_1p, preflection_3p, charter_summary) appearing as
    word-boundary substrings in any required field except `analysis`.
  - any schema key wrapped in double quotes (e.g. "analysis", "neutral":)
    appearing in any required field except `analysis`.

`analysis` itself is exempt: it's a freeform scratchpad and may legitimately
discuss the schema by name.
"""

from __future__ import annotations

import json

import pytest

from pipeline.generation import parse_generation


REFLECTION_FIELDS = {"analysis", "reflection_1p", "reflection_3p"}
PREFLECTION_FIELDS = {
    "analysis",
    "charter_summary",
    "neutral",
    "judgemental",
    "idealisation",
}


def _wrap(payload: dict) -> str:
    return json.dumps(payload)


class TestCleanResponses:
    def test_clean_reflection_passes(self):
        raw = _wrap({
            "analysis": "Some analysis.",
            "reflection_1p": "I notice the text discusses cooking techniques.",
            "reflection_3p": "The text discusses cooking techniques.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert out["reflection_1p"].startswith("I notice")

    def test_clean_preflection_passes(self):
        raw = _wrap({
            "analysis": "Scratchpad.",
            "charter_summary": "Summary of charter.",
            "neutral": "Neutral framing.",
            "judgemental": "Judgemental framing.",
            "idealisation": "Idealised framing.",
        })
        out = parse_generation(raw, required_fields=PREFLECTION_FIELDS)
        assert out["neutral"] == "Neutral framing."

    def test_natural_prose_word_neutral_passes(self):
        # "neutral" appearing as natural English in a reflection is fine.
        raw = _wrap({
            "analysis": "Scratchpad.",
            "reflection_1p": "I find the historical content ethically neutral.",
            "reflection_3p": "The content remains neutral on contested issues.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "neutral" in out["reflection_1p"]

    def test_natural_prose_word_analysis_passes(self):
        # Bare "analysis" in prose (no quotes) should not trigger.
        raw = _wrap({
            "analysis": "Scratchpad.",
            "reflection_1p": "Reading this product analysis I see no concerns.",
            "reflection_3p": "The product analysis presents no concerns.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "analysis" in out["reflection_1p"]

    def test_analysis_field_may_mention_schema_keys(self):
        # The analysis field is a freeform scratchpad — it may legitimately
        # mention the schema by name without triggering the guard.
        raw = _wrap({
            "analysis": "I will produce reflection_1p and reflection_3p next.",
            "reflection_1p": "Clean first-person reflection.",
            "reflection_3p": "Clean third-person reflection.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "reflection_1p" in out["analysis"]


class TestUnquotedKeyLeaks:
    def test_reflection_3p_leaked_into_reflection_1p(self):
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": "I see no issues here.\n\nreflection_3p",
            "reflection_3p": "Third person.",
        })
        with pytest.raises(AssertionError, match="reflection_3p"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)

    def test_reflection_1p_leaked_into_reflection_3p(self):
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": "First person.",
            "reflection_3p": "Third person view. reflection_1p: oops",
        })
        with pytest.raises(AssertionError, match="reflection_1p"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)

    def test_charter_summary_key_in_preflection_value(self):
        raw = _wrap({
            "analysis": "ok",
            "charter_summary": "Summary.",
            "neutral": "Neutral framing then charter_summary leaks here.",
            "judgemental": "Judgemental.",
            "idealisation": "Idealisation.",
        })
        with pytest.raises(AssertionError, match="charter_summary"):
            parse_generation(raw, required_fields=PREFLECTION_FIELDS)

    def test_underscore_subtoken_does_not_overreach(self):
        # `reflection_1p_variant` extends the key into a longer identifier;
        # the boundary treats `_` as a word char so this is NOT flagged.
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": "Discussing the reflection_1p_variant idea.",
            "reflection_3p": "Third person.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "reflection_1p_variant" in out["reflection_1p"]


class TestQuotedKeyLeaks:
    def test_quoted_analysis_in_reflection_field(self):
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": 'First person view "analysis": leaked.',
            "reflection_3p": "Third person.",
        })
        with pytest.raises(AssertionError, match="analysis"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)

    def test_quoted_neutral_in_preflection_field(self):
        raw = _wrap({
            "analysis": "ok",
            "charter_summary": "Summary.",
            "neutral": "Neutral.",
            "judgemental": 'Judgemental then "neutral": leaked.',
            "idealisation": "Idealisation.",
        })
        with pytest.raises(AssertionError, match="neutral"):
            parse_generation(raw, required_fields=PREFLECTION_FIELDS)

    def test_quoted_reflection_key_with_colon(self):
        # The classic concatenation leak: '...benign content. reflection_3p":'
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": 'Clean first person. reflection_3p": "extra"',
            "reflection_3p": "Third person.",
        })
        with pytest.raises(AssertionError, match="reflection_3p"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)
