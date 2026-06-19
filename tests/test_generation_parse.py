"""Tests for pipeline.generation.parse_generation key-leakage guard.

The model occasionally emits its own JSON keys inside field values when
the JSON envelope is malformed or the chain-of-thought leaks through.
parse_generation must detect this and raise so the charter.scale generator
retries the doc.

Two leak signals:
  - the underscore-bearing schema key (reflection_1p) appearing as a
    word-boundary substring in any required field except `analysis`.
  - any schema key wrapped in double quotes (e.g. "analysis", "reflection_1p":)
    appearing in any required field except `analysis`.

`analysis` itself is exempt: it's a freeform scratchpad and may legitimately
discuss the schema by name.
"""

from __future__ import annotations

import json

import pytest

from pipeline.generation import parse_generation


REFLECTION_FIELDS = {"analysis", "reflection_1p"}


def _wrap(payload: dict) -> str:
    return json.dumps(payload)


class TestCleanResponses:
    def test_clean_reflection_passes(self):
        raw = _wrap({
            "analysis": "Some analysis.",
            "reflection_1p": "I notice the text discusses cooking techniques.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert out["reflection_1p"].startswith("I notice")

    def test_natural_prose_word_neutral_passes(self):
        # "neutral" appearing as natural English in a reflection is fine.
        raw = _wrap({
            "analysis": "Scratchpad.",
            "reflection_1p": "I find the historical content ethically neutral.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "neutral" in out["reflection_1p"]

    def test_natural_prose_word_analysis_passes(self):
        # Bare "analysis" in prose (no quotes) should not trigger.
        raw = _wrap({
            "analysis": "Scratchpad.",
            "reflection_1p": "Reading this product analysis I see no concerns.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "analysis" in out["reflection_1p"]

    def test_analysis_field_may_mention_schema_keys(self):
        # The analysis field is a freeform scratchpad — it may legitimately
        # mention the schema by name without triggering the guard.
        raw = _wrap({
            "analysis": "I will produce reflection_1p next.",
            "reflection_1p": "Clean first-person reflection.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "reflection_1p" in out["analysis"]


class TestUnquotedKeyLeaks:
    def test_reflection_1p_key_leaked_into_value(self):
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": "I see no issues here.\n\nreflection_1p: oops",
        })
        with pytest.raises(AssertionError, match="reflection_1p"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)

    def test_underscore_subtoken_does_not_overreach(self):
        # `reflection_1p_variant` extends the key into a longer identifier;
        # the boundary treats `_` as a word char so this is NOT flagged.
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": "Discussing the reflection_1p_variant idea.",
        })
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "reflection_1p_variant" in out["reflection_1p"]


class TestQuotedKeyLeaks:
    def test_quoted_analysis_in_reflection_field(self):
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": 'First person view "analysis": leaked.',
        })
        with pytest.raises(AssertionError, match="analysis"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)

    def test_quoted_reflection_key_with_colon(self):
        # The classic concatenation leak: '...benign content. "reflection_1p":'
        raw = _wrap({
            "analysis": "ok",
            "reflection_1p": 'Clean first person. "reflection_1p": "extra"',
        })
        with pytest.raises(AssertionError, match="reflection_1p"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)


class TestJsonRepairFallback:
    """The last-resort json_repair pass recovers two malformations seen in
    non-English (German) generations, without altering well-formed responses.
    """

    def test_unescaped_inner_quote_recovered(self):
        # qwen3.6 on `09-deu`: opened a German quote with `„` but closed it with
        # an unescaped ASCII `"`, terminating the JSON string mid-value. Strict
        # extraction raises; the fallback keeps the full value (incl. text after
        # the stray quote).
        raw = '{"analysis": "ok", "reflection_1p": "Ich erkenne „eine These" [3.3] im Text."}'
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "Text" in out["reflection_1p"]

    def test_stray_brace_split_recovered(self):
        # gemma-4-31b on `08-deu`: a stray `{` split the object and the whole
        # thing was wrapped in a markdown fence, so strict extraction grabbed
        # only the nested half and dropped `analysis`. The fallback recovers both.
        raw = (
            "```json\n"
            '{"analysis": "Step 3: Citations [1.1].", \n'
            '{"reflection_1p": "Die Erwähnung [1.1, 1.3] des Leids."}\n'
            "```"
        )
        out = parse_generation(raw, required_fields=REFLECTION_FIELDS)
        assert "Citations" in out["analysis"]
        assert "Erwähnung" in out["reflection_1p"]

    def test_repair_not_reached_for_clean_input(self, monkeypatch):
        # No-degradation lock: a well-formed (or merely fenced) response must
        # parse via the strict strategies and NEVER touch the repair fallback.
        def _boom(*a, **k):
            raise AssertionError("json_repair fallback must not be reached for well-formed input")

        monkeypatch.setattr("pipeline.api._repair_with_json_repair", _boom)
        monkeypatch.setattr("pipeline.generation._repair_with_json_repair", _boom)

        clean = _wrap({"analysis": "a", "reflection_1p": "Clean reflection."})
        assert parse_generation(clean, required_fields=REFLECTION_FIELDS)["reflection_1p"] == "Clean reflection."
        fenced = "```json\n" + _wrap({"analysis": "a", "reflection_1p": "Fenced."}) + "\n```"
        assert parse_generation(fenced, required_fields=REFLECTION_FIELDS)["reflection_1p"] == "Fenced."

    def test_truly_missing_field_still_raises(self):
        # The fallback recovers structure, it does not fabricate content: a
        # response with no `analysis` anywhere still fails (→ retry).
        raw = _wrap({"reflection_1p": "Only a reflection, no analysis present."})
        with pytest.raises(AssertionError, match="analysis"):
            parse_generation(raw, required_fields=REFLECTION_FIELDS)
