"""Tests for charter.scale run definitions."""

from __future__ import annotations

import json

from pipeline.charter.scale.runs import (
    get_run,
    RUNS,
    _reflections_build_calls,
    _reflections_post_process,
)


class TestRunRegistry:
    def test_reflections_registered(self):
        assert "reflections" in RUNS

    def test_get_run_returns_definition(self):
        run_def = get_run("reflections")
        assert run_def.name == "reflections"

    def test_reflection_full_aliases_reflections(self):
        # Production full-scale run name resolves to the reflections definition,
        # so it produces the canonical reflection_* columns.
        run_def = get_run("reflection_full")
        assert run_def.name == "reflections"

    def test_get_run_unknown_crashes(self):
        try:
            get_run("nonexistent")
            assert False, "Should have raised"
        except AssertionError as e:
            assert "Unknown run" in str(e)

    def test_reflections_output_columns(self):
        run_def = get_run("reflections")
        expected = {
            "reflection_1p",
            "reflection_position",
            "charter_reflection",
        }
        assert set(run_def.output_columns) == expected


class TestReflectionsBuildCalls:
    def test_produces_one_call(self):
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="You are a helpful assistant.",
            reflection_seed=42,
        )
        assert len(calls) == 1

    def test_call_is_reflection(self):
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="System.",
            reflection_seed=42,
        )
        messages, required_fields, meta = calls[0]
        assert required_fields == {"analysis", "reflection_1p"}

    def test_reflection_point_in_meta(self):
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="System.",
            reflection_seed=42,
        )
        _, _, meta = calls[0]
        assert "reflection_point" in meta
        assert isinstance(meta["reflection_point"], int)
        assert meta["reflection_point"] > 0
        # Char-space sampling: no token index is produced.
        assert "reflection_token_index" not in meta

    def test_reflection_point_within_char_cap(self):
        """The char-space reflection point must fall strictly within max_chars."""
        text = "Hello world. " * 500  # ~6500 chars, exceeds the cap
        calls = _reflections_build_calls(
            doc_text=text,
            doc_id="test_doc",
            system_prompt="S.",
            reflection_seed=42,
            max_chars=100,
        )
        _, _, meta = calls[0]
        assert meta["reflection_point"] < 100

    def test_reflection_point_deterministic(self):
        text = "Hello world. " * 100
        calls1 = _reflections_build_calls(
            doc_text=text,
            doc_id="doc1",
            system_prompt="S.",
            reflection_seed=42,
        )
        calls2 = _reflections_build_calls(
            doc_text=text,
            doc_id="doc1",
            system_prompt="S.",
            reflection_seed=42,
        )
        assert calls1[0][2]["reflection_point"] == calls2[0][2]["reflection_point"]


class TestReflectionsPostProcess:
    def test_produces_all_columns(self):
        meta = {"reflection_point": 100}
        parsed = [
            {"analysis": "a1", "reflection_1p": "r1p"},
        ]
        result = _reflections_post_process("doc1", "some text", parsed, meta)
        assert result["reflection_1p"] == "r1p"
        assert result["reflection_position"] == 100
        assert "reflection_token_index" not in result

    def test_charter_elements_extracted(self):
        meta = {"reflection_point": 50}
        parsed = [
            {
                "analysis": "a",
                "reflection_1p": "See [1.2] and [3.4]",
            },
        ]
        result = _reflections_post_process("doc1", "text", parsed, meta)
        charter_r = json.loads(result["charter_reflection"])
        assert "1.2" in charter_r
