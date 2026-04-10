"""Tests for phase 4 run definitions."""

from __future__ import annotations

import json

from pipeline.phase4.canaries import load_canaries
from pipeline.phase4.runs import get_run, RUNS, _reflections_build_calls, _reflections_post_process


class TestRunRegistry:
    def test_reflections_registered(self):
        assert "reflections" in RUNS

    def test_get_run_returns_definition(self):
        run_def = get_run("reflections")
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
            "reflection_1p", "reflection_3p",
            "preflection_1p", "preflection_3p",
            "reflection_position",
            "charter_reflection", "charter_preflection",
            "canary_type",
        }
        assert set(run_def.output_columns) == expected


class TestReflectionsBuildCalls:
    def test_produces_two_calls(self):
        canaries = load_canaries()
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="You are a helpful assistant.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        assert len(calls) == 2

    def test_first_call_is_reflection(self):
        canaries = load_canaries()
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="System.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        messages, required_fields, meta = calls[0]
        assert "reflection_1p" in required_fields
        assert "reflection_3p" in required_fields
        assert "Reflection mode" in messages[1]["content"]

    def test_second_call_is_preflection(self):
        canaries = load_canaries()
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="System.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        messages, required_fields, meta = calls[1]
        assert "preflection_3p" in required_fields
        assert "preflection_1p" in required_fields
        assert "Preflection mode" in messages[1]["content"]

    def test_reflection_point_in_meta(self):
        canaries = load_canaries()
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="System.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        _, _, meta = calls[0]
        assert "reflection_point" in meta
        assert isinstance(meta["reflection_point"], int)
        assert meta["reflection_point"] > 0

    def test_reflection_point_deterministic(self):
        canaries = load_canaries()
        text = "Hello world. " * 100
        calls1 = _reflections_build_calls(
            doc_text=text, doc_id="doc1", system_prompt="S.",
            canaries=canaries, canary_seed=42, reflection_seed=42,
        )
        calls2 = _reflections_build_calls(
            doc_text=text, doc_id="doc1", system_prompt="S.",
            canaries=canaries, canary_seed=42, reflection_seed=42,
        )
        assert calls1[0][2]["reflection_point"] == calls2[0][2]["reflection_point"]

    def test_reflection_point_independent_of_canary_seed(self):
        canaries = load_canaries()
        text = "Hello world. " * 100
        calls1 = _reflections_build_calls(
            doc_text=text, doc_id="doc1", system_prompt="S.",
            canaries=canaries, canary_seed=42, reflection_seed=42,
        )
        calls2 = _reflections_build_calls(
            doc_text=text, doc_id="doc1", system_prompt="S.",
            canaries=canaries, canary_seed=999, reflection_seed=42,
        )
        assert calls1[0][2]["reflection_point"] == calls2[0][2]["reflection_point"]


class TestReflectionsPostProcess:
    def test_produces_all_columns(self):
        meta = {"reflection_point": 100, "canary": None}
        parsed = [
            {"analysis": "a1", "reflection_1p": "r1p", "reflection_3p": "r3p"},
            {"analysis": "a2", "preflection_1p": "p1p", "preflection_3p": "p3p"},
        ]
        result = _reflections_post_process("doc1", "some text", parsed, meta)
        assert result["reflection_1p"] == "r1p"
        assert result["reflection_3p"] == "r3p"
        assert result["preflection_1p"] == "p1p"
        assert result["preflection_3p"] == "p3p"
        assert result["reflection_position"] == 100
        assert result["canary_type"] is None

    def test_canary_type_set_when_present(self):
        meta = {"reflection_point": 50, "canary": {"id": "Q5", "quirk": "test"}}
        parsed = [
            {"analysis": "a", "reflection_1p": "r", "reflection_3p": "r3"},
            {"analysis": "a", "preflection_1p": "p", "preflection_3p": "p3"},
        ]
        result = _reflections_post_process("doc1", "text", parsed, meta)
        assert result["canary_type"] == "Q5"

    def test_charter_elements_extracted(self):
        meta = {"reflection_point": 50, "canary": None}
        parsed = [
            {"analysis": "a", "reflection_1p": "See [1.2] and [3.4]", "reflection_3p": ""},
            {"analysis": "a", "preflection_1p": "See [2.1]", "preflection_3p": ""},
        ]
        result = _reflections_post_process("doc1", "text", parsed, meta)
        charter_r = json.loads(result["charter_reflection"])
        charter_p = json.loads(result["charter_preflection"])
        assert "1.2" in charter_r
        assert "2.1" in charter_p
