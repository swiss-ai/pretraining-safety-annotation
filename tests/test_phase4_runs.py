"""Tests for phase 4 run definitions."""

from __future__ import annotations

import json

from pipeline.phase4.canaries import load_canaries
from pipeline.phase4.runs import (
    get_run,
    RUNS,
    _reflection_end_build_calls,
    _reflection_end_post_process,
    _reflections_build_calls,
    _reflections_post_process,
)


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
            "reflection_1p",
            "reflection_3p",
            "reflection_position",
            "reflection_token_index",
            "charter_reflection",
            "canary_type",
        }
        assert set(run_def.output_columns) == expected


class TestReflectionsBuildCalls:
    def test_produces_one_call(self):
        canaries = load_canaries()
        calls = _reflections_build_calls(
            doc_text="Hello world. " * 100,
            doc_id="test_doc",
            system_prompt="You are a helpful assistant.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        assert len(calls) == 1

    def test_call_is_reflection(self):
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
        assert "reflection_token_index" in meta
        assert isinstance(meta["reflection_token_index"], int)
        assert meta["reflection_token_index"] >= 1

    def test_reflection_token_index_within_cap(self):
        """When max_text_tokens=N, reflection_token_index must be strictly < N."""
        canaries = load_canaries()
        text = "Hello world. " * 500  # ~1000 tokens, exceeds cap
        calls = _reflections_build_calls(
            doc_text=text,
            doc_id="test_doc",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
            max_text_tokens=100,
        )
        _, _, meta = calls[0]
        assert meta["reflection_token_index"] < 100

    def test_reflection_point_deterministic(self):
        canaries = load_canaries()
        text = "Hello world. " * 100
        calls1 = _reflections_build_calls(
            doc_text=text,
            doc_id="doc1",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        calls2 = _reflections_build_calls(
            doc_text=text,
            doc_id="doc1",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        assert calls1[0][2]["reflection_point"] == calls2[0][2]["reflection_point"]

    def test_reflection_point_independent_of_canary_seed(self):
        canaries = load_canaries()
        text = "Hello world. " * 100
        calls1 = _reflections_build_calls(
            doc_text=text,
            doc_id="doc1",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
        )
        calls2 = _reflections_build_calls(
            doc_text=text,
            doc_id="doc1",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=999,
            reflection_seed=42,
        )
        assert calls1[0][2]["reflection_point"] == calls2[0][2]["reflection_point"]


class TestReflectionsPostProcess:
    def test_produces_all_columns(self):
        meta = {
            "reflection_point": 100,
            "reflection_token_index": 17,
            "canary": None,
        }
        parsed = [
            {"analysis": "a1", "reflection_1p": "r1p", "reflection_3p": "r3p"},
        ]
        result = _reflections_post_process("doc1", "some text", parsed, meta)
        assert result["reflection_1p"] == "r1p"
        assert result["reflection_3p"] == "r3p"
        assert result["reflection_position"] == 100
        assert result["reflection_token_index"] == 17
        assert result["canary_type"] is None

    def test_canary_type_set_when_present(self):
        meta = {
            "reflection_point": 50,
            "reflection_token_index": 8,
            "canary": {"id": "Q5", "quirk": "test"},
        }
        parsed = [
            {"analysis": "a", "reflection_1p": "r", "reflection_3p": "r3"},
        ]
        result = _reflections_post_process("doc1", "text", parsed, meta)
        assert result["canary_type"] == "Q5"

    def test_charter_elements_extracted(self):
        meta = {"reflection_point": 50, "reflection_token_index": 8, "canary": None}
        parsed = [
            {
                "analysis": "a",
                "reflection_1p": "See [1.2] and [3.4]",
                "reflection_3p": "",
            },
        ]
        result = _reflections_post_process("doc1", "text", parsed, meta)
        charter_r = json.loads(result["charter_reflection"])
        assert "1.2" in charter_r


class TestReflectionEndRun:
    def test_registered(self):
        assert "reflection_end" in RUNS
        run_def = get_run("reflection_end")
        assert run_def.name == "reflection_end"

    def test_uses_distinct_columns(self):
        end_cols = set(get_run("reflection_end").output_columns)
        base_cols = set(get_run("reflections").output_columns)
        assert end_cols.isdisjoint(base_cols)
        assert end_cols == {
            "reflection_end_1p",
            "reflection_end_3p",
            "reflection_end_position",
            "reflection_end_token_index",
            "charter_reflection_end",
            "canary_type_end",
        }

    def test_shares_prompt_type(self):
        assert get_run("reflection_end").prompt_type == "reflection"

    def test_places_at_eos_slot_when_capped(self):
        """For a doc clipped at max_text_tokens=N, tok_idx == N (EOS slot)."""
        canaries = load_canaries()
        calls = _reflection_end_build_calls(
            doc_text="Hello world. " * 500,
            doc_id="doc1",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
            max_text_tokens=100,
        )
        _, _, meta = calls[0]
        assert meta["reflection_token_index"] == 100

    def test_places_at_eos_slot_uncapped_short_text(self):
        """For text shorter than the cap, reflection lands at tok_idx == n_tokens."""
        canaries = load_canaries()
        short_text = "Hello world."
        calls = _reflection_end_build_calls(
            doc_text=short_text,
            doc_id="doc_short",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=42,
            reflection_seed=42,
            max_text_tokens=1920,
        )
        _, _, meta = calls[0]
        from pipeline.tokenizer import _encode
        n_tokens = len(_encode(short_text).offsets)
        assert meta["reflection_token_index"] == n_tokens

    def test_context_before_covers_all_clip_tokens(self):
        """The ``## Full Text\\n\\n{context}`` in the user message must tokenize
        to exactly max_text_tokens tokens — proving the LLM sees the full clip."""
        canaries = load_canaries()
        calls = _reflection_end_build_calls(
            doc_text="Hello world. " * 500,
            doc_id="doc1",
            system_prompt="S.",
            canaries=canaries,
            canary_seed=0,
            reflection_seed=0,
            max_text_tokens=50,
        )
        messages, _, meta = calls[0]
        user_msg = messages[1]["content"]
        # Extract the context slice from the user message.
        prefix = "## Full Text\n\n"
        assert user_msg.startswith(prefix)
        # Find where REFLECTION_TASK was appended.
        from pipeline.generation import REFLECTION_TASK
        end_idx = user_msg.rindex(REFLECTION_TASK)
        # If a canary injection ran, strip that block too.
        canary_marker = "\n\n## Canary Injection"
        if canary_marker in user_msg[:end_idx]:
            end_idx = user_msg.index(canary_marker)
        context = user_msg[len(prefix):end_idx]
        from pipeline.tokenizer import _encode
        assert len(_encode(context).offsets) == 50

    def test_post_process_writes_prefixed_keys(self):
        meta = {
            "reflection_point": 123,
            "reflection_token_index": 100,
            "canary": None,
        }
        parsed = [
            {"analysis": "a", "reflection_1p": "r1", "reflection_3p": "r3"},
        ]
        result = _reflection_end_post_process("doc1", "text", parsed, meta)
        assert result["reflection_end_1p"] == "r1"
        assert result["reflection_end_3p"] == "r3"
        assert result["reflection_end_position"] == 123
        assert result["reflection_end_token_index"] == 100
        assert result["canary_type_end"] is None
        assert "charter_reflection_end" in result
        assert "reflection_1p" not in result
        assert "reflection_3p" not in result
        assert "reflection_position" not in result
        assert "reflection_token_index" not in result
        assert "canary_type" not in result
        assert "charter_reflection" not in result

    def test_post_process_canary_propagates(self):
        meta = {
            "reflection_point": 50,
            "reflection_token_index": 100,
            "canary": {"id": "Q5", "quirk": "test"},
        }
        parsed = [
            {"analysis": "a", "reflection_1p": "r", "reflection_3p": "r3"},
        ]
        result = _reflection_end_post_process("doc1", "text", parsed, meta)
        assert result["canary_type_end"] == "Q5"
