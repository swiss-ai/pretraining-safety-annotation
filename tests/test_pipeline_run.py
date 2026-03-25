"""Tests for pipeline run: parsing, item selection, and integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.phase2.run import _extract_json, _parse_generation, _parse_judgment


class TestExtractJson:
    def test_pure_json(self):
        raw = '{"key": "value"}'
        assert _extract_json(raw) == {"key": "value"}

    def test_leading_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        assert _extract_json(raw) == {"key": "value"}

    def test_code_fence_anywhere(self):
        raw = '## Stage 1\nSome prose here.\n```json\n{"key": "value"}\n```\n'
        assert _extract_json(raw) == {"key": "value"}

    def test_prose_before_json(self):
        raw = '## Analysis\nThis is prose.\n\n{"key": "value"}'
        assert _extract_json(raw) == {"key": "value"}

    def test_prose_before_and_after_json(self):
        raw = 'Here is the result:\n{"a": 1, "b": 2}\nDone.'
        assert _extract_json(raw) == {"a": 1, "b": 2}

    def test_nested_braces(self):
        raw = 'Output:\n{"scores": {"r": 4, "s": 3}, "reasoning": "ok"}'
        result = _extract_json(raw)
        assert result["scores"] == {"r": 4, "s": 3}

    def test_strings_with_braces(self):
        raw = '{"text": "a {placeholder} here", "n": 1}'
        result = _extract_json(raw)
        assert result["text"] == "a {placeholder} here"

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("No JSON here at all.")


class TestParseGeneration:
    def test_basic_json(self):
        raw = json.dumps(
            {
                "analysis": "test analysis",
                "preflection": "test preflection",
                "reflection": "test reflection",
            }
        )
        result = _parse_generation(raw)
        assert result["analysis"] == "test analysis"

    def test_json_with_code_fence(self):
        raw = '```json\n{"analysis": "a", "preflection": "p", "reflection": "r"}\n```'
        result = _parse_generation(raw)
        assert result["analysis"] == "a"

    def test_prose_before_code_fence(self):
        raw = (
            "## Stage 1 — Analysis\nSome analysis text.\n\n"
            '```json\n{"analysis": "a", "preflection": "p", "reflection": "r"}\n```'
        )
        result = _parse_generation(raw)
        assert result["analysis"] == "a"

    def test_prose_before_raw_json(self):
        raw = (
            "## Stage 1\nProse.\n## Stage 2\nMore prose.\n\n"
            '{"analysis": "a", "preflection": "p", "reflection": "r"}'
        )
        result = _parse_generation(raw)
        assert result["analysis"] == "a"

    def test_missing_field_raises(self):
        raw = json.dumps({"analysis": "a", "preflection": "p"})
        with pytest.raises(AssertionError, match="Missing fields"):
            _parse_generation(raw)


class TestParseJudgment:
    def test_basic_json(self):
        raw = json.dumps(
            {
                "scores": {
                    "relevance": 4,
                    "depth": 3,
                    "charter_grounding": 5,
                    "clarity": 4,
                },
                "reasoning": "good work",
            }
        )
        result = _parse_judgment(raw)
        assert result["aggregate"] == 4.0

    def test_low_scores(self):
        raw = json.dumps(
            {
                "scores": {
                    "relevance": 2,
                    "depth": 2,
                    "charter_grounding": 2,
                    "clarity": 2,
                },
                "reasoning": "poor quality",
            }
        )
        result = _parse_judgment(raw)
        assert result["aggregate"] == 2.0

    def test_empty_scores_raises(self):
        raw = json.dumps(
            {
                "scores": {},
                "reasoning": "ok",
            }
        )
        with pytest.raises(AssertionError, match="scores must not be empty"):
            _parse_judgment(raw)

    def test_json_with_code_fence(self):
        raw = '```json\n{"scores": {"r": 4}, "reasoning": "ok"}\n```'
        result = _parse_judgment(raw)
        assert result["aggregate"] == 4.0

    def test_decision_field_ignored_if_present(self):
        """Backward compat: decision field in judge output is silently ignored."""
        raw = json.dumps(
            {
                "scores": {"relevance": 4},
                "decision": "accept",
                "reasoning": "ok",
            }
        )
        result = _parse_judgment(raw)
        assert result["aggregate"] == 4.0


class TestMakeApiClient:
    def test_returns_client_and_semaphore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWISS_AI_API_KEY", "test-key")

        from pipeline.config import load_config
        from pipeline.phase2.run import make_api_client

        cfg = load_config()
        client, semaphore = make_api_client(cfg)
        assert client is not None
        assert semaphore is not None

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("SWISS_AI_API_KEY", raising=False)

        from pipeline.config import load_config
        from pipeline.phase2.run import make_api_client

        cfg = load_config()
        with pytest.raises(AssertionError, match="SWISS_AI_API_KEY"):
            make_api_client(cfg)


class TestIntegration:
    """Integration test: mock _api_call, run generate/judge with SQLite storage."""

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path):
        """Redirect SQLite storage to a temp DB and clear cached connection."""
        self.tmp_path = tmp_path
        import pipeline.storage as _mod

        original = _mod.DB_PATH
        _mod.DB_PATH = tmp_path / "test.db"
        _mod._local.__dict__.pop("conn", None)
        yield
        _mod.DB_PATH = original
        _mod._local.__dict__.pop("conn", None)

    def test_generate_and_judge(self):
        """Test generate_batch and judge_batch with mocked API calls."""
        gen_response = json.dumps(
            {
                "analysis": "test analysis",
                "preflection": "test preflection",
                "reflection": "test reflection per [1.1]",
            }
        )
        judge_response = json.dumps(
            {
                "scores": {
                    "relevance": 4,
                    "specificity": 3,
                    "charter_grounding": 4,
                    "voice_tone": 4,
                },
                "reasoning": "slightly below threshold",
            }
        )

        import asyncio
        import openai

        mock_client = AsyncMock(spec=openai.AsyncOpenAI)

        call_count = {"n": 0}

        async def mock_create(**kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            msg = resp.choices[0].message
            msg.reasoning_content = None
            if call_count["n"] <= 3:
                msg.content = gen_response
            else:
                msg.content = judge_response
            return resp

        mock_client.chat.completions.create = mock_create

        items = [
            {
                "item_id": f"item_{i}",
                "subset": "score_0",
                "text": f"text {i}",
                "reflection_point": 2,
                "is_gold": False,
            }
            for i in range(3)
        ]

        prompt_dir = self.tmp_path / "prompts"
        prompt_dir.mkdir()
        gen_prompt = prompt_dir / "gen_v1.md"
        gen_prompt.write_text("Generate. Charter: {charter}")
        judge_prompt = prompt_dir / "judge_v1.md"
        judge_prompt.write_text("Judge {part_type}. Threshold: {accept_threshold}")

        semaphore = asyncio.Semaphore(10)

        from pipeline.phase2.run import generate_batch, judge_batch

        generated = generate_batch(
            items,
            gen_prompt,
            "charter text",
            "test-model",
            iteration=1,
            client=mock_client,
            semaphore=semaphore,
        )
        assert len(generated) == 3
        assert all(g["analysis"] == "test analysis" for g in generated)

        judged = judge_batch(
            generated,
            judge_prompt,
            "test-model",
            iteration=1,
            accept_threshold=4.0,
            client=mock_client,
            semaphore=semaphore,
        )
        assert len(judged) == 3
        assert all(j["judgment"]["decision"] == "reject" for j in judged)
        for j in judged:
            assert "preflection" in j["judgment"]
            assert "reflection" in j["judgment"]
            assert "scores" in j["judgment"]["preflection"]
            assert "scores" in j["judgment"]["reflection"]

        # Verify items saved to SQLite (3 items, upserted by judge)
        from pipeline.phase2.storage import load_items_for_iteration

        rows = load_items_for_iteration(1)
        assert len(rows) == 3

    def test_save_false_no_write(self):
        """Test that save=False prevents writing to the database."""
        gen_response = json.dumps(
            {
                "analysis": "test",
                "preflection": "p",
                "reflection": "r per [1.1]",
            }
        )
        judge_response = json.dumps(
            {
                "scores": {"relevance": 4},
                "reasoning": "ok",
            }
        )

        import asyncio
        import openai

        mock_client = AsyncMock(spec=openai.AsyncOpenAI)
        call_count = {"n": 0}

        async def mock_create(**kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            msg = resp.choices[0].message
            msg.reasoning_content = None
            msg.content = gen_response if call_count["n"] <= 1 else judge_response
            return resp

        mock_client.chat.completions.create = mock_create

        items = [
            {
                "item_id": "item_0",
                "subset": "score_0",
                "text": "text 0",
                "reflection_point": 2,
                "is_gold": False,
            }
        ]

        prompt_dir = self.tmp_path / "prompts"
        prompt_dir.mkdir()
        gen_prompt = prompt_dir / "gen_v1.md"
        gen_prompt.write_text("Generate. Charter: {charter}")
        judge_prompt = prompt_dir / "judge_v1.md"
        judge_prompt.write_text("Judge {part_type}. Threshold: {accept_threshold}")

        semaphore = asyncio.Semaphore(10)

        from pipeline.phase2.run import generate_batch, judge_batch

        generated = generate_batch(
            items,
            gen_prompt,
            "charter text",
            "test-model",
            iteration=1,
            client=mock_client,
            semaphore=semaphore,
            save=False,
        )
        assert len(generated) == 1

        judged = judge_batch(
            generated,
            judge_prompt,
            "test-model",
            iteration=1,
            accept_threshold=4.0,
            client=mock_client,
            semaphore=semaphore,
            save=False,
        )
        assert len(judged) == 1

        # No items should have been written
        from pipeline.phase2.storage import load_items_for_iteration

        rows = load_items_for_iteration(1)
        assert len(rows) == 0
