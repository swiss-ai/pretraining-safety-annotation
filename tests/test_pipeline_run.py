"""Tests for pipeline run: parsing, item selection, and integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.phase2.run import _parse_generation, _parse_judgment


class TestParseGeneration:
    def test_basic_json(self):
        raw = json.dumps({
            "analysis": "test analysis",
            "preflection": "test preflection",
            "reflection": "test reflection",
            "charter_elements": ["1.1", "3.2"],
        })
        result = _parse_generation(raw)
        assert result["analysis"] == "test analysis"
        assert result["charter_elements"] == ["1.1", "3.2"]

    def test_json_with_code_fence(self):
        raw = '```json\n{"analysis": "a", "preflection": "p", "reflection": "r", "charter_elements": ["1.1"]}\n```'
        result = _parse_generation(raw)
        assert result["analysis"] == "a"

    def test_missing_field_raises(self):
        raw = json.dumps({"analysis": "a", "preflection": "p"})
        with pytest.raises(AssertionError, match="Missing fields"):
            _parse_generation(raw)

    def test_charter_elements_must_be_list(self):
        raw = json.dumps({
            "analysis": "a", "preflection": "p", "reflection": "r",
            "charter_elements": "1.1, 3.2",
        })
        with pytest.raises(AssertionError, match="charter_elements must be a list"):
            _parse_generation(raw)


class TestParseJudgment:
    def test_basic_json(self):
        raw = json.dumps({
            "scores": {"relevance": 4, "depth": 3, "charter_grounding": 5, "clarity": 4},
            "aggregate": 4.0,
            "decision": "accept",
            "reasoning": "good work",
        })
        result = _parse_judgment(raw)
        assert result["aggregate"] == 4.0
        assert result["decision"] == "accept"

    def test_reject_decision(self):
        raw = json.dumps({
            "scores": {"relevance": 2, "depth": 2, "charter_grounding": 2, "clarity": 2},
            "aggregate": 2.0,
            "decision": "reject",
            "reasoning": "poor quality",
        })
        result = _parse_judgment(raw)
        assert result["decision"] == "reject"

    def test_invalid_decision_raises(self):
        raw = json.dumps({
            "scores": {"relevance": 3}, "decision": "maybe", "reasoning": "idk",
        })
        with pytest.raises(AssertionError, match="Invalid decision"):
            _parse_judgment(raw)

    def test_empty_scores_raises(self):
        raw = json.dumps({
            "scores": {}, "decision": "accept", "reasoning": "ok",
        })
        with pytest.raises(AssertionError, match="scores must not be empty"):
            _parse_judgment(raw)

    def test_json_with_code_fence(self):
        raw = '```json\n{"scores": {"r": 4}, "aggregate": 4.0, "decision": "accept", "reasoning": "ok"}\n```'
        result = _parse_judgment(raw)
        assert result["decision"] == "accept"


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
        # Prevent load_dotenv from re-loading the key
        monkeypatch.setattr("pipeline.phase2.run.load_dotenv", lambda: None)

        from pipeline.config import load_config
        from pipeline.phase2.run import make_api_client

        cfg = load_config()
        with pytest.raises(AssertionError, match="SWISS_AI_API_KEY"):
            make_api_client(cfg)


class TestIntegration:
    """Integration test: mock _api_call, run iteration with 3 items."""

    @pytest.fixture(autouse=True)
    def tmp_dirs(self, tmp_path):
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / "data" / "pipeline"
        self.data_dir.mkdir(parents=True)
        self.ann_dir = tmp_path / "data" / "annotation"
        self.ann_dir.mkdir(parents=True)

    def test_generate_and_judge(self):
        """Test generate_batch and judge_batch with mocked API calls."""
        gen_response = json.dumps({
            "analysis": "test analysis",
            "preflection": "test preflection",
            "reflection": "test reflection",
            "charter_elements": ["1.1"],
        })
        judge_response = json.dumps({
            "scores": {"relevance": 4, "specificity": 3, "charter_grounding": 4, "voice_tone": 4},
            "aggregate": 3.75,
            "decision": "reject",
            "reasoning": "slightly below threshold",
        })

        import asyncio
        import openai

        mock_client = AsyncMock(spec=openai.AsyncOpenAI)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]

        call_count = {"n": 0}

        async def mock_create(**kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            msg = resp.choices[0].message
            msg.reasoning_content = None  # simulate non-reasoning model
            # First 3 calls are generation, next 6 are judging (2 per item: preflection + reflection)
            if call_count["n"] <= 3:
                msg.content = gen_response
            else:
                msg.content = judge_response
            return resp

        mock_client.chat.completions.create = mock_create

        items = [
            {"item_id": f"item_{i}", "subset": "score_0", "text": f"text {i}",
             "reflection_point": 2, "is_gold": False}
            for i in range(3)
        ]

        # Write a minimal prompt file
        prompt_dir = self.tmp_path / "prompts"
        prompt_dir.mkdir()
        gen_prompt = prompt_dir / "gen_v1.md"
        gen_prompt.write_text("Generate. Charter: {charter}")
        judge_prompt = prompt_dir / "judge_v1.md"
        judge_prompt.write_text("Judge {part_type}. Threshold: {accept_threshold}")

        semaphore = asyncio.Semaphore(10)

        with patch("pipeline.phase2.storage.PIPELINE_DATA_DIR", self.data_dir):
            from pipeline.phase2.run import generate_batch, judge_batch

            generated = generate_batch(
                items, gen_prompt, "charter text", "test-model",
                iteration=1, client=mock_client, semaphore=semaphore,
            )
            assert len(generated) == 3
            assert all(g["analysis"] == "test analysis" for g in generated)

            judged = judge_batch(
                generated, judge_prompt, "test-model",
                iteration=1, accept_threshold=4.0,
                client=mock_client, semaphore=semaphore,
            )
            assert len(judged) == 3
            # aggregate=3.75 < threshold=4.0 -> reject
            assert all(j["judgment"]["decision"] == "reject" for j in judged)
            # Verify separate preflection/reflection judgments
            for j in judged:
                assert "preflection" in j["judgment"]
                assert "reflection" in j["judgment"]
                assert "scores" in j["judgment"]["preflection"]
                assert "scores" in j["judgment"]["reflection"]

            # Verify JSONL was written (3 gen + 3 judged = 6 records)
            items_file = self.data_dir / "items.jsonl"
            assert items_file.exists()
            lines = [l for l in items_file.read_text().splitlines() if l.strip()]
            assert len(lines) == 6

    def test_save_false_no_write(self):
        """Test that save=False prevents writing to items.jsonl."""
        gen_response = json.dumps({
            "analysis": "test", "preflection": "p", "reflection": "r",
            "charter_elements": ["1.1"],
        })
        judge_response = json.dumps({
            "scores": {"relevance": 4}, "aggregate": 4.0,
            "decision": "accept", "reasoning": "ok",
        })

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

        items = [{"item_id": "item_0", "subset": "score_0", "text": "text 0",
                   "reflection_point": 2, "is_gold": False}]

        prompt_dir = self.tmp_path / "prompts"
        prompt_dir.mkdir()
        gen_prompt = prompt_dir / "gen_v1.md"
        gen_prompt.write_text("Generate. Charter: {charter}")
        judge_prompt = prompt_dir / "judge_v1.md"
        judge_prompt.write_text("Judge {part_type}. Threshold: {accept_threshold}")

        semaphore = asyncio.Semaphore(10)

        with patch("pipeline.phase2.storage.PIPELINE_DATA_DIR", self.data_dir):
            from pipeline.phase2.run import generate_batch, judge_batch

            generated = generate_batch(
                items, gen_prompt, "charter text", "test-model",
                iteration=1, client=mock_client, semaphore=semaphore, save=False,
            )
            assert len(generated) == 1

            judged = judge_batch(
                generated, judge_prompt, "test-model",
                iteration=1, accept_threshold=4.0,
                client=mock_client, semaphore=semaphore, save=False,
            )
            assert len(judged) == 1

            # No JSONL should have been written
            items_file = self.data_dir / "items.jsonl"
            assert not items_file.exists()
