"""Tests for pipeline run: parsing, item selection, and integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.run import _parse_generation, _parse_judgment


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
            "scores": {}, "aggregate": 3.0,
            "decision": "maybe", "reasoning": "idk",
        })
        with pytest.raises(AssertionError, match="Invalid decision"):
            _parse_judgment(raw)

    def test_json_with_code_fence(self):
        raw = '```json\n{"scores": {"r": 4}, "aggregate": 4.0, "decision": "accept", "reasoning": "ok"}\n```'
        result = _parse_judgment(raw)
        assert result["decision"] == "accept"


class TestIntegration:
    """Integration test: mock _api_call, run iteration with 3 items."""

    @pytest.fixture(autouse=True)
    def tmp_dirs(self, tmp_path):
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / "data" / "pipeline"
        self.data_dir.mkdir(parents=True)
        self.ann_dir = tmp_path / "data" / "annotation"
        self.ann_dir.mkdir(parents=True)

    @pytest.mark.asyncio
    async def test_generate_and_judge(self):
        """Test generate_batch and judge_batch with mocked API calls."""
        gen_response = json.dumps({
            "analysis": "test analysis",
            "preflection": "test preflection",
            "reflection": "test reflection",
            "charter_elements": ["1.1"],
        })
        judge_response = json.dumps({
            "scores": {"relevance": 4, "depth": 3, "charter_grounding": 4, "clarity": 4},
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
            # First 3 calls are generation, next 3 are judging
            if call_count["n"] <= 3:
                resp.choices[0].message.content = gen_response
            else:
                resp.choices[0].message.content = judge_response
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
        judge_prompt.write_text("Judge. Threshold: {accept_threshold}")

        semaphore = asyncio.Semaphore(10)

        with patch("pipeline.storage.PIPELINE_DATA_DIR", self.data_dir):
            from pipeline.run import generate_batch, judge_batch

            generated = await generate_batch(
                items, gen_prompt, "charter text", "test-model",
                iteration=1, client=mock_client, semaphore=semaphore,
            )
            assert len(generated) == 3
            assert all(g["analysis"] == "test analysis" for g in generated)

            judged = await judge_batch(
                generated, judge_prompt, "test-model",
                iteration=1, accept_threshold=4.0,
                client=mock_client, semaphore=semaphore,
            )
            assert len(judged) == 3
            assert all(j["judgment"]["decision"] == "reject" for j in judged)

            # Verify JSONL was written (3 gen + 3 judged = 6 records)
            items_file = self.data_dir / "items.jsonl"
            assert items_file.exists()
            lines = [l for l in items_file.read_text().splitlines() if l.strip()]
            assert len(lines) == 6
