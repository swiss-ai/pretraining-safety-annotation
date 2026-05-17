"""Tests for pipeline run: parsing, item selection, and integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.api import extract_json
from pipeline.config import extract_charter_elements, union_charter_elements
from pipeline.charter.improve.run import _parse_generation, _parse_mode_judgment


class TestExtractJson:
    def test_pure_json(self):
        raw = '{"key": "value"}'
        assert extract_json(raw) == {"key": "value"}

    def test_leading_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        assert extract_json(raw) == {"key": "value"}

    def test_code_fence_anywhere(self):
        raw = '## Stage 1\nSome prose here.\n```json\n{"key": "value"}\n```\n'
        assert extract_json(raw) == {"key": "value"}

    def test_prose_before_json(self):
        raw = '## Analysis\nThis is prose.\n\n{"key": "value"}'
        assert extract_json(raw) == {"key": "value"}

    def test_prose_before_and_after_json(self):
        raw = 'Here is the result:\n{"a": 1, "b": 2}\nDone.'
        assert extract_json(raw) == {"a": 1, "b": 2}

    def test_nested_braces(self):
        raw = 'Output:\n{"scores": {"r": 4, "s": 3}, "reasoning": "ok"}'
        result = extract_json(raw)
        assert result["scores"] == {"r": 4, "s": 3}

    def test_strings_with_braces(self):
        raw = '{"text": "a {placeholder} here", "n": 1}'
        result = extract_json(raw)
        assert result["text"] == "a {placeholder} here"

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("No JSON here at all.")

    def test_latex_backslash_invalid_escape(self):
        # \l in \left is not a valid JSON escape; without the repair pass this
        # raises and the wrapper survives as literal text in saved summaries.
        # With the repair, the JSON parses and LaTeX intent is preserved.
        raw = r'{"summary": "Formula $\left(x\right)$ holds."}'
        result = extract_json(raw)
        assert result["summary"] == r"Formula $\left(x\right)$ holds."

    def test_repair_does_not_break_valid_escapes(self):
        # \" \\ \/ should survive untouched.
        raw = r'{"a": "quote \" backslash \\ slash \/"}'
        result = extract_json(raw)
        assert result["a"] == 'quote " backslash \\ slash /'


class TestExtractCharterElements:
    def test_single_bracket(self):
        assert extract_charter_elements("see [1.2] for context") == ["1.2"]

    def test_consecutive_brackets(self):
        # [1.2][1.4] form
        assert extract_charter_elements("relevant [1.2][1.4] here") == ["1.2", "1.4"]

    def test_comma_separated_in_one_bracket(self):
        # [1.2,1.4] form
        assert extract_charter_elements("relevant [1.2,1.4] here") == ["1.2", "1.4"]

    def test_comma_separated_with_whitespace(self):
        assert extract_charter_elements("[1.2, 1.4]") == ["1.2", "1.4"]

    def test_mixed_formats(self):
        text = "First [1.2], then [1.4][2.1] and [2.3, 2.4]."
        assert extract_charter_elements(text) == ["1.2", "1.4", "2.1", "2.3", "2.4"]

    def test_dedup_preserves_first_seen_order(self):
        text = "[1.4] then [1.2,1.4] and [1.2]"
        assert extract_charter_elements(text) == ["1.4", "1.2"]

    def test_unknown_id_filtered(self):
        # 99.99 is not in the charter
        assert extract_charter_elements("[99.99] [1.2]") == ["1.2"]

    def test_no_match_returns_empty(self):
        assert extract_charter_elements("nothing to cite") == []

    def test_brackets_without_ids(self):
        assert extract_charter_elements("[note] and [TODO]") == []


class TestUnionCharterElements:
    def test_union_dedupes_across_texts(self):
        a = "first [1.2] only"
        b = "second [1.2,1.4]"
        assert union_charter_elements(a, b) == ["1.2", "1.4"]

    def test_handles_none_inputs(self):
        assert union_charter_elements(None, "[1.2]", None) == ["1.2"]

    def test_all_none_returns_empty(self):
        assert union_charter_elements(None, None) == []

    def test_preserves_first_seen_order_across_args(self):
        assert union_charter_elements("[2.1]", "[1.2]") == ["2.1", "1.2"]


class TestParseGeneration:
    """Tests for _parse_generation.

    Covers the current schema (4-field preflection + 2-voice reflection) and
    legacy single-mode paths for backward-compat with older fixtures.
    """

    @staticmethod
    def _full_payload(**overrides) -> dict:
        """Build a complete current-schema payload, overriding any field."""
        base = {
            "analysis": "a",
            "charter_summary": "cs",
            "neutral": "n",
            "judgemental": "j",
            "idealisation": "i",
            "reflection_1p": "r1",
            "reflection_3p": "r3",
        }
        base.update(overrides)
        return base

    def test_basic_json(self):
        raw = json.dumps(
            self._full_payload(
                analysis="test analysis",
                neutral="test neutral",
                reflection_1p="test reflection 1p",
            )
        )
        result = _parse_generation(raw)
        assert result["analysis"] == "test analysis"
        assert result["neutral"] == "test neutral"
        assert result["reflection_1p"] == "test reflection 1p"

    def test_json_with_code_fence(self):
        raw = "```json\n" + json.dumps(self._full_payload()) + "\n```"
        result = _parse_generation(raw)
        assert result["analysis"] == "a"

    def test_prose_before_code_fence(self):
        raw = (
            "## Stage 1 — Analysis\nSome analysis text.\n\n"
            "```json\n" + json.dumps(self._full_payload()) + "\n```"
        )
        result = _parse_generation(raw)
        assert result["analysis"] == "a"

    def test_prose_before_raw_json(self):
        raw = "## Stage 1\nProse.\n## Stage 2\nMore prose.\n\n" + json.dumps(
            self._full_payload()
        )
        result = _parse_generation(raw)
        assert result["analysis"] == "a"

    def test_field_name_normalization_spelling(self):
        # US spellings → canonical names
        raw = json.dumps(
            self._full_payload(
                judgmental="j-via-alias",
                idealization="i-via-alias",
            )
        )
        # Remove the canonical names so the aliases take effect
        payload = json.loads(raw)
        payload.pop("judgemental")
        payload.pop("idealisation")
        result = _parse_generation(json.dumps(payload))
        assert result["judgemental"] == "j-via-alias"
        assert result["idealisation"] == "i-via-alias"

    def test_reflection_only_required_fields(self):
        # Single-mode parse with explicit required_fields subset (reflection)
        raw = json.dumps(
            {"analysis": "a", "reflection_1p": "r1", "reflection_3p": "r3"}
        )
        result = _parse_generation(
            raw, required_fields={"analysis", "reflection_1p", "reflection_3p"}
        )
        assert result["reflection_1p"] == "r1"

    def test_preflection_only_required_fields(self):
        # Single-mode parse with the new 4-field preflection schema
        raw = json.dumps(
            {
                "analysis": "a",
                "charter_summary": "cs",
                "neutral": "n",
                "judgemental": "j",
                "idealisation": "i",
            }
        )
        result = _parse_generation(
            raw,
            required_fields={
                "analysis",
                "charter_summary",
                "neutral",
                "judgemental",
                "idealisation",
            },
        )
        assert result["charter_summary"] == "cs"
        assert result["idealisation"] == "i"

    def test_missing_field_raises(self):
        raw = json.dumps({"analysis": "a", "neutral": "n"})
        with pytest.raises(AssertionError, match="Missing fields"):
            _parse_generation(raw)


class TestParseModeJudgment:
    def test_basic_json_reflection(self):
        raw = json.dumps(
            {
                "reflection_1p": {
                    "scores": {
                        "relevance": 4,
                        "depth": 3,
                        "charter_grounding": 5,
                        "clarity": 4,
                    },
                    "reasoning": "good 1p",
                },
                "reflection_3p": {
                    "scores": {
                        "relevance": 4,
                        "depth": 3,
                        "charter_grounding": 5,
                        "clarity": 4,
                    },
                    "reasoning": "good 3p",
                },
            }
        )
        result = _parse_mode_judgment(raw, "reflection")
        assert "reflection_1p" in result
        assert "reflection_3p" in result
        assert result["reflection_1p"]["aggregate"] == 4.0

    def test_basic_json_preflection(self):
        # Current 4-field × 3-dim preflection schema.
        raw = json.dumps(
            {
                "charter_summary": {
                    "scores": {
                        "relevance": 4,
                        "charter_grounding": 4,
                        "class_discipline": 4,
                    },
                    "reasoning": "ok cs",
                },
                "neutral": {
                    "scores": {
                        "relevance": 4,
                        "charter_grounding": 4,
                        "class_discipline": 4,
                    },
                    "reasoning": "ok n",
                },
                "judgemental": {
                    "scores": {
                        "relevance": 4,
                        "charter_grounding": 4,
                        "class_discipline": 4,
                    },
                    "reasoning": "ok j",
                },
                "idealisation": {
                    "scores": {
                        "relevance": 4,
                        "charter_grounding": 4,
                        "class_discipline": 4,
                    },
                    "reasoning": "ok i",
                },
            }
        )
        result = _parse_mode_judgment(raw, "preflection")
        assert {"charter_summary", "neutral", "judgemental", "idealisation"} <= set(
            result.keys()
        )
        assert result["charter_summary"]["aggregate"] == 4.0
        assert result["idealisation"]["aggregate"] == 4.0

    def test_empty_scores_raises(self):
        raw = json.dumps(
            {
                "reflection_1p": {
                    "scores": {},
                    "reasoning": "ok",
                },
                "reflection_3p": {
                    "scores": {"r": 4},
                    "reasoning": "ok",
                },
            }
        )
        with pytest.raises(AssertionError, match="scores must be a non-empty dict"):
            _parse_mode_judgment(raw, "reflection")

    def test_json_with_code_fence(self):
        raw = '```json\n{"reflection_1p": {"scores": {"r": 4}, "reasoning": "ok"}, "reflection_3p": {"scores": {"r": 4}, "reasoning": "ok"}}\n```'
        result = _parse_mode_judgment(raw, "reflection")
        assert result["reflection_1p"]["aggregate"] == 4.0

    def test_decision_field_ignored_if_present(self):
        """Backward compat: decision field in judge output is silently ignored."""
        raw = json.dumps(
            {
                "reflection_1p": {
                    "scores": {"relevance": 4},
                    "decision": "accept",
                    "reasoning": "ok",
                },
                "reflection_3p": {
                    "scores": {"relevance": 4},
                    "reasoning": "ok",
                },
            }
        )
        result = _parse_mode_judgment(raw, "reflection")
        assert result["reflection_1p"]["aggregate"] == 4.0


class TestMakeApiClient:
    def test_returns_client_and_semaphore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWISS_AI_API_KEY", "test-key")

        from pipeline.api import make_api_client

        client, semaphore = make_api_client("https://example.com/v1", 10)
        assert client is not None
        assert semaphore is not None

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("SWISS_AI_API_KEY", raising=False)

        from pipeline.api import make_api_client

        with pytest.raises(AssertionError, match="SWISS_AI_API_KEY"):
            make_api_client("https://example.com/v1", 10)


class TestIntegration:
    """Integration test: mock _api_call, run generate/judge with SQLite storage."""

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path):
        """Redirect SQLite storage to a temp DB and reset all thread-local
        state (connection, read cache, bump version) so cached query results
        from a previous test can't leak into this one."""
        self.tmp_path = tmp_path
        import pipeline.storage as _mod

        def _reset_thread_local():
            for attr in ("conn", "read_cache", "bump_version"):
                _mod._local.__dict__.pop(attr, None)

        original = _mod.DB_PATH
        _mod.DB_PATH = tmp_path / "test.db"
        _reset_thread_local()
        yield
        _mod.DB_PATH = original
        _reset_thread_local()

    @staticmethod
    def _make_mock_client(judge_response: str | None = None):
        """Build an AsyncOpenAI mock that returns the right shape for each call.

        Generate calls are answered with reflection or preflection payloads
        depending on which mode the user message asks for. Judge calls return
        the supplied judge_response.
        """
        import openai

        judge_refl_response = None
        judge_prefl_response = None
        if judge_response is None:
            judge_refl_response = json.dumps(
                {
                    "reflection_1p": {
                        "scores": {
                            "relevance": 4,
                            "specificity": 3,
                            "charter_grounding": 4,
                            "voice_tone": 4,
                        },
                        "reasoning": "slightly below threshold",
                    },
                    "reflection_3p": {
                        "scores": {
                            "relevance": 4,
                            "specificity": 3,
                            "charter_grounding": 4,
                            "voice_tone": 4,
                        },
                        "reasoning": "slightly below threshold",
                    },
                }
            )
            # Current 4-field × 3-dim preflection judge schema.
            judge_prefl_response = json.dumps(
                {
                    field: {
                        "scores": {
                            "relevance": 4,
                            "charter_grounding": 3,
                            "class_discipline": 4,
                        },
                        "reasoning": "slightly below threshold",
                    }
                    for field in (
                        "charter_summary",
                        "neutral",
                        "judgemental",
                        "idealisation",
                    )
                }
            )
        else:
            # Caller provided a custom judge response for both modes.
            judge_refl_response = judge_response
            judge_prefl_response = judge_response

        refl_response = json.dumps(
            {
                "analysis": "refl analysis",
                "reflection_1p": "test reflection 1p per [1.1]",
                "reflection_3p": "test reflection 3p per [1.1]",
            }
        )
        # Current 4-field preflection schema.
        prefl_response = json.dumps(
            {
                "analysis": "prefl analysis",
                "charter_summary": "cs content [1.1]",
                "neutral": "n content [1.1]",
                "judgemental": "j content [1.1]",
                "idealisation": "i content [1.1]",
            }
        )

        mock_client = AsyncMock(spec=openai.AsyncOpenAI)

        async def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = next((m["content"] for m in messages if m["role"] == "user"), "")

            resp = MagicMock()
            resp.choices = [MagicMock()]
            msg = resp.choices[0].message
            msg.reasoning_content = None

            if system.startswith("Generate"):
                if "Reflection mode" in user:
                    msg.content = refl_response
                else:
                    msg.content = prefl_response
            else:
                # Judge calls: detect mode from the fields rendered into the
                # user content. Preflection mode includes "## charter_summary"
                # etc. as its per-field sections; reflection mode uses the
                # two voice headers.
                if "## charter_summary" in user or "## neutral" in user:
                    msg.content = judge_prefl_response
                else:
                    msg.content = judge_refl_response

            # Numeric usage so api_call's token bookkeeping works.
            resp.usage = MagicMock()
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50
            resp.usage.reasoning_tokens = 0
            resp.usage.completion_tokens_details = {}
            return resp

        mock_client.chat.completions.create = mock_create
        return mock_client

    def test_generate_and_judge(self):
        """Test generate_batch and judge_batch end-to-end with mocked API calls."""
        import asyncio

        mock_client = self._make_mock_client()

        # Items need text long enough that the reflection point split leaves
        # both halves non-empty (RP=20 over 80 chars).
        items = [
            {
                "item_id": f"item_{i}",
                "subset": "score_0",
                "text": (f"text body {i} " * 8).strip(),
                "reflection_point": 20,
                "is_gold": False,
            }
            for i in range(3)
        ]

        prompt_dir = self.tmp_path / "prompts"
        prompt_dir.mkdir()
        gen_refl_prompt = prompt_dir / "gen_reflection_v1.md"
        gen_refl_prompt.write_text("Generate. Charter: {charter}")
        gen_prefl_prompt = prompt_dir / "gen_preflection_v1.md"
        gen_prefl_prompt.write_text("Generate. Charter: {charter}")
        judge_refl_prompt = prompt_dir / "judge_reflection_v1.md"
        judge_refl_prompt.write_text("Judge {part_type}. Threshold: {accept_threshold}")
        judge_prefl_prompt = prompt_dir / "judge_preflection_v1.md"
        judge_prefl_prompt.write_text(
            "Judge {part_type}. Threshold: {accept_threshold}"
        )

        semaphore = asyncio.Semaphore(10)

        from pipeline.charter.improve.run import generate_batch, judge_batch

        generated = generate_batch(
            items,
            gen_refl_prompt,
            gen_prefl_prompt,
            "charter text",
            "test-model",
            iteration=1,
            client=mock_client,
            semaphore=semaphore,
        )
        assert len(generated) == 3
        for g in generated:
            assert "REFLECTION ANALYSIS" in g["analysis"]
            assert "PREFLECTION ANALYSIS" in g["analysis"]
            assert g["charter_summary"] == "cs content [1.1]"
            assert g["neutral"] == "n content [1.1]"
            assert g["judgemental"] == "j content [1.1]"
            assert g["idealisation"] == "i content [1.1]"
            assert g["reflection_3p"] == "test reflection 3p per [1.1]"

        judged = judge_batch(
            generated,
            judge_refl_prompt,
            judge_prefl_prompt,
            "test-model",
            iteration=1,
            accept_threshold=4.0,
            client=mock_client,
            semaphore=semaphore,
        )
        assert len(judged) == 3
        # Aggregate across 6 parts: reflection 2 voices × 4 dims (mean 3.75)
        # and preflection 4 fields × 3 dims (mean ~3.67) → overall below 4
        # → reject.
        assert all(j["judgment"]["decision"] == "reject" for j in judged)
        for j in judged:
            for voice in (
                "charter_summary",
                "neutral",
                "judgemental",
                "idealisation",
                "reflection_1p",
                "reflection_3p",
            ):
                assert voice in j["judgment"]
                assert "scores" in j["judgment"][voice]

        # Items were saved by generate_batch (and overwritten by judge_batch)
        from pipeline.charter.improve.storage import load_items_for_iteration

        rows = load_items_for_iteration(1)
        assert len(rows) == 3

    def test_save_false_no_write(self):
        """Test that save=False prevents writing to the database."""
        import asyncio

        mock_client = self._make_mock_client()

        items = [
            {
                "item_id": "item_0",
                "subset": "score_0",
                "text": ("text body " * 8).strip(),
                "reflection_point": 20,
                "is_gold": False,
            }
        ]

        prompt_dir = self.tmp_path / "prompts"
        prompt_dir.mkdir()
        gen_refl_prompt = prompt_dir / "gen_reflection_v1.md"
        gen_refl_prompt.write_text("Generate. Charter: {charter}")
        gen_prefl_prompt = prompt_dir / "gen_preflection_v1.md"
        gen_prefl_prompt.write_text("Generate. Charter: {charter}")
        judge_refl_prompt = prompt_dir / "judge_reflection_v1.md"
        judge_refl_prompt.write_text("Judge {part_type}. Threshold: {accept_threshold}")
        judge_prefl_prompt = prompt_dir / "judge_preflection_v1.md"
        judge_prefl_prompt.write_text(
            "Judge {part_type}. Threshold: {accept_threshold}"
        )

        semaphore = asyncio.Semaphore(10)

        from pipeline.charter.improve.run import generate_batch, judge_batch

        generated = generate_batch(
            items,
            gen_refl_prompt,
            gen_prefl_prompt,
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
            judge_refl_prompt,
            judge_prefl_prompt,
            "test-model",
            iteration=1,
            accept_threshold=4.0,
            client=mock_client,
            semaphore=semaphore,
            save=False,
        )
        assert len(judged) == 1

        # No items should have been written.
        from pipeline.charter.improve.storage import load_items_for_iteration

        rows = load_items_for_iteration(1)
        assert len(rows) == 0
