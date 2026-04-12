"""Tests for pipeline.phase3.eval_judges.

These tests are written BEFORE the implementation exists. Collection of this
file must succeed (no ImportError at collection time) — individual tests are
expected to fail at run time with ImportError until
pipeline/phase3/eval_judges.py exists.

The eval_judges module exposes one public function:

    def run_judge_eval(cfg: AppConfig, run_id: str) -> None:
        '''Path-B runner.

        1. Build item pool, write items.jsonl
        2. Generate from cfg.phase3.judge_eval.generator once.
        3. For each judge in [gold_judge] + dedup(judge_eval.candidates),
           judge those generations once.
        4. If cfg.phase3.judge_eval.include_reviewed:
             a. Build reviewed_items.jsonl from load_reviewed_items.
             b. For each judge in the same set, score those rows.
        5. Set metadata status=done, finished_at.
        '''

Observed via on-disk file presence and captured fake-call args.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fake implementations of generate_batch / judge_batch
# ---------------------------------------------------------------------------


def make_fake_generate(captured_seeds, captured_failures, fail_ids=frozenset()):
    """Build a fake stand-in for phase2.run.generate_batch."""

    def _fake(
        items,
        refl_prompt_path,
        prefl_prompt_path,
        charter_text,
        model,
        iteration,
        client,
        semaphore,
        save=True,
        writing_guidelines_text="",
        thinking=False,
        json_mode=False,
        canary_rng_seed=None,
        on_failure=None,
        mode=None,
        **kw,
    ):
        captured_seeds.append(canary_rng_seed)
        out = []
        for it in items:
            if it["item_id"] in fail_ids:
                if on_failure is not None:
                    fr = {
                        "item_id": it["item_id"],
                        "stage": "reflection",
                        "category": "parse",
                        "reason": "json_parse",
                        "raw": "...",
                        "raw_reasoning": None,
                    }
                    on_failure(fr)
                    captured_failures.append(fr)
                continue
            out.append(
                {
                    **it,
                    "iteration": iteration,
                    "model": model,
                    "preflection_1p": "p1",
                    "preflection_3p": "p3",
                    "reflection_1p": "r1",
                    "reflection_3p": "r3",
                    "judgment": None,
                    "canary": None,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "reasoning_tokens": 0,
                }
            )
        return out

    return _fake


def make_fake_judge(captured_calls=None, fail_ids=frozenset()):
    """Build a fake stand-in for phase2.run.judge_batch."""

    def _fake(
        items,
        refl_prompt_path,
        prefl_prompt_path,
        model,
        iteration,
        accept_threshold,
        client,
        semaphore,
        save=True,
        **kw,
    ):
        if captured_calls is not None:
            captured_calls.append(
                {
                    "model": model,
                    "prompt": (
                        refl_prompt_path.name
                        if hasattr(refl_prompt_path, "name")
                        else str(refl_prompt_path)
                    ),
                    "n_items": len(items),
                }
            )
        on_failure = kw.get("on_failure")
        out = []
        for it in items:
            if it["item_id"] in fail_ids:
                if on_failure is not None:
                    on_failure(
                        {
                            "item_id": it["item_id"],
                            "stage": "judge_reflection",
                            "category": "parse",
                            "reason": "missing_field",
                            "raw": "raw judge text",
                            "raw_reasoning": None,
                        }
                    )
                continue
            judgment = {
                "preflection_3p": {
                    "scores": {"relevance": 4},
                    "aggregate": 4.0,
                    "reasoning": "",
                },
                "preflection_1p": {
                    "scores": {"relevance": 4},
                    "aggregate": 4.0,
                    "reasoning": "",
                },
                "reflection_1p": {
                    "scores": {"relevance": 4},
                    "aggregate": 4.0,
                    "reasoning": "",
                },
                "reflection_3p": {
                    "scores": {"relevance": 4},
                    "aggregate": 4.0,
                    "reasoning": "",
                },
                "aggregate": 4.0,
                "decision": "accept",
                "judge_prompt": getattr(
                    refl_prompt_path, "name", str(refl_prompt_path)
                ),
                "raw_responses": {"combined": "raw"},
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "reasoning_tokens": 0,
                },
                "latency_ms": 1,
                "timestamp": "2026-04-09T00:00:00",
            }
            out.append({**it, "judgment": judgment})
        return out

    return _fake


# ---------------------------------------------------------------------------
# cfg builder
# ---------------------------------------------------------------------------


def _build_cfg(tmp_path):
    from pipeline.config import CandidateModel, load_config

    cfg = load_config()
    cfg.phase3.eval_dir = str(tmp_path / "phase3_eval")
    cfg.phase3.gold_judge = CandidateModel(
        alias="gold", api_name="gold-api", prompt="judge_v1.md"
    )
    cfg.phase3.judge_eval.generator = CandidateModel(
        alias="genA", api_name="genA-api", prompt="generator_v1.md"
    )
    cfg.phase3.judge_eval.candidates = [
        CandidateModel(alias="cand1", api_name="cand1-api", prompt="judge_v2.md"),
        CandidateModel(alias="cand2", api_name="cand2-api", prompt="judge_v3.md"),
    ]
    cfg.phase3.judge_eval.n_items = 5
    cfg.phase3.judge_eval.seed = 42
    cfg.phase3.judge_eval.chunk_size = 200
    cfg.phase3.judge_eval.include_reviewed = False
    return cfg


def _install_common_patches(monkeypatch, tmp_path, n_items_default=5):
    """Patch the standard rebound symbols on pipeline.phase3.eval_judges."""
    prompts_dir = tmp_path / "fake_prompts"
    prompts_dir.mkdir(exist_ok=True)

    def _fake_resolve(fn, alias):
        # Materialize a stub file per (alias, filename) so the runner has
        # something to read AND so prompt_path.name preserves the original
        # filename — tests that filter by prompt name rely on this.
        p = prompts_dir / f"{alias}__{fn}"
        if not p.exists():
            p.write_text("dummy")
        # Use a hardlink-style trick: create a sibling whose name == fn so
        # that path.name == fn (the runner only reads .name).
        named = prompts_dir / fn
        if not named.exists():
            named.write_text("dummy")
        return named

    monkeypatch.setattr(
        "pipeline.phase3.eval_judges.make_api_client",
        lambda *a, **kw: (MagicMock(), asyncio.Semaphore(1)),
    )
    monkeypatch.setattr(
        "pipeline.phase3.eval_judges.ensure_item_pool",
        lambda store, n_items, seed, max_tokens: [
            {
                "item_id": f"i{i}",
                "text": f"text {i}",
                "reflection_point": 5,
                "safety_score": i % 3,
            }
            for i in range(n_items)
        ],
    )
    monkeypatch.setattr(
        "pipeline.phase3.eval_judges.resolve_prompt_path",
        _fake_resolve,
    )
    return prompts_dir


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunJudgeEval:
    """Contract tests for pipeline.phase3.eval_judges.run_judge_eval."""

    def test_full_run_creates_judgment_per_judge(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        captured_judge_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(captured_judge_calls))

        cfg = _build_cfg(tmp_path)
        mod.run_judge_eval(cfg, "run-full")

        run_dir = tmp_path / "phase3_eval" / "run-full"
        gens = _read_jsonl(run_dir / "generations" / "genA__generator_v1.md.jsonl")
        assert len(gens) == 5

        j_gold = _read_jsonl(
            run_dir / "judgments" / "gold__judge_v1.md__on__genA__generator_v1.md.jsonl"
        )
        j_c1 = _read_jsonl(
            run_dir
            / "judgments"
            / "cand1__judge_v2.md__on__genA__generator_v1.md.jsonl"
        )
        j_c2 = _read_jsonl(
            run_dir
            / "judgments"
            / "cand2__judge_v3.md__on__genA__generator_v1.md.jsonl"
        )
        assert len(j_gold) == 5
        assert len(j_c1) == 5
        assert len(j_c2) == 5

        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta.get("status") == "done"

    def test_gold_judge_dedup(self, tmp_path, monkeypatch):
        from pipeline.config import CandidateModel
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        captured_judge_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(captured_judge_calls))

        cfg = _build_cfg(tmp_path)
        # Include a candidate whose (alias, prompt) duplicates the gold judge.
        cfg.phase3.judge_eval.candidates = [
            CandidateModel(alias="gold", api_name="gold-api", prompt="judge_v1.md"),
            CandidateModel(alias="cand1", api_name="cand1-api", prompt="judge_v2.md"),
        ]

        mod.run_judge_eval(cfg, "run-dedup")

        # Count how many distinct judge_batch invocations used the gold prompt.
        gold_calls = [
            c
            for c in captured_judge_calls
            if c["model"] == "gold-api" and c["prompt"] == "judge_v1.md"
        ]
        # We expect the gold prompt to be judged ONCE total.
        assert len(gold_calls) == 1

        # And only one gold judgment file should exist.
        run_dir = tmp_path / "phase3_eval" / "run-dedup"
        judgments_dir = run_dir / "judgments"
        gold_files = [p for p in judgments_dir.glob("gold__judge_v1.md__on__*.jsonl")]
        assert len(gold_files) == 1

    def test_canary_seed_passed_to_generator(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        captured_judge_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(captured_judge_calls))

        cfg = _build_cfg(tmp_path)
        cfg.phase3.judge_eval.seed = 999

        mod.run_judge_eval(cfg, "run-seed")

        # Generation runs once, so exactly one seed captured, and it must be 999.
        assert 999 in captured_seeds
        assert captured_seeds.count(999) == 1

    def test_judge_batch_called_with_each_judge_model(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        captured_judge_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(captured_judge_calls))

        cfg = _build_cfg(tmp_path)
        mod.run_judge_eval(cfg, "run-models")

        models_seen = {c["model"] for c in captured_judge_calls}
        assert {"gold-api", "cand1-api", "cand2-api"} <= models_seen

    def test_resume_skips_done_judgments(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        first_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(first_calls))

        cfg = _build_cfg(tmp_path)
        mod.run_judge_eval(cfg, "run-resume")

        # Sanity: first run scored 5 items per judge.
        assert any(c["n_items"] == 5 for c in first_calls)

        # Second run with the same run_id should see 0 items in every
        # judge_batch invocation.
        second_calls: list = []
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(second_calls))
        mod.run_judge_eval(cfg, "run-resume")

        for c in second_calls:
            assert c["n_items"] == 0, f"Expected resume to skip all items, got call {c}"

    def test_include_reviewed_path_judges_reviewed_items(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        captured_judge_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(captured_judge_calls))

        reviewed_rows = [
            {
                "item_id": "r1",
                "iteration": 1,
                "text": "x",
                "reflection_point": 0,
                "preflection_1p": "p1",
                "preflection_3p": "p3",
                "reflection_1p": "r1v",
                "reflection_3p": "r3v",
                "human_review": {
                    "scores": {
                        "preflection_3p": {"relevance": 3},
                        "preflection_1p": {"relevance": 3},
                        "reflection_1p": {"relevance": 3},
                        "reflection_3p": {"relevance": 3},
                    },
                    "aggregate": 3.0,
                },
            },
            {
                "item_id": "r2",
                "iteration": 1,
                "text": "y",
                "reflection_point": 0,
                "preflection_1p": "p1",
                "preflection_3p": "p3",
                "reflection_1p": "r1v",
                "reflection_3p": "r3v",
                "human_review": {
                    "scores": {
                        "preflection_3p": {"relevance": 4},
                        "preflection_1p": {"relevance": 4},
                        "reflection_1p": {"relevance": 4},
                        "reflection_3p": {"relevance": 4},
                    },
                    "aggregate": 4.0,
                },
            },
        ]
        monkeypatch.setattr(
            "pipeline.phase3.eval_judges.load_reviewed_items",
            lambda reviewer_policy="average": list(reviewed_rows),
        )

        cfg = _build_cfg(tmp_path)
        cfg.phase3.judge_eval.include_reviewed = True

        mod.run_judge_eval(cfg, "run-reviewed")

        run_dir = tmp_path / "phase3_eval" / "run-reviewed"
        reviewed = _read_jsonl(run_dir / "reviewed_items.jsonl")
        assert len(reviewed) == 2

        j_gold = _read_jsonl(
            run_dir / "judgments" / "gold__judge_v1.md__on__reviewed.jsonl"
        )
        j_c1 = _read_jsonl(
            run_dir / "judgments" / "cand1__judge_v2.md__on__reviewed.jsonl"
        )
        j_c2 = _read_jsonl(
            run_dir / "judgments" / "cand2__judge_v3.md__on__reviewed.jsonl"
        )
        assert len(j_gold) == 2
        assert len(j_c1) == 2
        assert len(j_c2) == 2

    def test_reviewed_resume_uses_composite_key(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        first_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(first_calls))

        # Note: "r1" intentionally shares an item_id with nothing in the
        # generator items ("i0".."i4"), but we also add a reviewed row with
        # item_id "i0" that already appears in the generations file to prove
        # the composite key prevents leaking resume state across files.
        reviewed_rows = [
            {
                "item_id": "i0",
                "iteration": 1,
                "text": "dup with generator",
                "reflection_point": 0,
                "preflection_1p": "p1",
                "preflection_3p": "p3",
                "reflection_1p": "r1v",
                "reflection_3p": "r3v",
                "human_review": {
                    "scores": {
                        "preflection_3p": {"relevance": 3},
                        "preflection_1p": {"relevance": 3},
                        "reflection_1p": {"relevance": 3},
                        "reflection_3p": {"relevance": 3},
                    },
                    "aggregate": 3.0,
                },
            },
            {
                "item_id": "r2",
                "iteration": 1,
                "text": "y",
                "reflection_point": 0,
                "preflection_1p": "p1",
                "preflection_3p": "p3",
                "reflection_1p": "r1v",
                "reflection_3p": "r3v",
                "human_review": {
                    "scores": {
                        "preflection_3p": {"relevance": 4},
                        "preflection_1p": {"relevance": 4},
                        "reflection_1p": {"relevance": 4},
                        "reflection_3p": {"relevance": 4},
                    },
                    "aggregate": 4.0,
                },
            },
        ]
        monkeypatch.setattr(
            "pipeline.phase3.eval_judges.load_reviewed_items",
            lambda reviewer_policy="average": list(reviewed_rows),
        )

        cfg = _build_cfg(tmp_path)
        cfg.phase3.judge_eval.include_reviewed = True
        mod.run_judge_eval(cfg, "run-reviewed-resume")

        # Second run: all reviewed judgments should already be done.
        second_calls: list = []
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(second_calls))
        mod.run_judge_eval(cfg, "run-reviewed-resume")

        for c in second_calls:
            assert c["n_items"] == 0, f"Expected resume to skip all items, got call {c}"

    def test_failures_recorded_per_judge(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        captured_judge_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )

        # Dispatch: gold judge fails on i0/i1; candidates never fail.
        def _judge_dispatcher(
            items,
            refl_prompt_path,
            prefl_prompt_path,
            model,
            iteration,
            accept_threshold,
            client,
            semaphore,
            save=True,
            **kw,
        ):
            if model == "gold-api":
                fake = make_fake_judge(
                    captured_judge_calls, fail_ids=frozenset({"i0", "i1"})
                )
            else:
                fake = make_fake_judge(captured_judge_calls)
            return fake(
                items,
                refl_prompt_path,
                prefl_prompt_path,
                model,
                iteration,
                accept_threshold,
                client,
                semaphore,
                save=save,
                **kw,
            )

        monkeypatch.setattr(mod, "judge_batch", _judge_dispatcher)

        cfg = _build_cfg(tmp_path)
        mod.run_judge_eval(cfg, "run-fail")

        run_dir = tmp_path / "phase3_eval" / "run-fail"

        gold_failures = _read_jsonl(
            run_dir
            / "failures"
            / "jud_gold__judge_v1.md__on__genA__generator_v1.md.jsonl"
        )
        assert len(gold_failures) == 2
        for row in gold_failures:
            assert row.get("category") == "parse"
            assert row.get("raw") == "raw judge text"

        j_gold = _read_jsonl(
            run_dir / "judgments" / "gold__judge_v1.md__on__genA__generator_v1.md.jsonl"
        )
        assert len(j_gold) == 3

        j_c1 = _read_jsonl(
            run_dir
            / "judgments"
            / "cand1__judge_v2.md__on__genA__generator_v1.md.jsonl"
        )
        j_c2 = _read_jsonl(
            run_dir
            / "judgments"
            / "cand2__judge_v3.md__on__genA__generator_v1.md.jsonl"
        )
        assert len(j_c1) == 5
        assert len(j_c2) == 5

    def test_run_dir_under_eval_dir(self, tmp_path, monkeypatch):
        from pipeline.phase3 import eval_judges as mod

        _install_common_patches(monkeypatch, tmp_path)

        captured_seeds: list = []
        captured_failures: list = []
        captured_judge_calls: list = []
        monkeypatch.setattr(
            mod,
            "generate_batch",
            make_fake_generate(captured_seeds, captured_failures),
        )
        monkeypatch.setattr(mod, "judge_batch", make_fake_judge(captured_judge_calls))

        cfg = _build_cfg(tmp_path)
        mod.run_judge_eval(cfg, "run-x")

        assert (tmp_path / "phase3_eval" / "run-x").exists()
        # Must not have leaked into the repo's default data/pipeline path.
        from pipeline.config import PIPELINE_DATA_DIR

        assert not (PIPELINE_DATA_DIR / "phase3_eval" / "run-x").exists()
