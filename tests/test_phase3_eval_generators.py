"""Tests for pipeline.phase3.eval_generators.run_generator_eval.

These tests are written BEFORE the implementation exists. They describe the
contract the Path-A generator-eval runner must satisfy. Running them now
(before the module is created) is expected to fail with ImportError —
that's the test-first workflow.

The runner's contract (summary):

    run_generator_eval(cfg, run_id) -> None

    For each candidate generator:
      1. _generate_with_resume: produces
         generations/<gen_alias>__<prompt_id>.jsonl via generate_batch
      2. _judge_with_resume: produces
         judgments/<gold_alias>__<gold_pid>__on__<gen_alias>__<gen_pid>.jsonl
         via judge_batch

    - Uses cfg.phase3.eval_dir as the run-store root.
    - Per-item resume via the JSONL store.
    - Failures are recorded via store.record_failure under:
        failures/gen_<gen_alias>__<prompt_id>.jsonl
        failures/jud_<gold_alias>__<gold_pid>__on__<gen_alias>__<gen_pid>.jsonl
      with a normalized `category: "api" | "parse"` field.
    - The same canary_rng_seed is passed to every generator call so canary
      subsets match across candidates.
    - When finished: metadata["status"] == "done" and metadata["finished_at"]
      is populated.
    - failure_attempt_cap is honored across runs via the in-memory failures
      attempt counter backed by failures/*.jsonl.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fake_items(n: int = 5) -> list[dict]:
    return [
        {
            "item_id": f"i{i}",
            "text": f"text {i}",
            "reflection_point": 5,
            "safety_score": i % 3,
            "subset": "dolma3",
            "is_gold": False,
        }
        for i in range(n)
    ]


def _make_fake_ensure_item_pool(items: list[dict]):
    """Return a fake ensure_item_pool that seeds the store and returns items."""

    def _fake(store, n_items, seed, max_tokens):
        # Materialize items.jsonl so the runner (or resume logic) can
        # observe them, matching the real helper's contract.
        existing = store.read_all("items.jsonl")
        if not existing:
            for it in items:
                store.append("items.jsonl", it)
            store.flush()
            meta = store.read_metadata()
            meta.update(
                {
                    "n_items": n_items,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "dataset_revision": "fake-rev",
                    "_original_dataset_revision": "fake-rev",
                }
            )
            store.write_metadata(meta)
        return list(items)

    return _fake


def _make_fake_generate(
    captured_canary_seeds: list,
    captured_on_failure_calls: list,
    fail_item_ids: frozenset[str] = frozenset(),
    captured_items_lists: list | None = None,
):
    """Build a fake generate_batch.

    - Records every canary_rng_seed passed into `captured_canary_seeds`.
    - For items whose id is in `fail_item_ids`, invokes `on_failure`
      with a parse-failure record and drops the item.
    - For every other item, returns a minimally-shaped generation row.
    - If `captured_items_lists` is provided, each invocation's items
      argument is appended (as a list of item_ids) for later inspection.
    """

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
        captured_canary_seeds.append(canary_rng_seed)
        on_result = kw.get("on_result")
        if captured_items_lists is not None:
            captured_items_lists.append([it["item_id"] for it in items])
        out = []
        for it in items:
            if it["item_id"] in fail_item_ids:
                if on_failure is not None:
                    fr = {
                        "item_id": it["item_id"],
                        "stage": "reflection",
                        "category": "parse",
                        "reason": "json_parse",
                        "raw": "raw response text",
                        "raw_reasoning": None,
                    }
                    on_failure(fr)
                    captured_on_failure_calls.append(fr)
                continue
            record = {
                **it,
                "iteration": iteration,
                "model": model,
                "analysis": f"a-{it['item_id']}",
                "preflection_1p": "p1",
                "preflection_3p": "p3",
                "reflection_1p": "r1",
                "reflection_3p": "r3",
                "raw_response": "...",
                "reasoning": None,
                "judgment": None,
                "canary": None,
                "input_tokens": 1,
                "output_tokens": 1,
                "reasoning_tokens": 0,
            }
            if on_result is not None:
                on_result(record)
            out.append(record)
        return out

    return _fake


def _make_fake_judge(
    captured_on_failure_calls: list | None = None,
    fail_item_ids: frozenset[str] = frozenset(),
):
    """Build a fake judge_batch.

    - For items whose id is in `fail_item_ids`, invokes `on_failure` (from kw)
      with a parse-failure record and drops the item.
    - For every other item, returns a minimally-shaped judgment row.
    """

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
        on_failure = kw.get("on_failure")
        on_result = kw.get("on_result")
        out = []
        for it in items:
            if it["item_id"] in fail_item_ids:
                if on_failure is not None:
                    fr = {
                        "item_id": it["item_id"],
                        "stage": "judge_reflection",
                        "category": "parse",
                        "reason": "missing_field",
                        "raw": "judge raw text",
                        "raw_reasoning": None,
                    }
                    on_failure(fr)
                    if captured_on_failure_calls is not None:
                        captured_on_failure_calls.append(fr)
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
                "judge_prompt": Path(refl_prompt_path).name,
                "raw_responses": {"combined": "raw"},
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "reasoning_tokens": 0,
                },
                "latency_ms": 1,
                "timestamp": "2026-04-09T00:00:00",
            }
            record = {**it, "judgment": judgment}
            if on_result is not None:
                on_result(record)
            out.append(record)
        return out

    return _fake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_cfg(tmp_path):
    """A small AppConfig with two candidate generators, eval_dir under tmp_path."""
    from pipeline.config import CandidateModel, load_config

    cfg = load_config()
    cfg.phase3.eval_dir = str(tmp_path / "phase3_eval")
    cfg.phase3.gold_judge = CandidateModel(
        alias="gold-judge",
        api_name="api/gold-judge",
        prompt_reflection="judge_v1.md",
        prompt_preflection="judge_v1.md",
        thinking=False,
        json_mode=False,
    )
    cfg.phase3.generator_eval.candidates = [
        CandidateModel(
            alias="gen0",
            api_name="api/gen0",
            prompt_reflection="generator_v1.md",
            prompt_preflection="generator_v1.md",
            thinking=False,
            json_mode=False,
        ),
        CandidateModel(
            alias="gen1",
            api_name="api/gen1",
            prompt_reflection="generator_v2.md",
            prompt_preflection="generator_v2.md",
            thinking=False,
            json_mode=False,
        ),
    ]
    cfg.phase3.generator_eval.gold_prompt_reflection = ""
    cfg.phase3.generator_eval.gold_prompt_preflection = ""
    cfg.phase3.generator_eval.n_items = 5
    cfg.phase3.generator_eval.seed = 42
    cfg.phase3.generator_eval.failure_attempt_cap = 3
    return cfg


@pytest.fixture
def patched_runner(monkeypatch, tmp_path, eval_cfg):
    """Importable handle to the runner module with most internals pre-patched.

    Yields a SimpleNamespace with:
        - module: the eval_generators module
        - run: run_generator_eval callable
        - cfg: the eval_cfg fixture (forwarded)
        - tmp_path: tmp_path fixture (forwarded)
        - captured_canary_seeds: list populated by the fake generate_batch
        - captured_gen_items: list of per-call item_id lists
        - captured_gen_failures: list of failure records from generation
        - captured_jud_failures: list of failure records from judging
        - install_fakes(fail_gen_ids, fail_jud_ids): helper that (re)installs
          fakes with specified failure item-id sets
    """
    from types import SimpleNamespace

    # Import will fail at run time if the module doesn't exist — that's
    # exactly the signal we want the failing tests to produce. Collection
    # still succeeds because this import is inside the fixture body.
    from pipeline.phase3 import eval_generators as eg

    # Ensure a fake prompt file exists so any code path that stat-checks it
    # will pass — even though resolve_prompt_path is also monkey-patched.
    fake_prompt = tmp_path / "fake_prompt.md"
    fake_prompt.write_text("dummy prompt body")

    monkeypatch.setattr(
        eg,
        "resolve_prompt_path",
        lambda fn, alias: fake_prompt,
    )
    monkeypatch.setattr(
        eg,
        "make_api_client",
        lambda *a, **kw: (MagicMock(), asyncio.Semaphore(1)),
    )
    monkeypatch.setattr(
        eg,
        "ensure_item_pool",
        _make_fake_ensure_item_pool(_fake_items(5)),
    )

    captured_canary_seeds: list = []
    captured_gen_items: list = []
    captured_gen_failures: list = []
    captured_jud_failures: list = []

    def install_fakes(
        fail_gen_ids: frozenset[str] = frozenset(),
        fail_jud_ids: frozenset[str] = frozenset(),
    ) -> None:
        monkeypatch.setattr(
            eg,
            "generate_batch",
            _make_fake_generate(
                captured_canary_seeds,
                captured_gen_failures,
                fail_item_ids=fail_gen_ids,
                captured_items_lists=captured_gen_items,
            ),
        )
        monkeypatch.setattr(
            eg,
            "judge_batch",
            _make_fake_judge(
                captured_on_failure_calls=captured_jud_failures,
                fail_item_ids=fail_jud_ids,
            ),
        )

    # Default install: nothing fails.
    install_fakes()

    return SimpleNamespace(
        module=eg,
        run=eg.run_generator_eval,
        cfg=eval_cfg,
        tmp_path=tmp_path,
        captured_canary_seeds=captured_canary_seeds,
        captured_gen_items=captured_gen_items,
        captured_gen_failures=captured_gen_failures,
        captured_jud_failures=captured_jud_failures,
        install_fakes=install_fakes,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestRunGeneratorEval:
    """Contract tests for pipeline.phase3.eval_generators.run_generator_eval."""

    # ---- file naming helpers -----------------------------------------------

    # File naming preserves the full prompt filename (incl. ".md") to match
    # the convention in the plan and the rank tests.

    @staticmethod
    def _pid(c) -> str:
        from pipeline.phase3.eval_generators import _prompt_id

        return _prompt_id(c)

    @staticmethod
    def _gen_rel(cand) -> str:
        from pipeline.phase3.eval_generators import _prompt_id

        return f"generations/{cand.alias}__{_prompt_id(cand)}.jsonl"

    @staticmethod
    def _jud_rel(gold, cand) -> str:
        from pipeline.phase3.eval_generators import _prompt_id

        return (
            f"judgments/{gold.alias}__{_prompt_id(gold)}"
            f"__on__{cand.alias}__{_prompt_id(cand)}.jsonl"
        )

    @staticmethod
    def _fail_gen_rel(cand) -> str:
        from pipeline.phase3.eval_generators import _prompt_id

        return f"failures/gen_{cand.alias}__{_prompt_id(cand)}.jsonl"

    @staticmethod
    def _fail_jud_rel(gold, cand) -> str:
        from pipeline.phase3.eval_generators import _prompt_id

        return (
            f"failures/jud_{gold.alias}__{_prompt_id(gold)}"
            f"__on__{cand.alias}__{_prompt_id(cand)}.jsonl"
        )

    @staticmethod
    def _run_dir(cfg, run_id: str) -> Path:
        return Path(cfg.phase3.eval_dir) / run_id

    # ---- tests -------------------------------------------------------------

    def test_full_run_creates_expected_files(self, patched_runner):
        cfg = patched_runner.cfg
        run_id = "run1"
        patched_runner.run(cfg, run_id)

        run_dir = self._run_dir(cfg, run_id)
        assert run_dir.exists(), f"run dir not created: {run_dir}"
        assert (run_dir / "items.jsonl").exists(), "items.jsonl missing"

        g0, g1 = cfg.phase3.generator_eval.candidates
        gold = cfg.phase3.gold_judge

        gen_rows_0 = _read_jsonl(run_dir / self._gen_rel(g0))
        gen_rows_1 = _read_jsonl(run_dir / self._gen_rel(g1))
        assert (
            len(gen_rows_0) == 5
        ), f"expected 5 gen rows for g0, got {len(gen_rows_0)}"
        assert (
            len(gen_rows_1) == 5
        ), f"expected 5 gen rows for g1, got {len(gen_rows_1)}"

        jud_rows_0 = _read_jsonl(run_dir / self._jud_rel(gold, g0))
        jud_rows_1 = _read_jsonl(run_dir / self._jud_rel(gold, g1))
        assert (
            len(jud_rows_0) == 5
        ), f"expected 5 judgment rows for g0, got {len(jud_rows_0)}"
        assert (
            len(jud_rows_1) == 5
        ), f"expected 5 judgment rows for g1, got {len(jud_rows_1)}"

        meta_path = run_dir / "metadata.json"
        assert meta_path.exists(), "metadata.json missing"
        meta = json.loads(meta_path.read_text())
        assert (
            meta.get("status") == "done"
        ), f"metadata.status should be 'done', got {meta.get('status')!r}"
        assert meta.get(
            "finished_at"
        ), f"metadata.finished_at should be set, got {meta.get('finished_at')!r}"

    def test_resume_skips_done_items(self, patched_runner):
        cfg = patched_runner.cfg
        run_id = "run-resume"

        # First run: all 5 items succeed for both generators.
        patched_runner.run(cfg, run_id)

        run_dir = self._run_dir(cfg, run_id)
        g0, g1 = cfg.phase3.generator_eval.candidates
        gold = cfg.phase3.gold_judge

        first_gen0 = _read_jsonl(run_dir / self._gen_rel(g0))
        first_gen1 = _read_jsonl(run_dir / self._gen_rel(g1))
        first_jud0 = _read_jsonl(run_dir / self._jud_rel(gold, g0))
        first_jud1 = _read_jsonl(run_dir / self._jud_rel(gold, g1))
        assert len(first_gen0) == 5
        assert len(first_gen1) == 5
        assert len(first_jud0) == 5
        assert len(first_jud1) == 5

        # Reset capture lists and rerun. On resume, the runner should
        # discover everything is done and pass an empty items list (or not
        # invoke the fake at all) for both candidates.
        patched_runner.captured_canary_seeds.clear()
        patched_runner.captured_gen_items.clear()

        patched_runner.run(cfg, run_id)

        # Every invocation of generate_batch during the second run must
        # receive an empty items list — everything is already done.
        for items_list in patched_runner.captured_gen_items:
            assert (
                items_list == []
            ), f"resume leaked un-done items to generate_batch: {items_list}"

        # File row counts unchanged.
        assert _read_jsonl(run_dir / self._gen_rel(g0)) == first_gen0
        assert _read_jsonl(run_dir / self._gen_rel(g1)) == first_gen1
        assert _read_jsonl(run_dir / self._jud_rel(gold, g0)) == first_jud0
        assert _read_jsonl(run_dir / self._jud_rel(gold, g1)) == first_jud1

    def test_canary_seed_passed_through(self, patched_runner):
        cfg = patched_runner.cfg
        seed = cfg.phase3.generator_eval.seed  # 42 by default
        patched_runner.run(cfg, "run-canary")

        # Two generators -> two captured seeds, all equal to cfg seed.
        assert len(patched_runner.captured_canary_seeds) == 2, (
            f"expected 2 generate_batch calls, got "
            f"{len(patched_runner.captured_canary_seeds)}"
        )
        assert patched_runner.captured_canary_seeds == [seed, seed], (
            f"canary_rng_seed must match across generators: "
            f"{patched_runner.captured_canary_seeds}"
        )

    def test_canary_seed_change_changes_seed_passed(self, patched_runner):
        cfg = patched_runner.cfg
        cfg.phase3.generator_eval.seed = 999
        patched_runner.run(cfg, "run-canary-999")
        assert patched_runner.captured_canary_seeds == [
            999,
            999,
        ], f"expected [999, 999], got {patched_runner.captured_canary_seeds}"

    def test_failure_records_written_with_raw(self, patched_runner):
        cfg = patched_runner.cfg
        g0, g1 = cfg.phase3.generator_eval.candidates
        gold = cfg.phase3.gold_judge

        # Reinstall fakes: generator candidate g0 fails on {"i1", "i3"}.
        # The spec only requires g0's failure set; g1 is clean.
        # Our fake doesn't distinguish per-generator, so we set the fail set
        # globally and assert the g0 outputs specifically. g1 will also fail
        # on the same items here, which is acceptable — the assertions are
        # focused on g0 files.
        patched_runner.install_fakes(
            fail_gen_ids=frozenset({"i1", "i3"}),
        )

        patched_runner.run(cfg, "run-fail")
        run_dir = self._run_dir(cfg, "run-fail")

        fail_rows = _read_jsonl(run_dir / self._fail_gen_rel(g0))
        assert len(fail_rows) == 2, (
            f"expected 2 failure rows for g0, got {len(fail_rows)}: " f"{fail_rows}"
        )
        fail_ids = {r["item_id"] for r in fail_rows}
        assert fail_ids == {"i1", "i3"}, f"unexpected failure ids: {fail_ids}"

        for row in fail_rows:
            assert (
                row.get("category") == "parse"
            ), f"failure row missing/wrong category: {row}"
            assert "reason" in row, f"failure row missing reason: {row}"
            assert (
                row.get("raw") == "raw response text"
            ), f"failure row must preserve raw response text: {row}"
            assert "attempt" in row, f"failure row missing attempt: {row}"
            assert row["attempt"] >= 1, f"attempt must be >=1: {row}"

        gen_rows = _read_jsonl(run_dir / self._gen_rel(g0))
        assert (
            len(gen_rows) == 3
        ), f"generations should have 5-2=3 rows, got {len(gen_rows)}"

        jud_rows = _read_jsonl(run_dir / self._jud_rel(gold, g0))
        assert (
            len(jud_rows) == 3
        ), f"judgments should only cover succeeded items (3), got {len(jud_rows)}"

    def test_judge_failures_recorded(self, patched_runner):
        cfg = patched_runner.cfg
        g0, g1 = cfg.phase3.generator_eval.candidates
        gold = cfg.phase3.gold_judge

        # All generations succeed; judge fails on i0 for both generators.
        patched_runner.install_fakes(
            fail_gen_ids=frozenset(),
            fail_jud_ids=frozenset({"i0"}),
        )

        patched_runner.run(cfg, "run-judfail")
        run_dir = self._run_dir(cfg, "run-judfail")

        for cand in (g0, g1):
            gen_rows = _read_jsonl(run_dir / self._gen_rel(cand))
            assert (
                len(gen_rows) == 5
            ), f"gen rows for {cand.alias} should be 5, got {len(gen_rows)}"

            jud_rows = _read_jsonl(run_dir / self._jud_rel(gold, cand))
            assert len(jud_rows) == 4, (
                f"judgments should be 5-1=4 for {cand.alias}, " f"got {len(jud_rows)}"
            )

            jud_fail_rows = _read_jsonl(run_dir / self._fail_jud_rel(gold, cand))
            assert len(jud_fail_rows) == 1, (
                f"expected 1 judge-failure for {cand.alias}, "
                f"got {len(jud_fail_rows)}"
            )
            row = jud_fail_rows[0]
            assert row.get("item_id") == "i0"
            assert row.get("category") == "parse"
            assert row.get("raw") == "judge raw text"
            assert "attempt" in row

    def test_failure_attempt_cap_skips_repeated_failures(self, patched_runner):
        cfg = patched_runner.cfg
        assert cfg.phase3.generator_eval.failure_attempt_cap == 3
        g0, g1 = cfg.phase3.generator_eval.candidates

        # All runs: i0 parse-fails for every generator.
        patched_runner.install_fakes(fail_gen_ids=frozenset({"i0"}))

        run_id = "run-cap"
        for _ in range(3):
            patched_runner.run(cfg, run_id)

        # After 3 runs, attempt counter for i0 is at the cap for both gens.
        # On the FOURTH run, i0 must NOT be fed into generate_batch.
        patched_runner.captured_gen_items.clear()
        patched_runner.captured_canary_seeds.clear()

        patched_runner.run(cfg, run_id)

        # Each generator candidate should see at least one call whose items
        # list is observed. i0 must not appear in ANY of those items lists.
        assert (
            patched_runner.captured_gen_items
        ), "expected generate_batch to be invoked on the 4th run"
        for items_list in patched_runner.captured_gen_items:
            assert "i0" not in items_list, (
                f"i0 exceeded failure_attempt_cap but was still submitted: "
                f"{items_list}"
            )

    def test_run_dir_under_eval_dir(self, patched_runner):
        cfg = patched_runner.cfg
        run_id = "run-special"
        patched_runner.run(cfg, run_id)

        expected = Path(cfg.phase3.eval_dir) / run_id
        assert (
            expected.exists()
        ), f"run dir was not created under cfg.phase3.eval_dir: {expected}"

        # Also sanity-check that the runner did NOT create the run under the
        # default data/pipeline/phase3_eval path.
        from pipeline.config import PROJECT_ROOT

        default_bad = PROJECT_ROOT / "data" / "pipeline" / "phase3_eval" / run_id
        assert not default_bad.exists(), (
            f"runner ignored cfg.phase3.eval_dir and created a run under "
            f"the project default: {default_bad}"
        )

    def test_metadata_records_candidates(self, patched_runner):
        cfg = patched_runner.cfg
        g0, g1 = cfg.phase3.generator_eval.candidates
        run_id = "run-meta"
        patched_runner.run(cfg, run_id)

        meta_path = self._run_dir(cfg, run_id) / "metadata.json"
        assert meta_path.exists(), "metadata.json missing"
        meta = json.loads(meta_path.read_text())

        cands = meta.get("candidates")
        assert isinstance(
            cands, list
        ), f"metadata.candidates should be a list, got {type(cands).__name__}"
        assert (
            len(cands) == 2
        ), f"metadata.candidates should list both generators, got {cands}"
        aliases = {c.get("alias") for c in cands}
        assert aliases == {
            g0.alias,
            g1.alias,
        }, f"metadata.candidates aliases mismatch: {aliases}"
        for c in cands:
            assert (
                "prompt_reflection" in c
            ), f"candidate metadata missing prompt_reflection: {c}"
            assert (
                "prompt_preflection" in c
            ), f"candidate metadata missing prompt_preflection: {c}"
            assert (
                "prompt_reflection_sha256" in c
            ), f"candidate metadata missing prompt_reflection_sha256: {c}"
