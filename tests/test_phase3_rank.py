"""Tests for pipeline.phase3.rank.

These tests are written BEFORE the implementation exists. They describe the
contract the rank module must satisfy. Collection of this file must succeed
(no ImportError at collection time) — individual tests are expected to fail
at run time with ImportError until pipeline/phase3/rank.py exists.

The rank module exposes:
  - rank_generators(run_id: str, eval_dir: Path | None = None) -> list[dict]
  - rank_judges(run_id: str, eval_dir: Path | None = None) -> dict

Both functions read hand-crafted fake run dirs that the tests build on disk
under `tmp_path`. Tests prefer passing `eval_dir=tmp_path` as a kwarg, but
fall back to monkeypatching `pipeline.phase3.rank._eval_root` so the
implementer can pick either API.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _four_voice_scores(scores: dict, aggregate: float) -> dict:
    """Build a four-voice judgment dict where every voice has the same scores."""
    voice = {"scores": dict(scores), "aggregate": aggregate}
    return {
        "preflection_3p": dict(voice),
        "preflection_1p": dict(voice),
        "reflection_1p": dict(voice),
        "reflection_3p": dict(voice),
    }


def _judgment(
    item_id: str,
    *,
    aggregate: float,
    decision: str,
    iteration: int = 0,
    safety_score: int | None = None,
    per_dim: dict | None = None,
) -> dict:
    """Build one judgment row matching the spec's four-voice shape."""
    scores = (
        per_dim
        if per_dim is not None
        else {"relevance": aggregate, "specificity": aggregate}
    )
    voices = _four_voice_scores(scores, aggregate)
    return {
        "item_id": item_id,
        "iteration": iteration,
        "safety_score": safety_score,
        "judgment": {
            **voices,
            "aggregate": aggregate,
            "decision": decision,
        },
    }


def _generation(item_id: str, iteration: int = 0, text: str = "hello") -> dict:
    return {"item_id": item_id, "iteration": iteration, "text": text}


def _item(item_id: str, safety_score: int | None = None) -> dict:
    return {
        "item_id": item_id,
        "text": f"item text for {item_id}",
        "safety_score": safety_score,
        "reflection_point": 0,
    }


def _build_run_dir(
    tmp_path: Path,
    run_id: str,
    *,
    type: str,
    metadata_extra: dict | None = None,
    items: list[dict] | None = None,
    generations: dict[str, list[dict]] | None = None,
    judgments: dict[str, list[dict]] | None = None,
    reviewed_items: list[dict] | None = None,
    failures: dict[str, list[dict]] | None = None,
) -> Path:
    """Write a complete fake run dir under tmp_path/run_id and return its Path.

    Arguments:
        type: "generator_eval" or "judge_eval"
        items: list of items.jsonl rows
        generations: {relative filename -> list of rows} under generations/
        judgments: {relative filename -> list of rows} under judgments/
        reviewed_items: list of reviewed_items.jsonl rows
        failures: {relative filename -> list of rows} under failures/
        metadata_extra: merged into metadata.json
    """
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "type": type,
        "gold_judge": {
            "alias": "gold",
            "prompt_reflection": "judge_v1.md",
            "prompt_preflection": "judge_v1.md",
        },
        "n_items": len(items) if items is not None else 0,
    }
    if metadata_extra:
        # shallow merge is enough for the test cases
        for k, v in metadata_extra.items():
            if k in metadata and isinstance(metadata[k], dict) and isinstance(v, dict):
                merged = dict(metadata[k])
                merged.update(v)
                metadata[k] = merged
            else:
                metadata[k] = v
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if items is not None:
        _write_jsonl(run_dir / "items.jsonl", items)

    if generations:
        for name, rows in generations.items():
            _write_jsonl(run_dir / "generations" / name, rows)

    if judgments:
        for name, rows in judgments.items():
            _write_jsonl(run_dir / "judgments" / name, rows)

    if reviewed_items is not None:
        _write_jsonl(run_dir / "reviewed_items.jsonl", reviewed_items)

    if failures:
        for name, rows in failures.items():
            _write_jsonl(run_dir / "failures" / name, rows)

    return run_dir


def _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch):
    """Try eval_dir kwarg first, fall back to _eval_root monkeypatch."""
    try:
        return rank_mod.rank_generators(run_id, eval_dir=tmp_path)
    except TypeError:
        monkeypatch.setattr("pipeline.phase3.rank._eval_root", lambda: tmp_path)
        return rank_mod.rank_generators(run_id)


def _call_rank_judges(rank_mod, run_id, tmp_path, monkeypatch):
    """Try eval_dir kwarg first, fall back to _eval_root monkeypatch."""
    try:
        return rank_mod.rank_judges(run_id, eval_dir=tmp_path)
    except TypeError:
        monkeypatch.setattr("pipeline.phase3.rank._eval_root", lambda: tmp_path)
        return rank_mod.rank_judges(run_id)


@pytest.fixture
def rank_mod():
    """Import pipeline.phase3.rank lazily inside each test.

    Returning the module from a fixture means collection succeeds even when
    the module doesn't exist yet; the ImportError surfaces at test run time.
    """
    return importlib.import_module("pipeline.phase3.rank")


# ---------------------------------------------------------------------------
# rank_generators
# ---------------------------------------------------------------------------


class TestRankGenerators:
    """Contract tests for pipeline.phase3.rank.rank_generators."""

    def test_rank_generators_basic(self, tmp_path, monkeypatch, rank_mod):
        run_id = "rank-test-1"
        gen_name = "gen_a__gen_v1.md.jsonl"
        jud_name = "gold__judge_v1.md__on__gen_a__gen_v1.md.jsonl"
        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=[_item("i0", safety_score=0), _item("i1", safety_score=0)],
            generations={
                gen_name: [
                    _generation("i0"),
                    _generation("i1"),
                ],
            },
            judgments={
                jud_name: [
                    _judgment("i0", aggregate=4.5, decision="accept"),
                    _judgment("i1", aggregate=2.0, decision="reject"),
                ],
            },
        )

        result = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)

        assert isinstance(result, list)
        assert len(result) == 1
        row = result[0]
        assert row["generator"] == "gen_a__gen_v1.md"
        assert row["n_pool"] == 2
        assert row["n_succeeded"] == 2
        assert row["mean_aggregate"] == pytest.approx(3.25)
        assert row["accept_rate"] == pytest.approx(0.5)
        assert row["failure_rates"]["total_dropped"] == pytest.approx(0.0)

    def test_rank_generators_sorts_descending(self, tmp_path, monkeypatch, rank_mod):
        run_id = "rank-test-2"
        gen_a = "gen_a__v1.md.jsonl"
        gen_b = "gen_b__v1.md.jsonl"
        jud_a = "gold__judge_v1.md__on__gen_a__v1.md.jsonl"
        jud_b = "gold__judge_v1.md__on__gen_b__v1.md.jsonl"
        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=[_item("i0"), _item("i1")],
            generations={
                gen_a: [_generation("i0"), _generation("i1")],
                gen_b: [_generation("i0"), _generation("i1")],
            },
            judgments={
                # gen_a mean = 3.0
                jud_a: [
                    _judgment("i0", aggregate=3.0, decision="accept"),
                    _judgment("i1", aggregate=3.0, decision="accept"),
                ],
                # gen_b mean = 4.5
                jud_b: [
                    _judgment("i0", aggregate=4.5, decision="accept"),
                    _judgment("i1", aggregate=4.5, decision="accept"),
                ],
            },
        )

        result = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)

        assert len(result) == 2
        assert result[0]["generator"] == "gen_b__v1.md"
        assert result[1]["generator"] == "gen_a__v1.md"
        assert result[0]["mean_aggregate"] == pytest.approx(4.5)
        assert result[1]["mean_aggregate"] == pytest.approx(3.0)

    def test_rank_generators_per_dim_mean(self, tmp_path, monkeypatch, rank_mod):
        run_id = "rank-test-3"
        gen_name = "gen__v1.md.jsonl"
        jud_name = "gold__judge_v1.md__on__gen__v1.md.jsonl"

        # Every voice on every row has {"relevance": 5, "specificity": 3}.
        scores = {"relevance": 5, "specificity": 3}
        rows = []
        for i in range(3):
            rows.append(
                _judgment(
                    f"i{i}",
                    aggregate=4.0,
                    decision="accept",
                    per_dim=scores,
                )
            )

        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=[_item(f"i{i}") for i in range(3)],
            generations={
                gen_name: [_generation(f"i{i}") for i in range(3)],
            },
            judgments={jud_name: rows},
        )

        result = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)
        assert len(result) == 1
        per_dim_mean = result[0]["per_dim_mean"]
        assert per_dim_mean == pytest.approx({"relevance": 5.0, "specificity": 3.0})

    def test_rank_generators_accept_by_safety_score(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-test-4"
        gen_name = "gen__v1.md.jsonl"
        jud_name = "gold__judge_v1.md__on__gen__v1.md.jsonl"

        items = [
            _item("i0", safety_score=0),
            _item("i1", safety_score=0),
            _item("i2", safety_score=1),
            _item("i3", safety_score=2),
        ]
        jud_rows = [
            _judgment("i0", aggregate=4.0, decision="accept", safety_score=0),
            _judgment("i1", aggregate=2.0, decision="reject", safety_score=0),
            _judgment("i2", aggregate=4.0, decision="accept", safety_score=1),
            _judgment("i3", aggregate=4.0, decision="accept", safety_score=2),
        ]

        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=items,
            generations={gen_name: [_generation(i["item_id"]) for i in items]},
            judgments={jud_name: jud_rows},
        )

        result = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)
        assert len(result) == 1
        buckets = result[0]["accept_by_safety_score"]
        assert set(buckets.keys()) == {"0", "1", "2"}
        assert buckets["0"]["n"] == 2
        assert buckets["0"]["accept_rate"] == pytest.approx(0.5)
        assert buckets["1"]["n"] == 1
        assert buckets["1"]["accept_rate"] == pytest.approx(1.0)
        assert buckets["2"]["n"] == 1
        assert buckets["2"]["accept_rate"] == pytest.approx(1.0)

    def test_rank_generators_failure_rates_split_api_parse(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-test-5"
        gen_name = "gen_a__gen_v1.md.jsonl"
        jud_name = "gold__judge_v1.md__on__gen_a__gen_v1.md.jsonl"
        gen_fail_name = "gen_gen_a__gen_v1.md.jsonl"
        jud_fail_name = "jud_gold__judge_v1.md__on__gen_a__gen_v1.md.jsonl"

        items = [_item(f"i{i}") for i in range(10)]
        # 7 generations
        gen_rows = [_generation(f"i{i}") for i in range(7)]
        jud_rows = [
            _judgment(f"i{i}", aggregate=3.0, decision="accept") for i in range(7)
        ]
        # 3 gen failures: 2 api (distinct item_ids), 1 parse
        gen_failures = [
            {
                "item_id": "i7",
                "category": "api",
                "reason": "timeout",
                "raw": "",
                "attempt": 1,
                "ts": "2026-04-09T00:00:00Z",
            },
            {
                "item_id": "i8",
                "category": "api",
                "reason": "500",
                "raw": "",
                "attempt": 1,
                "ts": "2026-04-09T00:00:00Z",
            },
            {
                "item_id": "i9",
                "category": "parse",
                "reason": "bad json",
                "raw": "",
                "attempt": 1,
                "ts": "2026-04-09T00:00:00Z",
            },
        ]

        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=items,
            generations={gen_name: gen_rows},
            judgments={jud_name: jud_rows},
            failures={
                gen_fail_name: gen_failures,
                jud_fail_name: [],
            },
        )

        result = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)
        assert len(result) == 1
        fr = result[0]["failure_rates"]
        assert fr["gen_api"] == pytest.approx(0.2)
        assert fr["gen_parse"] == pytest.approx(0.1)
        assert fr["judge_api"] == pytest.approx(0.0)
        assert fr["judge_parse"] == pytest.approx(0.0)
        assert fr["total_dropped"] == pytest.approx(0.3)

    def test_rank_generators_dedups_failures_by_item_id_per_category(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-test-6"
        gen_name = "gen_a__v1.md.jsonl"
        jud_name = "gold__judge_v1.md__on__gen_a__v1.md.jsonl"
        gen_fail_name = "gen_gen_a__v1.md.jsonl"

        items = [_item(f"i{i}") for i in range(10)]
        gen_rows = [_generation(f"i{i}") for i in range(9)]
        jud_rows = [
            _judgment(f"i{i}", aggregate=3.0, decision="accept") for i in range(9)
        ]
        # Three retries for the SAME item_id, all category=api.
        gen_failures = [
            {
                "item_id": "i9",
                "category": "api",
                "reason": "timeout",
                "raw": "",
                "attempt": n,
                "ts": "2026-04-09T00:00:00Z",
            }
            for n in (1, 2, 3)
        ]

        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=items,
            generations={gen_name: gen_rows},
            judgments={jud_name: jud_rows},
            failures={gen_fail_name: gen_failures},
        )

        result = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)
        assert len(result) == 1
        fr = result[0]["failure_rates"]
        # 1 distinct item, not 3
        assert fr["gen_api"] == pytest.approx(1 / 10)
        assert fr["gen_parse"] == pytest.approx(0.0)

    def test_rank_generators_missing_failures_file_treated_as_zero(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-test-7"
        gen_name = "gen_a__v1.md.jsonl"
        jud_name = "gold__judge_v1.md__on__gen_a__v1.md.jsonl"

        items = [_item("i0"), _item("i1")]
        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=items,
            generations={
                gen_name: [_generation("i0"), _generation("i1")],
            },
            judgments={
                jud_name: [
                    _judgment("i0", aggregate=3.0, decision="accept"),
                    _judgment("i1", aggregate=3.0, decision="accept"),
                ],
            },
            # no failures dict, so no failures/ dir written at all
        )

        # Sanity: no failures dir on disk
        assert not (tmp_path / run_id / "failures").exists()

        result = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)
        assert len(result) == 1
        fr = result[0]["failure_rates"]
        assert fr["gen_api"] == pytest.approx(0.0)
        assert fr["gen_parse"] == pytest.approx(0.0)
        assert fr["judge_api"] == pytest.approx(0.0)
        assert fr["judge_parse"] == pytest.approx(0.0)
        assert fr["total_dropped"] == pytest.approx(0.0)

    def test_rank_generators_missing_gold_judgment_file_skips(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-test-8"
        gen_name = "gen_a__v1.md.jsonl"

        _build_run_dir(
            tmp_path,
            run_id,
            type="generator_eval",
            items=[_item("i0")],
            generations={gen_name: [_generation("i0")]},
            # NOTE: no judgments/ at all; the gold judge file is missing
            judgments={},
        )

        rows = _call_rank_generators(rank_mod, run_id, tmp_path, monkeypatch)
        # Generator with no judgment file is skipped, not raised
        assert rows == []


# ---------------------------------------------------------------------------
# rank_judges
# ---------------------------------------------------------------------------


class TestRankJudges:
    """Contract tests for pipeline.phase3.rank.rank_judges."""

    def _judge_eval_metadata_extra(self) -> dict:
        return {
            "generator": {
                "alias": "gen",
                "prompt_reflection": "gen_v1.md",
                "prompt_preflection": "gen_v1.md",
            },
        }

    def test_rank_judges_vs_gold_basic(self, tmp_path, monkeypatch, rank_mod):
        run_id = "rank-judge-1"
        gold_name = "gold__judge_v1.md__on__gen__gen_v1.md.jsonl"
        cand_name = "candidate__judge_v2.md__on__gen__gen_v1.md.jsonl"

        items = [_item("i0"), _item("i1"), _item("i2")]
        gold_rows = [
            _judgment("i0", aggregate=4.0, decision="accept"),
            _judgment("i1", aggregate=3.0, decision="reject"),
            _judgment("i2", aggregate=5.0, decision="accept"),
        ]
        cand_rows = [
            _judgment("i0", aggregate=4.0, decision="accept"),
            _judgment("i1", aggregate=3.0, decision="reject"),
            _judgment("i2", aggregate=5.0, decision="accept"),
        ]

        _build_run_dir(
            tmp_path,
            run_id,
            type="judge_eval",
            metadata_extra=self._judge_eval_metadata_extra(),
            items=items,
            generations={
                "gen__gen_v1.md.jsonl": [_generation(i["item_id"]) for i in items],
            },
            judgments={
                gold_name: gold_rows,
                cand_name: cand_rows,
            },
        )

        blocks = _call_rank_judges(rank_mod, run_id, tmp_path, monkeypatch)
        assert "vs_gold" in blocks
        vs_gold = blocks["vs_gold"]
        assert isinstance(vs_gold, list)
        candidate_rows = [r for r in vs_gold if "candidate" in r["judge"]]
        assert len(candidate_rows) >= 1
        crow = candidate_rows[0]
        assert crow["spearman"] == pytest.approx(1.0)
        assert isinstance(crow["per_dim"], dict)
        assert "failure_rates" in crow
        for key in ("api", "parse", "total_dropped"):
            assert key in crow["failure_rates"]

    def test_rank_judges_vs_gold_excludes_gold_against_itself(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-judge-2"
        gold_name = "gold__judge_v1.md__on__gen__gen_v1.md.jsonl"
        cand_name = "candidate__judge_v2.md__on__gen__gen_v1.md.jsonl"

        items = [_item("i0"), _item("i1")]
        gold_rows = [
            _judgment("i0", aggregate=4.0, decision="accept"),
            _judgment("i1", aggregate=3.0, decision="reject"),
        ]
        cand_rows = [
            _judgment("i0", aggregate=4.0, decision="accept"),
            _judgment("i1", aggregate=3.0, decision="reject"),
        ]

        _build_run_dir(
            tmp_path,
            run_id,
            type="judge_eval",
            metadata_extra=self._judge_eval_metadata_extra(),
            items=items,
            generations={
                "gen__gen_v1.md.jsonl": [_generation(i["item_id"]) for i in items],
            },
            judgments={
                gold_name: gold_rows,
                cand_name: cand_rows,
            },
        )

        blocks = _call_rank_judges(rank_mod, run_id, tmp_path, monkeypatch)
        vs_gold = blocks["vs_gold"]
        # The gold judge (alias "gold") must not appear in vs_gold evaluated
        # against itself.
        gold_rows_in_output = [r for r in vs_gold if r["judge"].startswith("gold__")]
        assert gold_rows_in_output == []

    def test_rank_judges_vs_gold_pairs_on_item_id(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-judge-3"
        gold_name = "gold__judge_v1.md__on__gen__gen_v1.md.jsonl"
        cand_name = "candidate__judge_v2.md__on__gen__gen_v1.md.jsonl"

        items = [_item("i0"), _item("i1"), _item("i2"), _item("i3")]
        gold_rows = [
            _judgment("i0", aggregate=4.0, decision="accept"),
            _judgment("i1", aggregate=3.0, decision="reject"),
            _judgment("i2", aggregate=5.0, decision="accept"),
        ]
        cand_rows = [
            _judgment("i0", aggregate=4.0, decision="accept"),
            _judgment("i1", aggregate=3.0, decision="reject"),
            _judgment("i3", aggregate=2.0, decision="reject"),
        ]

        _build_run_dir(
            tmp_path,
            run_id,
            type="judge_eval",
            metadata_extra=self._judge_eval_metadata_extra(),
            items=items,
            generations={
                "gen__gen_v1.md.jsonl": [_generation(i["item_id"]) for i in items],
            },
            judgments={
                gold_name: gold_rows,
                cand_name: cand_rows,
            },
        )

        blocks = _call_rank_judges(rank_mod, run_id, tmp_path, monkeypatch)
        vs_gold = blocks["vs_gold"]
        candidate_rows = [r for r in vs_gold if "candidate" in r["judge"]]
        assert len(candidate_rows) == 1
        # Only i0 and i1 are paired.
        assert candidate_rows[0]["n_succeeded"] == 2

    def test_rank_judges_vs_human_basic(self, tmp_path, monkeypatch, rank_mod):
        run_id = "rank-judge-4"
        cand_reviewed_name = "candidate__judge_v2.md__on__reviewed.jsonl"

        # Reviewed items with a human_review dict mirroring a judgment shape.
        reviewed_items = []
        cand_rows = []
        for idx, agg in enumerate([4.0, 3.0, 5.0]):
            iid = f"i{idx}"
            review_voices = _four_voice_scores(
                {"relevance": agg, "specificity": agg}, agg
            )
            reviewed_items.append(
                {
                    "item_id": iid,
                    "iteration": 0,
                    "text": f"item {iid}",
                    "human_review": {
                        **review_voices,
                        "aggregate": agg,
                        "decision": "accept" if agg >= 3.5 else "reject",
                    },
                }
            )
            cand_rows.append(
                _judgment(
                    iid,
                    aggregate=agg,
                    decision="accept" if agg >= 3.5 else "reject",
                    iteration=0,
                )
            )

        _build_run_dir(
            tmp_path,
            run_id,
            type="judge_eval",
            metadata_extra=self._judge_eval_metadata_extra(),
            items=[_item(f"i{i}") for i in range(3)],
            reviewed_items=reviewed_items,
            judgments={cand_reviewed_name: cand_rows},
        )

        blocks = _call_rank_judges(rank_mod, run_id, tmp_path, monkeypatch)
        assert "vs_human" in blocks
        vs_human = blocks["vs_human"]
        candidate_rows = [r for r in vs_human if "candidate" in r["judge"]]
        assert len(candidate_rows) >= 1
        crow = candidate_rows[0]
        assert crow["spearman"] == pytest.approx(1.0)
        assert crow["n_succeeded"] == 3

    def test_rank_judges_vs_human_pairs_on_item_id_and_iteration(
        self, tmp_path, monkeypatch, rank_mod
    ):
        run_id = "rank-judge-5"
        cand_reviewed_name = "candidate__judge_v2.md__on__reviewed.jsonl"

        reviewed_items = []
        cand_rows = []
        pairs = [("i0", 1, 4.0), ("i0", 2, 3.0)]
        for iid, iteration, agg in pairs:
            review_voices = _four_voice_scores(
                {"relevance": agg, "specificity": agg}, agg
            )
            reviewed_items.append(
                {
                    "item_id": iid,
                    "iteration": iteration,
                    "text": f"item {iid}",
                    "human_review": {
                        **review_voices,
                        "aggregate": agg,
                        "decision": "accept" if agg >= 3.5 else "reject",
                    },
                }
            )
            cand_rows.append(
                _judgment(
                    iid,
                    aggregate=agg,
                    decision="accept" if agg >= 3.5 else "reject",
                    iteration=iteration,
                )
            )

        _build_run_dir(
            tmp_path,
            run_id,
            type="judge_eval",
            metadata_extra=self._judge_eval_metadata_extra(),
            items=[_item("i0")],
            reviewed_items=reviewed_items,
            judgments={cand_reviewed_name: cand_rows},
        )

        blocks = _call_rank_judges(rank_mod, run_id, tmp_path, monkeypatch)
        vs_human = blocks["vs_human"]
        candidate_rows = [r for r in vs_human if "candidate" in r["judge"]]
        assert len(candidate_rows) == 1
        # Both (i0, 1) and (i0, 2) pair up.
        assert candidate_rows[0]["n_succeeded"] == 2

    def test_rank_judges_failure_rates_split(self, tmp_path, monkeypatch, rank_mod):
        run_id = "rank-judge-6"
        gold_name = "gold__judge_v1.md__on__gen__gen_v1.md.jsonl"
        cand_name = "candidate_a__cand_v1.md__on__gen__gen_v1.md.jsonl"
        cand_fail_name = "jud_candidate_a__cand_v1.md__on__gen__gen_v1.md.jsonl"

        items = [_item(f"i{i}") for i in range(10)]
        gold_rows = [
            _judgment(f"i{i}", aggregate=3.0, decision="accept") for i in range(10)
        ]
        cand_rows = [
            _judgment(f"i{i}", aggregate=3.0, decision="accept") for i in range(8)
        ]

        cand_failures = [
            {
                "item_id": "i8",
                "category": "api",
                "reason": "timeout",
                "raw": "",
                "attempt": 1,
                "ts": "2026-04-09T00:00:00Z",
            },
            {
                "item_id": "i9",
                "category": "parse",
                "reason": "bad json",
                "raw": "",
                "attempt": 1,
                "ts": "2026-04-09T00:00:00Z",
            },
        ]

        _build_run_dir(
            tmp_path,
            run_id,
            type="judge_eval",
            metadata_extra=self._judge_eval_metadata_extra(),
            items=items,
            generations={
                "gen__gen_v1.md.jsonl": [_generation(i["item_id"]) for i in items],
            },
            judgments={
                gold_name: gold_rows,
                cand_name: cand_rows,
            },
            failures={cand_fail_name: cand_failures},
        )

        blocks = _call_rank_judges(rank_mod, run_id, tmp_path, monkeypatch)
        vs_gold = blocks["vs_gold"]
        candidate_rows = [r for r in vs_gold if "candidate_a" in r["judge"]]
        assert len(candidate_rows) == 1
        fr = candidate_rows[0]["failure_rates"]
        assert fr["api"] == pytest.approx(0.1)
        assert fr["parse"] == pytest.approx(0.1)
        assert "total_dropped" in fr
