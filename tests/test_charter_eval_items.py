"""Tests for pipeline.charter.eval.items.

These tests are written BEFORE the implementation exists. They describe the
contract the items module must satisfy. Running them now (before the module
is created) is expected to fail with ImportError — that's the test-first
workflow.

The items module provides:
  - build_item_pool(n_items, seed, max_tokens) -> (items, dataset_revision)
  - ensure_item_pool(store, n_items, seed, max_tokens) -> items
  - load_reviewed_items(reviewer_policy="average") -> items
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fakes / fixtures shared across tests
# ---------------------------------------------------------------------------


def _fake_sample_diverse_factory(items, revision="rev-abc"):
    """Build a fake sample_diverse that returns the given items + revision.

    The implementer's `pipeline.data.sample_diverse` is rebound inside
    `pipeline.charter.eval.items` (e.g. `from pipeline.data import sample_diverse`),
    so monkeypatching `pipeline.charter.eval.items.sample_diverse` is the right
    handle.

    The fake matches the real `sample_diverse(n, seed, max_tokens)` signature
    and returns (items, revision) — `build_item_pool` accepts the tuple form.
    """
    call_log: list[dict] = []

    def _fake(n, seed, max_tokens, **kwargs):
        call_log.append(
            {"n": n, "seed": seed, "max_tokens": max_tokens, **kwargs}
        )
        return list(items), revision

    _fake.calls = call_log  # type: ignore[attr-defined]
    return _fake


def _fake_compute_reflection_point(text, **kwargs):
    """Trivial reflection point: half-way through the text length."""
    return max(0, len(text) // 2)


def _make_item(item_id="i0", text="hello world", safety_score=0):
    return {
        "item_id": item_id,
        "text": text,
        "safety_score": safety_score,
    }


# ===========================================================================
# build_item_pool
# ===========================================================================


class TestBuildItemPool:
    """Tests for pipeline.charter.eval.items.build_item_pool."""

    def test_build_item_pool_deterministic(self, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        fake_items = [_make_item(f"i{i}", f"text {i}") for i in range(10)]
        fake = _fake_sample_diverse_factory(fake_items, revision="rev-1")
        monkeypatch.setattr(items_mod, "sample_diverse", fake)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        out1, rev1 = items_mod.build_item_pool(
            n_items=10, seed=42, max_tokens=2048
        )
        out2, rev2 = items_mod.build_item_pool(
            n_items=10, seed=42, max_tokens=2048
        )
        assert out1 == out2
        assert rev1 == rev2 == "rev-1"

    def test_build_item_pool_returns_dataset_revision(self, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        fake_items = [_make_item(f"i{i}", f"text {i}") for i in range(3)]
        fake = _fake_sample_diverse_factory(fake_items, revision="rev-xyz")
        monkeypatch.setattr(items_mod, "sample_diverse", fake)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        items, revision = items_mod.build_item_pool(
            n_items=3, seed=1, max_tokens=512
        )
        assert revision == "rev-xyz"
        assert len(items) == 3

    def test_build_item_pool_attaches_reflection_point(self, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        fake_items = [
            _make_item("a", "alpha beta gamma delta"),
            _make_item("b", "x"),
            _make_item("c", "another piece of text here"),
        ]
        fake = _fake_sample_diverse_factory(fake_items, revision="rev-r")
        monkeypatch.setattr(items_mod, "sample_diverse", fake)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        items, _ = items_mod.build_item_pool(n_items=3, seed=1, max_tokens=512)
        assert len(items) == 3
        for it in items:
            assert "reflection_point" in it
            rp = it["reflection_point"]
            assert isinstance(rp, int)
            assert rp >= 0
            assert rp <= len(it["text"])


# ===========================================================================
# ensure_item_pool
# ===========================================================================


class TestEnsureItemPool:
    """Tests for pipeline.charter.eval.items.ensure_item_pool."""

    def _make_store(self, tmp_path, run_id):
        from pipeline.charter.eval.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, run_id)
        store.open(create=True)
        return store

    def test_ensure_item_pool_first_call_writes(self, tmp_path, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        fake_items = [_make_item(f"i{i}", f"text {i}") for i in range(10)]
        fake = _fake_sample_diverse_factory(fake_items, revision="rev-first")
        monkeypatch.setattr(items_mod, "sample_diverse", fake)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        store = self._make_store(tmp_path, "items-001")
        try:
            out = items_mod.ensure_item_pool(
                store, n_items=10, seed=42, max_tokens=2048
            )
            store.flush()

            assert len(out) == 10
            on_disk = store.read_all("items.jsonl")
            assert len(on_disk) == 10

            meta = store.read_metadata()
            assert meta.get("dataset_revision") == "rev-first"
        finally:
            store.close()

    def test_ensure_item_pool_resume_returns_existing(
        self, tmp_path, monkeypatch
    ):
        from pipeline.charter.eval import items as items_mod
        from pipeline.charter.eval.storage import JsonlRunStore

        fake_items_a = [_make_item(f"a{i}", f"alpha {i}") for i in range(5)]
        fake_a = _fake_sample_diverse_factory(fake_items_a, revision="rev-a")
        monkeypatch.setattr(items_mod, "sample_diverse", fake_a)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        store = self._make_store(tmp_path, "items-002")
        try:
            first = items_mod.ensure_item_pool(
                store, n_items=5, seed=7, max_tokens=512
            )
            store.flush()
        finally:
            store.close()

        # Now patch with a DIFFERENT fake that would yield different items
        # if invoked. The resume call must NOT call sample_diverse.
        fake_items_b = [_make_item(f"b{i}", f"beta {i}") for i in range(5)]
        fake_b = _fake_sample_diverse_factory(fake_items_b, revision="rev-a")
        monkeypatch.setattr(items_mod, "sample_diverse", fake_b)

        store2 = JsonlRunStore(tmp_path, "items-002")
        store2.open(create=False)
        try:
            second = items_mod.ensure_item_pool(
                store2, n_items=5, seed=7, max_tokens=512
            )
            assert len(second) == len(first)
            assert second == first
            assert fake_b.calls == []
        finally:
            store2.close()

    def test_ensure_item_pool_resume_mismatched_n_items_raises(
        self, tmp_path, monkeypatch
    ):
        from pipeline.charter.eval import items as items_mod
        from pipeline.charter.eval.storage import JsonlRunStore

        fake_items = [_make_item(f"i{i}", f"text {i}") for i in range(10)]
        fake = _fake_sample_diverse_factory(fake_items, revision="rev-1")
        monkeypatch.setattr(items_mod, "sample_diverse", fake)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        store = self._make_store(tmp_path, "items-003")
        try:
            items_mod.ensure_item_pool(
                store, n_items=10, seed=42, max_tokens=2048
            )
            store.flush()
        finally:
            store.close()

        store2 = JsonlRunStore(tmp_path, "items-003")
        store2.open(create=False)
        try:
            with pytest.raises(Exception):
                items_mod.ensure_item_pool(
                    store2, n_items=20, seed=42, max_tokens=2048
                )
        finally:
            try:
                store2.close()
            except Exception:
                pass

    def test_ensure_item_pool_resume_stale_reflection_policy_raises(
        self, tmp_path, monkeypatch
    ):
        """A pool built under an old/absent reflection-point policy must fail
        loudly on resume rather than silently serving stale reflection points."""
        from pipeline.charter.eval import items as items_mod
        from pipeline.charter.eval.storage import JsonlRunStore

        fake_items = [_make_item(f"i{i}", f"text {i}") for i in range(5)]
        fake = _fake_sample_diverse_factory(fake_items, revision="rev-1")
        monkeypatch.setattr(items_mod, "sample_diverse", fake)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        store = self._make_store(tmp_path, "items-005")
        try:
            items_mod.ensure_item_pool(
                store, n_items=5, seed=1, max_tokens=512
            )
            store.flush()

            # Simulate an old-policy pool: drop the reflection_policy marker
            # (pools built before the marker existed have no such key).
            meta = store.read_metadata()
            meta.pop("reflection_policy", None)
            store.write_metadata(meta)
            store.flush()
        finally:
            store.close()

        store2 = JsonlRunStore(tmp_path, "items-005")
        store2.open(create=False)
        try:
            with pytest.raises(ValueError, match="reflection_policy"):
                items_mod.ensure_item_pool(
                    store2, n_items=5, seed=1, max_tokens=512
                )
        finally:
            try:
                store2.close()
            except Exception:
                pass

    def test_ensure_item_pool_resume_mismatched_dataset_revision_raises(
        self, tmp_path, monkeypatch
    ):
        from pipeline.charter.eval import items as items_mod
        from pipeline.charter.eval.storage import JsonlRunStore

        fake_items = [_make_item(f"i{i}", f"text {i}") for i in range(5)]
        fake = _fake_sample_diverse_factory(fake_items, revision="abc")
        monkeypatch.setattr(items_mod, "sample_diverse", fake)
        monkeypatch.setattr(
            items_mod, "compute_reflection_point", _fake_compute_reflection_point
        )

        store = self._make_store(tmp_path, "items-004")
        try:
            items_mod.ensure_item_pool(
                store, n_items=5, seed=1, max_tokens=512
            )
            store.flush()

            # Corrupt the recorded dataset_revision in metadata to simulate
            # a divergent revision on resume.
            meta = store.read_metadata()
            meta["dataset_revision"] = "xyz"
            store.write_metadata(meta)
            store.flush()
        finally:
            store.close()

        # On resume, the function should detect the mismatch (the rebuilt
        # revision would be "abc" but metadata says "xyz") and raise.
        store2 = JsonlRunStore(tmp_path, "items-004")
        store2.open(create=False)
        try:
            with pytest.raises(Exception):
                items_mod.ensure_item_pool(
                    store2, n_items=5, seed=1, max_tokens=512
                )
        finally:
            try:
                store2.close()
            except Exception:
                pass


# ===========================================================================
# load_reviewed_items
# ===========================================================================


def _voice_scores(base=3):
    """Build a reflection_1p scores dict."""
    return {
        "reflection_1p": {"relevance": base, "specificity": base},
    }


def _make_review(
    item_id, iteration, reviewer_id, ts, scores=None, base_score=3
):
    return {
        "item_id": item_id,
        "iteration": iteration,
        "reviewer_id": reviewer_id,
        "scores": scores if scores is not None else _voice_scores(base_score),
        "aggregate": float(base_score),
        "decision": "accept",
        "notes": "",
        "timestamp": ts,
    }


def _make_items_table_row(item_id, iteration):
    return {
        "item_id": item_id,
        "iteration": iteration,
        "text": f"text for {item_id} iter {iteration}",
        "reflection": "r",
        "reflection_1p": "r1",
        "subset": "score_3",
        "is_gold": False,
        "model": "test",
        "gen_prompt": "g.md",
        "analysis": "a",
        "charter_elements": [],
        "raw_response": "raw",
        "latency_ms": 1,
        "timestamp": "2026-01-01T00:00:00",
        "judgment": None,
        "reflection_point": 1,
    }


def _patch_review_loaders(monkeypatch, reviews, items_table):
    """Patch the symbols rebound inside pipeline.charter.eval.items.

    The implementer is expected to do:
        from pipeline.charter.improve.storage import load_reviews, load_items_for_iteration
    so we patch the rebound names in pipeline.charter.eval.items.
    """
    from pipeline.charter.eval import items as items_mod

    def _fake_load_reviews():
        return list(reviews)

    def _fake_load_items_for_iteration(iteration):
        return [r for r in items_table if r["iteration"] == iteration]

    monkeypatch.setattr(items_mod, "load_reviews", _fake_load_reviews)
    monkeypatch.setattr(
        items_mod, "load_items_for_iteration", _fake_load_items_for_iteration
    )


class TestLoadReviewedItems:
    """Tests for pipeline.charter.eval.items.load_reviewed_items."""

    def test_load_reviewed_items_average_policy(self, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        reviews = [
            _make_review("item-1", 1, "alice", "2026-04-01T00:00:00", base_score=2),
            _make_review("item-1", 1, "bob", "2026-04-02T00:00:00", base_score=4),
        ]
        items_table = [_make_items_table_row("item-1", 1)]
        _patch_review_loaders(monkeypatch, reviews, items_table)

        out = items_mod.load_reviewed_items(reviewer_policy="average")
        assert len(out) == 1
        row = out[0]
        assert row["item_id"] == "item-1"
        assert row["iteration"] == 1
        hr = row["human_review"]
        scores = hr["scores"]
        # Each voice present, each per-dim score averaged.
        for voice in ("reflection_1p",):
            assert voice in scores
            for dim in ("relevance", "specificity"):
                assert scores[voice][dim] == pytest.approx(3.0)

    def test_load_reviewed_items_first_policy(self, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        reviews = [
            _make_review("item-1", 1, "alice", "2026-04-01T00:00:00", base_score=2),
            _make_review("item-1", 1, "bob", "2026-04-02T00:00:00", base_score=4),
        ]
        items_table = [_make_items_table_row("item-1", 1)]
        _patch_review_loaders(monkeypatch, reviews, items_table)

        out = items_mod.load_reviewed_items(reviewer_policy="first")
        assert len(out) == 1
        hr = out[0]["human_review"]
        scores = hr["scores"]
        # Earliest reviewer is alice (base_score=2).
        for voice in ("reflection_1p",):
            for dim in ("relevance", "specificity"):
                assert scores[voice][dim] == 2

    def test_load_reviewed_items_all_policy(self, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        reviews = [
            _make_review("item-1", 1, "alice", "2026-04-01T00:00:00", base_score=2),
            _make_review("item-1", 1, "bob", "2026-04-02T00:00:00", base_score=4),
        ]
        items_table = [_make_items_table_row("item-1", 1)]
        _patch_review_loaders(monkeypatch, reviews, items_table)

        out = items_mod.load_reviewed_items(reviewer_policy="all")
        assert len(out) == 2
        # Each output row should still carry item_id == "item-1".
        assert all(r["item_id"] == "item-1" for r in out)
        # Each row should reference its own reviewer.
        reviewer_ids = set()
        for r in out:
            hr = r["human_review"]
            assert "reviewer_id" in hr
            reviewer_ids.add(hr["reviewer_id"])
        assert reviewer_ids == {"alice", "bob"}

    def test_load_reviewed_items_rejects_non_dict_scores(self, monkeypatch):
        from pipeline.charter.eval import items as items_mod

        # `scores` must be a dict — anything else fails loud. The old hard
        # 4-voice schema check was removed (reviews now span multiple schema
        # generations) but a genuinely malformed row still raises.
        reviews = [
            _make_review(
                "item-1", 1, "alice", "2026-04-01T00:00:00", scores="not-a-dict"
            ),
        ]
        items_table = [_make_items_table_row("item-1", 1)]
        _patch_review_loaders(monkeypatch, reviews, items_table)

        with pytest.raises(ValueError, match="non-dict scores"):
            items_mod.load_reviewed_items(reviewer_policy="average")

    def test_load_reviewed_items_accepts_arbitrary_score_keys(
        self, monkeypatch
    ):
        """The reflection_1p schema must load and average without any
        hard-coded voice validation (reviews span multiple schema
        generations, so the loader is schema-agnostic)."""
        from pipeline.charter.eval import items as items_mod

        scores_a = {
            "reflection_1p": {"relevance": 4, "specificity": 4},
        }
        scores_b = {k: {dim: 5 for dim in v} for k, v in scores_a.items()}
        reviews = [
            _make_review(
                "item-1", 1, "alice", "2026-04-14T00:00:00", scores=scores_a
            ),
            _make_review(
                "item-1", 1, "bob", "2026-04-14T01:00:00", scores=scores_b
            ),
        ]
        items_table = [_make_items_table_row("item-1", 1)]
        _patch_review_loaders(monkeypatch, reviews, items_table)

        out = items_mod.load_reviewed_items(reviewer_policy="average")
        assert len(out) == 1
        hr_scores = out[0]["human_review"]["scores"]
        # Averaged across reviewers for each voice/dim.
        assert set(hr_scores.keys()) == set(scores_a.keys())
        assert hr_scores["reflection_1p"]["relevance"] == 4.5
        assert hr_scores["reflection_1p"]["specificity"] == 4.5

    def test_load_reviewed_items_drops_orphans(self, monkeypatch, caplog):
        from pipeline.charter.eval import items as items_mod

        reviews = [
            _make_review(
                "item-1", 1, "alice", "2026-04-01T00:00:00", base_score=3
            ),
            _make_review(
                "ghost", 99, "bob", "2026-04-02T00:00:00", base_score=4
            ),
        ]
        items_table = [_make_items_table_row("item-1", 1)]
        _patch_review_loaders(monkeypatch, reviews, items_table)

        out = items_mod.load_reviewed_items(reviewer_policy="average")
        # Only item-1 has a matching items row; ghost is dropped.
        assert len(out) == 1
        assert out[0]["item_id"] == "item-1"
