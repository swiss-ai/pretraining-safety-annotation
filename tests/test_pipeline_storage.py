"""Tests for pipeline storage: write/read/dedup for runs, items, reviews."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path):
    """Redirect pipeline storage to a temp directory."""
    with patch("pipeline.storage.PIPELINE_DATA_DIR", tmp_path):
        yield tmp_path


def test_save_and_load_run():
    from pipeline.storage import load_runs, save_run

    save_run(
        iteration=1, gen_prompt="gen_v1.md", judge_prompt="judge_v1.md",
        model="test-model", n_items=50, n_gold=12,
        config={"accept_threshold": 4}, analysis="test analysis",
    )
    runs = load_runs()
    assert len(runs) == 1
    assert runs[0]["iteration"] == 1
    assert runs[0]["analysis"] == "test analysis"
    assert "timestamp" in runs[0]


def test_save_and_load_item():
    from pipeline.storage import load_items, load_latest_items, save_item

    record = {
        "item_id": "abc123", "iteration": 1, "is_gold": True,
        "subset": "score_3", "text": "hello", "reflection_point": 2,
        "gen_prompt": "gen_v1.md", "model": "test",
        "analysis": "a", "preflection": "p", "reflection": "r",
        "charter_elements": ["1.1"], "raw_response": "raw",
        "latency_ms": 100, "timestamp": "2025-01-01T00:00:00",
        "judgment": None,
    }
    save_item(record)
    items = load_items()
    assert len(items) == 1
    assert items[0]["item_id"] == "abc123"

    # Dedup: update with judgment
    updated = {**record, "judgment": {
        "preflection": {"scores": {"relevance": 4}, "aggregate": 4.0, "reasoning": "good pre"},
        "reflection": {"scores": {"relevance": 4}, "aggregate": 4.0, "reasoning": "good ref"},
        "aggregate": 4.0, "decision": "accept",
    }}
    save_item(updated)
    latest = load_latest_items()
    assert len(latest) == 1
    assert latest[("abc123", 1)]["judgment"] is not None


def test_items_dedup_by_iteration():
    from pipeline.storage import load_latest_items, save_item

    for iteration in [1, 2]:
        save_item({
            "item_id": "abc123", "iteration": iteration,
            "is_gold": False, "subset": "score_0", "text": "x",
            "reflection_point": 0, "gen_prompt": "gen_v1.md",
            "model": "test", "analysis": "a", "preflection": "p",
            "reflection": "r", "charter_elements": [],
            "raw_response": "raw", "latency_ms": 100,
            "timestamp": "2025-01-01", "judgment": None,
        })

    latest = load_latest_items()
    assert len(latest) == 2
    assert ("abc123", 1) in latest
    assert ("abc123", 2) in latest


def test_load_items_for_iteration():
    from pipeline.storage import load_items_for_iteration, save_item

    for i, iteration in enumerate([1, 1, 2]):
        save_item({
            "item_id": f"item_{i}", "iteration": iteration,
            "is_gold": False, "subset": "score_0", "text": "x",
            "reflection_point": 0, "gen_prompt": "gen_v1.md",
            "model": "test", "analysis": "a", "preflection": "p",
            "reflection": "r", "charter_elements": [],
            "raw_response": "raw", "latency_ms": 100,
            "timestamp": "2025-01-01", "judgment": None,
        })

    iter1 = load_items_for_iteration(1)
    assert len(iter1) == 2
    iter2 = load_items_for_iteration(2)
    assert len(iter2) == 1


def test_save_and_load_review():
    from pipeline.storage import load_latest_reviews, load_reviews, save_review

    save_review(
        item_id="abc123", iteration=1, reviewer_id="alice",
        scores={"relevance": 4, "specificity": 3, "charter_grounding": 5, "voice_tone": 4},
        aggregate=4.0, decision="accept", notes="good",
    )
    reviews = load_reviews()
    assert len(reviews) == 1
    assert reviews[0]["reviewer_id"] == "alice"

    # Dedup by (item_id, iteration, reviewer_id)
    save_review(
        item_id="abc123", iteration=1, reviewer_id="alice",
        scores={"relevance": 5, "specificity": 4, "charter_grounding": 5, "voice_tone": 5},
        aggregate=4.75, decision="accept", notes="updated",
    )
    latest = load_latest_reviews()
    assert len(latest) == 1
    assert latest[("abc123", 1, "alice")]["aggregate"] == 4.75


def test_empty_loads():
    from pipeline.storage import load_items, load_reviews, load_runs

    assert load_runs() == []
    assert load_items() == []
    assert load_reviews() == []
