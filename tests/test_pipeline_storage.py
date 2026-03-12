"""Tests for pipeline storage: write/read/dedup for runs, items, reviews, test results."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path):
    """Redirect SQLite storage to a temp directory."""
    test_db = tmp_path / "test.db"
    with patch("pipeline.storage.DB_PATH", test_db):
        # Clear any cached thread-local connection so a new one is created
        from pipeline.storage import _local
        _local.conn = None
        yield tmp_path
        _local.conn = None


def test_save_and_load_run():
    from pipeline.phase2.storage import load_runs, save_run

    save_run(
        iteration=1, gen_prompt="gen_v1.md", judge_prompt="judge_v1.md",
        generator_model="glm45", judge_model="glm45", n_items=50, n_gold=12,
        config={"accept_threshold": 4}, analysis="test analysis",
    )
    runs = load_runs()
    assert len(runs) == 1
    assert runs[0]["iteration"] == 1
    assert runs[0]["analysis"] == "test analysis"
    assert "timestamp" in runs[0]


def test_save_and_load_item():
    from pipeline.phase2.storage import load_items, load_latest_items, save_item

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
    from pipeline.phase2.storage import load_latest_items, save_item

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
    from pipeline.phase2.storage import load_items_for_iteration, save_item

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
    from pipeline.phase2.storage import load_latest_reviews, load_reviews, save_review

    per_part_scores = {
        "preflection": {"relevance": 4, "specificity": 3, "charter_grounding": 5, "voice_tone": 4},
        "reflection": {"relevance": 3, "specificity": 4, "charter_grounding": 4, "voice_tone": 3},
    }
    save_review(
        item_id="abc123", iteration=1, reviewer_id="alice",
        scores=per_part_scores, aggregate=3.75, decision="accept", notes="good",
    )
    reviews = load_reviews()
    assert len(reviews) == 1
    assert reviews[0]["reviewer_id"] == "alice"
    assert "preflection" in reviews[0]["scores"]
    assert "reflection" in reviews[0]["scores"]

    # Dedup by (item_id, iteration, reviewer_id)
    updated_scores = {
        "preflection": {"relevance": 5, "specificity": 4, "charter_grounding": 5, "voice_tone": 5},
        "reflection": {"relevance": 5, "specificity": 5, "charter_grounding": 5, "voice_tone": 4},
    }
    save_review(
        item_id="abc123", iteration=1, reviewer_id="alice",
        scores=updated_scores, aggregate=4.75, decision="accept", notes="updated",
    )
    latest = load_latest_reviews()
    assert len(latest) == 1
    assert latest[("abc123", 1, "alice")]["aggregate"] == 4.75


def test_empty_loads():
    from pipeline.phase2.storage import load_items, load_reviews, load_runs

    assert load_runs() == []
    assert load_items() == []
    assert load_reviews() == []


def test_save_and_load_test_result():
    from pipeline.phase2.storage import load_test_results, save_test_result

    record = {
        "test_id": "tg_20260312_143022",
        "type": "generate",
        "phase": "A",
        "prompt": "judge_v3.md",
        "model_alias": "glm45",
        "items": [{"item_id": "abc123"}],
        "summary": {"n_items": 1, "mean_score": 3.4, "n_accepted": 0},
        "timestamp": "2026-03-12T14:30:22",
    }
    save_test_result(record)
    results = load_test_results()
    assert len(results) == 1
    assert results[0]["test_id"] == "tg_20260312_143022"
    assert results[0]["type"] == "generate"


def test_save_and_load_loop_history():
    from pipeline.phase2.storage import load_loop_history, save_loop_run

    record = {
        "started_at": "2026-03-12T14:00:00",
        "finished_at": "2026-03-12T15:00:00",
        "phase_a": {"status": "done", "reasoning": "improved judge"},
        "phase_b": {"status": "done", "reasoning": "improved generator"},
        "error": None,
        "model_alias": "glm45",
        "prompts_before": {"judge_v1.md": "old judge", "generator_v1.md": "old gen"},
        "prompts_after": {"judge_v1.md": "old judge", "judge_v2.md": "new judge", "generator_v1.md": "old gen"},
    }
    save_loop_run(record)
    history = load_loop_history()
    assert len(history) == 1
    assert history[0]["model_alias"] == "glm45"
    assert "judge_v2.md" in history[0]["prompts_after"]
    assert "judge_v2.md" not in history[0]["prompts_before"]


def test_load_test_results_filter_by_phase():
    from pipeline.phase2.storage import load_test_results, save_test_result

    save_test_result({"test_id": "t1", "type": "generate", "phase": "A", "timestamp": "t"})
    save_test_result({"test_id": "t2", "type": "judge", "phase": "B", "timestamp": "t"})
    save_test_result({"test_id": "t3", "type": "batch", "phase": "A", "timestamp": "t"})

    all_results = load_test_results()
    assert len(all_results) == 3

    phase_a = load_test_results(phase="A")
    assert len(phase_a) == 2
    assert all(r["phase"] == "A" for r in phase_a)

    phase_b = load_test_results(phase="B")
    assert len(phase_b) == 1
    assert phase_b[0]["test_id"] == "t2"


# --- New tests for comment features ---

def test_comment_target_part():
    from pipeline.phase1.storage import load_comments_by_annotation, save_comment

    save_comment("item1", "alice", "bob", "general comment", target_part="general")
    save_comment("item1", "alice", "bob", "preflection note", target_part="preflection")
    save_comment("item1", "alice", "bob", "reflection feedback", target_part="reflection")

    by_ann = load_comments_by_annotation()
    comments = by_ann[("item1", "alice")]
    assert len(comments) == 3

    general = [c for c in comments if c["target_part"] == "general"]
    assert len(general) == 1
    assert general[0]["comment"] == "general comment"

    pre = [c for c in comments if c["target_part"] == "preflection"]
    assert len(pre) == 1

    refl = [c for c in comments if c["target_part"] == "reflection"]
    assert len(refl) == 1


def test_delete_comment():
    from pipeline.phase1.storage import delete_comment, load_comments_by_annotation, save_comment

    save_comment("item1", "alice", "bob", "to delete", target_part="general")
    save_comment("item1", "alice", "bob", "to keep", target_part="general")

    by_ann = load_comments_by_annotation()
    comments = by_ann[("item1", "alice")]
    assert len(comments) == 2

    delete_comment(comments[0]["id"])

    by_ann = load_comments_by_annotation()
    comments = by_ann[("item1", "alice")]
    assert len(comments) == 1
    assert comments[0]["comment"] == "to keep"


def test_annotation_roundtrip():
    from pipeline.phase1.storage import load_latest_annotations, save_annotation

    save_annotation(
        item_id="item1", annotator_id="alice", subset="score_3",
        text="hello world", reflection_point=5,
        analysis="analysis text", preflection="pre text",
        reflection="refl text",
        reflection_charter_elements=["1.1", "2.3"],
        presentation_order=0,
    )
    latest = load_latest_annotations()
    assert len(latest) == 1
    ann = latest[("item1", "alice")]
    assert ann["reflection_charter_elements"] == ["1.1", "2.3"]
    assert ann["analysis"] == "analysis text"


def test_migration_script(tmp_path):
    """Test that the migration script correctly converts JSONL to SQLite."""
    import sqlite3

    # Create fake JSONL files
    ann_dir = tmp_path / "annotation"
    ann_dir.mkdir()
    pipe_dir = tmp_path / "pipeline"
    pipe_dir.mkdir()

    # Annotations with duplicate (last wins)
    anns = [
        {"item_id": "i1", "annotator_id": "a1", "subset": "s0", "text": "t",
         "reflection_point": 1, "analysis": "old", "preflection": "p",
         "reflection": "r", "reflection_charter_elements": [], "presentation_order": 0,
         "timestamp": "2025-01-01T00:00:00"},
        {"item_id": "i1", "annotator_id": "a1", "subset": "s0", "text": "t",
         "reflection_point": 1, "analysis": "new", "preflection": "p",
         "reflection": "r", "reflection_charter_elements": ["1.1"], "presentation_order": 1,
         "timestamp": "2025-01-01T01:00:00"},
    ]
    (ann_dir / "annotations.jsonl").write_text("\n".join(json.dumps(a) for a in anns))

    # Comments
    comments = [
        {"item_id": "i1", "target_annotator_id": "a1", "commenter_id": "b1",
         "comment": "nice", "timestamp": "2025-01-01T02:00:00"},
    ]
    (ann_dir / "comments.jsonl").write_text("\n".join(json.dumps(c) for c in comments))

    # Items
    items = [
        {"item_id": "x1", "iteration": 1, "is_gold": False, "subset": "s0",
         "text": "hello", "reflection_point": 2, "gen_prompt": "g.md",
         "model": "m", "analysis": "a", "preflection": "p", "reflection": "r",
         "charter_elements": [], "raw_response": "raw", "reasoning": None,
         "latency_ms": 50, "timestamp": "2025-01-01", "judgment": None},
    ]
    (pipe_dir / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items))

    # Empty files for the rest
    (pipe_dir / "reviews.jsonl").write_text("")
    (pipe_dir / "runs.jsonl").write_text("")
    (pipe_dir / "test_results.jsonl").write_text("")
    (pipe_dir / "loop_history.jsonl").write_text("")

    # Run migration with patched paths
    with patch("pipeline.config.ANNOTATION_DATA_DIR", ann_dir), \
         patch("pipeline.config.PIPELINE_DATA_DIR", pipe_dir), \
         patch("scripts.migrate_jsonl_to_sqlite.ANNOTATION_DATA_DIR", ann_dir), \
         patch("scripts.migrate_jsonl_to_sqlite.PIPELINE_DATA_DIR", pipe_dir):
        from scripts.migrate_jsonl_to_sqlite import migrate
        migrate()

    # Verify
    from pipeline.storage import _get_conn
    conn = _get_conn()
    ann_rows = conn.execute("SELECT * FROM annotations").fetchall()
    assert len(ann_rows) == 1
    assert ann_rows[0]["analysis"] == "new"  # last-wins dedup

    comment_rows = conn.execute("SELECT * FROM comments").fetchall()
    assert len(comment_rows) == 1
    assert comment_rows[0]["target_part"] == "general"

    item_rows = conn.execute("SELECT * FROM items").fetchall()
    assert len(item_rows) == 1
