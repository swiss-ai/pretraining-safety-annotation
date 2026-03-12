"""One-time migration from JSONL files to SQLite.

Usage:
    uv run python scripts/migrate_jsonl_to_sqlite.py
"""

import json
from pathlib import Path

from pipeline.config import ANNOTATION_DATA_DIR, DATA_DIR, PIPELINE_DATA_DIR
from pipeline.storage import _get_conn, _init_schema


def _read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def migrate():
    conn = _get_conn()

    # --- Annotations ---
    ann_path = ANNOTATION_DATA_DIR / "annotations.jsonl"
    raw_anns = _read_jsonl(ann_path)
    # Dedup: last entry per (item_id, annotator_id) wins
    deduped: dict[tuple[str, str], dict] = {}
    for r in raw_anns:
        deduped[(r["item_id"], r["annotator_id"])] = r
    for r in deduped.values():
        charter_elems = r.get("reflection_charter_elements", r.get("charter_elements", []))
        conn.execute(
            """INSERT OR REPLACE INTO annotations
               (item_id, annotator_id, subset, text, reflection_point,
                analysis, preflection, reflection, reflection_charter_elements,
                presentation_order, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["item_id"], r["annotator_id"], r["subset"], r["text"],
                r["reflection_point"], r["analysis"], r["preflection"],
                r["reflection"], json.dumps(charter_elems),
                r.get("presentation_order", 0), r["timestamp"],
            ),
        )
    conn.commit()
    n_ann = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    print(f"Annotations: {len(raw_anns)} JSONL rows -> {n_ann} SQLite rows (deduped from {len(raw_anns)})")

    # --- Comments ---
    comments_path = ANNOTATION_DATA_DIR / "comments.jsonl"
    raw_comments = _read_jsonl(comments_path)
    for r in raw_comments:
        conn.execute(
            """INSERT INTO comments
               (item_id, target_annotator_id, commenter_id, target_part, comment, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                r["item_id"], r["target_annotator_id"], r["commenter_id"],
                r.get("target_part", "general"), r["comment"], r["timestamp"],
            ),
        )
    conn.commit()
    n_comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    print(f"Comments: {len(raw_comments)} JSONL rows -> {n_comments} SQLite rows")

    # --- Items ---
    items_path = PIPELINE_DATA_DIR / "items.jsonl"
    raw_items = _read_jsonl(items_path)
    for r in raw_items:
        conn.execute(
            """INSERT OR REPLACE INTO items
               (item_id, iteration, is_gold, subset, text, reflection_point,
                gen_prompt, model, analysis, preflection, reflection,
                charter_elements, raw_response, reasoning, latency_ms,
                timestamp, judgment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["item_id"], r["iteration"],
                int(r.get("is_gold", False)), r["subset"],
                r["text"], r["reflection_point"],
                r["gen_prompt"], r["model"],
                r["analysis"], r["preflection"], r["reflection"],
                json.dumps(r.get("charter_elements", [])),
                r["raw_response"], r.get("reasoning"),
                r["latency_ms"], r["timestamp"],
                json.dumps(r["judgment"]) if r.get("judgment") is not None else None,
            ),
        )
    conn.commit()
    n_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"Items: {len(raw_items)} JSONL rows -> {n_items} SQLite rows (natural dedup)")

    # --- Reviews ---
    reviews_path = PIPELINE_DATA_DIR / "reviews.jsonl"
    raw_reviews = _read_jsonl(reviews_path)
    for r in raw_reviews:
        conn.execute(
            """INSERT OR REPLACE INTO reviews
               (item_id, iteration, reviewer_id, scores, aggregate, decision, notes, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["item_id"], r["iteration"], r["reviewer_id"],
                json.dumps(r["scores"]), r["aggregate"],
                r["decision"], r["notes"], r["timestamp"],
            ),
        )
    conn.commit()
    n_reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    print(f"Reviews: {len(raw_reviews)} JSONL rows -> {n_reviews} SQLite rows (natural dedup)")

    # --- Runs ---
    runs_path = PIPELINE_DATA_DIR / "runs.jsonl"
    raw_runs = _read_jsonl(runs_path)
    for r in raw_runs:
        conn.execute(
            """INSERT INTO runs
               (iteration, gen_prompt, judge_prompt, generator_model, judge_model,
                n_items, n_gold, config, analysis, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["iteration"], r["gen_prompt"], r["judge_prompt"],
                r["generator_model"], r["judge_model"],
                r["n_items"], r["n_gold"],
                json.dumps(r.get("config", {})), r["analysis"], r["timestamp"],
            ),
        )
    conn.commit()
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    print(f"Runs: {len(raw_runs)} JSONL rows -> {n_runs} SQLite rows")

    # --- Test Results ---
    tr_path = PIPELINE_DATA_DIR / "test_results.jsonl"
    raw_tr = _read_jsonl(tr_path)
    for r in raw_tr:
        conn.execute(
            "INSERT INTO test_results (data, timestamp) VALUES (?, ?)",
            (json.dumps(r), r.get("timestamp", "")),
        )
    conn.commit()
    n_tr = conn.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]
    print(f"Test results: {len(raw_tr)} JSONL rows -> {n_tr} SQLite rows")

    # --- Loop History ---
    lh_path = PIPELINE_DATA_DIR / "loop_history.jsonl"
    raw_lh = _read_jsonl(lh_path)
    for r in raw_lh:
        conn.execute(
            "INSERT INTO loop_history (data, timestamp) VALUES (?, ?)",
            (json.dumps(r), r.get("finished_at", r.get("timestamp", ""))),
        )
    conn.commit()
    n_lh = conn.execute("SELECT COUNT(*) FROM loop_history").fetchone()[0]
    print(f"Loop history: {len(raw_lh)} JSONL rows -> {n_lh} SQLite rows")

    print(f"\nMigration complete. Database: {DATA_DIR / 'storage.db'}")
    print("Verify the counts above, then manually delete the JSONL files:")
    print(f"  rm {ann_path} {comments_path}")
    print(f"  rm {items_path} {reviews_path} {runs_path}")
    print(f"  rm {tr_path} {lh_path}")


if __name__ == "__main__":
    migrate()
