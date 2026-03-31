"""Summary-specific SQLite helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pipeline.storage import _get_conn


def save_summary_run(run: dict) -> None:
    """Insert or replace a summary run record."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO summary_runs
           (run_id, generator_model, judge_model, gen_prompt, judge_prompt,
            n_items, mean_score, source, config, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run["run_id"],
            run["generator_model"],
            run["judge_model"],
            run["gen_prompt"],
            run["judge_prompt"],
            run["n_items"],
            run.get("mean_score"),
            run.get("source", "benchmark"),
            json.dumps(run.get("config", {})),
            run.get("timestamp", datetime.now(timezone.utc).isoformat()),
        ),
    )
    conn.commit()


def save_summary_item(item: dict) -> None:
    """Insert or replace a summary item record."""
    save_summary_items([item])


def save_summary_items(items: list[dict]) -> None:
    """Batch insert or replace summary item records in a single transaction."""
    if not items:
        return
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """INSERT OR REPLACE INTO summary_items
           (item_id, run_id, text, summary, summary_tokens,
            raw_gen_response, gen_latency_ms, scores, aggregate,
            judge_reasoning, raw_judge_response, judge_latency_ms,
            gen_tokens, judge_tokens, safety_score, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                item["item_id"],
                item["run_id"],
                item["text"],
                item["summary"],
                item.get("summary_tokens"),
                item["raw_gen_response"],
                item["gen_latency_ms"],
                json.dumps(item["scores"]),
                item["aggregate"],
                item["judge_reasoning"],
                item["raw_judge_response"],
                item["judge_latency_ms"],
                item.get("gen_tokens"),
                item.get("judge_tokens"),
                item.get("safety_score"),
                item.get("timestamp", now),
            )
            for item in items
        ],
    )
    conn.commit()


def load_summary_items(run_id: str) -> list[dict]:
    """Load all summary items for a given run."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM summary_items WHERE run_id = ? ORDER BY item_id",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def load_summary_runs(model: str | None = None) -> list[dict]:
    """Load summary runs, optionally filtered by generator model."""
    conn = _get_conn()
    if model:
        rows = conn.execute(
            "SELECT * FROM summary_runs WHERE generator_model = ? ORDER BY timestamp",
            (model,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM summary_runs ORDER BY timestamp"
        ).fetchall()
    return [dict(r) for r in rows]
