"""Phase 2 pipeline persistence via SQLite."""

import hashlib
import json
from datetime import datetime, timezone

from pipeline.storage import _get_conn, bump_cache_version, cached_load


def review_split(item_id: str) -> str:
    """Deterministic train/validation split for a review by item_id.

    Returns "validation" for 25% of items, "train" for the rest.
    Uses SHA-256 so the split is stable across sessions and machines.
    """
    h = int(hashlib.sha256(item_id.encode()).hexdigest(), 16)
    return "validation" if h % 4 == 0 else "train"


_EXCLUDED_REVIEWERS = {"Bob"}


def build_review_lookup(
    split: str | None = None,
) -> dict[tuple[str, int], dict]:
    """Build a review lookup dict keyed by (item_id, iteration).

    Loads all reviews, optionally filters by split ("train" or "validation"),
    and deduplicates to one review per (item_id, iteration).
    """
    reviews = load_latest_reviews()
    lookup: dict[tuple[str, int], dict] = {}
    for (item_id, iteration, _reviewer), rev in reviews.items():
        if _reviewer in _EXCLUDED_REVIEWERS:
            continue
        if split and review_split(item_id) != split:
            continue
        key = (item_id, iteration)
        if key not in lookup:
            lookup[key] = rev
    return lookup


# --- Runs ---


def load_runs() -> list[dict]:
    """Load all iteration run records."""

    def _load() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        return [_row_to_run(r) for r in rows]

    return cached_load("runs", _load)


def next_iteration() -> int:
    """Atomically claim the next iteration number (process-safe).

    Uses a dedicated counter table with BEGIN EXCLUSIVE to prevent concurrent
    processes from getting the same iteration number. The table is created
    and seeded from existing runs during schema migration.
    """
    conn = _get_conn()
    conn.execute("BEGIN EXCLUSIVE")
    try:
        conn.execute("UPDATE iteration_counter SET value = value + 1 WHERE id = 1")
        row = conn.execute(
            "SELECT value FROM iteration_counter WHERE id = 1"
        ).fetchone()
        conn.commit()
        bump_cache_version()
    except Exception:
        conn.rollback()
        raise
    return row[0]


def save_run(
    iteration: int,
    gen_prompt: str,
    judge_prompt: str,
    generator_model: str,
    judge_model: str,
    n_items: int,
    n_gold: int,
    config: dict,
    analysis: str,
    source: str = "manual",
    group_id: str | None = None,
    phase: str = "phase2",
    gen_reflection_prompt: str | None = None,
    gen_preflection_prompt: str | None = None,
    judge_reflection_prompt: str | None = None,
    judge_preflection_prompt: str | None = None,
) -> None:
    """Append a completed iteration run record.

    source: one of "manual", "improve_judge", "improve_generator".
    group_id: shared UUID linking cross-iteration runs in the same batch.
    phase: pipeline phase ("phase2" or "phase3").
    gen_reflection_prompt / gen_preflection_prompt: per-mode generator prompts.
    judge_reflection_prompt / judge_preflection_prompt: per-mode judge prompts.
    """
    conn = _get_conn()
    conn.execute(
        """INSERT INTO runs
           (iteration, gen_prompt, judge_prompt,
            gen_reflection_prompt, gen_preflection_prompt,
            judge_reflection_prompt, judge_preflection_prompt,
            generator_model, judge_model,
            n_items, n_gold, config, analysis, timestamp, source, group_id, phase)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            iteration,
            gen_prompt,
            judge_prompt,
            gen_reflection_prompt,
            gen_preflection_prompt,
            judge_reflection_prompt,
            judge_preflection_prompt,
            generator_model,
            judge_model,
            n_items,
            n_gold,
            json.dumps(config),
            analysis,
            datetime.now(timezone.utc).isoformat(),
            source,
            group_id,
            phase,
        ),
    )
    conn.commit()
    bump_cache_version()


# --- Items ---


def load_items() -> list[dict]:
    """Load all item records."""

    def _load() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM items ORDER BY timestamp").fetchall()
        return [_row_to_item(r) for r in rows]

    return cached_load("items_all", _load)


def load_latest_items() -> dict[tuple[str, int], dict]:
    """Load items keyed by (item_id, iteration). Dedup is implicit via PRIMARY KEY."""

    def _load() -> dict[tuple[str, int], dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM items").fetchall()
        return {(r["item_id"], r["iteration"]): _row_to_item(r) for r in rows}

    return cached_load("latest_items", _load)


def load_items_for_iteration(iteration: int) -> list[dict]:
    """Load items for a specific iteration."""

    def _load() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM items WHERE iteration = ?", (iteration,)
        ).fetchall()
        return [_row_to_item(r) for r in rows]

    return cached_load(("items_iter", iteration), _load)


def load_item_across_iterations(item_id: str, iterations: list[int]) -> list[dict]:
    """Load a specific item from multiple iterations in a single query."""
    if not iterations:
        return []
    conn = _get_conn()
    placeholders = ",".join("?" for _ in iterations)
    rows = conn.execute(
        f"SELECT * FROM items WHERE item_id = ? AND iteration IN ({placeholders})",
        [item_id] + iterations,
    ).fetchall()
    return [_row_to_item(r) for r in rows]


def save_item(record: dict) -> None:
    """Upsert a single item record (generation or generation+judgment)."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO items
           (item_id, iteration, is_gold, subset, text, reflection_point,
            gen_prompt, model, analysis, preflection, reflection,
            preflection_1p, reflection_3p,
            preflection_charter_elements, reflection_charter_elements,
            raw_response, reasoning, latency_ms,
            timestamp, judgment, input_tokens, output_tokens, reasoning_tokens,
            safety_score, canary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record["item_id"],
            record["iteration"],
            int(record.get("is_gold", False)),
            record["subset"],
            record["text"],
            record["reflection_point"],
            record["gen_prompt"],
            record["model"],
            record.get("analysis"),
            record.get("preflection"),
            record.get("reflection"),
            record.get("preflection_1p"),
            record.get("reflection_3p"),
            json.dumps(record.get("preflection_charter_elements", [])),
            json.dumps(record.get("reflection_charter_elements", [])),
            record.get("raw_response"),
            record.get("reasoning"),
            record["latency_ms"],
            record["timestamp"],
            (
                json.dumps(record["judgment"])
                if record.get("judgment") is not None
                else None
            ),
            record.get("input_tokens"),
            record.get("output_tokens"),
            record.get("reasoning_tokens"),
            record.get("safety_score"),
            record.get("canary"),
        ),
    )
    conn.commit()
    bump_cache_version()


# --- Reviews ---


def load_reviews() -> list[dict]:
    """Load all human review records."""

    def _load() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM reviews ORDER BY timestamp").fetchall()
        return [_row_to_review(r) for r in rows]

    return cached_load("reviews", _load)


def load_latest_reviews() -> dict[tuple[str, int, str], dict]:
    """Load reviews keyed by (item_id, iteration, reviewer_id). Dedup is implicit."""

    def _load() -> dict[tuple[str, int, str], dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM reviews").fetchall()
        return {
            (r["item_id"], r["iteration"], r["reviewer_id"]): _row_to_review(r)
            for r in rows
        }

    return cached_load("latest_reviews", _load)


def save_review(
    item_id: str,
    iteration: int,
    reviewer_id: str,
    scores: dict[str, dict[str, int]],
    aggregate: float,
    decision: str,
    notes: str,
    reflection_decision: str | None = None,
    preflection_decision: str | None = None,
    reflection_aggregate: float | None = None,
    preflection_aggregate: float | None = None,
) -> None:
    """Upsert a human review record.

    scores is keyed by part: {"preflection": {dim: int}, "reflection": {dim: int}}.
    reflection_decision / preflection_decision: per-mode accept/reject decisions.
    reflection_aggregate / preflection_aggregate: per-mode aggregate scores.
    """
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO reviews
           (item_id, iteration, reviewer_id, scores, aggregate, decision, notes,
            reflection_decision, preflection_decision,
            reflection_aggregate, preflection_aggregate,
            timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            iteration,
            reviewer_id,
            json.dumps(scores),
            aggregate,
            decision,
            notes,
            reflection_decision,
            preflection_decision,
            reflection_aggregate,
            preflection_aggregate,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    bump_cache_version()


# --- Review Comments ---


def load_review_comments() -> dict[tuple[str, int, str], list[dict]]:
    """Load review comments grouped by (item_id, iteration, reviewer_id), sorted by timestamp."""

    def _load() -> dict[tuple[str, int, str], list[dict]]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM review_comments ORDER BY timestamp"
        ).fetchall()
        by_review: dict[tuple[str, int, str], list[dict]] = {}
        for r in rows:
            key = (r["item_id"], r["iteration"], r["reviewer_id"])
            by_review.setdefault(key, []).append(dict(r))
        return by_review

    return cached_load("review_comments", _load)


def save_review_comment(
    item_id: str,
    iteration: int,
    reviewer_id: str,
    commenter_id: str,
    comment: str,
) -> None:
    """Insert a comment on a review."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO review_comments
           (item_id, iteration, reviewer_id, commenter_id, comment, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            iteration,
            reviewer_id,
            commenter_id,
            comment,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    bump_cache_version()


def delete_review_comment(comment_id: int) -> None:
    """Hard-delete a review comment by its row id."""
    conn = _get_conn()
    conn.execute("DELETE FROM review_comments WHERE id = ?", (comment_id,))
    conn.commit()
    bump_cache_version()


def delete_review(item_id: str, iteration: int, reviewer_id: str) -> None:
    """Delete a review and all its comments in one transaction."""
    conn = _get_conn()
    conn.execute(
        "DELETE FROM review_comments WHERE item_id = ? AND iteration = ? AND reviewer_id = ?",
        (item_id, iteration, reviewer_id),
    )
    conn.execute(
        "DELETE FROM reviews WHERE item_id = ? AND iteration = ? AND reviewer_id = ?",
        (item_id, iteration, reviewer_id),
    )
    conn.commit()
    bump_cache_version()


# --- Test Results ---


def save_test_result(record: dict) -> None:
    """Append a test result record (from test_generate, test_judge, or run_batch)."""
    conn = _get_conn()
    ts = record.get("timestamp", datetime.now(timezone.utc).isoformat())
    conn.execute(
        "INSERT INTO test_results (data, timestamp) VALUES (?, ?)",
        (json.dumps(record), ts),
    )
    conn.commit()
    bump_cache_version()


def load_test_results(phase: str | None = None, role: str | None = None) -> list[dict]:
    """Load test results, optionally filtered by phase/role.

    Supports both old phase format ('A'/'B') and new role format ('judge'/'generator').
    """

    def _load_all() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM test_results ORDER BY id").fetchall()
        return [json.loads(r["data"]) for r in rows]

    results = cached_load("test_results", _load_all)
    if phase is not None:
        results = [r for r in results if r.get("phase") == phase]
    if role is not None:
        results = [r for r in results if r.get("role") == role]
    return results


# --- Loop History ---


def save_loop_run(record: dict) -> None:
    """Append a completed loop run record."""
    conn = _get_conn()
    ts = record.get("finished_at", datetime.now(timezone.utc).isoformat())
    conn.execute(
        "INSERT INTO loop_history (data, timestamp) VALUES (?, ?)",
        (json.dumps(record), ts),
    )
    conn.commit()
    bump_cache_version()


def load_loop_history() -> list[dict]:
    """Load all loop run records."""

    def _load() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM loop_history ORDER BY id").fetchall()
        return [json.loads(r["data"]) for r in rows]

    return cached_load("loop_history", _load)


# --- Judge Correlations ---


def save_judge_correlation(
    item_id: str,
    iteration: int,
    judge_prompt: str,
    judge_model: str,
    judgment: dict,
) -> None:
    """Upsert a re-judgment record for judge-human correlation tracking."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO judge_correlations
           (item_id, iteration, judge_prompt, judge_model, judgment, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            iteration,
            judge_prompt,
            judge_model,
            json.dumps(judgment),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    bump_cache_version()


def load_judge_correlations() -> list[dict]:
    """Load all judge correlation entries."""

    def _load() -> list[dict]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM judge_correlations ORDER BY timestamp"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["judgment"] = json.loads(d["judgment"])
            result.append(d)
        return result

    return cached_load("judge_correlations", _load)


# --- Row converters ---


def _row_to_run(row) -> dict:
    d = dict(row)
    d["config"] = json.loads(d["config"])
    d.pop("id", None)
    return d


def _row_to_item(row) -> dict:
    d = dict(row)
    d["is_gold"] = bool(d["is_gold"])
    d["preflection_charter_elements"] = json.loads(d["preflection_charter_elements"])
    d["reflection_charter_elements"] = json.loads(d["reflection_charter_elements"])
    d["judgment"] = json.loads(d["judgment"]) if d["judgment"] is not None else None
    return d


def _row_to_review(row) -> dict:
    d = dict(row)
    d["scores"] = json.loads(d["scores"])
    return d
