"""Phase 1 annotation persistence via SQLite."""

import json
from datetime import datetime, timezone

from pipeline.backup import notify_critical_write
from pipeline.storage import _get_conn


def load_annotations() -> list[dict]:
    """Load all annotation records."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM annotations ORDER BY timestamp").fetchall()
    return [_row_to_annotation(r) for r in rows]


def load_latest_annotations() -> dict[tuple[str, str], dict]:
    """Load annotations keyed by (item_id, annotator_id).

    Dedup is implicit via PRIMARY KEY — each key has exactly one row.
    """
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM annotations").fetchall()
    return {(r["item_id"], r["annotator_id"]): _row_to_annotation(r) for r in rows}


def load_annotator_ids() -> list[str]:
    """Return sorted list of unique annotator IDs from existing annotations."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT annotator_id FROM annotations ORDER BY annotator_id"
    ).fetchall()
    return [r["annotator_id"] for r in rows]


def save_annotation(
    item_id: str,
    annotator_id: str,
    subset: str,
    text: str,
    reflection_point: int,
    analysis: str,
    preflection: str,
    reflection: str,
    reflection_charter_elements: list[str],
    presentation_order: int,
) -> None:
    """Upsert a single annotation record."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO annotations
           (item_id, annotator_id, subset, text, reflection_point,
            analysis, preflection, reflection, reflection_charter_elements,
            presentation_order, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            annotator_id,
            subset,
            text,
            reflection_point,
            analysis,
            preflection,
            reflection,
            json.dumps(reflection_charter_elements),
            presentation_order,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    notify_critical_write()


def load_annotations_by_item() -> dict[str, list[dict]]:
    """Load all annotations grouped by item_id."""
    latest = load_latest_annotations()
    by_item: dict[str, list[dict]] = {}
    for (item_id, _), record in latest.items():
        by_item.setdefault(item_id, []).append(record)
    return by_item


# --- Comments ---


def load_comments_by_annotation() -> dict[tuple[str, str], list[dict]]:
    """Load comments keyed by (item_id, target_annotator_id), sorted by timestamp."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM comments ORDER BY timestamp").fetchall()
    by_annotation: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["item_id"], r["target_annotator_id"])
        by_annotation.setdefault(key, []).append(dict(r))
    return by_annotation


def save_comment(
    item_id: str,
    target_annotator_id: str,
    commenter_id: str,
    comment: str,
    target_part: str = "general",
) -> None:
    """Insert a comment on an annotation."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO comments
           (item_id, target_annotator_id, commenter_id, target_part, comment, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            target_annotator_id,
            commenter_id,
            target_part,
            comment,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    notify_critical_write()


def delete_comment(comment_id: int) -> None:
    """Hard-delete a comment by its row id."""
    conn = _get_conn()
    conn.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    conn.commit()
    notify_critical_write()


def _row_to_annotation(row: dict) -> dict:
    """Convert a sqlite3.Row to a plain dict, deserializing JSON fields."""
    d = dict(row)
    d["reflection_charter_elements"] = json.loads(d["reflection_charter_elements"])
    return d
