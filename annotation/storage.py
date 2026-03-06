"""Annotation persistence via append-only JSONL files."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ANNOTATION_DIR = Path(__file__).parent


def annotations_path() -> Path:
    """Return the JSONL file path for annotations."""
    return ANNOTATION_DIR / "annotations.jsonl"


def load_annotations() -> list[dict]:
    """Load all annotation records (no dedup)."""
    path = annotations_path()
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_latest_annotations() -> dict[tuple[str, str], dict]:
    """Load annotations keyed by (item_id, annotator_id). Last entry per key wins."""
    latest: dict[tuple[str, str], dict] = {}
    for record in load_annotations():
        key = (record["item_id"], record["annotator_id"])
        latest[key] = record
    return latest


def load_annotator_ids() -> list[str]:
    """Return sorted list of unique annotator IDs from existing annotations."""
    annotations = load_annotations()
    return sorted({r["annotator_id"] for r in annotations})


def save_annotation(
    item_id: str,
    annotator_id: str,
    subset: str,
    reflection_point: int,
    analysis: str,
    preflection: str,
    reflection: str,
    charter_elements: list[str],
    presentation_order: int,
) -> None:
    """Append a single annotation record."""
    record = {
        "item_id": item_id,
        "annotator_id": annotator_id,
        "subset": subset,
        "reflection_point": reflection_point,
        "analysis": analysis,
        "preflection": preflection,
        "reflection": reflection,
        "charter_elements": charter_elements,
        "presentation_order": presentation_order,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(annotations_path(), "a") as f:
        f.write(json.dumps(record) + "\n")


def load_annotations_by_item() -> dict[str, list[dict]]:
    """Load all annotations grouped by item_id (latest per annotator)."""
    latest = load_latest_annotations()
    by_item: dict[str, list[dict]] = {}
    for (item_id, _), record in latest.items():
        by_item.setdefault(item_id, []).append(record)
    return by_item


def compute_item_id(text: str) -> str:
    """Compute a stable item ID from the text (first 200 chars)."""
    return hashlib.sha256(text[:200].encode()).hexdigest()[:16]


# --- Comments ---

def comments_path() -> Path:
    """Return the JSONL file path for annotation comments."""
    return ANNOTATION_DIR / "comments.jsonl"


def load_comments() -> list[dict]:
    """Load all comment records."""
    path = comments_path()
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_comments_by_annotation() -> dict[tuple[str, str], list[dict]]:
    """Load comments keyed by (item_id, target_annotator_id), sorted by timestamp."""
    by_annotation: dict[tuple[str, str], list[dict]] = {}
    for comment in load_comments():
        key = (comment["item_id"], comment["target_annotator_id"])
        by_annotation.setdefault(key, []).append(comment)
    for comments in by_annotation.values():
        comments.sort(key=lambda c: c["timestamp"])
    return by_annotation


def save_comment(
    item_id: str,
    target_annotator_id: str,
    commenter_id: str,
    comment: str,
) -> None:
    """Append a comment on an annotation."""
    record = {
        "item_id": item_id,
        "target_annotator_id": target_annotator_id,
        "commenter_id": commenter_id,
        "comment": comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(comments_path(), "a") as f:
        f.write(json.dumps(record) + "\n")
