"""Pipeline persistence via append-only JSONL files."""

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import PIPELINE_DATA_DIR


def _ensure_dir() -> Path:
    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PIPELINE_DATA_DIR


# --- Paths ---

def runs_path() -> Path:
    """Return the JSONL file path for iteration run logs."""
    return _ensure_dir() / "runs.jsonl"


def items_path() -> Path:
    """Return the JSONL file path for generated items."""
    return _ensure_dir() / "items.jsonl"


def reviews_path() -> Path:
    """Return the JSONL file path for human reviews."""
    return _ensure_dir() / "reviews.jsonl"


# --- Generic JSONL helpers ---

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# --- Runs ---

def load_runs() -> list[dict]:
    """Load all iteration run records."""
    return _load_jsonl(runs_path())


def save_run(
    iteration: int,
    gen_prompt: str,
    judge_prompt: str,
    model: str,
    n_items: int,
    n_gold: int,
    config: dict,
    analysis: str,
) -> None:
    """Append a completed iteration run record."""
    record = {
        "iteration": iteration,
        "gen_prompt": gen_prompt,
        "judge_prompt": judge_prompt,
        "model": model,
        "n_items": n_items,
        "n_gold": n_gold,
        "config": config,
        "analysis": analysis,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(runs_path(), record)


# --- Items ---

def load_items() -> list[dict]:
    """Load all item records (no dedup)."""
    return _load_jsonl(items_path())


def load_latest_items() -> dict[tuple[str, int], dict]:
    """Load items deduped by (item_id, iteration). Last record per key wins."""
    latest: dict[tuple[str, int], dict] = {}
    for record in load_items():
        key = (record["item_id"], record["iteration"])
        latest[key] = record
    return latest


def load_items_for_iteration(iteration: int) -> list[dict]:
    """Load latest items for a specific iteration."""
    latest = load_latest_items()
    return [v for k, v in latest.items() if k[1] == iteration]


def save_item(record: dict) -> None:
    """Append a single item record (generation or generation+judgment)."""
    _append_jsonl(items_path(), record)


# --- Reviews ---

def load_reviews() -> list[dict]:
    """Load all human review records (cumulative, never deleted)."""
    return _load_jsonl(reviews_path())


def load_latest_reviews() -> dict[tuple[str, int, str], dict]:
    """Load reviews deduped by (item_id, iteration, reviewer_id). Last wins."""
    latest: dict[tuple[str, int, str], dict] = {}
    for record in load_reviews():
        key = (record["item_id"], record["iteration"], record["reviewer_id"])
        latest[key] = record
    return latest


def save_review(
    item_id: str,
    iteration: int,
    reviewer_id: str,
    scores: dict[str, int],
    aggregate: float,
    decision: str,
    notes: str,
) -> None:
    """Append a human review record."""
    record = {
        "item_id": item_id,
        "iteration": iteration,
        "reviewer_id": reviewer_id,
        "scores": scores,
        "aggregate": aggregate,
        "decision": decision,
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(reviews_path(), record)
