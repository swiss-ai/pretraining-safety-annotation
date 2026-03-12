"""Shared SQLite storage backend used by both phase1 and phase2 storage."""

import hashlib
import json
import sqlite3
import threading

from pipeline.config import DATA_DIR

DB_PATH = DATA_DIR / "storage.db"

_local = threading.local()

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS annotations (
    item_id TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    subset TEXT NOT NULL,
    text TEXT NOT NULL,
    reflection_point INTEGER NOT NULL,
    analysis TEXT NOT NULL,
    preflection TEXT NOT NULL,
    reflection TEXT NOT NULL,
    reflection_charter_elements TEXT NOT NULL,
    presentation_order INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (item_id, annotator_id)
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    target_annotator_id TEXT NOT NULL,
    commenter_id TEXT NOT NULL,
    target_part TEXT NOT NULL DEFAULT 'general',
    comment TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comments_annotation ON comments(item_id, target_annotator_id);

CREATE TABLE IF NOT EXISTS items (
    item_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    is_gold INTEGER NOT NULL,
    subset TEXT NOT NULL,
    text TEXT NOT NULL,
    reflection_point INTEGER NOT NULL,
    gen_prompt TEXT NOT NULL,
    model TEXT NOT NULL,
    analysis TEXT NOT NULL,
    preflection TEXT NOT NULL,
    reflection TEXT NOT NULL,
    charter_elements TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    reasoning TEXT,
    latency_ms INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    judgment TEXT,
    PRIMARY KEY (item_id, iteration)
);

CREATE TABLE IF NOT EXISTS reviews (
    item_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    reviewer_id TEXT NOT NULL,
    scores TEXT NOT NULL,
    aggregate REAL NOT NULL,
    decision TEXT NOT NULL,
    notes TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (item_id, iteration, reviewer_id)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER NOT NULL,
    gen_prompt TEXT NOT NULL,
    judge_prompt TEXT NOT NULL,
    generator_model TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    n_items INTEGER NOT NULL,
    n_gold INTEGER NOT NULL,
    config TEXT NOT NULL,
    analysis TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    """Run CREATE TABLE IF NOT EXISTS for all tables."""
    conn.executescript(_SCHEMA)


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode and Row factory."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    _local.conn = conn
    return conn


def checkpoint() -> None:
    """Flush WAL to main database file for backup consistency."""
    conn = _get_conn()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def compute_item_id(text: str) -> str:
    """Compute a stable item ID from the text (first 200 chars)."""
    return hashlib.sha256(text[:200].encode()).hexdigest()[:16]
