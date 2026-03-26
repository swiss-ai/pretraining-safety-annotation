"""Shared SQLite storage backend used by both phase1 and phase2 storage."""

import hashlib
import json
import sqlite3
import threading

from pipeline.config import DATA_DIR
from pipeline.log import logger

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
CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator_id);

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
CREATE INDEX IF NOT EXISTS idx_items_iteration ON items(iteration);

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

CREATE TABLE IF NOT EXISTS review_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    reviewer_id TEXT NOT NULL,
    commenter_id TEXT NOT NULL,
    comment TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_comments_review
    ON review_comments(item_id, iteration, reviewer_id);

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
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    group_id TEXT
);

CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    gold_model TEXT NOT NULL,
    target_model TEXT NOT NULL,
    role TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer_notes TEXT,
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

CREATE TABLE IF NOT EXISTS judge_correlations (
    item_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    judge_prompt TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    judgment TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (item_id, iteration, judge_prompt, judge_model)
);
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    """Run CREATE TABLE IF NOT EXISTS for all tables, then apply migrations."""
    conn.executescript(_SCHEMA)
    _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply incremental migrations for columns added after initial schema."""
    # Add source column to runs (added 2026-03-13)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "source" not in cols:
        conn.execute(
            "ALTER TABLE runs ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'"
        )

    # Add group_id column to runs (added 2026-03-13)
    if "group_id" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN group_id TEXT")

    # Create iteration_counter table, seeded from existing runs (added 2026-03-16)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "iteration_counter" not in tables:
        conn.execute(
            "CREATE TABLE iteration_counter "
            "(id INTEGER PRIMARY KEY CHECK(id = 1), value INTEGER NOT NULL DEFAULT 0)"
        )
        row = conn.execute("SELECT MAX(iteration) FROM runs").fetchone()
        max_iter = row[0] or 0
        conn.execute("INSERT INTO iteration_counter VALUES (1, ?)", (max_iter,))
        conn.commit()

    # Add token usage columns to items (added 2026-03-16)
    item_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    for col in ("input_tokens", "output_tokens", "reasoning_tokens"):
        if col not in item_cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} INTEGER")

    # Add phase column to runs (added 2026-03-16)
    if "phase" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN phase TEXT NOT NULL DEFAULT 'phase2'")

    # Add safety_score column to items (added 2026-03-25)
    if "safety_score" not in item_cols:
        conn.execute("ALTER TABLE items ADD COLUMN safety_score INTEGER")

    # Add canary column to items (added 2026-03-26)
    if "canary" not in item_cols:
        conn.execute("ALTER TABLE items ADD COLUMN canary TEXT")

    # Add judge_model column to judge_correlations and update PK (added 2026-03-13)
    jc_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(judge_correlations)").fetchall()
    }
    if "judge_model" not in jc_cols:
        conn.execute("ALTER TABLE judge_correlations RENAME TO judge_correlations_old")
        conn.execute("""
            CREATE TABLE judge_correlations (
                item_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                judge_prompt TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                judgment TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (item_id, iteration, judge_prompt, judge_model)
            )
        """)
        conn.execute("""
            INSERT INTO judge_correlations (item_id, iteration, judge_prompt, judge_model, judgment, timestamp)
            SELECT item_id, iteration, judge_prompt, 'unknown', judgment, timestamp
            FROM judge_correlations_old
        """)
        conn.execute("DROP TABLE judge_correlations_old")
        conn.commit()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode and Row factory.

    Sets a 10-second busy timeout to handle concurrent access from dashboard,
    backup, and iteration threads without "database is locked" errors.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    _local.conn = conn
    logger.debug(
        "Opened SQLite connection (thread={})", threading.current_thread().name
    )
    return conn


def checkpoint() -> None:
    """Flush WAL to main database file for backup consistency."""
    conn = _get_conn()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def compute_item_id(text: str) -> str:
    """Compute a stable item ID from the text (first 200 chars)."""
    return hashlib.sha256(text[:200].encode()).hexdigest()[:16]
