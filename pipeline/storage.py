"""Shared SQLite storage backend used by both phase1 and phase2 storage."""

import hashlib
import json
import sqlite3
import threading
import time

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
    analysis TEXT,
    preflection TEXT,
    reflection TEXT,
    preflection_charter_elements TEXT NOT NULL DEFAULT '[]',
    reflection_charter_elements TEXT NOT NULL DEFAULT '[]',
    raw_response TEXT,
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
    reflection_decision TEXT,
    preflection_decision TEXT,
    reflection_aggregate REAL,
    preflection_aggregate REAL,
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
    gen_reflection_prompt TEXT,
    gen_preflection_prompt TEXT,
    judge_reflection_prompt TEXT,
    judge_preflection_prompt TEXT,
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

    # Add alternate-voice annotation columns to items (added 2026-03-27)
    # preflection_1p: first-person variant of preflection (existing preflection column is 3rd-person)
    # reflection_3p: third-person variant of reflection (existing reflection column is 1st-person)
    for col in ("preflection_1p", "reflection_3p"):
        if col not in item_cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} TEXT")

    # Add phase column to runs (added 2026-03-16)
    if "phase" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN phase TEXT NOT NULL DEFAULT 'improve'")

    # Add safety_score column to items (added 2026-03-25)
    if "safety_score" not in item_cols:
        conn.execute("ALTER TABLE items ADD COLUMN safety_score INTEGER")

    # Add canary column to items (added 2026-03-26)
    if "canary" not in item_cols:
        conn.execute("ALTER TABLE items ADD COLUMN canary TEXT")

    # Split charter_elements into separate preflection/reflection sets (added 2026-04-09)
    # Old behaviour: a single `charter_elements` column populated by extracting
    # from `reflection_1p` only. New behaviour: two columns, each the union of
    # the citations the model wrote in the 1p and 3p variants of that part.
    item_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    if "charter_elements" in item_cols and (
        "reflection_charter_elements" not in item_cols
        or "preflection_charter_elements" not in item_cols
    ):
        from pipeline.config import union_charter_elements

        if "preflection_charter_elements" not in item_cols:
            conn.execute(
                "ALTER TABLE items ADD COLUMN preflection_charter_elements "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        if "reflection_charter_elements" not in item_cols:
            conn.execute(
                "ALTER TABLE items ADD COLUMN reflection_charter_elements "
                "TEXT NOT NULL DEFAULT '[]'"
            )

        rows = conn.execute(
            "SELECT item_id, iteration, preflection, preflection_1p, "
            "reflection, reflection_3p, charter_elements FROM items"
        ).fetchall()
        for r in rows:
            refl_union = union_charter_elements(r["reflection"], r["reflection_3p"])
            pref_union = union_charter_elements(r["preflection"], r["preflection_1p"])
            # Fall back to legacy column if reflection text yielded nothing
            # (rare case where the old extractor saw something the new one did not).
            if not refl_union and r["charter_elements"]:
                try:
                    refl_union = json.loads(r["charter_elements"]) or []
                except (TypeError, ValueError):
                    refl_union = []
            conn.execute(
                "UPDATE items SET reflection_charter_elements = ?, "
                "preflection_charter_elements = ? "
                "WHERE item_id = ? AND iteration = ?",
                (
                    json.dumps(refl_union),
                    json.dumps(pref_union),
                    r["item_id"],
                    r["iteration"],
                ),
            )

        conn.execute("ALTER TABLE items DROP COLUMN charter_elements")
        conn.commit()

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

    # Summary pipeline tables (added 2026-03-30)
    if "summary_runs" not in tables:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summary_runs (
                run_id TEXT PRIMARY KEY,
                generator_model TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                gen_prompt TEXT NOT NULL,
                judge_prompt TEXT NOT NULL,
                n_items INTEGER NOT NULL,
                mean_score REAL,
                source TEXT NOT NULL DEFAULT 'benchmark',
                config TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL
            )
        """)
    if "summary_items" not in tables:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summary_items (
                item_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                text TEXT NOT NULL,
                summary TEXT NOT NULL,
                summary_tokens INTEGER,
                raw_gen_response TEXT NOT NULL,
                gen_latency_ms INTEGER NOT NULL,
                scores TEXT NOT NULL,
                aggregate REAL NOT NULL,
                judge_reasoning TEXT NOT NULL,
                raw_judge_response TEXT NOT NULL,
                judge_latency_ms INTEGER NOT NULL,
                gen_tokens INTEGER,
                judge_tokens INTEGER,
                safety_score INTEGER,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (item_id, run_id)
            )
        """)

    # Add per-mode review columns (added 2026-04-12)
    review_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
    }
    for col, coltype in [
        ("reflection_decision", "TEXT"),
        ("preflection_decision", "TEXT"),
        ("reflection_aggregate", "REAL"),
        ("preflection_aggregate", "REAL"),
    ]:
        if col not in review_cols:
            conn.execute(f"ALTER TABLE reviews ADD COLUMN {col} {coltype}")

    # Add per-mode prompt columns to runs (added 2026-04-12)
    # Re-fetch cols since we may have added 'source'/'group_id'/'phase' above.
    run_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    for col in (
        "gen_reflection_prompt",
        "gen_preflection_prompt",
        "judge_reflection_prompt",
        "judge_preflection_prompt",
    ):
        if col not in run_cols:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")

    # Make preflection/reflection/analysis/raw_response nullable (added 2026-04-12)
    # SQLite cannot ALTER COLUMN, so we recreate the table if any of these are NOT NULL.
    item_col_info = conn.execute("PRAGMA table_info(items)").fetchall()
    not_null_map = {row[1]: bool(row[3]) for row in item_col_info}
    needs_rebuild = any(
        not_null_map.get(c) for c in ("preflection", "reflection", "analysis")
    )
    if needs_rebuild:
        # Get current column names/types to rebuild with nullable columns
        col_names = [row[1] for row in item_col_info]
        cols_csv = ", ".join(col_names)
        # Build new CREATE TABLE with the correct schema (nullable where needed)
        conn.execute("ALTER TABLE items RENAME TO items_old")
        # Re-run schema for items table (already nullable in _SCHEMA)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                is_gold INTEGER NOT NULL,
                subset TEXT NOT NULL,
                text TEXT NOT NULL,
                reflection_point INTEGER NOT NULL,
                gen_prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                analysis TEXT,
                preflection TEXT,
                reflection TEXT,
                preflection_charter_elements TEXT NOT NULL DEFAULT '[]',
                reflection_charter_elements TEXT NOT NULL DEFAULT '[]',
                raw_response TEXT,
                reasoning TEXT,
                latency_ms INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                judgment TEXT,
                PRIMARY KEY (item_id, iteration)
            );
            """)
        # Add optional columns that may exist in old table
        new_item_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()
        }
        for col in col_names:
            if col not in new_item_cols:
                conn.execute(f"ALTER TABLE items ADD COLUMN {col}")
        # Copy data — only columns that exist in both
        new_item_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()
        }
        shared = [c for c in col_names if c in new_item_cols]
        shared_csv = ", ".join(shared)
        conn.execute(
            f"INSERT INTO items ({shared_csv}) SELECT {shared_csv} FROM items_old"
        )
        conn.execute("DROP TABLE items_old")
        conn.commit()

    # Four-field preflection columns (added 2026-04-16). Replaces the old
    # two-voice preflection (preflection / preflection_1p) with four fields:
    # charter_summary, neutral, judgemental, idealisation. Legacy columns stay
    # for historical data.
    item_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    for col in ("charter_summary", "neutral", "judgemental", "idealisation"):
        if col not in item_cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} TEXT")


# Connections older than this are closed and reopened so the new connection
# picks up WAL frames written by other processes.  This is critical when the
# dashboard (Docker) and pipeline (host) share the DB via a bind mount —
# the mmapped -shm file can go stale across the mount boundary, causing
# PRAGMA data_version to miss external writes.
_CONN_MAX_AGE_S = 5.0


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode and Row factory.

    Sets a 10-second busy timeout to handle concurrent access from dashboard,
    backup, and iteration threads without "database is locked" errors.

    Connections are recycled after ``_CONN_MAX_AGE_S`` seconds so that readers
    in a Docker container reliably see writes made by the host process.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        if time.monotonic() - getattr(_local, "conn_opened_at", 0) < _CONN_MAX_AGE_S:
            return conn
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    db_str = str(DB_PATH)
    if (
        not getattr(_local, "schema_done", False)
        or getattr(_local, "schema_db", None) != db_str
    ):
        _init_schema(conn)
        _local.schema_done = True
        _local.schema_db = db_str
    _local.conn = conn
    _local.conn_opened_at = time.monotonic()
    logger.debug(
        "Opened SQLite connection (thread={})", threading.current_thread().name
    )
    return conn


# --- Thread-local read cache, invalidated by PRAGMA data_version ---
#
# `PRAGMA data_version` increments whenever any *other* connection (in this
# process or another) modifies the database. We additionally bump a per-thread
# counter inside our own write helpers to invalidate when *this* connection
# is the writer. The combination gives correct invalidation across:
#   - the dashboard's own writes (save_review etc.)  → bump_cache_version
#   - the pipeline process's writes (save_item etc.) → PRAGMA data_version
#   - other dashboard threads / background jobs      → PRAGMA data_version
#
# Returned objects are cached by reference — callers must not mutate them.


def _cache_token() -> tuple[int, int]:
    conn = _get_conn()
    sv = conn.execute("PRAGMA data_version").fetchone()[0]
    bv = getattr(_local, "bump_version", 0)
    return (sv, bv)


def cached_load(key, loader):
    """Thread-local memoization for read-only loaders.

    Cache is invalidated whenever the database is modified by any other
    connection (via PRAGMA data_version) or by our own write helpers
    (via bump_cache_version()). Returned values are shared by reference;
    callers must treat them as read-only.
    """
    cache = getattr(_local, "read_cache", None)
    if cache is None:
        cache = {}
        _local.read_cache = cache
    token = _cache_token()
    entry = cache.get(key)
    if entry is not None and entry[0] == token:
        return entry[1]
    result = loader()
    cache[key] = (token, result)
    return result


def force_reconnect() -> None:
    """Close the DB connection, clear the read cache, and force a fresh connect.

    Use this when external processes (e.g. the pipeline on the host) have written
    to the DB and the dashboard (in Docker) needs to see the new data immediately.
    Closing the connection forces SQLite to re-read the WAL and -shm files on the
    next query, bypassing any stale mmap state from the bind mount.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None
    _local.conn_opened_at = 0
    _local.read_cache = {}
    _local.bump_version = getattr(_local, "bump_version", 0) + 1


def bump_cache_version() -> None:
    """Invalidate this thread's read cache after a write on the same connection.

    Must be called after committing any write — PRAGMA data_version does not
    increment for changes made by the same connection, so we track our own
    writes separately.
    """
    _local.bump_version = getattr(_local, "bump_version", 0) + 1


def checkpoint() -> None:
    """Best-effort WAL flush for backup consistency.

    Uses PASSIVE mode so we never block or require exclusive access — critical
    when the dashboard (Docker) and pipeline (host) share the DB via a bind
    mount, because TRUNCATE checkpoints can corrupt WAL state across the mount
    boundary.  PASSIVE may leave some pages in the WAL, but the backup already
    tolerates a slightly-behind snapshot.
    """
    conn = _get_conn()
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")


def compute_item_id(text: str) -> str:
    """Compute a stable item ID from the text (first 200 chars)."""
    return hashlib.sha256(text[:200].encode()).hexdigest()[:16]
