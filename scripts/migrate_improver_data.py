"""Migrate old phase_a/phase_b improver data to the new cross-model format.

Idempotent: safe to run multiple times.

Usage:
    uv run python scripts/migrate_improver_data.py
"""

import json
from pathlib import Path

from pipeline.storage import _get_conn

PROJECT_ROOT = Path(__file__).parent.parent

PHASE_TO_ROLE = {"phase_a": "improve_judge", "phase_b": "improve_generator"}
TEST_PHASE_MAP = {"A": "judge", "B": "generator"}


def migrate_runs(conn):
    """Convert runs.source from phase_a/phase_b to improve_judge/improve_generator.

    Also adds group_id column if missing.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "group_id" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN group_id TEXT")
        print("runs: added group_id column")

    updated = 0
    for old, new in PHASE_TO_ROLE.items():
        cur = conn.execute("UPDATE runs SET source = ? WHERE source = ?", (new, old))
        updated += cur.rowcount
    conn.commit()
    print(f"runs: updated {updated} rows (source rename)")


def migrate_loop_history(conn):
    """Convert loop_history JSON blobs from phase_a/phase_b to improvers dict."""
    rows = conn.execute("SELECT id, data FROM loop_history").fetchall()
    updated = 0
    for row_id, raw in rows:
        data = json.loads(raw)
        if "phase_a" not in data and "phase_b" not in data:
            continue

        alias = data.get("model_alias", "unknown")
        phase_a = data.pop("phase_a", None)
        phase_b = data.pop("phase_b", None)
        data.pop("model_alias", None)

        improvers = {}
        if phase_a is not None:
            improvers[f"judge_{alias}"] = phase_a
        if phase_b is not None:
            improvers[f"generator_{alias}"] = phase_b
        data["improvers"] = improvers
        data["role"] = "judge"

        old_logs = data.get("logs", {})
        new_logs = {}
        if "phase_a" in old_logs:
            new_logs[f"judge_{alias}"] = old_logs["phase_a"]
        if "phase_b" in old_logs:
            new_logs[f"generator_{alias}"] = old_logs["phase_b"]
        if new_logs:
            data["logs"] = new_logs

        conn.execute("UPDATE loop_history SET data = ? WHERE id = ?", (json.dumps(data), row_id))
        updated += 1

    conn.commit()
    print(f"loop_history: updated {updated}/{len(rows)} rows")


def delete_loop_status():
    """Delete old-format loop_status.json if present."""
    path = PROJECT_ROOT / "data" / "pipeline" / "loop_status.json"
    if path.exists():
        path.unlink()
        print(f"loop_status.json: deleted {path}")
    else:
        print("loop_status.json: not present, skipping")


def migrate_test_results(conn):
    """Convert test_results JSON: rename phase field to role, map A->judge, B->generator."""
    rows = conn.execute("SELECT id, data FROM test_results").fetchall()
    updated = 0
    for row_id, raw in rows:
        data = json.loads(raw)
        old_phase = data.get("phase")
        if old_phase not in TEST_PHASE_MAP:
            continue
        data["role"] = TEST_PHASE_MAP[old_phase]
        data.pop("phase", None)
        conn.execute("UPDATE test_results SET data = ? WHERE id = ?", (json.dumps(data), row_id))
        updated += 1

    conn.commit()
    print(f"test_results: updated {updated}/{len(rows)} rows")


def main():
    """Run all migrations."""
    conn = _get_conn()
    migrate_runs(conn)
    migrate_loop_history(conn)
    delete_loop_status()
    migrate_test_results(conn)
    print("\nMigration complete.")


if __name__ == "__main__":
    main()
