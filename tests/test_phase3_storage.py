"""Tests for pipeline.phase3.storage.JsonlRunStore.

These tests are written BEFORE the implementation exists. They describe the
contract the JsonlRunStore must satisfy. Running them now (before the module
is created) is expected to fail with ImportError — that's the test-first
workflow.

See the spec document for details. The store is an append-only JSONL run
store with a writer thread, used by the phase 3 evaluation pipeline.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _baseline_metadata(**overrides) -> dict:
    """A reasonable default metadata dict matching the expected schema."""
    base = {
        "n_items": 100,
        "seed": 42,
        "max_tokens": 2048,
        "dataset_revision": "abc123",
        "gold_judge": {"prompt_sha256": "g0"},
        "candidates": [
            {"alias": "A", "prompt_sha256": "X"},
            {"alias": "B", "prompt_sha256": "Y"},
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJsonlRunStore:
    """Contract tests for pipeline.phase3.storage.JsonlRunStore."""

    # ---- 1. open(create=True) writes metadata ----------------------------

    def test_open_create_writes_metadata(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-001")
        try:
            store.open(create=True, expected_metadata=_baseline_metadata())
            store.write_metadata(_baseline_metadata())
            store.flush()

            run_dir = tmp_path / "run-001"
            assert run_dir.exists()
            meta_path = run_dir / "metadata.json"
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            for key in ("n_items", "seed", "max_tokens", "dataset_revision"):
                assert key in meta
        finally:
            store.close()

    # ---- 2. round trip via append ---------------------------------------

    def test_append_and_read_round_trip(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-002")
        try:
            store.open(create=True)
            rel = "generations/foo.jsonl"
            rows = [{"item_id": f"i{i}", "value": i} for i in range(5)]
            for row in rows:
                store.append(rel, row)
            store.flush()

            read_rows = list(store.iter_rows(rel))
            assert read_rows == rows
        finally:
            store.close()

    # ---- 3. round trip via append_many ----------------------------------

    def test_append_many_round_trip(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-003")
        try:
            store.open(create=True)
            rel = "generations/bar.jsonl"
            rows = [{"item_id": f"i{i}", "value": i} for i in range(7)]
            store.append_many(rel, rows)
            store.flush()

            read_rows = list(store.iter_rows(rel))
            assert read_rows == rows
        finally:
            store.close()

    # ---- 4. done_keys (single key) --------------------------------------

    def test_done_keys_single(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-004")
        try:
            store.open(create=True)
            rel = "generations/baz.jsonl"
            ids = ["a", "b", "c"]
            for item_id in ids:
                store.append(rel, {"item_id": item_id, "value": 1})
            store.flush()

            keys = store.done_keys(rel)
            assert keys == set(ids)
        finally:
            store.close()

    # ---- 5. done_keys (composite key) -----------------------------------

    def test_done_keys_composite(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-005")
        try:
            store.open(create=True)
            rel = "generations/qux.jsonl"
            entries = [("a", 1), ("a", 2), ("b", 1)]
            for item_id, iteration in entries:
                store.append(
                    rel,
                    {"item_id": item_id, "iteration": iteration, "value": 1},
                )
            store.flush()

            keys = store.done_keys(rel, key=("item_id", "iteration"))
            assert keys == set(entries)
        finally:
            store.close()

    # ---- 6. done_keys handles torn final line ---------------------------

    def test_done_keys_handles_torn_last_line(self, tmp_path, caplog):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-006")
        try:
            store.open(create=True)
            rel = "generations/torn.jsonl"
            good_rows = [{"item_id": f"i{i}", "value": i} for i in range(3)]
            for row in good_rows:
                store.append(rel, row)
            store.flush()

            file_path = tmp_path / "run-006" / rel
            data = file_path.read_bytes()

            # Append a half-written final row (no trailing newline, malformed JSON).
            with file_path.open("ab") as f:
                f.write(b'{"item_id": "tor')

            with caplog.at_level(logging.WARNING):
                keys = store.done_keys(rel)
            assert keys == {f"i{i}" for i in range(3)}
        finally:
            store.close()

    # ---- 7. close is idempotent -----------------------------------------

    def test_close_is_idempotent(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-007")
        store.open(create=True)
        store.close()
        # Calling close twice should not raise.
        store.close()

    # ---- 8. close drains queue ------------------------------------------

    def test_close_drains_queue(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-008")
        store.open(create=True)
        rel = "generations/drain.jsonl"
        rows = [{"item_id": f"i{i}", "value": i} for i in range(10)]
        for row in rows:
            store.append(rel, row)
        # Close without explicit flush — should still drain the queue.
        store.close()

        on_disk = _read_jsonl(tmp_path / "run-008" / rel)
        assert on_disk == rows

    # ---- 9. resume reads existing metadata ------------------------------

    def test_resume_reads_existing_metadata(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        m1 = _baseline_metadata()
        store = JsonlRunStore(tmp_path, "run-009")
        store.open(create=True, expected_metadata=m1)
        store.write_metadata(m1)
        store.close()

        store2 = JsonlRunStore(tmp_path, "run-009")
        try:
            # Should not raise.
            store2.open(create=False, expected_metadata=m1)
            loaded = store2.read_metadata()
            assert loaded["n_items"] == m1["n_items"]
            assert loaded["seed"] == m1["seed"]
        finally:
            store2.close()

    # ---- 10. resume rejects metadata mismatch ---------------------------

    def test_resume_rejects_metadata_mismatch(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        m1 = _baseline_metadata(n_items=100)
        store = JsonlRunStore(tmp_path, "run-010")
        store.open(create=True, expected_metadata=m1)
        store.write_metadata(m1)
        store.close()

        m2 = _baseline_metadata(n_items=200)
        store2 = JsonlRunStore(tmp_path, "run-010")
        try:
            with pytest.raises(ValueError, match="n_items"):
                store2.open(create=False, expected_metadata=m2)
        finally:
            try:
                store2.close()
            except Exception:
                pass

    # ---- 11. resume allows new candidate --------------------------------

    def test_resume_allows_new_candidate(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        m1 = _baseline_metadata()
        m1["candidates"] = [
            {"alias": "A", "prompt_sha256": "X"},
            {"alias": "B", "prompt_sha256": "Y"},
        ]
        store = JsonlRunStore(tmp_path, "run-011")
        store.open(create=True, expected_metadata=m1)
        store.write_metadata(m1)
        store.close()

        m2 = _baseline_metadata()
        m2["candidates"] = [
            {"alias": "A", "prompt_sha256": "X"},
            {"alias": "B", "prompt_sha256": "Y"},
            {"alias": "C", "prompt_sha256": "Z"},
        ]
        store2 = JsonlRunStore(tmp_path, "run-011")
        try:
            # Should not raise — adding a NEW candidate is allowed.
            store2.open(create=False, expected_metadata=m2)
        finally:
            store2.close()

    # ---- 12. resume rejects modified candidate --------------------------

    def test_resume_rejects_modified_candidate(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        m1 = _baseline_metadata()
        m1["candidates"] = [{"alias": "A", "prompt_sha256": "X"}]
        store = JsonlRunStore(tmp_path, "run-012")
        store.open(create=True, expected_metadata=m1)
        store.write_metadata(m1)
        store.close()

        m2 = _baseline_metadata()
        m2["candidates"] = [{"alias": "A", "prompt_sha256": "Y"}]
        store2 = JsonlRunStore(tmp_path, "run-012")
        try:
            with pytest.raises(ValueError):
                store2.open(create=False, expected_metadata=m2)
        finally:
            try:
                store2.close()
            except Exception:
                pass

    # ---- 13. record_failure increments attempts -------------------------

    def test_record_failure_increments_attempts(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-013")
        try:
            store.open(create=True)
            counts = []
            for _ in range(3):
                n = store.record_failure("genA", "item-1", "boom")
                counts.append(n)
            assert counts == [1, 2, 3]
        finally:
            store.close()

    # ---- 14. writer thread batches by size ------------------------------

    def test_writer_thread_batches_by_size(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-014")
        try:
            store.open(create=True)
            rel = "generations/big.jsonl"
            rows = [{"item_id": f"i{i}", "value": i} for i in range(150)]
            for row in rows:
                store.append(rel, row)
            store.flush()

            on_disk = _read_jsonl(tmp_path / "run-014" / rel)
            assert on_disk == rows
        finally:
            store.close()

    # ---- 15. writer thread batches by time ------------------------------

    @pytest.mark.slow
    def test_writer_thread_batches_by_time(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-015")
        try:
            store.open(create=True)
            rel = "generations/slow.jsonl"
            rows = [{"item_id": "i0", "value": 0}, {"item_id": "i1", "value": 1}]
            for row in rows:
                store.append(rel, row)
            # Wait long enough for the time-based flush (>2s).
            time.sleep(3.0)

            on_disk = _read_jsonl(tmp_path / "run-015" / rel)
            assert on_disk == rows
        finally:
            store.close()

    # ---- 16. fsync called at chunk boundary -----------------------------

    def test_fsync_called_at_chunk_boundary(self, tmp_path):
        from pipeline.phase3 import storage as storage_mod
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-016")
        try:
            store.open(create=True)
            rel = "generations/sync.jsonl"
            for i in range(5):
                store.append(rel, {"item_id": f"i{i}", "value": i})

            # Patch os.fsync inside the storage module so we capture calls.
            with patch.object(storage_mod.os, "fsync") as mock_fsync:
                store.flush(fsync=True)
                assert mock_fsync.called
                # The first positional arg should look like a file descriptor (int).
                first_call = mock_fsync.call_args_list[0]
                args = first_call.args if first_call.args else first_call[0]
                assert len(args) >= 1
                assert isinstance(args[0], int)
        finally:
            store.close()

    # ---- 17. iter_rows is a generator -----------------------------------

    def test_iter_rows_streams(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-017")
        try:
            store.open(create=True)
            rel = "generations/stream.jsonl"
            rows = [{"item_id": f"i{i}", "value": i} for i in range(4)]
            for row in rows:
                store.append(rel, row)
            store.flush()

            gen = store.iter_rows(rel)
            assert inspect.isgenerator(gen)
            collected = list(gen)
            assert collected == rows
        finally:
            store.close()

    # ---- 18. read_all returns list --------------------------------------

    def test_read_all_returns_list(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-018")
        try:
            store.open(create=True)
            rel = "generations/all.jsonl"
            rows = [{"item_id": f"i{i}", "value": i} for i in range(3)]
            for row in rows:
                store.append(rel, row)
            store.flush()

            result = store.read_all(rel)
            assert isinstance(result, list)
            assert all(isinstance(r, dict) for r in result)
            assert result == rows
        finally:
            store.close()

    # ---- 19. update_heartbeat merges ------------------------------------

    def test_update_heartbeat_merges(self, tmp_path):
        from pipeline.phase3.storage import JsonlRunStore

        store = JsonlRunStore(tmp_path, "run-019")
        try:
            store.open(create=True, expected_metadata=_baseline_metadata())
            store.write_metadata(_baseline_metadata())

            store.update_heartbeat(rows_written=5, last_write_ts="2026-04-09T00:00:00")
            meta = store.read_metadata()
            hb = meta.get("heartbeat", {})
            assert hb.get("rows_written") == 5
            assert hb.get("last_write_ts") == "2026-04-09T00:00:00"
        finally:
            store.close()
