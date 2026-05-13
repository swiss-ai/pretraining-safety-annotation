"""Append-only JSONL run store with a writer thread for charter.eval runs.

A `JsonlRunStore` owns a single run directory under the charter.eval root
and provides:

- atomic, durable metadata snapshots (with resume validation)
- bounded-queue writer thread that batches per-file
- per-file `done_keys` for per-item resume
- chunk-boundary `flush(fsync=True)` for crash durability
- failures sidecar with attempt counting

The store does NOT install signal handlers — callers wrap their work in
`try/finally: store.close()`.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from pipeline.log import logger

# Per-file batch parameters: flush every BATCH_SIZE rows or BATCH_SECONDS,
# whichever first.
BATCH_SIZE = 64
BATCH_SECONDS = 2.0
QUEUE_MAXSIZE = 4096


class _Sentinel:
    """Single sentinel instance enqueued by close() to drain the writer."""


_SENTINEL = _Sentinel()


class JsonlRunStore:
    """Charter eval eval run store. See module docstring for the contract."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / run_id

        # Writer-thread state. None until open() is called.
        self._queue: queue.Queue | None = None
        self._writer_thread: threading.Thread | None = None
        self._closed = False
        self._opened = False

        # File handles cached lazily, keyed by absolute path.
        self._handles: dict[Path, Any] = {}
        self._handles_lock = threading.Lock()

        # Per-file batch state used by the writer thread.
        self._pending: dict[Path, list[str]] = {}
        self._last_flush_ts: dict[Path, float] = {}

        # Failure attempt counter (in-memory mirror of failures/<name>.jsonl).
        # Keyed by (name, item_id).  Accessed via get_failure_count /
        # set_failure_count so external callers don't touch the dict directly.
        self._failure_attempts: dict[tuple[str, str], int] = {}

        # Flush coordination: main thread sets _flush_requested + (optionally)
        # _flush_fsync_requested under _flush_lock, then waits on
        # _flush_done_event. Writer thread checks _flush_requested between
        # queue items, drains, fsyncs if requested, and signals _flush_done_event.
        self._flush_lock = threading.Lock()
        self._flush_requested = False
        self._flush_fsync_requested = False
        self._flush_done_event = threading.Event()

    # ------------------------------------------------------------------ open

    def open(
        self,
        *,
        create: bool,
        expected_metadata: dict | None = None,
    ) -> None:
        """Open or resume a run dir.

        See module docstring for create vs resume semantics.
        """
        if self._opened:
            return

        if create:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            meta_path = self.run_dir / "metadata.json"
            if meta_path.exists():
                # Re-opening an existing dir with create=True is fine, but if
                # the existing metadata clashes with the expected one, refuse.
                if expected_metadata is not None:
                    self._validate_metadata(
                        json.loads(meta_path.read_text()), expected_metadata
                    )
            else:
                # Bootstrap a minimal metadata file so resume works even if
                # the caller never calls write_metadata.
                self._atomic_write_metadata(expected_metadata or {})
        else:
            meta_path = self.run_dir / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"No metadata.json in run dir: {self.run_dir}")
            existing = json.loads(meta_path.read_text())
            if expected_metadata is not None:
                self._validate_metadata(existing, expected_metadata)

        self._queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"JsonlRunStoreWriter[{self.run_id}]",
            daemon=False,
        )
        self._writer_thread.start()
        self._opened = True

    # ----------------------------------------------------- metadata helpers

    def _atomic_write_metadata(self, m: dict) -> None:
        """Atomic write of metadata.json via temp file + rename."""
        meta_path = self.run_dir / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(meta_path.parent),
            prefix=".metadata.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(m, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, meta_path)

    def write_metadata(self, m: dict) -> None:
        self._atomic_write_metadata(m)

    def read_metadata(self) -> dict:
        meta_path = self.run_dir / "metadata.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())

    def update_heartbeat(self, **fields) -> None:
        """Merge fields into metadata['heartbeat']."""
        meta = self.read_metadata()
        hb = dict(meta.get("heartbeat") or {})
        hb.update(fields)
        meta["heartbeat"] = hb
        self._atomic_write_metadata(meta)

    @staticmethod
    def _validate_metadata(existing: dict, expected: dict) -> None:
        """Raise ValueError if any expected scalar field doesn't match existing.

        - Scalar fields (n_items, seed, max_tokens, dataset_revision) must match
          if present in `expected`.
        - ``gold_judge`` prompt SHA256 fields must match.
        - ``candidates``: every entry in expected.candidates that has the SAME
          alias as one in existing.candidates must have matching prompt SHA256s.
          New aliases (not in existing) are allowed.

        Supports both old (``prompt_sha256``) and new
        (``prompt_reflection_sha256`` + ``prompt_preflection_sha256``) formats.
        """
        for key in ("n_items", "seed", "max_tokens", "dataset_revision"):
            if key in expected and existing.get(key) != expected[key]:
                raise ValueError(
                    f"Resume metadata mismatch on {key}: "
                    f"existing={existing.get(key)!r} expected={expected[key]!r}"
                )

        sha_keys = (
            "prompt_sha256",
            "prompt_reflection_sha256",
            "prompt_preflection_sha256",
        )

        if "gold_judge" in expected:
            ex_gj = expected.get("gold_judge") or {}
            cur_gj = existing.get("gold_judge") or {}
            for sk in sha_keys:
                if sk in ex_gj and sk in cur_gj:
                    if cur_gj[sk] != ex_gj[sk]:
                        raise ValueError(
                            f"Resume metadata mismatch on gold_judge.{sk}: "
                            f"existing={cur_gj[sk]!r} expected={ex_gj[sk]!r}"
                        )

        if "candidates" in expected:
            existing_by_alias = {
                c.get("alias"): c for c in (existing.get("candidates") or [])
            }
            for cand in expected["candidates"]:
                alias = cand.get("alias")
                if alias in existing_by_alias:
                    cur = existing_by_alias[alias]
                    for sk in sha_keys:
                        if sk in cand and sk in cur:
                            if cur[sk] != cand[sk]:
                                raise ValueError(
                                    f"Resume metadata mismatch on candidate "
                                    f"'{alias}' {sk}: existing={cur[sk]!r} "
                                    f"expected={cand[sk]!r}"
                                )

    # --------------------------------------------------------------- append

    def _check_open(self) -> None:
        if not self._opened or self._queue is None:
            raise RuntimeError("JsonlRunStore not opened")
        if self._closed:
            raise RuntimeError("JsonlRunStore is closed")

    def append(self, rel_path: str, row: dict) -> None:
        """Enqueue a row for the writer thread."""
        self._check_open()
        line = json.dumps(row, ensure_ascii=False)
        # blocks if the queue is full → backpressure
        self._queue.put((rel_path, line))  # type: ignore[union-attr]

    def append_many(self, rel_path: str, rows: Iterable[dict]) -> None:
        for row in rows:
            self.append(rel_path, row)

    # --------------------------------------------------------- read helpers

    def _abs_path(self, rel_path: str) -> Path:
        return self.run_dir / rel_path

    def iter_rows(self, rel_path: str) -> Iterator[dict]:
        """Stream rows from a JSONL file. Tolerates a torn final line."""
        path = self._abs_path(rel_path)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            line_no = 0
            buffered_last: tuple[int, str] | None = None
            for line in f:
                line_no += 1
                if buffered_last is not None:
                    prev_no, prev = buffered_last
                    yield self._parse_row_or_skip(rel_path, prev_no, prev)
                buffered_last = (line_no, line)
            if buffered_last is not None:
                prev_no, prev = buffered_last
                # Tolerate torn last line
                stripped = prev.rstrip("\n")
                if not stripped.strip():
                    return
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "torn last line in {} (line {}): {}; discarding {} bytes",
                        rel_path,
                        prev_no,
                        e,
                        len(prev),
                    )

    def _parse_row_or_skip(self, rel_path: str, line_no: int, line: str) -> dict:
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(
                "skipping unparseable line {} in {}: {}", line_no, rel_path, e
            )
            return {}

    def read_all(self, rel_path: str) -> list[dict]:
        return [r for r in self.iter_rows(rel_path) if r]

    def done_keys(
        self,
        rel_path: str,
        key: str | tuple[str, ...] = "item_id",
    ) -> set:
        """Return the set of resume keys present in `rel_path`.

        Tolerates a torn final line. The default key is the per-item id;
        for files keyed by (item_id, iteration) pass `("item_id","iteration")`.
        """
        path = self._abs_path(rel_path)
        keys: set = set()
        if not path.exists():
            return keys
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, raw in enumerate(lines):
            stripped = raw.rstrip("\n")
            if not stripped.strip():
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                if i == len(lines) - 1:
                    logger.warning(
                        "torn last line in {}: {}; discarding {} bytes",
                        rel_path,
                        e,
                        len(raw),
                    )
                    continue
                logger.warning(
                    "skipping unparseable line {} in {}: {}", i + 1, rel_path, e
                )
                continue
            if isinstance(key, str):
                if key in row:
                    keys.add(row[key])
            else:
                if all(k in row for k in key):
                    keys.add(tuple(row[k] for k in key))
        return keys

    # -------------------------------------------------------------- failures

    def record_failure(self, name: str, item_id: str, reason: str) -> int:
        """Append a failure record and return total attempts so far for item_id.

        The returned count INCLUDES the call that just happened, so a fresh
        item starts at 1, the next call returns 2, etc.
        """
        self._check_open()
        rel = f"failures/{name}.jsonl"
        attempts = self._failure_attempts.get((name, item_id), 0) + 1
        self._failure_attempts[(name, item_id)] = attempts
        record = {
            "item_id": item_id,
            "reason": reason,
            "attempt": attempts,
            "ts": _now_iso(),
        }
        self.append(rel, record)
        return attempts

    def get_failure_count(self, name: str, item_id: str) -> int:
        """Return the current in-memory failure attempt count for (name, item_id)."""
        return self._failure_attempts.get((name, item_id), 0)

    def set_failure_count(self, name: str, item_id: str, count: int) -> None:
        """Set the in-memory failure attempt count for (name, item_id)."""
        self._failure_attempts[(name, item_id)] = count

    # ------------------------------------------------------------- writer thread

    def _writer_loop(self) -> None:
        """Background writer: drains queue, batches per file, periodic flushes."""
        try:
            while True:
                # Try to get an item with a short timeout so we can do
                # time-based batch flushes.
                try:
                    item = self._queue.get(timeout=0.5)  # type: ignore[union-attr]
                except queue.Empty:
                    self._maybe_time_flush()
                    self._maybe_handle_flush_request()
                    continue

                if item is _SENTINEL:
                    # Drain anything left, flush all, exit.
                    self._drain_queue()
                    self._flush_all_pending(fsync=False)
                    self._maybe_handle_flush_request()
                    return

                rel_path, line = item
                self._enqueue_pending(rel_path, line)
                self._maybe_size_flush(rel_path)
                self._maybe_handle_flush_request()
        except Exception:
            logger.exception("JsonlRunStore writer thread crashed")
            raise

    def _drain_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()  # type: ignore[union-attr]
            except queue.Empty:
                return
            if item is _SENTINEL:
                return
            rel_path, line = item
            self._enqueue_pending(rel_path, line)

    def _enqueue_pending(self, rel_path: str, line: str) -> None:
        path = self._abs_path(rel_path)
        if path not in self._pending:
            self._pending[path] = []
            self._last_flush_ts[path] = time.monotonic()
        self._pending[path].append(line)

    def _maybe_size_flush(self, rel_path: str) -> None:
        path = self._abs_path(rel_path)
        if len(self._pending.get(path, [])) >= BATCH_SIZE:
            self._flush_one(path, fsync=False)

    def _maybe_time_flush(self) -> None:
        now = time.monotonic()
        # iterate over a copy because _flush_one drops the entry
        for path in list(self._pending.keys()):
            if now - self._last_flush_ts.get(path, now) >= BATCH_SECONDS:
                self._flush_one(path, fsync=False)

    def _flush_one(self, path: Path, fsync: bool) -> None:
        lines = self._pending.get(path)
        if not lines:
            self._last_flush_ts[path] = time.monotonic()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = self._get_handle(path)
        fh.write("\n".join(lines) + "\n")
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())
        self._pending[path] = []
        self._last_flush_ts[path] = time.monotonic()

    def _flush_all_pending(self, fsync: bool) -> None:
        for path in list(self._pending.keys()):
            self._flush_one(path, fsync=fsync)
        if fsync:
            with self._handles_lock:
                for fh in self._handles.values():
                    try:
                        os.fsync(fh.fileno())
                    except OSError as e:
                        logger.warning("fsync failed: {}", e)

    def _get_handle(self, path: Path):
        with self._handles_lock:
            fh = self._handles.get(path)
            if fh is None or fh.closed:
                fh = path.open("a", encoding="utf-8")
                self._handles[path] = fh
            return fh

    def _maybe_handle_flush_request(self) -> None:
        with self._flush_lock:
            if not self._flush_requested:
                return
            fsync = self._flush_fsync_requested
            self._flush_requested = False
        # Drain everything that's still in the queue (items the producer
        # enqueued before requesting the flush).
        self._drain_queue()
        self._flush_all_pending(fsync=fsync)
        self._flush_done_event.set()

    # ------------------------------------------------------------- public flush

    def flush(self, fsync: bool = True) -> None:
        """Synchronously drain the queue and flush all pending batches."""
        if not self._opened or self._queue is None:
            return
        if self._closed:
            return
        with self._flush_lock:
            self._flush_done_event.clear()
            self._flush_fsync_requested = fsync
            self._flush_requested = True
        # Wait until the writer thread sees the request and signals completion.
        # If the writer is blocked in queue.get(timeout=0.5), it wakes within
        # 500ms and processes the flush request.
        if not self._flush_done_event.wait(timeout=30.0):
            logger.warning("flush timed out after 30s; queue may be wedged")

    # -------------------------------------------------------------- close

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is not None and self._writer_thread is not None:
            try:
                self._queue.put(_SENTINEL)
            except Exception:
                pass
            self._writer_thread.join(timeout=30.0)
            if self._writer_thread.is_alive():
                logger.warning("writer thread did not exit in 30s")
        # final flush + fsync of any straggler batches (sentinel handler did
        # this already, but be defensive)
        try:
            self._flush_all_pending(fsync=True)
        except Exception as e:
            logger.warning("close-time flush failed: {}", e)
        with self._handles_lock:
            for fh in self._handles.values():
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                    fh.close()
                except Exception as e:
                    logger.warning("close-time handle close failed: {}", e)
            self._handles.clear()


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()
