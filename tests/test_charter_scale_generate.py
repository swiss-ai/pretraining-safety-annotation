"""Tests for generate.py pure-Python helpers: doc_id-keyed resume + save loop.

The generator's run() needs a live sglang server, but its resume done-set and
the async save loop (batching + serialize-failure routing to failures.jsonl)
are pure Python and carry real logic — covered here.
"""

from __future__ import annotations

import json
import queue
import threading

from pipeline.charter.scale.generate import _load_done_set, _save_loop


def test_load_done_set_keys_on_doc_id_with_torn_line(tmp_path):
    p = tmp_path / "results.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"doc_id": "<urn:a>", "reflection_1p": "x"}) + "\n")
        f.write(json.dumps({"doc_id": "<urn:b>", "reflection_1p": "y"}) + "\n")
        f.write('{"doc_id": "<urn:torn>", "reflec')  # torn last line, no newline
    done = _load_done_set(p)
    assert done == {"<urn:a>", "<urn:b>"}  # torn line skipped


def test_load_done_set_missing_file_is_empty(tmp_path):
    assert _load_done_set(tmp_path / "nope.jsonl") == set()


def test_save_loop_writes_good_rows_and_routes_serialize_failures(tmp_path):
    results = tmp_path / "results.jsonl"
    failures = tmp_path / "failures.jsonl"
    q: queue.Queue = queue.Queue()
    q.put({"doc_id": "<urn:ok1>", "reflection_1p": "a"})
    q.put({"doc_id": "<urn:ok2>", "reflection_1p": "b"})
    # A set is not JSON-serializable -> serialize fails; has doc_id -> failures.jsonl.
    q.put({"doc_id": "<urn:bad>", "reflection_1p": {1, 2}})

    done = threading.Event()
    done.set()  # all items already enqueued; loop drains then exits
    _save_loop(q, results, batch_size=2, done_event=done, failures_path=failures)

    ok_ids = {json.loads(line)["doc_id"] for line in results.read_text().splitlines() if line.strip()}
    assert ok_ids == {"<urn:ok1>", "<urn:ok2>"}
    fail = [json.loads(line) for line in failures.read_text().splitlines() if line.strip()]
    assert len(fail) == 1 and fail[0]["doc_id"] == "<urn:bad>"
    assert "serialize" in fail[0]["error"]
