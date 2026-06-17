"""Tests for the export step (per-rank JSONL -> doc_id-keyed parquet dataset)."""

from __future__ import annotations

import json

import pyarrow.parquet as pq

from pipeline.charter.scale.export import export_run


def _result(doc_id, reflection="ok", **over):
    row = {
        "doc_id": doc_id,
        "language": "en",
        "safety_score": 5,
        "source_shard": "000_00000.parquet",
        "reflection_1p": reflection,
        "reflection_position": 12,
        "reflection_token_index": 7,
        "charter_reflection": json.dumps(["1.2"]),
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_tokens": 0,
    }
    row.update(over)
    return row


def _write_rank(run_dir, rank, lines, *, completed=True, failures=False):
    rank_dir = run_dir / rank
    rank_dir.mkdir(parents=True)
    with open(rank_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    if completed:
        (run_dir / "completions").mkdir(exist_ok=True)
        (run_dir / "completions" / rank).write_text("")
    if failures:
        (rank_dir / "failures.jsonl").write_text('{"doc_id":"x","error":"boom"}\n')


def _load(dataset_dir, rank):
    return pq.read_table(dataset_dir + f"/{rank}.parquet")


class TestExport:
    def test_basic_roundtrip_and_types(self, tmp_path):
        run_dir = tmp_path / "reflections"
        lines = [json.dumps(_result(f"<urn:{i}>")) for i in range(3)]
        _write_rank(run_dir, "00000", lines)

        dataset_dir = export_run(str(tmp_path), "reflections", corpus="dclm-edu")
        t = _load(dataset_dir, "00000")

        assert t.num_rows == 3
        names = set(t.column_names)
        assert {"doc_id", "corpus", "source_shard", "language", "safety_score"} <= names
        assert {"reflection_1p", "reflection_position", "reflection_token_index", "charter_reflection"} <= names
        assert {"input_tokens", "output_tokens", "reasoning_tokens"} <= names
        schema = t.schema
        # output-column types via the meta map
        assert schema.field("reflection_position").type == __import__("pyarrow").int32()
        assert schema.field("reflection_token_index").type == __import__("pyarrow").int32()
        # provenance/usage explicitly typed (NOT coerced to large_string)
        assert schema.field("safety_score").type == __import__("pyarrow").int64()
        assert schema.field("input_tokens").type == __import__("pyarrow").int64()
        # corpus filled from the param
        assert set(t.column("corpus").to_pylist()) == {"dclm-edu"}

    def test_dedup_by_doc_id_keep_last(self, tmp_path):
        run_dir = tmp_path / "reflections"
        lines = [
            json.dumps(_result("<urn:dup>", reflection="first")),
            json.dumps(_result("<urn:other>")),
            json.dumps(_result("<urn:dup>", reflection="LAST")),
        ]
        _write_rank(run_dir, "00000", lines)
        dataset_dir = export_run(str(tmp_path), "reflections", corpus="dclm-edu")
        t = _load(dataset_dir, "00000")
        rows = {r["doc_id"]: r["reflection_1p"] for r in t.to_pylist()}
        assert t.num_rows == 2
        assert rows["<urn:dup>"] == "LAST"

    def test_torn_last_line_and_missing_doc_id_skipped(self, tmp_path):
        run_dir = tmp_path / "reflections"
        lines = [
            json.dumps(_result("<urn:ok>")),
            json.dumps({"reflection_1p": "no id here"}),  # missing doc_id -> skip
            '{"doc_id": "<urn:torn>", "reflection_1p": "tr',  # torn JSON -> skip
        ]
        _write_rank(run_dir, "00000", lines)
        dataset_dir = export_run(str(tmp_path), "reflections", corpus="dclm-edu")
        t = _load(dataset_dir, "00000")
        assert t.column("doc_id").to_pylist() == ["<urn:ok>"]

    def test_surrogate_sanitization(self, tmp_path):
        run_dir = tmp_path / "reflections"
        # ensure_ascii=True (as the generator writes) encodes the lone surrogate
        # as \udXXX; json.loads turns it back into a lone surrogate at export.
        line = json.dumps(_result("<urn:s>", reflection="bad\ud800text"), ensure_ascii=True)
        _write_rank(run_dir, "00000", [line])
        dataset_dir = export_run(str(tmp_path), "reflections", corpus="dclm-edu")
        t = _load(dataset_dir, "00000")
        val = t.column("reflection_1p").to_pylist()[0]
        assert "\ud800" not in val and "bad" in val and "text" in val

    def test_multiple_ranks_each_one_shard(self, tmp_path):
        run_dir = tmp_path / "reflections"
        _write_rank(run_dir, "00000", [json.dumps(_result("<urn:a>"))])
        _write_rank(run_dir, "00001", [json.dumps(_result("<urn:b>"))])
        dataset_dir = export_run(str(tmp_path), "reflections", corpus="dclm-edu")
        import os

        shards = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".parquet"))
        assert shards == ["00000.parquet", "00001.parquet"]

    def test_idempotent_reexport(self, tmp_path):
        run_dir = tmp_path / "reflections"
        _write_rank(run_dir, "00000", [json.dumps(_result("<urn:a>"))])
        d1 = export_run(str(tmp_path), "reflections", corpus="dclm-edu")
        d2 = export_run(str(tmp_path), "reflections", corpus="dclm-edu")
        assert d1 == d2
        assert _load(d2, "00000").num_rows == 1
