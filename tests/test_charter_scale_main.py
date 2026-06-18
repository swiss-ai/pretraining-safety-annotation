"""Tests for the scale CLI helpers: n_tasks derivation, paths freezing, and the
frozen-config guard that protects resume/rerun/status."""

from __future__ import annotations

import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pipeline.charter.scale.__main__ as scale_main
from pipeline.charter.scale.__main__ import (
    DEFAULT_MAX_TASKS,
    _check_or_write_run_config,
    _derive_n_tasks,
    _freeze_paths_file,
    _list_source_shards,
    _paths_fingerprint,
    _resolve_annotation_inputs,
)
from pipeline.config import load_config
from pipeline.corpus import get_corpus


class TestDeriveNTasks:
    def test_explicit_override(self):
        assert _derive_n_tasks(10, 8) == 8

    def test_auto_caps_at_default(self):
        assert _derive_n_tasks(10_000, 0) == DEFAULT_MAX_TASKS

    def test_auto_below_cap_uses_shard_count(self):
        assert _derive_n_tasks(5, 0) == 5

    def test_never_zero(self):
        assert _derive_n_tasks(0, 0) == 1


class TestPathsFreeze:
    def test_fingerprint_order_independent(self):
        assert _paths_fingerprint(["b", "a"]) == _paths_fingerprint(["a", "b"])

    def test_fingerprint_detects_added_shard(self):
        assert _paths_fingerprint(["a", "b"]) != _paths_fingerprint(["a", "b", "c"])

    def test_freeze_writes_sorted(self, tmp_path):
        p = tmp_path / "shards.txt"
        _freeze_paths_file(["c", "a", "b"], p)
        assert p.read_text().split() == ["a", "b", "c"]


class TestRunConfigGuard:
    def test_first_write_then_match(self, tmp_path):
        p = tmp_path / "run_config.json"
        cfg = {"n_tasks": 4, "paths_fingerprint": {"n": 4}}
        _check_or_write_run_config(p, cfg, ["n_tasks", "paths_fingerprint"])
        # identical -> no exit
        _check_or_write_run_config(p, dict(cfg), ["n_tasks", "paths_fingerprint"])

    def test_guarded_change_exits(self, tmp_path):
        p = tmp_path / "run_config.json"
        _check_or_write_run_config(p, {"n_tasks": 4}, ["n_tasks"])
        with pytest.raises(SystemExit):
            _check_or_write_run_config(p, {"n_tasks": 8}, ["n_tasks"])


def _cfg(tmp_path, n_tasks=0):
    return load_config(
        [
            f"charter.scale.filtered_dir={tmp_path}/filtered",
            f"charter.scale.output_dir={tmp_path}/out",
            f"charter.scale.n_tasks={n_tasks}",
        ]
    )


def _write_dense(d, n):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        t = pa.table(
            {
                "id": [f"x{i}"],
                "text": ["t"],
                "safety_score": [5],
                "language": ["en"],
                "source_shard": ["s"],
            }
        )
        pq.write_table(t, d / f"{i:05d}.parquet")


class TestResolveAnnotationInputs:
    def test_freezes_then_reads_back_frozen_n_tasks(self, tmp_path):
        _write_dense(tmp_path / "filtered", 5)
        paths_file, n_tasks = _resolve_annotation_inputs(_cfg(tmp_path, 0), "reflections")
        assert n_tasks == 5  # min(5, DEFAULT_MAX_TASKS)
        assert json.loads((tmp_path / "out" / "reflections" / "run_config.json").read_text())["n_tasks"] == 5

        # A later config edit to n_tasks must NOT change the frozen value.
        _, n_tasks2 = _resolve_annotation_inputs(_cfg(tmp_path, 3), "reflections")
        assert n_tasks2 == 5

    def test_changed_dense_dataset_trips_guard(self, tmp_path):
        _write_dense(tmp_path / "filtered", 5)
        _resolve_annotation_inputs(_cfg(tmp_path, 0), "reflections")
        # Materially change the dense dataset (different shard fingerprint).
        _write_dense(tmp_path / "filtered", 6)
        with pytest.raises(SystemExit):
            _resolve_annotation_inputs(_cfg(tmp_path, 0), "reflections")

    def test_missing_dense_dataset_asserts(self, tmp_path):
        (tmp_path / "filtered").mkdir(parents=True)
        with pytest.raises(AssertionError):
            _resolve_annotation_inputs(_cfg(tmp_path, 0), "reflections")


def _probs6(score, conf=0.95):
    p = [(1.0 - conf) / 5.0] * 6
    p[score] = conf
    return p


def _write_source_shard(path, n=4, lang="en"):
    """A source-schema shard (top-level safety_score/safety_probs + metadata struct)."""
    rows = []
    for i in range(n):
        score = 5 if i % 2 == 0 else 1  # mix of passing / failing rows
        rows.append(
            {
                "text": f"src {path.name} {i} " * 3,
                "id": f"<urn:{path.name}-{i}>",
                "safety_score": score,
                "safety_probs": _probs6(score),
                "metadata": {"language": lang, "embeddings": [0.0] * 4, "file_path": "up.warc"},
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def capture_executor(monkeypatch):
    """Stub _ExclusiveSlurmExecutor.create so cmd_* assemble a pipeline without sbatch."""
    cap = {}

    class _Stub:
        def run(self):
            cap["ran"] = True

    def fake_create(**kwargs):
        cap.update(kwargs)
        return _Stub()

    monkeypatch.setattr(scale_main._ExclusiveSlurmExecutor, "create", staticmethod(fake_create))
    return cap


class TestCliWiring:
    def test_prefilter_pipeline_shape(self, tmp_path, capture_executor):
        src = tmp_path / "src"
        src.mkdir()
        for s in range(5):
            _write_source_shard(src / f"000_{s:05d}.parquet")
        scale_main.cmd_prefilter(
            argparse.Namespace(),
            [
                f"charter.scale.source_dir={src}",
                f"charter.scale.filtered_dir={tmp_path}/filtered",
                "charter.scale.prefilter_max_shards=2",
                "charter.scale.n_tasks=2",
            ],
        )
        assert [type(s).__name__ for s in capture_executor["pipeline"]] == [
            "CorpusReader",
            "SafetyLanguageFilter",
            "ParquetWriter",
        ]
        assert capture_executor["tasks"] == 2
        assert capture_executor["gpus_per_task"] == 0  # no-sglang prefilter
        assert "sglang" not in capture_executor["env_command"]
        # prefilter_max_shards capped the frozen source universe to 2.
        frozen = (tmp_path / "filtered" / "_source_shards.txt").read_text().split()
        assert len(frozen) == 2

    def test_submit_pipeline_shape(self, tmp_path, capture_executor):
        _write_dense(tmp_path / "filtered", 4)
        scale_main.cmd_submit(
            argparse.Namespace(run="reflections"),
            [
                f"charter.scale.filtered_dir={tmp_path}/filtered",
                f"charter.scale.output_dir={tmp_path}/out",
                "charter.scale.n_tasks=1",
            ],
        )
        pipe = capture_executor["pipeline"]
        assert [type(s).__name__ for s in pipe] == ["ParquetReader", "AnnotationGenerator"]
        cfg = load_config([f"charter.scale.filtered_dir={tmp_path}/filtered"])
        sg = cfg.charter.scale.sglang
        assert capture_executor["gpus_per_task"] == sg.tp_size * sg.dp_size
        assert "sglang" in capture_executor["env_command"]
        # The reader reads the frozen dense paths_file.
        assert "reflections/filtered_shards.txt" in str(pipe[0].paths_file)


class TestListSourceShards:
    def test_fineweb_per_language_dir_selection(self, tmp_path):
        src = tmp_path / "fw"
        src.mkdir()
        for d in ("deu_Latn", "fra_Latn", "spa_Latn"):  # spa is not a target
            (src / d).mkdir()
            _write_source_shard(src / d / "000_00000.parquet")
        cfg = load_config(
            [
                "charter.scale.corpus=fineweb-2",
                f"charter.scale.source_dir={src}",
                "charter.scale.language_filter=[deu,fra]",
            ]
        )
        shards = _list_source_shards(cfg, get_corpus("fineweb-2"))
        assert shards == ["deu_Latn/000_00000.parquet", "fra_Latn/000_00000.parquet"]

    def test_unmapped_language_asserts(self, tmp_path):
        src = tmp_path / "fw"
        src.mkdir()
        cfg = load_config(
            [
                "charter.scale.corpus=fineweb-2",
                f"charter.scale.source_dir={src}",
                "charter.scale.language_filter=[xyz]",
            ]
        )
        with pytest.raises(AssertionError):
            _list_source_shards(cfg, get_corpus("fineweb-2"))
