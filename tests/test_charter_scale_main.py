"""Tests for the scale CLI helpers: n_tasks derivation, paths freezing, and the
frozen-config guard that protects resume/rerun/status."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.charter.scale.__main__ import (
    DEFAULT_MAX_TASKS,
    _check_or_write_run_config,
    _derive_n_tasks,
    _freeze_paths_file,
    _paths_fingerprint,
    _resolve_annotation_inputs,
)
from pipeline.config import load_config


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
