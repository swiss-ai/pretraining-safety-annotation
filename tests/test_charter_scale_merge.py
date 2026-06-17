"""Tests for charter.scale streaming additive merge."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.charter.scale.merge import merge_shards


@pytest.fixture
def sidecar_and_results(tmp_path):
    """Create a sidecar parquet and matching results JSONL."""
    # Create sidecar with 20 rows, 2 row groups of 10
    sidecar_path = tmp_path / "sidecar.parquet"
    table = pa.table(
        {
            "doc_id": [f"doc_{i:04d}" for i in range(20)],
            "text": [f"text {i}" for i in range(20)],
            "token_length": list(range(20)),
            "safety_score": [0.9] * 20,
            "reflection": [""] * 20,  # placeholder to be renamed
            "reflection_position": [0] * 20,
            "is_bad": [False] * 20,
        }
    )
    pq.write_table(table, str(sidecar_path), row_group_size=10)

    # Create results for the reflections run
    output_dir = tmp_path / "output"

    refl_run_dir = output_dir / "reflections" / "00000"
    refl_run_dir.mkdir(parents=True)
    refl_results = []
    for i in range(20):
        refl_results.append(
            {
                "global_row_idx": i,
                "doc_id": f"doc_{i:04d}",
                "reflection_1p": f"r1p_{i}",
                "reflection_position": 100 + i,
                "reflection_token_index": i,
                "charter_reflection": json.dumps(["1.1"]),
            }
        )
    with open(refl_run_dir / "results.jsonl", "w") as f:
        for r in refl_results:
            f.write(json.dumps(r) + "\n")

    return str(sidecar_path), str(output_dir)


class TestMergeShards:
    def test_merge_adds_columns(self, sidecar_and_results, tmp_path):
        sidecar_path, output_dir = sidecar_and_results
        out_path = str(tmp_path / "merged.parquet")

        merge_shards(output_dir, "reflections", sidecar_path, out_path)

        merged = pq.read_table(out_path)
        # Should have reflection columns
        assert "reflection_1p" in merged.column_names
        assert "charter_reflection" in merged.column_names
        assert "reflection_token_index" in merged.column_names

    def test_old_placeholders_dropped(self, sidecar_and_results, tmp_path):
        sidecar_path, output_dir = sidecar_and_results

        # Reflections merge drops old "reflection" placeholder
        out_refl = str(tmp_path / "merged_refl.parquet")
        merge_shards(output_dir, "reflections", sidecar_path, out_refl)
        merged_refl = pq.read_table(out_refl)
        assert "reflection" not in merged_refl.column_names

    def test_existing_columns_preserved(self, sidecar_and_results, tmp_path):
        sidecar_path, output_dir = sidecar_and_results
        out_path = str(tmp_path / "merged.parquet")

        merge_shards(output_dir, "reflections", sidecar_path, out_path)

        merged = pq.read_table(out_path)
        assert "doc_id" in merged.column_names
        assert "text" in merged.column_names
        assert "token_length" in merged.column_names
        assert "safety_score" in merged.column_names

    def test_data_values_correct(self, sidecar_and_results, tmp_path):
        sidecar_path, output_dir = sidecar_and_results
        out_path = str(tmp_path / "merged.parquet")

        merge_shards(output_dir, "reflections", sidecar_path, out_path)

        merged = pq.read_table(out_path)
        assert merged.column("reflection_1p")[0].as_py() == "r1p_0"
        assert merged.column("reflection_1p")[5].as_py() == "r1p_5"
        assert merged.column("reflection_position")[3].as_py() == 103

    def test_row_count_preserved(self, sidecar_and_results, tmp_path):
        sidecar_path, output_dir = sidecar_and_results
        out_path = str(tmp_path / "merged.parquet")

        merge_shards(output_dir, "reflections", sidecar_path, out_path)

        merged = pq.read_table(out_path)
        assert merged.num_rows == 20

    def test_missing_rows_fails_by_default(self, sidecar_and_results, tmp_path):
        sidecar_path, output_dir = sidecar_and_results
        # Delete some results
        import pathlib

        results_file = (
            pathlib.Path(output_dir) / "reflections" / "00000" / "results.jsonl"
        )
        with open(results_file) as f:
            lines = f.readlines()
        with open(results_file, "w") as f:
            f.writelines(lines[:10])  # only 10 of 20

        out_path = str(tmp_path / "merged.parquet")
        with pytest.raises(AssertionError, match="Missing"):
            merge_shards(output_dir, "reflections", sidecar_path, out_path)

    def test_out_of_order_results(self, tmp_path):
        """Results arrive in async completion order, not row order."""
        sidecar_path = tmp_path / "sidecar.parquet"
        table = pa.table({
            "doc_id": [f"doc_{i:04d}" for i in range(20)],
            "text": [f"text {i}" for i in range(20)],
            "token_length": list(range(20)),
            "safety_score": [0.9] * 20,
            "reflection": [""] * 20,
            "reflection_position": [0] * 20,
            "is_bad": [False] * 20,
        })
        pq.write_table(table, str(sidecar_path), row_group_size=10)

        output_dir = tmp_path / "output"
        # Two shards, each with 10 results in shuffled order
        for rank, indices in [(0, [7, 2, 9, 0, 4, 8, 1, 5, 3, 6]),
                              (1, [18, 12, 15, 10, 19, 13, 16, 11, 17, 14])]:
            rank_dir = output_dir / "reflections" / f"{rank:05d}"
            rank_dir.mkdir(parents=True)
            with open(rank_dir / "results.jsonl", "w") as f:
                for i in indices:
                    f.write(json.dumps({
                        "global_row_idx": i,
                        "doc_id": f"doc_{i:04d}",
                        "reflection_1p": f"r1p_{i}",
                        "reflection_position": 100 + i,
                        "reflection_token_index": i,
                        "charter_reflection": "[]",
                    }) + "\n")

        out_path = str(tmp_path / "merged.parquet")
        merge_shards(str(output_dir), "reflections", str(sidecar_path), out_path)

        merged = pq.read_table(out_path)
        assert merged.num_rows == 20
        # Every row should have data
        for i in range(20):
            assert merged.column("reflection_1p")[i].as_py() == f"r1p_{i}"
            assert merged.column("reflection_position")[i].as_py() == 100 + i

    def test_missing_rows_allowed(self, sidecar_and_results, tmp_path):
        sidecar_path, output_dir = sidecar_and_results
        import pathlib

        results_file = (
            pathlib.Path(output_dir) / "reflections" / "00000" / "results.jsonl"
        )
        with open(results_file) as f:
            lines = f.readlines()
        with open(results_file, "w") as f:
            f.writelines(lines[:10])

        out_path = str(tmp_path / "merged.parquet")
        merge_shards(
            output_dir, "reflections", sidecar_path, out_path, allow_missing=True
        )

        merged = pq.read_table(out_path)
        assert merged.num_rows == 20
        # Missing rows should have empty strings for text cols
        assert merged.column("reflection_1p")[15].as_py() == ""
