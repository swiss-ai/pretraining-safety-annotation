"""Tests for charter.scale SidecarReader."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.charter.scale.reader import SidecarReader


@pytest.fixture
def sidecar_parquet(tmp_path):
    """Create a small sidecar parquet with 100 rows across 2 row groups."""
    path = tmp_path / "sidecar.parquet"
    rows = []
    for i in range(100):
        rows.append({
            "doc_id": f"doc_{i:04d}",
            "text": f"Text content for document {i}. " * 10,
            "token_length": 50 + i,
            "safety_score": 0.9 - (i * 0.001),
            "reflection": "",
            "preflection": "",
            "reflection_position": 0,
            "is_bad": False,
        })

    table = pa.table({
        "doc_id": [r["doc_id"] for r in rows],
        "text": [r["text"] for r in rows],
        "token_length": [r["token_length"] for r in rows],
        "safety_score": [r["safety_score"] for r in rows],
        "reflection": [r["reflection"] for r in rows],
        "preflection": [r["preflection"] for r in rows],
        "reflection_position": [r["reflection_position"] for r in rows],
        "is_bad": [r["is_bad"] for r in rows],
    })

    # Write with small row group size to get multiple row groups
    pq.write_table(table, path, row_group_size=50)
    return str(path)


class TestSidecarReader:
    def test_reads_correct_row_range(self, sidecar_parquet):
        reader = SidecarReader(sidecar_parquet, rows_per_task=30)
        docs = list(reader.run(rank=0))
        assert len(docs) == 30
        assert docs[0].id == "doc_0000"
        assert docs[0].metadata["global_row_idx"] == 0
        assert docs[29].id == "doc_0029"
        assert docs[29].metadata["global_row_idx"] == 29

    def test_second_rank(self, sidecar_parquet):
        reader = SidecarReader(sidecar_parquet, rows_per_task=30)
        docs = list(reader.run(rank=1))
        assert len(docs) == 30
        assert docs[0].id == "doc_0030"
        assert docs[0].metadata["global_row_idx"] == 30

    def test_last_rank_clamps(self, sidecar_parquet):
        reader = SidecarReader(sidecar_parquet, rows_per_task=30)
        # rank=3 should get rows 90-99 (10 rows, not 30)
        docs = list(reader.run(rank=3))
        assert len(docs) == 10
        assert docs[0].id == "doc_0090"
        assert docs[-1].id == "doc_0099"

    def test_rank_beyond_file_yields_nothing(self, sidecar_parquet):
        reader = SidecarReader(sidecar_parquet, rows_per_task=30)
        docs = list(reader.run(rank=10))
        assert len(docs) == 0

    def test_metadata_has_safety_score(self, sidecar_parquet):
        reader = SidecarReader(sidecar_parquet, rows_per_task=10)
        docs = list(reader.run(rank=0))
        assert docs[0].metadata["safety_score"] is not None

    def test_metadata_has_token_length(self, sidecar_parquet):
        reader = SidecarReader(sidecar_parquet, rows_per_task=10)
        docs = list(reader.run(rank=0))
        # Fixture sets token_length = 50 + i
        assert docs[0].metadata["token_length"] == 50
        assert docs[9].metadata["token_length"] == 59

    def test_reads_across_row_groups(self, sidecar_parquet):
        # rows_per_task=60 spans both row groups (50 each)
        reader = SidecarReader(sidecar_parquet, rows_per_task=60)
        docs = list(reader.run(rank=0))
        assert len(docs) == 60
        assert docs[49].id == "doc_0049"  # last row of first row group
        assert docs[50].id == "doc_0050"  # first row of second row group

    def test_all_rows_covered(self, sidecar_parquet):
        reader = SidecarReader(sidecar_parquet, rows_per_task=30)
        all_docs = []
        for rank in range(4):  # ceil(100/30) = 4
            all_docs.extend(reader.run(rank=rank))
        assert len(all_docs) == 100
        ids = [d.id for d in all_docs]
        assert ids == [f"doc_{i:04d}" for i in range(100)]
