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
            "reflection_position": 0,
            # Every odd-indexed row is is_bad — gives reader-filter tests a
            # deterministic 50/50 split to exercise.
            "is_bad": bool(i % 2),
        })

    table = pa.table({
        "doc_id": [r["doc_id"] for r in rows],
        "text": [r["text"] for r in rows],
        "token_length": [r["token_length"] for r in rows],
        "safety_score": [r["safety_score"] for r in rows],
        "reflection": [r["reflection"] for r in rows],
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


class TestSidecarReaderFilter:
    """The reader's optional ``filter_column`` skips rows where the named
    boolean column is False (e.g. gating on ``is_bad`` so a run only
    processes docs with safety_score ≥ 3).

    Skipped rows must still advance global_row_idx — the merge step relies
    on that index to align results back to sidecar rows."""

    def test_filter_skips_false_rows(self, sidecar_parquet):
        reader = SidecarReader(
            sidecar_parquet, rows_per_task=30, filter_column="is_bad"
        )
        docs = list(reader.run(rank=0))
        # Rows 0-29 with is_bad = i%2: rows 1,3,5,...,29 → 15 docs.
        assert len(docs) == 15
        # Yielded docs are exactly the odd-indexed rows.
        for d in docs:
            assert d.metadata["global_row_idx"] % 2 == 1
        assert docs[0].metadata["global_row_idx"] == 1
        assert docs[-1].metadata["global_row_idx"] == 29

    def test_filter_preserves_global_row_idx(self, sidecar_parquet):
        # Skipping rows must NOT renumber global_row_idx — the merge step
        # joins rephrasals to sidecar rows by this index.
        reader = SidecarReader(
            sidecar_parquet, rows_per_task=100, filter_column="is_bad"
        )
        docs = list(reader.run(rank=0))
        observed = [d.metadata["global_row_idx"] for d in docs]
        assert observed == list(range(1, 100, 2))

    def test_filter_works_across_row_groups(self, sidecar_parquet):
        # rows_per_task=60 spans both row groups (50 each)
        reader = SidecarReader(
            sidecar_parquet, rows_per_task=60, filter_column="is_bad"
        )
        docs = list(reader.run(rank=0))
        assert len(docs) == 30  # half of 60
        assert all(d.metadata["global_row_idx"] % 2 == 1 for d in docs)

    def test_none_filter_yields_all_rows(self, sidecar_parquet):
        # Explicit None is the same as omitting filter_column.
        reader = SidecarReader(
            sidecar_parquet, rows_per_task=30, filter_column=None
        )
        docs = list(reader.run(rank=0))
        assert len(docs) == 30
