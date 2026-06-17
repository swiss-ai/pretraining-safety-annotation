"""Tests for the corpus dataloader: CorpusReader projection + datatrove sharding.

Fixtures mimic the source schema (top-level text/id/safety_score/safety_probs +
a metadata struct that includes the heavy `embeddings` field) so we can verify
the reader projects it away. Built in tmp_path — no committed binary fixtures.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.corpus import CorpusReader, get_corpus


def _probs(score: int, conf: float = 0.95) -> list[float]:
    p = [(1.0 - conf) / 5.0] * 6
    p[score] = conf
    return p


def _shard_rows(prefix: str, n: int, lang: str = "en"):
    """Source-schema rows with a metadata struct carrying 8-dim embeddings."""
    rows = []
    for i in range(n):
        score = i % 6
        rows.append(
            {
                "text": f"{prefix} document {i} " * 4,
                "id": f"<urn:uuid:{prefix}-{i:04d}>",
                "safety_score": score,
                "safety_probs": _probs(score),
                "metadata": {
                    "language": lang,
                    "embeddings": [0.1] * 8,
                    "file_path": "s3://upstream/original.warc.gz",
                    "url": f"http://example.com/{prefix}/{i}",
                },
            }
        )
    return rows


@pytest.fixture
def source_dir(tmp_path):
    """A directory of 5 source-like shards (named like the real corpus)."""
    d = tmp_path / "src"
    d.mkdir()
    for s in range(5):
        name = f"000_{s:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(_shard_rows(name, 6)), d / name)
    return d


def _reader(source_dir, **kw):
    c = get_corpus("dclm-edu")
    return CorpusReader(
        data_folder=str(source_dir),
        adapter=c.adapter,
        projection=c.projection,
        text_key="text",
        id_key="id",
        batch_size=4,
        **kw,
    )


class TestProjectionAndAdapter:
    def test_drops_embeddings_and_token_length(self, source_dir):
        docs = list(_reader(source_dir).read_file("000_00000.parquet"))
        assert len(docs) == 6
        md = docs[0].metadata
        # Projection ["...","metadata.language"] never reads the embeddings.
        assert "embeddings" not in md
        # No tokenization for these corpora -> generator falls back to cfg.max_tokens.
        assert "token_length" not in md

    def test_canonical_fields(self, source_dir):
        docs = list(_reader(source_dir).read_file("000_00002.parquet"))
        d = docs[0]
        assert d.id == "<urn:uuid:000_00002.parquet-0000>"
        assert d.text.startswith("000_00002.parquet document 0")
        assert d.metadata["language"] == "en"
        assert d.metadata["safety_score"] == 0
        # safety_probs is needed by the filter; it's read at top level.
        assert d.metadata["safety_probs"] is not None

    def test_source_shard_is_relative_path(self, source_dir):
        docs = list(_reader(source_dir).read_file("000_00003.parquet"))
        # NOT the upstream metadata.file_path, and NOT absolute.
        assert docs[0].metadata["source_shard"] == "000_00003.parquet"


class TestFileSharding:
    """datatrove strides whole files across tasks: sorted(files)[rank::world_size]."""

    def _shards_for_rank(self, source_dir, rank, world_size):
        docs = list(_reader(source_dir).run(rank=rank, world_size=world_size))
        return {d.metadata["source_shard"] for d in docs}

    def test_ranks_disjoint_and_complete(self, source_dir):
        r0 = self._shards_for_rank(source_dir, 0, 2)
        r1 = self._shards_for_rank(source_dir, 1, 2)
        assert r0.isdisjoint(r1)
        all_shards = {f"000_{s:05d}.parquet" for s in range(5)}
        assert r0 | r1 == all_shards
        # Strided over sorted files: rank 0 -> shards 0,2,4 ; rank 1 -> 1,3.
        assert r0 == {"000_00000.parquet", "000_00002.parquet", "000_00004.parquet"}
        assert r1 == {"000_00001.parquet", "000_00003.parquet"}

    def test_single_task_reads_everything(self, source_dir):
        docs = list(_reader(source_dir).run(rank=0, world_size=1))
        assert len(docs) == 5 * 6
        assert len({d.id for d in docs}) == 5 * 6

    def test_frozen_paths_file_strides_in_file_order(self, source_dir, tmp_path):
        # A paths_file pins the universe + order; get_shard_from_paths_file does
        # NOT re-sort, so the file must already be sorted (as our freeze writes it).
        paths = sorted(p.name for p in source_dir.glob("*.parquet"))
        paths_file = tmp_path / "shards.txt"
        paths_file.write_text("\n".join(paths) + "\n")
        c = get_corpus("dclm-edu")
        reader = CorpusReader(
            data_folder=str(source_dir),
            paths_file=str(paths_file),
            adapter=c.adapter,
            projection=c.projection,
            text_key="text",
            id_key="id",
            batch_size=4,
        )
        r0 = {d.metadata["source_shard"] for d in reader.run(rank=0, world_size=2)}
        assert r0 == {"000_00000.parquet", "000_00002.parquet", "000_00004.parquet"}
