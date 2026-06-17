"""Tests for the prefilter pipeline: source -> SafetyLanguageFilter -> dense parquet.

Drives the datatrove pipeline steps directly (no SLURM) and reads the dense
output back with the default ParquetReader exactly as the annotation run does,
so the round-trip (flat dense schema -> Document.metadata) is covered too.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

from pipeline.corpus import (
    DENSE_SCHEMA,
    CorpusReader,
    SafetyLanguageFilter,
    dense_writer_adapter,
    get_corpus,
)


def _probs(score: int, conf: float) -> list[float]:
    p = [(1.0 - conf) / 5.0] * 6
    p[score] = conf
    return p


@pytest.fixture
def source_dir(tmp_path):
    """One source shard with a known pass/fail split and a non-target language."""
    d = tmp_path / "src"
    d.mkdir()
    rows = [
        # id, language, score, conf, expected-keep
        ("<urn:keep-1>", "en", 5, 0.95),  # keep
        ("<urn:keep-2>", "en", 4, 0.92),  # keep
        ("<urn:low-score>", "en", 3, 0.99),  # drop: score
        ("<urn:low-conf>", "en", 5, 0.50),  # drop: confidence
        ("<urn:wrong-lang>", "fra", 5, 0.99),  # drop: language
        ("<urn:null-id>", "en", 5, 0.99),  # drop: id below set to None
    ]
    records = []
    for doc_id, lang, score, conf in rows:
        records.append(
            {
                "text": f"text for {doc_id} " * 4,
                "id": None if doc_id == "<urn:null-id>" else doc_id,
                "safety_score": score,
                "safety_probs": _probs(score, conf),
                "metadata": {
                    "language": lang,
                    "embeddings": [0.1] * 8,
                    "file_path": "s3://upstream/orig.warc.gz",
                },
            }
        )
    pq.write_table(pa.Table.from_pylist(records), d / "000_00000.parquet")
    return d


def _run_prefilter(source_dir, dense_dir, *, min_score=4, min_confidence=0.9, languages=("en",)):
    c = get_corpus("dclm-edu")
    reader = CorpusReader(
        data_folder=str(source_dir),
        adapter=c.adapter,
        projection=c.projection,
        text_key="text",
        id_key="id",
        batch_size=4,
    )
    flt = SafetyLanguageFilter(min_score=min_score, min_confidence=min_confidence, languages=list(languages))
    writer = ParquetWriter(
        output_folder=str(dense_dir),
        adapter=dense_writer_adapter,
        schema=DENSE_SCHEMA,
        compression="snappy",
    )
    docs = reader.run(rank=0, world_size=1)
    # Exhaust the writer generator to write + close the parquet file.
    list(writer.run(flt.run(docs), rank=0, world_size=1))


class TestPrefilter:
    def test_dense_dataset_has_only_passing_rows(self, source_dir, tmp_path):
        dense = tmp_path / "filtered"
        _run_prefilter(source_dir, dense)
        table = pq.read_table(str(dense))
        ids = set(table.column("id").to_pylist())
        assert ids == {"<urn:keep-1>", "<urn:keep-2>"}

    def test_dense_schema_is_flat_with_provenance(self, source_dir, tmp_path):
        dense = tmp_path / "filtered"
        _run_prefilter(source_dir, dense)
        table = pq.read_table(str(dense))
        assert set(table.column_names) == {"id", "text", "safety_score", "language", "source_shard"}
        assert "embeddings" not in table.column_names
        assert table.schema.field("safety_score").type == pa.int64()
        assert set(table.column("source_shard").to_pylist()) == {"000_00000.parquet"}

    def test_threshold_is_configurable(self, source_dir, tmp_path):
        # Loosen the confidence gate: the score-5/conf-0.5 doc now survives.
        dense = tmp_path / "filtered_loose"
        _run_prefilter(source_dir, dense, min_confidence=0.0)
        ids = set(pq.read_table(str(dense)).column("id").to_pylist())
        assert "<urn:low-conf>" in ids  # kept now
        assert "<urn:low-score>" not in ids  # still below the score floor

    def test_annotation_reader_roundtrip(self, source_dir, tmp_path):
        # The annotation run reads the dense dataset with the DEFAULT adapter;
        # the leftover flat columns must land in Document.metadata.
        dense = tmp_path / "filtered"
        _run_prefilter(source_dir, dense)
        reader = ParquetReader(data_folder=str(dense), text_key="text", id_key="id")
        docs = sorted(reader.run(rank=0, world_size=1), key=lambda d: d.id)
        assert [d.id for d in docs] == ["<urn:keep-1>", "<urn:keep-2>"]
        md = docs[0].metadata
        assert md["safety_score"] == 5
        assert md["language"] == "en"
        assert md["source_shard"] == "000_00000.parquet"
        # No tokenization for these corpora -> generator falls back to cfg.max_tokens.
        assert "token_length" not in md
