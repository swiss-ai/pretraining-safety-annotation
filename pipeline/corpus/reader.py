"""CorpusReader: a ParquetReader that projects only the columns we need.

The stock ``ParquetReader`` reads every column (``read_metadata=True``), which
would haul the 768-dim ``embeddings`` struct (~30% of bytes) through memory for
every document. We override ``read_file`` to pass an explicit pyarrow column
projection (dotted sub-selection of the metadata struct is supported) and reuse
datatrove's file-sharding/adapter machinery unchanged.
"""

from __future__ import annotations

from datatrove.pipeline.readers import ParquetReader


class CorpusReader(ParquetReader):
    """ParquetReader with an explicit column projection (drops embeddings)."""

    name = "📒 Corpus"

    def __init__(self, *args, projection: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def read_file(self, filepath: str):
        """Yield Documents for one parquet file, reading only ``self.projection``.

        Mirrors ``ParquetReader.read_file`` but with our projection instead of
        the all-columns / text+id default.
        """
        import pyarrow.parquet as pq

        with self.data_folder.open(filepath, "rb") as f:
            with pq.ParquetFile(f) as pqf:
                li = 0
                for batch in pqf.iter_batches(batch_size=self.batch_size, columns=self.projection):
                    documents = []
                    with self.track_time("batch"):
                        for line in batch.to_pylist():
                            document = self.get_document_from_dict(line, filepath, li)
                            if not document:
                                continue
                            documents.append(document)
                            li += 1
                    yield from documents
