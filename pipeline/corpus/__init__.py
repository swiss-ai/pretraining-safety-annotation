"""General corpus dataloader for scale annotation (DCLM-Edu, FineWeb-2, ...).

Scale-only: a canonical row→Document adapter per corpus, a projecting
``CorpusReader`` (drops embeddings), and a configurable safety predicate +
prefilter step. No sampling API — sampling/subsetting of the produced annotation
dataset is done downstream.
"""

from pipeline.corpus.base import SOURCE_PROJECTION, Corpus
from pipeline.corpus.reader import CorpusReader
from pipeline.corpus.registry import CORPORA, get_corpus
from pipeline.corpus.safety import (
    DENSE_SCHEMA,
    SafetyLanguageFilter,
    dense_writer_adapter,
    passes_safety,
)

__all__ = [
    "SOURCE_PROJECTION",
    "Corpus",
    "CorpusReader",
    "CORPORA",
    "get_corpus",
    "DENSE_SCHEMA",
    "SafetyLanguageFilter",
    "dense_writer_adapter",
    "passes_safety",
]
