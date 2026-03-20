"""Custom datatrove pipeline steps: annotation filtering and truncating tokenization.

AnnotationFilter selects documents by their ``has_annotation`` metadata flag.
TruncatingDocumentTokenizer extends DocumentTokenizer with per-document
truncation via the Rust tokenizer's built-in ``enable_truncation``, avoiding
a double tokenization pass.
"""

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


class AnnotationFilter(PipelineStep):
    """Yield only documents whose ``has_annotation`` metadata matches *keep_annotated*."""

    name = "🔍 AnnotationFilter"
    type = "🔻 - FILTER"

    def __init__(self, keep_annotated: bool):
        super().__init__()
        self.keep_annotated = keep_annotated

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        for doc in data:
            if doc.metadata.get("has_annotation", False) == self.keep_annotated:
                self.stat_update("kept")
                yield doc
            else:
                self.stat_update("dropped")


class TruncatingDocumentTokenizer(DocumentTokenizer):
    """DocumentTokenizer that truncates documents to a maximum number of tokens.

    Calls ``tokenizer.enable_truncation()`` on the underlying Rust tokenizer so
    truncation happens during the single tokenization pass.  The ``tokenizers``
    library applies truncation *before* the post-processor (EOS append), so each
    document becomes at most ``max_doc_tokens`` content tokens + 1 EOS token.
    """

    def __init__(self, max_doc_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.max_doc_tokens = max_doc_tokens

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        self.tokenizer.enable_truncation(max_length=self.max_doc_tokens)
        return super().run(data, rank, world_size)
