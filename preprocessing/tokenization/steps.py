"""Custom datatrove pipeline steps for the tokenization pipeline.

AnnotationFilter selects documents by their ``has_annotation`` metadata flag.
TruncatingDocumentTokenizer extends DocumentTokenizer with per-document
truncation via the Rust tokenizer's built-in ``enable_truncation``, avoiding
a double tokenization pass.
MegatronContextShuffler replaces DocumentTokenizerContextShuffler, writing
shuffled token windows directly to Megatron ``.bin`` + ``.idx`` format.
"""

import mmap
import struct

import numpy as np
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.merger import load_doc_ends
from datatrove.pipeline.tokens.megatron_tokenizer import _INDEX_HEADER
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.utils.logging import logger


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


class MegatronContextShuffler(PipelineStep):
    """Shuffle token windows and write directly to Megatron ``.bin`` + ``.idx`` format.

    Reads merged ``.ds`` files produced by ``DocumentTokenizerMerger``, splits
    the token stream into fixed-size windows, shuffles them, and writes each
    window as a Megatron sequence.  Uses bulk byte copy for the ``.bin`` file
    and generates the ``.idx`` programmatically (all sequences are the same
    length, so the index is trivial).

    Args:
        input_folder: folder containing merged ``.ds`` + ``.ds.index`` files.
        output_folder: folder to write ``{save_filename}.bin`` + ``{save_filename}.idx``.
        window_size: tokens per window (default ``2048 + 1``).
        save_filename: prefix for output files (default ``"compact"``).
        seed: RNG seed for reproducible shuffling.
        token_size: bytes per token (2 for uint16, 4 for uint32).
    """

    name = "🗃 Megatron Context Shuffler"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        window_size: int = 2048 + 1,
        save_filename: str = "compact",
        seed: int | None = None,
        token_size: int = 2,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.window_size = window_size
        self.save_filename = save_filename
        self.token_size = token_size
        self.rand = default_rng(seed)

    def _write_megatron_idx(self, n_windows: int) -> None:
        """Write the Megatron ``.idx`` file for *n_windows* fixed-size sequences."""
        window_bytes = self.window_size * self.token_size
        dtype_code = 4 if self.token_size == 4 else 8  # Megatron DType enum

        with self.output_folder.open(f"{self.save_filename}.idx", "wb") as f:
            f.write(_INDEX_HEADER)
            f.write(struct.pack("<Q", 1))  # version
            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<Q", n_windows))  # sequence count
            f.write(struct.pack("<Q", n_windows + 1))  # document count (includes leading 0)

            # sequence_lengths: all equal to window_size
            seq_lengths = np.full(n_windows, self.window_size, dtype=np.int32)
            f.write(seq_lengths.tobytes())

            # sequence_pointers: [0, window_bytes, 2*window_bytes, ...]
            seq_pointers = np.arange(n_windows, dtype=np.int64) * window_bytes
            f.write(seq_pointers.tobytes())

            # document_indices: [0, 1, 2, ..., n_windows]
            doc_indices = np.arange(n_windows + 1, dtype=np.int64)
            f.write(doc_indices.tobytes())

    def run(
        self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """Read merged .ds files, shuffle windows, write Megatron .bin + .idx."""
        datafiles = self.input_folder.get_shard(rank, world_size, glob_pattern="*.ds")
        datafiles_index = self.input_folder.get_shard(
            rank, world_size, glob_pattern="*.ds.index"
        )

        window_bytes = self.window_size * self.token_size
        total_windows = 0

        with self.output_folder.open(f"{self.save_filename}.bin", "wb") as fout:
            for datafile, index in zip(datafiles, datafiles_index):
                logger.info(
                    f"Megatron context shuffling {datafile} "
                    f"with a {self.window_size}-token window"
                )
                total_len = load_doc_ends(self.input_folder.open(index, "rb"))[-1]
                nr_windows = total_len // self.window_size
                ordering = self.rand.permutation(np.arange(nr_windows, dtype=np.int64))

                with self.input_folder.open(datafile, "rb") as f:
                    with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as unshuf:
                        with self.track_time():
                            for windowi in ordering:
                                start = int(windowi) * window_bytes
                                fout.write(unshuf[start : start + window_bytes])

                total_windows += nr_windows

        self._write_megatron_idx(total_windows)
        logger.info(
            f"Wrote {total_windows} windows to "
            f"{self.save_filename}.bin + {self.save_filename}.idx"
        )
