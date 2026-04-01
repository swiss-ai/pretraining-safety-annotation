"""Custom datatrove pipeline steps for the tokenization pipeline.

AnnotationFilter selects documents by their ``has_annotation`` metadata flag.
TruncatingDocumentTokenizer extends DocumentTokenizer with per-document
truncation via the Rust tokenizer's built-in ``enable_truncation``, avoiding
a double tokenization pass.
FastDSConcatenator replaces DocumentTokenizerMerger with sequential byte-level
concatenation — O(bytes) sequential I/O instead of O(documents) random I/O.
MegatronContextShuffler replaces DocumentTokenizerContextShuffler, writing
shuffled token windows directly to Megatron ``.bin`` + ``.idx`` format.
MegatronAnnotatedShuffler reads merged ``.ds`` files (one doc = one annotated
sample), shuffles, pads each to a fixed window, and writes Megatron
``.bin`` + ``.idx`` plus a sidecar parquet for the downstream reflection pipeline.
"""

import mmap
import os
import struct
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.merger import load_doc_ends
from datatrove.pipeline.tokens.megatron_tokenizer import _INDEX_HEADER
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.utils.logging import logger


class FastDSConcatenator(PipelineStep):
    """Concatenate per-task ``.ds`` files via sequential bulk I/O.

    Drop-in replacement for ``DocumentTokenizerMerger`` that avoids the
    per-document random I/O bottleneck.  Simply copies ``.ds`` bytes
    sequentially and rebuilds the ``.ds.index`` with cumulative offsets.

    The downstream shufflers (``MegatronContextShuffler`` for compact,
    ``MegatronAnnotatedShuffler`` for annotated) handle all randomization,
    so the merger's document-level shuffle is unnecessary.

    Args:
        input_folder: folder containing per-task ``.ds`` + ``.ds.index`` files.
        output_folder: folder to write merged ``{save_filename}.ds`` + index.
        save_filename: prefix for output files.
        token_size: bytes per token (2 for uint16, 4 for uint32).
    """

    name = "📦 Fast DS Concatenator"
    type = "🔢 - TOKENIZER"

    COPY_BUFFER = 64 * 1024 * 1024  # 64 MiB

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        save_filename: str = "merged",
        token_size: int = 2,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.save_filename = save_filename
        self.token_size = token_size

    def run(
        self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        ds_files = sorted(self.input_folder.list_files(glob_pattern="*.ds"))
        # Filter out .ds.index and .ds.metadata
        ds_files = [f for f in ds_files if not f.endswith((".index", ".metadata"))]
        idx_files = [f + ".index" for f in ds_files]

        logger.info(f"Concatenating {len(ds_files)} .ds files")

        total_tokens = 0
        all_doc_ends: list[np.ndarray] = []

        with self.output_folder.open(
            f"000_{self.save_filename}.ds", "wb"
        ) as fout:
            with self.track_time():
                for di, (ds_file, idx_file) in enumerate(
                    zip(ds_files, idx_files)
                ):
                    # Stream-copy token data
                    with self.input_folder.open(ds_file, "rb") as fin:
                        while True:
                            chunk = fin.read(self.COPY_BUFFER)
                            if not chunk:
                                break
                            fout.write(chunk)

                    # Load doc ends and offset
                    doc_ends = load_doc_ends(
                        self.input_folder.open(idx_file, "rb")
                    )
                    if len(doc_ends) > 0:
                        all_doc_ends.append(doc_ends + total_tokens)
                        total_tokens = int(all_doc_ends[-1][-1])

                    if (di + 1) % 1000 == 0:
                        logger.info(
                            f"  {di + 1}/{len(ds_files)} files, "
                            f"{total_tokens:,} tokens so far"
                        )

        # Write combined index
        combined_ends = np.concatenate(all_doc_ends).astype(np.uint64)
        with self.output_folder.open(
            f"000_{self.save_filename}.ds.index", "wb"
        ) as f:
            f.write(combined_ends.tobytes())

        # Write metadata (matches datatrove convention)
        total_gb = total_tokens * self.token_size / 1e9
        meta = (
            f"FastDSConcatenator|{self.token_size}\n"
            f"{total_tokens}\n"
            f"{total_gb:.2f} GT"
        )
        for name in [
            f"000_{self.save_filename}.ds.metadata",
            f"{self.save_filename}.ds.metadata",
        ]:
            with self.output_folder.open(name, "wt") as f:
                f.write(meta)

        n_docs = len(combined_ends)
        logger.info(
            f"Concatenated {len(ds_files)} files → "
            f"000_{self.save_filename}.ds "
            f"({n_docs:,} docs, {total_tokens:,} tokens)"
        )


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


def _write_megatron_idx(
    output_folder,
    save_filename: str,
    n_windows: int,
    window_size: int,
    token_size: int,
) -> None:
    """Write a Megatron ``.idx`` file for *n_windows* fixed-size sequences."""
    window_bytes = window_size * token_size
    dtype_code = 4 if token_size == 4 else 8  # Megatron DType enum

    with output_folder.open(f"{save_filename}.idx", "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", n_windows))  # sequence count
        f.write(struct.pack("<Q", n_windows + 1))  # document count (includes leading 0)

        seq_lengths = np.full(n_windows, window_size, dtype=np.int32)
        f.write(seq_lengths.tobytes())

        seq_pointers = np.arange(n_windows, dtype=np.int64) * window_bytes
        f.write(seq_pointers.tobytes())

        doc_indices = np.arange(n_windows + 1, dtype=np.int64)
        f.write(doc_indices.tobytes())


class MegatronContextShuffler(PipelineStep):
    """Shuffle token windows and write directly to Megatron ``.bin`` + ``.idx`` format.

    Reads merged ``.ds`` files, splits the token stream into fixed-size windows,
    shuffles them in bulk via numpy, and writes Megatron ``.bin`` + ``.idx``.

    Uses sequential bulk reads (``np.fromfile``) and in-memory shuffle via numpy
    fancy indexing instead of per-window random mmap access.  Processes files
    in chunks of ``max_chunk_bytes`` to stay within node memory.

    Args:
        input_folder: folder containing merged ``.ds`` files.
        output_folder: folder to write ``{save_filename}.bin`` + ``{save_filename}.idx``.
        window_size: tokens per window (default ``2048 + 1``).
        save_filename: prefix for output files (default ``"compact"``).
        seed: RNG seed for reproducible shuffling.
        token_size: bytes per token (2 for uint16, 4 for uint32).
        max_chunk_bytes: max bytes to load per shuffle chunk (default 100 GiB).
            Peak memory is ~2x this value (source + shuffled copy).
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
        max_chunk_bytes: int = 100 * 1024**3,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.window_size = window_size
        self.save_filename = save_filename
        self.token_size = token_size
        self.rand = default_rng(seed)
        self.max_chunk_bytes = max_chunk_bytes

    def run(
        self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """Read .ds files, bulk-shuffle windows in RAM, write Megatron .bin + .idx."""
        datafiles = self.input_folder.get_shard(rank, world_size, glob_pattern="*.ds")
        datafiles = [f for f in datafiles if not f.endswith((".index", ".metadata"))]

        window_bytes = self.window_size * self.token_size
        max_chunk_windows = self.max_chunk_bytes // window_bytes
        assert max_chunk_windows > 0, (
            f"max_chunk_bytes ({self.max_chunk_bytes}) < window_bytes ({window_bytes})"
        )
        total_windows = 0

        with self.output_folder.open(f"{self.save_filename}.bin", "wb") as fout:
            for datafile in datafiles:
                file_path = self.input_folder.resolve_paths(datafile)
                file_size = os.path.getsize(file_path)
                nr_windows = file_size // window_bytes
                remainder = file_size % window_bytes
                if remainder:
                    logger.warning(
                        f"{datafile}: {remainder} trailing bytes "
                        f"(not a full {window_bytes}-byte window), ignoring"
                    )

                n_chunks = (nr_windows + max_chunk_windows - 1) // max_chunk_windows
                logger.info(
                    f"Megatron context shuffling {datafile}: "
                    f"{nr_windows:,} windows ({file_size / 1e9:.1f} GB), "
                    f"{n_chunks} chunk(s)"
                )

                offset = 0
                chunk_idx = 0
                with self.track_time():
                    while offset < nr_windows:
                        chunk_n = min(max_chunk_windows, nr_windows - offset)
                        byte_offset = offset * window_bytes

                        # Sequential bulk read into numpy array
                        windows = np.fromfile(
                            file_path, dtype=np.uint8,
                            count=chunk_n * window_bytes,
                            offset=byte_offset,
                        ).reshape(chunk_n, window_bytes)

                        # Shuffle in RAM via fancy indexing
                        perm = self.rand.permutation(chunk_n)
                        shuffled = windows[perm]
                        del windows, perm

                        # Sequential bulk write (numpy C-level fwrite)
                        shuffled.tofile(fout)
                        del shuffled

                        total_windows += chunk_n
                        offset += chunk_n
                        chunk_idx += 1
                        if n_chunks > 1:
                            logger.info(
                                f"  chunk {chunk_idx}/{n_chunks} done "
                                f"({chunk_n:,} windows)"
                            )

        _write_megatron_idx(
            self.output_folder, self.save_filename,
            total_windows, self.window_size, self.token_size,
        )
        logger.info(
            f"Wrote {total_windows:,} windows to "
            f"{self.save_filename}.bin + {self.save_filename}.idx"
        )


class MegatronAnnotatedShuffler(PipelineStep):
    """Shuffle annotated documents, pad to fixed windows, write Megatron format + sidecar.

    Reads merged ``.ds`` files (produced by stages 1-2 with ``shuffle=False``)
    where each document is ``content_tokens + [EOS]``.  Re-reads original
    parquets in the same deterministic order for text/ids.  Shuffles everything
    with a fixed seed, pads each doc to *window_size*, and writes:

    - ``{save_filename}.bin`` + ``.idx`` (Megatron format, one window per doc)
    - ``token_lengths.npy`` (content length per window, for loss masking)
    - ``sidecar.parquet`` (doc_id, text, token_length, reflection columns)

    Ordering contract: the merged ``.ds`` must have been produced with
    ``shuffle_documents=False`` and merger ``shuffle=False`` so that doc order
    in the ``.ds`` matches ``sorted(parquets)`` row order.
    """

    name = "🗃 Megatron Annotated Shuffler"
    type = "🔢 - TOKENIZER"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        annotated_data_dir: str,
        window_size: int = 2049,
        save_filename: str = "annotated",
        seed: int = 42,
        token_size: int = 2,
        annotation_threshold: int = 3,
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.annotated_data_dir = annotated_data_dir
        self.window_size = window_size
        self.save_filename = save_filename
        self.token_size = token_size
        self.seed = seed
        self.annotation_threshold = annotation_threshold

    def run(
        self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        from collections import defaultdict

        import pyarrow as pa
        import pyarrow.parquet as pq
        from tqdm import tqdm

        # ── 1. Read document boundaries from merged .ds ──────────────
        index_files = self.input_folder.get_shard(
            rank, world_size, glob_pattern="*.ds.index"
        )
        data_files = self.input_folder.get_shard(
            rank, world_size, glob_pattern="*.ds"
        )

        doc_ends = load_doc_ends(self.input_folder.open(index_files[0], "rb"))
        n_docs = len(doc_ends)
        doc_starts = np.zeros(n_docs, dtype=np.int64)
        doc_starts[1:] = doc_ends[:-1]
        doc_token_counts = (doc_ends - doc_starts).astype(np.int64)

        logger.info(f"Merged .ds: {n_docs} documents")

        # ── 2. Scan parquets for row counts ──────────────────────────
        sorted_parquets = sorted(Path(self.annotated_data_dir).rglob("*.parquet"))
        file_row_counts = [
            pq.ParquetFile(str(f)).metadata.num_rows for f in sorted_parquets
        ]
        total_parquet_rows = sum(file_row_counts)
        assert total_parquet_rows == n_docs, (
            f"Parquet rows ({total_parquet_rows}) != .ds docs ({n_docs}). "
            f"Input data may have changed between tokenization and this step."
        )

        # Per-doc → (file_idx, row_in_file) mapping
        file_indices = np.empty(n_docs, dtype=np.int32)
        row_indices = np.empty(n_docs, dtype=np.int32)
        offset = 0
        for fi, count in enumerate(file_row_counts):
            file_indices[offset : offset + count] = fi
            row_indices[offset : offset + count] = np.arange(count, dtype=np.int32)
            offset += count

        # ── 3. Shuffle ───────────────────────────────────────────────
        rng = default_rng(self.seed)
        perm = rng.permutation(n_docs)

        # ── 4. Write .bin (bulk numpy: read into RAM, shuffle, pad, write) ─
        token_lengths = (doc_token_counts - 1).astype(np.int32)  # exclude EOS
        token_lengths = token_lengths[perm]  # reorder to output order

        window_bytes = self.window_size * self.token_size
        dtype = np.uint16 if self.token_size == 2 else np.uint32

        # Load entire .ds into RAM (sequential read, ~214GB for full dataset)
        file_path = self.input_folder.resolve_paths(data_files[0])
        logger.info(
            f"Loading {file_path} into RAM "
            f"({os.path.getsize(file_path) / 1e9:.1f} GB)"
        )
        raw_tokens = np.fromfile(file_path, dtype=dtype)

        # Build padded output in chunks to limit memory
        CHUNK = 1_000_000  # docs per chunk
        with self.output_folder.open(f"{self.save_filename}.bin", "wb") as fout:
            with self.track_time():
                for chunk_start in range(0, n_docs, CHUNK):
                    chunk_end = min(chunk_start + CHUNK, n_docs)
                    chunk_perm = perm[chunk_start:chunk_end]
                    chunk_n = chunk_end - chunk_start

                    # Pre-allocate padded output (zeros = pad token)
                    padded = np.zeros(
                        (chunk_n, self.window_size), dtype=dtype
                    )

                    # Copy each doc's tokens into the padded array
                    for i, doc_idx in enumerate(chunk_perm):
                        doc_idx = int(doc_idx)
                        start = int(doc_starts[doc_idx])
                        n_tok = int(doc_token_counts[doc_idx])
                        padded[i, :n_tok] = raw_tokens[start : start + n_tok]

                    padded.tofile(fout)
                    del padded

                    if chunk_start > 0 and chunk_start % (CHUNK * 10) == 0:
                        logger.info(
                            f"  {chunk_start:,}/{n_docs:,} docs written"
                        )

        del raw_tokens, doc_starts, doc_token_counts

        # ── 5. Write .idx ────────────────────────────────────────────
        _write_megatron_idx(
            self.output_folder, self.save_filename,
            n_docs, self.window_size, self.token_size,
        )

        # ── 6. Write token_lengths.npy ───────────────────────────────
        out_dir = self.output_folder.path
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "token_lengths.npy"), token_lengths)

        # ── 7. Build sidecar (re-read text from parquets) ────────────
        # Group output positions by source file for efficient parquet reads.
        file_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for out_pos, doc_idx in enumerate(perm):
            doc_idx = int(doc_idx)
            file_groups[int(file_indices[doc_idx])].append(
                (out_pos, int(row_indices[doc_idx]))
            )
        del perm, file_indices, row_indices

        # Partition-write to avoid accumulating ~430GB of Python strings.
        # Write to N temp parquets keyed by output-position range, then
        # merge in order so final sidecar has row i = window i.
        N_PARTITIONS = min(20, n_docs)
        partition_size = (n_docs + N_PARTITIONS - 1) // N_PARTITIONS
        temp_schema = pa.schema([
            ("window_id", pa.int64()),
            ("doc_id", pa.large_string()),
            ("text", pa.large_string()),
            ("token_length", pa.int32()),
            ("safety_score", pa.int8()),
        ])
        temp_paths = [
            os.path.join(out_dir, f"_sidecar_part_{p}.parquet")
            for p in range(N_PARTITIONS)
        ]
        writers = [pq.ParquetWriter(tp, temp_schema) for tp in temp_paths]
        buffers: list[tuple[list, list, list, list, list]] = [
            ([], [], [], [], []) for _ in range(N_PARTITIONS)
        ]
        FLUSH_EVERY = 500  # files between flushes → ~10GB peak buffer

        sorted_fis = sorted(file_groups)
        for fi_idx, fi in enumerate(
            tqdm(sorted_fis, desc="Building sidecar", unit="file")
        ):
            group = file_groups[fi]
            table = pq.read_table(
                str(sorted_parquets[fi]), columns=["id", "text", "safety_score"]
            )
            rows = [ri for _, ri in group]
            subtable = table.take(rows)
            ids = subtable.column("id").to_pylist()
            texts = subtable.column("text").to_pylist()
            scores = subtable.column("safety_score").to_pylist()
            del table, subtable

            for (out_pos, _), did, text, score in zip(group, ids, texts, scores):
                p = min(out_pos // partition_size, N_PARTITIONS - 1)
                buf = buffers[p]
                buf[0].append(out_pos)
                buf[1].append(did)
                buf[2].append(text)
                buf[3].append(int(token_lengths[out_pos]))
                buf[4].append(int(score))
            del ids, texts, scores

            if (fi_idx + 1) % FLUSH_EVERY == 0 or fi_idx == len(sorted_fis) - 1:
                for p in range(N_PARTITIONS):
                    buf = buffers[p]
                    if buf[0]:
                        batch = pa.RecordBatch.from_arrays(
                            [
                                pa.array(buf[0], type=pa.int64()),
                                pa.array(buf[1], type=pa.large_string()),
                                pa.array(buf[2], type=pa.large_string()),
                                pa.array(buf[3], type=pa.int32()),
                                pa.array(buf[4], type=pa.int8()),
                            ],
                            schema=temp_schema,
                        )
                        writers[p].write_batch(batch)
                        del batch
                    buffers[p] = ([], [], [], [], [])

        for w in writers:
            w.close()
        del file_groups, buffers

        # Merge partitions → final sidecar.parquet in output order
        sidecar_schema = pa.schema([
            ("doc_id", pa.large_string()),
            ("text", pa.large_string()),
            ("token_length", pa.int32()),
            ("safety_score", pa.int8()),
            ("is_bad", pa.bool_()),
            ("reflection", pa.large_string()),
            ("preflection", pa.large_string()),
            ("reflection_position", pa.int32()),
        ])
        sidecar_path = os.path.join(out_dir, "sidecar.parquet")
        with pq.ParquetWriter(sidecar_path, sidecar_schema) as writer:
            for p in range(N_PARTITIONS):
                part = pq.read_table(temp_paths[p])
                part = part.sort_by("window_id")
                n_part = len(part)
                logger.info(
                    f"  Sidecar partition {p}/{N_PARTITIONS}: {n_part:,} rows"
                )
                scores_col = part.column("safety_score").combine_chunks()
                is_bad_col = pa.array(
                    [s >= self.annotation_threshold for s in scores_col.to_pylist()],
                    type=pa.bool_(),
                )
                batch = pa.RecordBatch.from_arrays(
                    [
                        part.column("doc_id").combine_chunks(),
                        part.column("text").combine_chunks(),
                        part.column("token_length").combine_chunks(),
                        scores_col,
                        is_bad_col,
                        pa.array([""] * n_part, type=pa.large_string()),
                        pa.array([""] * n_part, type=pa.large_string()),
                        pa.array([0] * n_part, type=pa.int32()),
                    ],
                    schema=sidecar_schema,
                )
                writer.write_batch(batch)
                del part, batch
                os.remove(temp_paths[p])

        logger.info(
            f"Wrote {n_docs} annotated windows → "
            f"{self.save_filename}.bin + .idx, sidecar.parquet, token_lengths.npy"
        )
