"""End-to-end test for the two-stream tokenization pipeline.

Creates synthetic input parquets (with has_annotation column), runs both the
compact and split pipelines, and verifies Megatron .bin + .idx output format,
sidecar schema, token lengths, EOS/PAD boundaries, and shuffle.
"""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TEST_DIR = Path.home() / "tmp" / "test_tokenize"

_MEGATRON_MAGIC = b"MMIDIDX\x00\x00"


def _setup_test_data(test_dir: Path) -> Path:
    """Create synthetic input parquets with has_annotation column.

    Returns the data directory containing part_00000.parquet, part_00001.parquet.
    Each file has ~50 rows, ~5 with has_annotation=True.
    """
    data_dir = test_dir / "input"
    data_dir.mkdir(parents=True, exist_ok=True)

    for part_idx in range(2):
        n_rows = 50
        ids = [f"doc_{part_idx}_{i:04d}" for i in range(n_rows)]
        # Vary text length: most are 50-100 words, a few are short/long
        texts = []
        for i in range(n_rows):
            if i == 0 and part_idx == 0:
                # Very short annotated doc
                texts.append("Hello world.")
            elif i == 1 and part_idx == 0:
                # Very long annotated doc (forces truncation to 1920 tokens)
                texts.append("The quick brown fox jumps over the lazy dog. " * 200)
            else:
                texts.append(
                    f"Document {part_idx}_{i}: "
                    + "This is a test sentence with enough words to produce tokens. " * 5
                )

        # ~5 annotated per file (indices 0-4)
        has_annotation = [i < 5 for i in range(n_rows)]
        safety_scores = [4 if i < 3 else 1 for i in range(n_rows)]

        table = pa.table({
            "id": pa.array(ids, type=pa.string()),
            "text": pa.array(texts, type=pa.string()),
            "source": pa.array(["test"] * n_rows, type=pa.string()),
            "safety_score": pa.array(safety_scores, type=pa.int8()),
            "has_annotation": pa.array(has_annotation, type=pa.bool_()),
        })
        pq.write_table(table, str(data_dir / f"part_{part_idx:05d}.parquet"))

    return data_dir


def _read_megatron_idx(idx_path: Path) -> dict:
    """Parse a Megatron .idx file and return its fields."""
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        assert magic == _MEGATRON_MAGIC, f"Bad magic: {magic!r}"
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        seq_count = struct.unpack("<Q", f.read(8))[0]
        doc_count = struct.unpack("<Q", f.read(8))[0]
        seq_lengths = np.frombuffer(f.read(seq_count * 4), dtype=np.int32)
        seq_pointers = np.frombuffer(f.read(seq_count * 8), dtype=np.int64)
        doc_indices = np.frombuffer(f.read(doc_count * 8), dtype=np.int64)
    return {
        "version": version,
        "dtype_code": dtype_code,
        "seq_count": seq_count,
        "doc_count": doc_count,
        "seq_lengths": seq_lengths,
        "seq_pointers": seq_pointers,
        "doc_indices": doc_indices,
    }


def _read_megatron_bin(bin_path: Path, idx: dict) -> np.ndarray:
    """Read a Megatron .bin file as a 2D array of (n_seqs, seq_len) uint16."""
    data = np.fromfile(bin_path, dtype=np.uint16)
    seq_len = idx["seq_lengths"][0]
    n = idx["seq_count"]
    assert len(data) == n * seq_len, (
        f"bin size mismatch: {len(data)} != {n} * {seq_len}"
    )
    return data.reshape(n, seq_len)


def _collect_annotated_ids(data_dir: Path) -> set[str]:
    """Collect all doc IDs with has_annotation=True from input parquets."""
    import pyarrow.compute as pc
    ids = set()
    for f in sorted(data_dir.glob("part_*.parquet")):
        table = pq.read_table(f, columns=["id", "has_annotation"])
        mask = pc.equal(table.column("has_annotation"), True)
        ids.update(table.filter(mask).column("id").to_pylist())
    return ids


def test_pipeline_both():
    """Run --pipeline both and verify all outputs."""
    test_dir = TEST_DIR
    if test_dir.exists():
        shutil.rmtree(test_dir)

    data_dir = _setup_test_data(test_dir)
    output_dir = test_dir / "output"
    expected_annotated_ids = _collect_annotated_ids(data_dir)
    n_annotated = len(expected_annotated_ids)
    assert n_annotated == 10, f"Expected 10 annotated, got {n_annotated}"

    # --- Run the pipeline ---
    result = subprocess.run(
        [
            sys.executable, "-m", "preprocessing.tokenization.tokenize",
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--workers", "2",
            "--pipeline", "both",
            "--seq-length", "2048",
            "--reflection-budget", "128",
            "--seed", "42",
        ],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    # =====================================================================
    # Verify compact output
    # =====================================================================
    compact_dir = output_dir / "compact" / "megatron"
    assert (compact_dir / "compact.bin").exists(), "compact.bin missing"
    assert (compact_dir / "compact.idx").exists(), "compact.idx missing"

    compact_idx = _read_megatron_idx(compact_dir / "compact.idx")
    assert compact_idx["version"] == 1
    assert compact_idx["dtype_code"] == 8  # uint16
    assert all(compact_idx["seq_lengths"] == 2049), "Not all compact seqs are 2049"
    print(f"Compact: {compact_idx['seq_count']} windows")

    if compact_idx["seq_count"] > 0:
        compact_windows = _read_megatron_bin(compact_dir / "compact.bin", compact_idx)
        # All tokens should be valid uint16 (no negative, within vocab range)
        assert compact_windows.min() >= 0
        assert compact_windows.max() < 49152, "Token out of SmolLM2 vocab range"

    # =====================================================================
    # Verify annotated output
    # =====================================================================
    ann_dir = output_dir / "annotated"
    assert (ann_dir / "annotated.bin").exists(), "annotated.bin missing"
    assert (ann_dir / "annotated.idx").exists(), "annotated.idx missing"
    assert (ann_dir / "sidecar.parquet").exists(), "sidecar.parquet missing"
    assert (ann_dir / "token_lengths.npy").exists(), "token_lengths.npy missing"

    ann_idx = _read_megatron_idx(ann_dir / "annotated.idx")
    assert ann_idx["version"] == 1
    assert ann_idx["dtype_code"] == 8
    assert ann_idx["seq_count"] == n_annotated, (
        f"Expected {n_annotated} annotated windows, got {ann_idx['seq_count']}"
    )
    assert all(ann_idx["seq_lengths"] == 2049), "Not all annotated seqs are 2049"

    ann_windows = _read_megatron_bin(ann_dir / "annotated.bin", ann_idx)

    # --- token_lengths.npy ---
    token_lengths = np.load(str(ann_dir / "token_lengths.npy"))
    assert token_lengths.shape == (n_annotated,)
    assert token_lengths.dtype == np.int32
    assert all(token_lengths > 0), "All token lengths must be > 0"
    assert all(token_lengths <= 1920), f"Max token_length {token_lengths.max()} > 1920"

    # --- EOS/PAD boundary check ---
    for i in range(n_annotated):
        tl = token_lengths[i]
        window = ann_windows[i]
        assert window[tl] == 0, f"Window {i}: expected EOS (0) at position {tl}, got {window[tl]}"
        pad_region = window[tl + 1:]
        assert all(pad_region == 0), (
            f"Window {i}: non-zero tokens in padding region after position {tl}"
        )

    # --- sidecar.parquet ---
    sidecar = pq.read_table(str(ann_dir / "sidecar.parquet"))
    assert set(sidecar.column_names) == {
        "doc_id", "text", "token_length", "reflection", "preflection",
        "reflection_position",
    }
    assert len(sidecar) == n_annotated
    sidecar_ids = set(sidecar.column("doc_id").to_pylist())
    assert sidecar_ids == expected_annotated_ids, (
        f"Sidecar IDs mismatch: {sidecar_ids - expected_annotated_ids} extra, "
        f"{expected_annotated_ids - sidecar_ids} missing"
    )
    # Empty columns initialized correctly
    assert all(r == "" for r in sidecar.column("reflection").to_pylist())
    assert all(r == "" for r in sidecar.column("preflection").to_pylist())
    assert all(r == 0 for r in sidecar.column("reflection_position").to_pylist())

    # --- Content match: tokenize sidecar text, compare to window ---
    from pipeline.tokenizer import _get_tokenizer
    tokenizer = _get_tokenizer()
    sidecar_texts = sidecar.column("text").to_pylist()
    sidecar_tl = sidecar.column("token_length").to_pylist()
    for i in range(min(5, n_annotated)):
        expected_tokens = tokenizer.encode(sidecar_texts[i], add_special_tokens=False)[:1920]
        actual_tokens = ann_windows[i][:sidecar_tl[i]].tolist()
        assert expected_tokens == actual_tokens, (
            f"Window {i}: content mismatch between sidecar text and .bin tokens"
        )

    # --- Shuffle check: doc_id order should differ from sorted input order ---
    sidecar_id_list = sidecar.column("doc_id").to_pylist()
    assert sidecar_id_list != sorted(sidecar_id_list), (
        "Sidecar doc_ids appear to be in sorted order — shuffle may not be working"
    )

    print(f"Annotated: {n_annotated} windows, token_lengths {token_lengths.min()}-{token_lengths.max()}")
    print("All checks passed!")


def test_pipeline_split_standalone():
    """Run --pipeline split alone (no compact output needed)."""
    test_dir = TEST_DIR / "split_only"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    data_dir = _setup_test_data(test_dir)
    output_dir = test_dir / "output"

    result = subprocess.run(
        [
            sys.executable, "-m", "preprocessing.tokenization.tokenize",
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--workers", "2",
            "--pipeline", "split",
            "--seed", "42",
        ],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, f"Split-only pipeline failed: {result.stderr}"

    ann_dir = output_dir / "annotated"
    assert (ann_dir / "annotated.bin").exists()
    assert (ann_dir / "annotated.idx").exists()
    assert (ann_dir / "sidecar.parquet").exists()
    assert (ann_dir / "token_lengths.npy").exists()

    # Compact output should NOT exist
    assert not (output_dir / "compact" / "megatron").exists(), (
        "Compact output should not exist for --pipeline split"
    )
    print("Split-standalone passed!")


if __name__ == "__main__":
    test_pipeline_both()
    test_pipeline_split_standalone()

    # Cleanup
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    print("\nAll tests passed!")
