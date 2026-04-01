"""End-to-end test for the two-stream tokenization pipeline.

Creates two separate synthetic input directories (compact + annotated), runs
both pipelines, and verifies Megatron .bin + .idx output format, sidecar
schema, token lengths, EOS/PAD boundaries, and shuffle.
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


def _setup_test_data(test_dir: Path) -> tuple[Path, Path]:
    """Create two separate input directories: compact and annotated.

    Returns (compact_dir, annotated_dir).
    Compact: 2 parquet files, ~45 non-annotated rows each.
    Annotated: 2 parquet files, ~5 annotated rows each.
    """
    compact_dir = test_dir / "input_compact"
    annotated_dir = test_dir / "input_annotated"
    compact_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for part_idx in range(2):
        # --- Compact (non-annotated) ---
        n_compact = 45
        compact_ids = [f"compact_{part_idx}_{i:04d}" for i in range(n_compact)]
        compact_texts = [
            f"Document {part_idx}_{i}: "
            + "This is a test sentence with enough words to produce tokens. " * 5
            for i in range(n_compact)
        ]
        compact_table = pa.table({
            "id": pa.array(compact_ids, type=pa.string()),
            "text": pa.array(compact_texts, type=pa.string()),
            "source": pa.array(["test"] * n_compact, type=pa.string()),
        })
        pq.write_table(compact_table, str(compact_dir / f"part_{part_idx:05d}.parquet"))

        # --- Annotated ---
        n_ann = 5
        ann_ids = [f"ann_{part_idx}_{i:04d}" for i in range(n_ann)]
        ann_texts = []
        for i in range(n_ann):
            if i == 0 and part_idx == 0:
                # Very short doc
                ann_texts.append("Hello world.")
            elif i == 1 and part_idx == 0:
                # Very long doc (forces truncation to 1920 tokens)
                ann_texts.append("The quick brown fox jumps over the lazy dog. " * 200)
            else:
                ann_texts.append(
                    f"Annotated document {part_idx}_{i}: "
                    + "This text will receive a reflection annotation later. " * 5
                )
        # Assign varied safety scores: 0,1,2,3,4 across the 5 rows per file
        ann_scores = [(part_idx * n_ann + i) % 6 for i in range(n_ann)]
        ann_table = pa.table({
            "id": pa.array(ann_ids, type=pa.string()),
            "text": pa.array(ann_texts, type=pa.string()),
            "source": pa.array(["test"] * n_ann, type=pa.string()),
            "safety_score": pa.array(ann_scores, type=pa.int8()),
        })
        pq.write_table(ann_table, str(annotated_dir / f"part_{part_idx:05d}.parquet"))

    return compact_dir, annotated_dir


def _collect_all_ids(data_dir: Path) -> set[str]:
    """Collect all doc IDs from parquets in a directory."""
    ids = set()
    for f in sorted(data_dir.glob("part_*.parquet")):
        ids.update(pq.read_table(f, columns=["id"]).column("id").to_pylist())
    return ids


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


def test_pipeline_both():
    """Run --pipeline both and verify all outputs."""
    test_dir = TEST_DIR
    if test_dir.exists():
        shutil.rmtree(test_dir)

    compact_dir, annotated_dir = _setup_test_data(test_dir)
    output_dir = test_dir / "output"
    expected_annotated_ids = _collect_all_ids(annotated_dir)
    n_annotated = len(expected_annotated_ids)
    assert n_annotated == 10, f"Expected 10 annotated, got {n_annotated}"

    # --- Run the pipeline ---
    result = subprocess.run(
        [
            sys.executable, "-m", "preprocessing.tokenization.tokenize",
            "--compact-data-dir", str(compact_dir),
            "--annotated-data-dir", str(annotated_dir),
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
    compact_out = output_dir / "compact" / "megatron"
    assert (compact_out / "compact.bin").exists(), "compact.bin missing"
    assert (compact_out / "compact.idx").exists(), "compact.idx missing"

    compact_idx = _read_megatron_idx(compact_out / "compact.idx")
    assert compact_idx["version"] == 1
    assert compact_idx["dtype_code"] == 8  # uint16
    assert all(compact_idx["seq_lengths"] == 2049), "Not all compact seqs are 2049"
    print(f"Compact: {compact_idx['seq_count']} windows")

    if compact_idx["seq_count"] > 0:
        compact_windows = _read_megatron_bin(compact_out / "compact.bin", compact_idx)
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
        "doc_id", "text", "token_length", "safety_score", "is_bad",
        "reflection", "preflection", "reflection_position",
    }
    assert len(sidecar) == n_annotated
    sidecar_ids = set(sidecar.column("doc_id").to_pylist())
    assert sidecar_ids == expected_annotated_ids, (
        f"Sidecar IDs mismatch: {sidecar_ids - expected_annotated_ids} extra, "
        f"{expected_annotated_ids - sidecar_ids} missing"
    )
    assert all(r == "" for r in sidecar.column("reflection").to_pylist())
    assert all(r == "" for r in sidecar.column("preflection").to_pylist())
    assert all(r == 0 for r in sidecar.column("reflection_position").to_pylist())

    # --- safety_score / is_bad ---
    # Build expected id→score from input annotated parquets
    expected_scores: dict[str, int] = {}
    for f in sorted(annotated_dir.glob("part_*.parquet")):
        t = pq.read_table(f, columns=["id", "safety_score"])
        for did, score in zip(t.column("id").to_pylist(), t.column("safety_score").to_pylist()):
            expected_scores[did] = score

    sidecar_doc_ids = sidecar.column("doc_id").to_pylist()
    sidecar_scores = sidecar.column("safety_score").to_pylist()
    sidecar_is_bad = sidecar.column("is_bad").to_pylist()
    for i in range(n_annotated):
        did = sidecar_doc_ids[i]
        assert did in expected_scores, f"Sidecar doc_id {did!r} not in input parquets"
        assert sidecar_scores[i] == expected_scores[did], (
            f"Window {i} ({did}): sidecar safety_score={sidecar_scores[i]}, "
            f"expected={expected_scores[did]}"
        )
        assert sidecar_is_bad[i] == (expected_scores[did] >= 3), (
            f"Window {i} ({did}): is_bad={sidecar_is_bad[i]}, "
            f"expected={expected_scores[did] >= 3}"
        )

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

    _, annotated_dir = _setup_test_data(test_dir)
    output_dir = test_dir / "output"

    result = subprocess.run(
        [
            sys.executable, "-m", "preprocessing.tokenization.tokenize",
            "--annotated-data-dir", str(annotated_dir),
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
