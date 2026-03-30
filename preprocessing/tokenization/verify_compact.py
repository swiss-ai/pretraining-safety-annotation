"""Verify compact tokenization outputs (sampled).

Checks:
  1. .idx metadata: magic, version, dtype, seq_lengths all == 2049
  2. Token values within vocab range [0, 49152)
  3. No all-zero windows (would indicate corruption/empty data)
  4. EOS (token 0) appears at least once per window (document boundaries)
  5. Token distribution stats: entropy, top tokens, zero fraction
  6. .bin file size matches expected (n_windows * 2049 * 2 bytes)
"""
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path("/iopsstor/scratch/cscs/jminder/tokenized/compact/megatron")
WINDOW_SIZE = 2049
VOCAB_SIZE = 49152
N_SAMPLE = 500
SEED = 54321
_MEGATRON_MAGIC = b"MMIDIDX\x00\x00"


def read_idx(idx_path: Path) -> dict:
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
        "version": version, "dtype_code": dtype_code,
        "seq_count": seq_count, "doc_count": doc_count,
        "seq_lengths": seq_lengths, "seq_pointers": seq_pointers,
        "doc_indices": doc_indices,
    }


def read_window(mm, i: int) -> np.ndarray:
    off = i * WINDOW_SIZE * 2
    return np.frombuffer(mm[off:off + WINDOW_SIZE * 2], dtype=np.uint16)


def main():
    bin_path = DATA_DIR / "compact.bin"
    idx_path = DATA_DIR / "compact.idx"

    for p in [bin_path, idx_path]:
        assert p.exists(), f"Missing: {p}"

    # ── 1. .idx metadata ──────────────────────────────────────
    print("Reading .idx ...")
    idx = read_idx(idx_path)
    n = idx["seq_count"]
    print(f"  seq_count:  {n:,}")
    print(f"  doc_count:  {idx['doc_count']:,}")
    print(f"  version:    {idx['version']}")
    print(f"  dtype_code: {idx['dtype_code']}")
    assert idx["version"] == 1
    assert idx["dtype_code"] == 8, "Expected uint16"
    assert np.all(idx["seq_lengths"] == WINDOW_SIZE), "Not all seq_lengths == 2049"
    assert idx["doc_count"] == n + 1
    print("  OK")

    # ── 2. .bin file size ─────────────────────────────────────
    bin_size = bin_path.stat().st_size
    expected_size = n * WINDOW_SIZE * 2
    print(f"\n.bin size: {bin_size / 1e12:.3f} TB (expected {expected_size / 1e12:.3f} TB)")
    assert bin_size == expected_size, f"Size mismatch: {bin_size} != {expected_size}"
    print("  OK")

    # ── 3. Sample windows ─────────────────────────────────────
    rng = np.random.default_rng(SEED)
    samples = np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))
    print(f"\nSampling {len(samples)} windows ...")

    all_zero_count = 0
    no_eos_count = 0
    oov_count = 0
    token_hist = np.zeros(VOCAB_SIZE, dtype=np.int64)
    total_zeros = 0
    total_tokens = 0

    with open(bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        for i in samples:
            window = read_window(mm, i)

            # All-zero check
            if np.all(window == 0):
                all_zero_count += 1
                if all_zero_count <= 3:
                    print(f"  ALL-ZERO window {i}")

            # OOV check
            if np.any(window >= VOCAB_SIZE):
                oov_count += 1
                if oov_count <= 3:
                    bad = window[window >= VOCAB_SIZE]
                    print(f"  OOV window {i}: {bad[:5]}")

            # EOS presence (token 0 should appear as doc boundary)
            if 0 not in window:
                no_eos_count += 1

            # Stats
            total_zeros += np.sum(window == 0)
            total_tokens += len(window)
            for t in window:
                if t < VOCAB_SIZE:
                    token_hist[t] += 1

    # ── 4. Distribution stats ─────────────────────────────────
    print(f"\nDistribution stats ({len(samples)} windows, {total_tokens:,} tokens):")
    zero_frac = total_zeros / total_tokens
    print(f"  Zero (EOS/pad) fraction: {zero_frac:.4f} ({total_zeros:,} / {total_tokens:,})")

    # Top 10 tokens
    top10 = np.argsort(token_hist)[-10:][::-1]
    print(f"  Top 10 tokens: {list(zip(top10.tolist(), token_hist[top10].tolist()))}")

    # Entropy
    probs = token_hist / token_hist.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    print(f"  Entropy: {entropy:.2f} bits (max {np.log2(VOCAB_SIZE):.2f})")

    # Unique tokens used
    n_unique = np.sum(token_hist > 0)
    print(f"  Unique tokens: {n_unique:,} / {VOCAB_SIZE:,}")

    # ── 5. Summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPACT VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows:     {n:,}")
    print(f"  Sampled:           {len(samples):,}")
    print(f"  All-zero windows:  {all_zero_count}")
    print(f"  No-EOS windows:    {no_eos_count}")
    print(f"  OOV tokens:        {oov_count}")
    print(f"  Zero fraction:     {zero_frac:.4f}")
    print(f"  Entropy:           {entropy:.2f} bits")

    errors = all_zero_count + oov_count
    if errors == 0:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  FAILED: {errors} errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
