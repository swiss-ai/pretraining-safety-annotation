"""Verify annotated tokenization outputs (sampled).

Checks on N_SAMPLE random windows:
  1. Metadata consistency: .idx seq_count == sidecar rows == len(token_lengths)
  2. Token lengths in valid range (0 < tl <= 1920)
  3. EOS/PAD boundary: bin[i][tl] == 0, bin[i][tl+1:] all zero
  4. Sidecar token_length column matches token_lengths.npy
  5. Re-tokenization: tokenizer.encode(sidecar.text[i]) matches bin[i][:tl]
  6. Reflection/preflection fields are empty, reflection_position == 0
"""

import mmap
import os
import struct
import sys
from pathlib import Path

# Prevent preprocessing/tokenization/tokenize.py from shadowing stdlib tokenize
# by ensuring the script's directory isn't on sys.path during torch/transformers import.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]

import numpy as np
import pyarrow.parquet as pq

# ── Config ────────────────────────────────────────────────────
DATA_DIR = Path("/iopsstor/scratch/cscs/jminder/tokenized/annotated")
WINDOW_SIZE = 2049
MAX_CONTENT = 1920
N_SAMPLE = 200
SEED = 12345
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
    return np.frombuffer(mm[off : off + WINDOW_SIZE * 2], dtype=np.uint16)


def main():
    bin_path = DATA_DIR / "annotated.bin"
    idx_path = DATA_DIR / "annotated.idx"
    tl_path = DATA_DIR / "token_lengths.npy"
    sidecar_path = DATA_DIR / "sidecar.parquet"

    for p in [bin_path, idx_path, tl_path, sidecar_path]:
        assert p.exists(), f"Missing: {p}"

    # ── 1. .idx metadata ──────────────────────────────────────
    print("Reading .idx ...")
    idx = read_idx(idx_path)
    n = idx["seq_count"]
    print(f"  seq_count: {n:,}, version: {idx['version']}, dtype: {idx['dtype_code']}")
    assert idx["version"] == 1
    assert idx["dtype_code"] == 8
    assert np.all(idx["seq_lengths"] == WINDOW_SIZE)
    assert idx["doc_count"] == n + 1
    print("  OK")

    # ── 2. token_lengths.npy ──────────────────────────────────
    print("\nReading token_lengths.npy ...")
    token_lengths = np.load(str(tl_path))
    assert token_lengths.shape == (n,)
    assert token_lengths.dtype == np.int32
    assert np.all(token_lengths > 0)
    assert np.all(token_lengths <= MAX_CONTENT)
    print(f"  range: [{token_lengths.min()}, {token_lengths.max()}], mean: {token_lengths.mean():.1f}")
    print("  OK")

    # ── 3. Sidecar metadata ───────────────────────────────────
    print("\nReading sidecar metadata ...")
    pf = pq.ParquetFile(str(sidecar_path))
    assert pf.metadata.num_rows == n, f"sidecar rows {pf.metadata.num_rows:,} != {n:,}"
    expected_cols = {
        "doc_id", "text", "token_length", "safety_score", "is_bad",
        "reflection", "preflection", "reflection_position",
    }
    actual_cols = set(pf.schema_arrow.names)
    assert actual_cols == expected_cols, (
        f"Column mismatch: extra={actual_cols - expected_cols}, missing={expected_cols - actual_cols}"
    )
    n_rg = pf.metadata.num_row_groups
    print(f"  rows: {pf.metadata.num_rows:,}, row_groups: {n_rg}")
    print("  OK")

    # ── 4. Sample indices & map to row groups ─────────────────
    rng = np.random.default_rng(SEED)
    samples = np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))

    # Build row-group boundaries
    rg_starts = []
    cumulative = 0
    for rg_i in range(n_rg):
        rg_starts.append(cumulative)
        cumulative += pf.metadata.row_group(rg_i).num_rows
    rg_starts = np.array(rg_starts)

    # Map each sample to its row group
    rg_for_sample = np.searchsorted(rg_starts, samples, side="right") - 1
    # Group samples by row group
    from collections import defaultdict
    rg_to_samples: dict[int, list[int]] = defaultdict(list)
    for s, rg_i in zip(samples, rg_for_sample):
        rg_to_samples[int(rg_i)].append(int(s))

    print(f"\n{len(samples):,} samples across {len(rg_to_samples)} row groups")

    # ── 5. Load tokenizer ─────────────────────────────────────
    # Use the Rust tokenizers library directly (same as datatrove pipeline)
    # to avoid \n\n merge differences vs transformers.AutoTokenizer.
    print("Loading tokenizer ...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    # ── 6. Verify sampled windows ─────────────────────────────
    eos_pad_err = 0
    tl_err = 0
    retok_err = 0
    refl_err = 0
    checked = 0

    with open(bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        for rg_i in sorted(rg_to_samples):
            rg_start = int(rg_starts[rg_i])
            rg = pf.read_row_group(rg_i)
            sample_idxs = rg_to_samples[rg_i]

            texts = rg.column("text").to_pylist()
            doc_ids = rg.column("doc_id").to_pylist()
            sc_tl = rg.column("token_length").to_pylist()
            refls = rg.column("reflection").to_pylist()
            prefls = rg.column("preflection").to_pylist()
            rpos = rg.column("reflection_position").to_pylist()

            for i in sample_idxs:
                local = i - rg_start
                tl = int(token_lengths[i])
                window = read_window(mm, i)

                # EOS/PAD
                if window[tl] != 0:
                    eos_pad_err += 1
                    if eos_pad_err <= 3:
                        print(f"  EOS ERROR window {i}: pos {tl} = {window[tl]}")
                elif not np.all(window[tl + 1:] == 0):
                    eos_pad_err += 1
                    if eos_pad_err <= 3:
                        print(f"  PAD ERROR window {i}: non-zero after pos {tl}")

                # token_length match
                if sc_tl[local] != tl:
                    tl_err += 1
                    if tl_err <= 3:
                        print(f"  TL MISMATCH window {i}: sidecar={sc_tl[local]}, npy={tl}")

                # reflection fields
                if refls[local] != "" or prefls[local] != "" or rpos[local] != 0:
                    refl_err += 1

                # re-tokenization
                expected = tokenizer.encode(texts[local], add_special_tokens=False).ids[:tl]
                actual = window[:tl].tolist()
                if expected != actual:
                    retok_err += 1
                    if retok_err <= 5:
                        diff_pos = next((j for j, (a, b) in enumerate(zip(expected, actual)) if a != b), "len_diff")
                        print(f"  RETOK MISMATCH window {i}: "
                              f"expected {len(expected)} toks, got {len(actual)}, diff@{diff_pos}")

                checked += 1

            if checked % 500 == 0:
                print(f"  checked {checked}/{len(samples)} ...")

    # ── 7. Summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows:           {n:,}")
    print(f"  Sampled & checked:       {checked:,}")
    print(f"  EOS/PAD errors:          {eos_pad_err}")
    print(f"  token_length mismatches: {tl_err}")
    print(f"  Re-tokenization errors:  {retok_err}")
    print(f"  Reflection field errors: {refl_err}")

    total = eos_pad_err + tl_err + retok_err + refl_err
    if total == 0:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  FAILED: {total} total errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
