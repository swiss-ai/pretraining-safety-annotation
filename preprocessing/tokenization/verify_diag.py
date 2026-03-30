"""Quick diagnostic: why do re-tokenized sidecar texts differ from .bin tokens?
Samples 20 windows, prints detailed mismatch info."""
import mmap, struct, sys, os
from pathlib import Path

# Prevent tokenize.py shadowing stdlib
_sd = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if os.path.abspath(p) != _sd]

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from collections import defaultdict

DATA = Path("/iopsstor/scratch/cscs/jminder/tokenized/annotated")
WINDOW, MAX_C = 2049, 1920

# Read idx header
with open(DATA / "annotated.idx", "rb") as f:
    f.read(9); f.read(8); f.read(1)
    n = struct.unpack("<Q", f.read(8))[0]
print(f"n_windows: {n:,}")

tl = np.load(str(DATA / "token_lengths.npy"))
print(f"token_lengths range: [{tl.min()}, {tl.max()}], mean: {tl.mean():.1f}")

# Sample
rng = np.random.default_rng(12345)
samples = sorted(rng.choice(n, size=20, replace=False))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

pf = pq.ParquetFile(str(DATA / "sidecar.parquet"))
n_rg = pf.metadata.num_row_groups
rg_starts = []
cum = 0
for i in range(n_rg):
    rg_starts.append(cum)
    cum += pf.metadata.row_group(i).num_rows
rg_starts = np.array(rg_starts)

rg_map = defaultdict(list)
for s in samples:
    rg_i = int(np.searchsorted(rg_starts, s, side="right") - 1)
    rg_map[rg_i].append(int(s))

print(f"20 samples across {len(rg_map)} row groups")
match_count = 0
mismatch_count = 0

with open(DATA / "annotated.bin", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    for rg_i in sorted(rg_map):
        rg_start = int(rg_starts[rg_i])
        print(f"\nReading row_group {rg_i} (start={rg_start:,})...")
        rg = pf.read_row_group(rg_i)
        texts = rg.column("text").to_pylist()

        for idx in rg_map[rg_i]:
            local = idx - rg_start
            tok_len = int(tl[idx])
            off = idx * WINDOW * 2
            actual = np.frombuffer(mm[off:off + WINDOW * 2], dtype=np.uint16)[:tok_len].tolist()
            text = texts[local]
            expected = tokenizer.encode(text, add_special_tokens=False)[:MAX_C]

            if expected == actual:
                print(f"  window {idx}: MATCH (len={tok_len})")
                match_count += 1
            else:
                mismatch_count += 1
                min_len = min(len(expected), len(actual))
                first_diff = next((j for j in range(min_len) if expected[j] != actual[j]), min_len)
                print(f"  window {idx}: MISMATCH exp_len={len(expected)} act_len={len(actual)} first_diff@{first_diff}")

                if first_diff < min_len:
                    print(f"    expected[{first_diff}:{first_diff+5}] = {expected[first_diff:first_diff+5]}")
                    print(f"    actual  [{first_diff}:{first_diff+5}] = {actual[first_diff:first_diff+5]}")
                    print(f"    exp decoded: {repr(tokenizer.decode(expected[max(0,first_diff-2):first_diff+3]))}")
                    print(f"    act decoded: {repr(tokenizer.decode(actual[max(0,first_diff-2):first_diff+3]))}")
                else:
                    print(f"    exp tail: {expected[-3:]} -> {repr(tokenizer.decode(expected[-3:]))}")
                    print(f"    act tail: {actual[-3:]} -> {repr(tokenizer.decode(actual[-3:]))}")

                # Check if it's an off-by-one at the end
                if len(expected) == len(actual) + 1 and expected[:len(actual)] == actual:
                    print(f"    >>> actual is expected[:-1] (missing last token)")
                elif len(actual) == len(expected) + 1 and actual[:len(expected)] == expected:
                    print(f"    >>> actual has one extra token vs expected")

                # Show text length
                n_tok_full = len(tokenizer.encode(text, add_special_tokens=False))
                print(f"    text chars={len(text)}, full_tokens={n_tok_full}, truncated={n_tok_full > MAX_C}")

print(f"\nMatch: {match_count}, Mismatch: {mismatch_count}")
