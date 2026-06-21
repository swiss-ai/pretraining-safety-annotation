"""Estimate token-length statistics of the dclm-edu corpus with the
Apertus-70B-2509 tokenizer. Samples documents across shards (spread over the
full corpus) for a representative estimate, not just the first shard."""
import glob
import statistics as st

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

TOK_PATH = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-70B-2509"
DATA_DIR = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/safety_labels/dclm-edu-filterrobots_fine/data"
N_SHARDS_SAMPLED = 16        # spread across the corpus
DOCS_PER_SHARD = 3500        # ~56k docs total
PIPELINE_MAX_CHARS = 8000    # the scale schematic truncates input to this

shards = sorted(glob.glob(f"{DATA_DIR}/*.parquet"))
n_total_shards = len(shards)
pick = [shards[i] for i in np.linspace(0, n_total_shards - 1, N_SHARDS_SAMPLED, dtype=int)]
print(f"corpus: {n_total_shards} shards; sampling {N_SHARDS_SAMPLED} of them, {DOCS_PER_SHARD} docs each")

tok = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)

texts = []
rows_per_shard = None
for f in pick:
    pf = pq.ParquetFile(f)
    if rows_per_shard is None:
        rows_per_shard = pf.metadata.num_rows
    batch = next(pf.iter_batches(batch_size=DOCS_PER_SHARD, columns=["text"]))
    texts.extend(batch.column("text").to_pylist())
print(f"sampled {len(texts)} docs (rows/shard ~{rows_per_shard})")

def tok_lengths(strs):
    out = []
    for i in range(0, len(strs), 1000):
        enc = tok(strs[i : i + 1000], add_special_tokens=False)["input_ids"]
        out.extend(len(x) for x in enc)
    return np.array(out)

full = tok_lengths(texts)
trunc = tok_lengths([t[:PIPELINE_MAX_CHARS] for t in texts])
chars = np.array([len(t) for t in texts])

def report(name, a):
    pct = lambda p: int(np.percentile(a, p))
    print(f"\n=== {name} (n={len(a)}) ===")
    print(f"  mean={a.mean():.1f}  std={a.std():.1f}  min={a.min()}  max={a.max()}")
    print(f"  p1={pct(1)}  p5={pct(5)}  p25={pct(25)}  p50={pct(50)}  "
          f"p75={pct(75)}  p90={pct(90)}  p95={pct(95)}  p99={pct(99)}")

report("FULL-document token length", full)
report("PIPELINE input token length (text[:8000 chars])", trunc)
print(f"\nchars/token (full docs): {chars.sum()/full.sum():.2f}")
est_docs = n_total_shards * (rows_per_shard or 0)
print(f"est. total corpus docs: ~{est_docs/1e9:.2f}B  "
      f"=> est. total tokens: ~{est_docs*full.mean()/1e12:.2f}T (full docs)")
frac_over = (full > 2000).mean()
print(f"frac docs > 8000 chars (truncated by pipeline): {(chars>PIPELINE_MAX_CHARS).mean():.1%}")
