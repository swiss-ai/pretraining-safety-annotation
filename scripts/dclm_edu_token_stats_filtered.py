"""Token-length stats of dclm-edu AFTER the production safety filter
(score>=4 AND safety_probs[score]>=0.9), with the Apertus-70B-2509 tokenizer.
Matches pipeline/corpus/safety.py::passes_safety. Scans more raw docs per shard
because the filter thins the sample; also reports the pass rate."""
import glob

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

TOK_PATH = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-70B-2509"
DATA_DIR = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/safety_labels/dclm-edu-filterrobots_fine/data"
MIN_SCORE, MIN_CONF = 4, 0.9
N_SHARDS_SAMPLED = 24
SCAN_PER_SHARD = 200000       # raw docs scanned per shard (then filtered)
MAX_FILTERED = 45000          # cap on tokenized filtered docs
PIPELINE_MAX_CHARS = 8000

shards = sorted(glob.glob(f"{DATA_DIR}/*.parquet"))
pick = [shards[i] for i in np.linspace(0, len(shards) - 1, N_SHARDS_SAMPLED, dtype=int)]
tok = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)

texts, n_raw, n_pass, rows_per_shard = [], 0, 0, None
for f in pick:
    pf = pq.ParquetFile(f)
    rows_per_shard = rows_per_shard or pf.metadata.num_rows
    seen = 0
    for b in pf.iter_batches(batch_size=20000, columns=["text", "safety_score", "safety_probs"]):
        scores = b.column("safety_score").to_numpy(zero_copy_only=False)
        n_raw += len(scores)
        seen += len(scores)
        # Cheap numpy pre-filter on score; only materialize probs/text for score>=4 rows.
        cand = np.nonzero(scores >= MIN_SCORE)[0]
        if len(cand):
            probs = b.column("safety_probs")
            txt = b.column("text")
            for i in cand:
                s = int(scores[i])
                p = probs[int(i)].as_py()
                if p is not None and 0 <= s < len(p) and p[s] >= MIN_CONF:
                    n_pass += 1
                    if len(texts) < MAX_FILTERED:
                        texts.append(txt[int(i)].as_py())
        if seen >= SCAN_PER_SHARD:
            break

pass_rate = n_pass / n_raw
print(f"scanned {n_raw} raw docs across {N_SHARDS_SAMPLED} shards; "
      f"{n_pass} passed (score>={MIN_SCORE} & conf>={MIN_CONF}) -> pass rate {pass_rate:.2%}")
print(f"tokenizing {len(texts)} filtered docs")

def tok_lengths(strs):
    out = []
    for i in range(0, len(strs), 1000):
        out.extend(len(x) for x in tok(strs[i : i + 1000], add_special_tokens=False)["input_ids"])
    return np.array(out)

full = tok_lengths(texts)
trunc = tok_lengths([t[:PIPELINE_MAX_CHARS] for t in texts])
chars = np.array([len(t) for t in texts])

def report(name, a):
    p = lambda q: int(np.percentile(a, q))
    print(f"\n=== {name} (n={len(a)}) ===")
    print(f"  mean={a.mean():.1f}  std={a.std():.1f}  min={a.min()}  max={a.max()}")
    print(f"  p1={p(1)}  p5={p(5)}  p25={p(25)}  p50={p(50)}  p75={p(75)}  p90={p(90)}  p95={p(95)}  p99={p(99)}")

report("FULL-document token length (filtered)", full)
report("PIPELINE input token length, text[:8000 chars] (filtered)", trunc)
print(f"\nchars/token (full docs): {chars.sum()/full.sum():.2f}")
est_docs = len(shards) * (rows_per_shard or 0)
est_filtered = est_docs * pass_rate
print(f"est. total corpus docs: ~{est_docs/1e9:.2f}B  =>  est. filtered docs to annotate: ~{est_filtered/1e6:.0f}M")
print(f"est. filtered tokens (full docs): ~{est_filtered*full.mean()/1e9:.0f}B")
print(f"frac filtered docs > 8000 chars (truncated): {(chars>PIPELINE_MAX_CHARS).mean():.1%}")
