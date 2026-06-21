"""Estimate what % of dclm docs in the safety bracket [score>=4, 0.8<=conf<0.9]
"require a citation" — operationalized as: the production reflection generator
(Qwen3.6-A3B, v7 prompt, apertus-3800 schematic) emits >=1 inline [X.Y] charter
citation in reflection_1p."""
import argparse
import asyncio
import glob

import numpy as np
import pyarrow.parquet as pq

from pipeline.api import make_api_client, api_call
from pipeline.charter.improve.run import _parse_generation
from pipeline.config import CHARTER_PATH, load_config, union_charter_elements
from pipeline.generation import REFLECTION_TASK
from pipeline.tokenizer import compute_reflection_point_apertus

DATA_DIR = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/safety_labels/dclm-edu-filterrobots_fine/data"
PROMPT_PATH = "/users/jminder/repositories/pretraining-safety-annotation/pipeline/prompts/models/qwen3.6-35b-a3b/generator_reflection_v7.md"
API_NAME = "qwen/qwen3.6-35b-a3b"
ENDPOINT = "https://openrouter.ai/api/v1"
MIN_SCORE, CONF_LO, CONF_HI = 4, 0.8, 0.9

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--scan-per-shard", type=int, default=40000)
ap.add_argument("--n-shards", type=int, default=24)
ap.add_argument("--concurrent", type=int, default=32)
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

# --- sample the bracket ---
shards = sorted(glob.glob(f"{DATA_DIR}/*.parquet"))
pick = [shards[i] for i in np.linspace(0, len(shards) - 1, args.n_shards, dtype=int)]
texts, n_raw, n_bracket = [], 0, 0
per_shard = max(1, -(-args.n // args.n_shards))  # ceil: spread the sample across shards
for f in pick:
    pf = pq.ParquetFile(f)
    seen = got_this = 0
    for b in pf.iter_batches(batch_size=20000, columns=["text", "safety_score", "safety_probs"]):
        scores = b.column("safety_score").to_numpy(zero_copy_only=False)
        n_raw += len(scores); seen += len(scores)
        for i in np.nonzero(scores >= MIN_SCORE)[0]:
            s = int(scores[i]); p = b.column("safety_probs")[int(i)].as_py()
            if p is not None and 0 <= s < len(p) and CONF_LO <= p[s] < CONF_HI:
                n_bracket += 1
                if got_this < per_shard and len(texts) < args.n:
                    texts.append(b.column("text")[int(i)].as_py()); got_this += 1
        if seen >= args.scan_per_shard or got_this >= per_shard or len(texts) >= args.n:
            break
    if len(texts) >= args.n:
        break
print(f"scanned {n_raw} raw; bracket [score>={MIN_SCORE}, {CONF_LO}<=conf<{CONF_HI}] seen {n_bracket}; using {len(texts)}")

# --- system prompt (v7 + charter) ---
charter = CHARTER_PATH.read_text(encoding="utf-8")
system_prompt = open(PROMPT_PATH, encoding="utf-8").read().replace("{charter}", charter)

cfg = load_config()
client, sem = make_api_client(ENDPOINT, max_concurrent=args.concurrent, api_keys=cfg.api_keys)

async def one(idx, text):
    rp = compute_reflection_point_apertus(text)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"## Full Text\n\n{text[:rp]}" + REFLECTION_TASK},
    ]
    try:
        raw, reasoning, usage = await api_call(client, API_NAME, msgs, sem, thinking=True, max_tokens=8192)
    except Exception as e:
        return {"ok": False, "err": str(e)[:200]}
    try:
        parsed = _parse_generation(raw, required_fields={"analysis", "reflection_1p"})
    except Exception:
        parsed = {"analysis": raw, "reflection_1p": ""}
    r1p = parsed.get("reflection_1p", "")
    try:
        cites = union_charter_elements(r1p)
    except Exception:
        cites = []
    return {"ok": True, "n_cites": len(cites), "cites": cites, "r1p": r1p, "raw": raw}

async def main():
    res = await asyncio.gather(*[one(i, t) for i, t in enumerate(texts)])
    ok = [r for r in res if r.get("ok")]
    fail = [r for r in res if not r.get("ok")]
    if fail:
        print(f"FAILED {len(fail)}/{len(res)}; e.g. {fail[0].get('err')}")
    if not ok:
        return
    with_cite = [r for r in ok if r["n_cites"] >= 1]
    pct = 100 * len(with_cite) / len(ok)
    print(f"\n=== bracket citation analysis (n={len(ok)} annotated) ===")
    print(f"  require a citation (>=1 [X.Y]): {len(with_cite)}/{len(ok)} = {pct:.1f}%")
    counts = [r["n_cites"] for r in ok]
    print(f"  citations/doc: mean={np.mean(counts):.2f} median={int(np.median(counts))} max={max(counts)}")
    from collections import Counter
    dist = Counter(counts)
    print("  count distribution:", {k: dist[k] for k in sorted(dist)})
    topvals = Counter(c for r in ok for c in r["cites"])
    print("  most-cited values:", topvals.most_common(8))
    if args.verbose:
        for i, r in enumerate(ok[:8]):
            print(f"\n  --- doc {i}: cites={r['cites']}\n   r1p: {r['r1p'][:300]}")

asyncio.run(main())
