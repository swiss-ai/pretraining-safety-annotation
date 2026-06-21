"""Generation sweep: run the per-model reflection prompts over English safety
docs (locuslab/safety_data_annotated) via OpenRouter — all models in parallel,
reasoning ON. Dumps per-model JSONL (for judging) and prints a summary.

Usage: uv run python scripts/run_generation_sweep.py [N] [model_id ...]
"""

import asyncio
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from pipeline.config import CHARTER_PATH, PROJECT_ROOT, load_config
from pipeline.tokenizer import compute_reflection_point_tokens
from smoke_common import DEFAULT_CONCURRENCY, generate_batch, latest_version, load_system_prompt, make_client

load_dotenv(PROJECT_ROOT / ".env")

DATASET = "locuslab/safety_data_annotated"  # English safety-annotated web text
ALIAS = {
    "qwen/qwen3.5-35b-a3b": "qwen3.5-35b-a3b",
    "google/gemma-4-31b-it": "gemma-4-31b",
    "google/gemma-4-26b-a4b-it": "gemma-4-26b-a4b",
}
OUT_DIR = Path.home() / "tmp"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODELS = sys.argv[2:] or list(ALIAS)


def pick_items(n: int, max_tokens: int, seed: int = 42) -> list[dict]:
    """Stream rows and select `n` spanning the safety_score range."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    by_score: dict[int, list[dict]] = {}
    seen = 0
    for row in ds:
        if len((row.get("text") or "").strip()) < 200 or row.get("safety_score") is None:
            continue
        by_score.setdefault(int(row["safety_score"]), []).append(row)
        seen += 1
        if seen >= 600:
            break
    picked, buckets, i = [], [by_score[k] for k in sorted(by_score)], 0
    while len(picked) < n and any(buckets):
        b = buckets[i % len(buckets)]
        if b:
            picked.append(b.pop(0))
        i += 1
        if i > n * (len(buckets) + 1):
            break
    items = []
    for row in picked[:n]:
        text = row["text"].strip()
        rng = random.Random(f"{seed}_{row['id']}")
        rp, _ = compute_reflection_point_tokens(text, rng, max_tokens=max_tokens)
        items.append({"item_id": str(row["id"]), "text": text, "reflection_point": rp,
                      "safety_score": int(row["safety_score"]), "ds_reason": row.get("reason")})
    return items


def summarize(model: str, items: list[dict], recs: dict) -> None:
    ok = [r for r in recs.values() if not r.get("error")]
    errs = [r for r in recs.values() if r.get("error")]
    cited: Counter = Counter()
    n_cite = n_think = miss = 0
    outs, reas = [], []
    for r in ok:
        c = r.get("citations") or []
        cited.update(c)
        n_cite += bool(c)
        n_think += bool((r.get("reasoning_tokens") or 0) > 0)
        outs.append(r.get("output_tokens") or 0)
        reas.append(r.get("reasoning_tokens") or 0)
        if re.search(r"\b[1-8A]\.\d\b", r.get("reflection_1p") or "") and not c:
            miss += 1
    n = max(1, len(ok))
    print(f"  SUMMARY {model}: ok {len(ok)}/{len(items)} (err {len(errs)}) | thinking {n_think}/{len(ok)} | "
          f"cited {n_cite}/{len(ok)} | bracket-miss {miss} | avg_out {sum(outs)//n} | avg_reas {sum(reas)//n}")
    print(f"  cited values: {dict(cited.most_common())}")


def dump(alias: str, items: list[dict], recs: dict) -> Path:
    out = OUT_DIR / f"smoke{len(items)}_{alias}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            r = recs.get(it["item_id"], {})
            f.write(json.dumps({
                "item_id": it["item_id"], "safety_score": it["safety_score"],
                "ds_reason": it.get("ds_reason"), "seen_text": it["text"][: it["reflection_point"]],
                "analysis": r.get("analysis"), "reflection_1p": r.get("reflection_1p"),
                "citations": r.get("citations"), "output_tokens": r.get("output_tokens"),
                "reasoning_tokens": r.get("reasoning_tokens"), "error": r.get("error"),
            }) + "\n")
    return out


async def run_all(cfg, charter_text, items):
    client, sem = make_client(cfg, DEFAULT_CONCURRENCY)
    specs = [(m, ALIAS.get(m, m.replace("/", "_"))) for m in MODELS]
    prompts = {a: load_system_prompt(a, charter_text) for _, a in specs}  # latest version per alias
    results = await asyncio.gather(
        *[generate_batch(client, sem, m, prompts[a], items, a) for m, a in specs])
    return list(zip(specs, results))


def main():
    cfg = load_config()
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    items = pick_items(N, max_tokens=cfg.max_tokens)
    print(f"Selected {len(items)} items (safety {[it['safety_score'] for it in items]})  "
          f"models in parallel: {len(MODELS)}  reasoning: ON\n")
    for (model, alias), recs in asyncio.run(run_all(cfg, charter_text, items)):
        print("=" * 90)
        print(f"MODEL: {model}   PROMPT: {alias}/generator_reflection_v{latest_version(alias)}.md")
        print("=" * 90)
        summarize(model, items, recs)
        print(f"  wrote {dump(alias, items, recs)}")


if __name__ == "__main__":
    main()
