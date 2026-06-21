"""Cross-lingual consistency test for the reflection generator.

The multilingual dataset (VityaVitalich/multilingual-safety-data) holds 1-1
translations of each document into 7 languages, linked by `uid`. We run the
generator on every language version of the SAME document — with the reflection
fixed to English — and measure how consistent the cited [X.Y] values are across
languages. High consistency ⇒ the annotation is robust to source language.

Usage: uv run python scripts/run_consistency.py [n_groups] [model_id]
"""

import asyncio
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from pipeline.config import CHARTER_PATH, PROJECT_ROOT, load_config
from smoke_common import DEFAULT_CONCURRENCY, generate_batch, load_system_prompt, make_client

load_dotenv(PROJECT_ROOT / ".env")

DATASET = "VityaVitalich/multilingual-safety-data"
CONFIG = "clean"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "google/gemma-4-31b-it"
ALIAS = "gemma-4-31b"
CHAR_CAP = 8000  # annotate each translation in full (docs are short); cap is a guard
OUT = Path.home() / "tmp" / "smoke_consistency.jsonl"
EN_INSTR = ("The source text may be in any language. Write both `analysis` and "
            "`reflection_1p` in English, regardless of the language of the source text.")

N_GROUPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def pick_groups(n: int, seed: int = 42):
    """Return (uids, items) for n complete 7-language translation groups."""
    from datasets import load_dataset

    df = load_dataset(DATASET, CONFIG, split="train").to_pandas()
    df = df[df["text"].str.len() >= 200]
    counts = df.groupby("uid")["lang"].nunique()
    uids = list(counts[counts == 7].index)
    random.Random(seed).shuffle(uids)
    chosen = uids[:n]

    items = []
    sub = df[df["uid"].isin(chosen)]
    for _, r in sub.iterrows():
        text = r["text"].strip()
        items.append({
            "item_id": f"{r['uid']}:{r['lang']}",
            "uid": r["uid"], "lang": r["lang"], "safety_score": int(r["safety_score"]),
            "text": text, "reflection_point": min(len(text), CHAR_CAP),
        })
    return chosen, items


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def main():
    cfg = load_config()
    charter = CHARTER_PATH.read_text(encoding="utf-8")
    sp = load_system_prompt(ALIAS, charter, extra_section=EN_INSTR)
    chosen, items = pick_groups(N_GROUPS)
    print(f"Model: {MODEL}  reasoning: ON  groups: {len(chosen)}  "
          f"items: {len(items)} (7 langs each)\n")

    client, sem = make_client(cfg, DEFAULT_CONCURRENCY)
    recs = asyncio.run(generate_batch(client, sem, MODEL, sp, items, tag="cons"))

    by_uid: dict[str, dict] = {}
    for it in items:
        r = recs.get(it["item_id"]) or {}
        cites = None if r.get("error") else r.get("citations")
        by_uid.setdefault(it["uid"], {})[it["lang"]] = cites

    group_jac, identical = [], 0
    support_hist = {k: 0 for k in range(1, 8)}  # (doc,value) cited in exactly k/7 langs
    with OUT.open("w", encoding="utf-8") as f:
        for uid, langmap in by_uid.items():
            sets = [set(c) for c in langmap.values() if c is not None]
            if len(sets) < 2:
                continue
            pj = mean(jaccard(a, b) for a, b in combinations(sets, 2))
            group_jac.append(pj)
            if len(sets) == 7 and len({frozenset(s) for s in sets}) == 1:
                identical += 1
            for v in set().union(*sets):
                support_hist[sum(v in s for s in sets)] += 1
            f.write(json.dumps({"uid": uid, "n_langs": len(sets),
                                "pairwise_jaccard": round(pj, 3), "by_lang": langmap}) + "\n")

    total = sum(support_hist.values()) or 1
    print("=" * 80)
    print(f"mean pairwise Jaccard of citation sets across languages: {mean(group_jac):.3f}")
    print(f"groups with IDENTICAL citations across all 7 langs:      {identical}/{len(group_jac)}")
    print("value stability — of all (doc,value) pairs cited in ≥1 lang, # langs that cite it:")
    for k in range(7, 0, -1):
        print(f"   {k}/7 langs: {support_hist[k]:4d} ({100 * support_hist[k] / total:.0f}%)")
    lang_n = {}
    for it in items:
        r = recs.get(it["item_id"]) or {}
        lang_n.setdefault(it["lang"], []).append(0 if r.get("error") else len(r.get("citations") or []))
    print("per-language avg #citations:", {l: round(mean(v), 2) for l, v in sorted(lang_n.items())})
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
