"""Multilinguality test: gemma-4-31b on non-English docs
(VityaVitalich/multilingual-safety-data), two conditions run in parallel on the
SAME docs — reflect in English vs reflect in the source document's language.
Reasoning ON. A crude script-of() hint flags whether each output's writing system
matches the expected language.

Usage: uv run python scripts/run_multilingual.py [N]
"""

import asyncio
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline.config import CHARTER_PATH, PROJECT_ROOT, load_config
from pipeline.tokenizer import compute_reflection_point_tokens
from smoke_common import DEFAULT_CONCURRENCY, generate_batch, load_system_prompt, make_client

load_dotenv(PROJECT_ROOT / ".env")

DATASET = "VityaVitalich/multilingual-safety-data"
CONFIG = "clean"
MODEL = "google/gemma-4-31b-it"
ALIAS = "gemma-4-31b"
OUT = Path.home() / "tmp" / "smoke_multilingual.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

LANG_INSTR = {
    "en": "The source text may be in any language. Write both `analysis` and "
          "`reflection_1p` in English, regardless of the language of the source text.",
    "src": "The source text may be in any language. Write `reflection_1p` in the SAME "
           "language as the source text. (Your `analysis` may be in English.)",
}


def pick_items(n: int, max_tokens: int, seed: int = 42) -> list[dict]:
    """Spread `n` docs across all languages (dataset is grouped by language)."""
    from datasets import load_dataset

    df = load_dataset(DATASET, CONFIG, split="train").to_pandas()
    df = df[df["text"].str.len() >= 200]
    per = max(1, n // df["lang"].nunique() + 1)
    buckets = {lang: g.head(per).to_dict("records") for lang, g in df.groupby("lang")}
    picked, langs, i = [], sorted(buckets), 0
    while len(picked) < n and any(buckets[l] for l in langs):
        b = buckets[langs[i % len(langs)]]
        if b:
            picked.append(b.pop(0))
        i += 1
    items = []
    for r in picked[:n]:
        text = r["text"].strip()
        rng = random.Random(f"{seed}_{r['uid']}")
        rp, _ = compute_reflection_point_tokens(text, rng, max_tokens=max_tokens)
        items.append({"item_id": str(r["uid"]), "lang": r["lang"], "text": text,
                      "reflection_point": rp, "safety_score": int(r["safety_score"])})
    return items


async def run(cfg, charter_text, items):
    client, sem = make_client(cfg, DEFAULT_CONCURRENCY)
    sp_en = load_system_prompt(ALIAS, charter_text, extra_section=LANG_INSTR["en"])
    sp_src = load_system_prompt(ALIAS, charter_text, extra_section=LANG_INSTR["src"])
    return await asyncio.gather(
        generate_batch(client, sem, MODEL, sp_en, items, "en"),
        generate_batch(client, sem, MODEL, sp_src, items, "src"),
    )


def main():
    cfg = load_config()
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    items = pick_items(N, max_tokens=cfg.max_tokens)
    print(f"Model: {MODEL}  reasoning: ON  docs: {len(items)}  "
          f"langs: {sorted({it['lang'] for it in items})}\n")
    en, src = asyncio.run(run(cfg, charter_text, items))

    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "item_id": it["item_id"], "lang": it["lang"], "safety_score": it["safety_score"],
                "seen_text": it["text"][: it["reflection_point"]],
                "en": en.get(it["item_id"]), "src": src.get(it["item_id"]),
            }) + "\n")
    print(f"\nwrote {OUT}\n" + "=" * 90)
    for it in items:
        e, s = en.get(it["item_id"], {}), src.get(it["item_id"], {})
        print("-" * 90)
        print(f"[{it['item_id'][:10]}] lang={it['lang']} safety={it['safety_score']}")
        print(f"  [EN ] {e.get('script', '-')} {e.get('citations')}: {(e.get('reflection_1p') or e.get('error') or '').strip()}")
        print(f"  [SRC] {s.get('script', '-')} {s.get('citations')}: {(s.get('reflection_1p') or s.get('error') or '').strip()}")


if __name__ == "__main__":
    main()
