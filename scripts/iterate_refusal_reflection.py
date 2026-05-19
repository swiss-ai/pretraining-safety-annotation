# %% [markdown]
# # Refusal Reflection — Interactive Iteration
#
# Vibe-check the refusal-augmented reflection prompt on a sample of
# gold + fresh Dolma3 docs.

# %%
from __future__ import annotations

import asyncio
import json
import textwrap

import dotenv

dotenv.load_dotenv()

import pipeline.api
from pipeline.api import api_call, make_api_client, resolve_sampling_params

pipeline.api.MAX_RETRIES = 3
pipeline.api.MAX_RETRIES_RATE_LIMIT = 3
from pipeline.charter.improve.run import select_items
from pipeline.config import CHARTER_PATH, WRITING_GUIDELINES_PATH, load_config, resolve_prompt_path
from pipeline.generation import REFUSAL_REFLECTION_TASK, parse_generation

# %%  — configuration
MODEL = "Qwen/Qwen3.5-35B-A3B"
ALIAS = "qwen3.5-35b-a3b"
ENDPOINT = "https://openrouter.ai/api/v1"
MAX_CONCURRENT = 50

N_ITEMS = 100
N_GOLD = 5
SEED = 42
MAX_TOKENS = 1920

REFUSAL_PROMPT = "generator_reflection_refusal_v2.md"

# %%  — load prompt and resources
charter_text = CHARTER_PATH.read_text(encoding="utf-8")
guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")


def load_prompt(filename: str) -> str:
    path = resolve_prompt_path(filename, ALIAS)
    template = path.read_text(encoding="utf-8")
    return template.replace("{charter}", charter_text).replace(
        "{writing_guidelines}", guidelines_text
    )


refusal_system = load_prompt(REFUSAL_PROMPT)
print(f"Refusal prompt: {len(refusal_system):,} chars")

# %%  — sample documents
items = select_items(N_ITEMS, N_GOLD, SEED, MAX_TOKENS)
print(f"Sampled {len(items)} items ({sum(i.get('is_gold', False) for i in items)} gold)")
for it in items[:3]:
    print(f"  {it['item_id'][:20]}… {len(it['text'])} chars")

# %%  — generation helpers
cfg = load_config()
api_keys = dict(cfg.api_keys) if cfg.api_keys else None
client, semaphore = make_api_client(ENDPOINT, MAX_CONCURRENT, api_keys=api_keys)
sampling = resolve_sampling_params(MODEL, ALIAS)


async def generate_one(system_prompt: str, item: dict) -> dict:
    rp = item["reflection_point"]
    context = item["text"][:rp]
    user_msg = f"## Full Text\n\n{context}" + REFUSAL_REFLECTION_TASK
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    try:
        raw, reasoning, usage = await api_call(
            client, MODEL, messages, semaphore,
            sampling_params=sampling,
            max_tokens=4096,
        )
    except RuntimeError as e:
        return {
            "item_id": item["item_id"],
            "parsed": {"_error": f"api_failed: {e}"},
            "raw": None,
            "reasoning": None,
            "usage": {},
        }
    try:
        parsed = parse_generation(raw, required_fields={"analysis", "reflection_1p"})
    except (json.JSONDecodeError, AssertionError):
        parsed = {"_raw": raw, "_error": "parse_failed"}
    return {
        "item_id": item["item_id"],
        "parsed": parsed,
        "raw": raw,
        "reasoning": reasoning,
        "usage": usage,
    }


async def generate_batch(system_prompt: str, items: list[dict], desc: str) -> list[dict]:
    from tqdm.asyncio import tqdm_asyncio
    coros = [generate_one(system_prompt, it) for it in items]
    return await tqdm_asyncio.gather(*coros, desc=desc)

# %%  — run refusal variant
loop = asyncio.new_event_loop()
results = loop.run_until_complete(generate_batch(refusal_system, items, "refusal"))
print(f"Refusal: {len(results)} results, {sum(1 for r in results if '_error' in r['parsed'])} errors")

# %%  — display results
WIDTH = 100


def wrap(text: str, w: int = WIDTH) -> str:
    return "\n".join(textwrap.wrap(text, w))


for r in results:
    p = r["parsed"]
    print("=" * WIDTH)
    print(f"DOC: {r['item_id']}")
    print("-" * WIDTH)

    if "_error" in p:
        print(f"[ERROR] {p['_error']}")
        if r["raw"]:
            print(f"  raw: {r['raw'][:300]}")
        continue

    print(f"\nANALYSIS:\n{wrap(p.get('analysis', ''))}\n")
    print(f"1P:\n{wrap(p.get('reflection_1p', ''))}\n")

# %%  — dump to markdown
from pathlib import Path

item_by_id = {it["item_id"]: it for it in items}
md_path = Path("data/refusal_reflection_vibecheck_v2.md")
md_lines = ["# Refusal Reflection Vibe Check\n"]
md_lines.append(f"Prompt: `{REFUSAL_PROMPT}` | Model: `{MODEL}` | N={len(results)}\n")

import re as _re

successful = [r for r in results if "_error" not in r["parsed"]]
cited_results = [r for r in successful if _re.search(r"\[\d+\.\d+\]", r["parsed"].get("reflection_1p", ""))]
benign_results = [r for r in successful if r not in cited_results]

# Interleave: alternate cited and benign
interleaved = []
ci, bi = 0, 0
while ci < len(cited_results) or bi < len(benign_results):
    if ci < len(cited_results):
        interleaved.append(cited_results[ci]); ci += 1
    if bi < len(benign_results):
        interleaved.append(benign_results[bi]); bi += 1

for r in interleaved:
    item_id = r["item_id"]
    p = r["parsed"]
    item = item_by_id[item_id]
    source_text = item["text"][:item["reflection_point"]]
    has_cite = _re.search(r"\[\d+\.\d+\]", p.get("reflection_1p", ""))

    md_lines.append(f"\n---\n\n## {item_id} {'[CITED]' if has_cite else '[BENIGN]'}\n")

    # Source text (truncated to reflection point)
    md_lines.append("\n<details><summary>Source text (click to expand)</summary>\n")
    md_lines.append(f"\n```\n{source_text}\n```\n")
    md_lines.append("\n</details>\n")

    md_lines.append(f"\n### Analysis\n\n{p.get('analysis', '')}\n")
    md_lines.append(f"\n### reflection_1p\n\n{p.get('reflection_1p', '')}\n")

md_path.write_text("\n".join(md_lines), encoding="utf-8")
print(f"\nMarkdown written to {md_path} ({len(md_lines)} lines)")
