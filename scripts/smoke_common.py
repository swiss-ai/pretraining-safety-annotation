"""Shared helpers for the ad-hoc OpenRouter dev scripts in ``scripts/``.

Run from the repo root: ``uv run python scripts/<name>.py``. These scripts
generate reflections with reasoning ENABLED via OpenRouter's native
``reasoning`` param — ``pipeline.api.api_call`` only knows the sglang-style
``enable_thinking`` path, which OpenRouter ignores. On OpenRouter reasoning is
returned separately, so ``reasoning_tokens`` is trustworthy here (unlike the
local sglang cluster, where thinking leaks into ``output_tokens``).
"""

import asyncio
import random
import re

from pipeline.api import make_api_client
from pipeline.config import PROJECT_ROOT, extract_charter_elements
from pipeline.generation import REFLECTION_TASK, parse_generation
from pipeline.tokenizer import compute_reflection_point_tokens

ENDPOINT = "https://openrouter.ai/api/v1"
MAX_TOKENS = 12000  # room for the thinking trace + JSON
REQUEST_TIMEOUT = 240  # seconds; bound a single hung/slow request
DEFAULT_CONCURRENCY = 100
# Per-model iteration history lives here; the current best of each is promoted
# (copied) into ``final_prompts/<alias>/``. The smoke sweeps always iterate on
# the latest version in the history dir.
ITER_PROMPTS = PROJECT_ROOT / "pipeline" / "prompts" / "generator_reflection"


def reasoning_tokens(usage) -> int:
    """Pull reasoning-token count out of an OpenAI/OpenRouter usage object."""
    rt = getattr(usage, "reasoning_tokens", 0) or 0
    if rt:
        return rt
    d = getattr(usage, "completion_tokens_details", None)
    if d is None:
        return 0
    if isinstance(d, dict):
        return d.get("reasoning_tokens", 0) or 0
    return getattr(d, "reasoning_tokens", 0) or 0


def script_of(text: str) -> str:
    """Crude writing-system hint, for eyeballing the language of an output."""
    if re.search(r"[一-鿿぀-ヿ]", text):
        return "CJK"
    if re.search(r"[Ѐ-ӿ]", text):
        return "cyrillic"
    if re.search(r"[a-zA-Z]", text):
        return "latin"
    return "?"


def latest_version(alias: str) -> int:
    """Highest generator_reflection_vN.md version on disk for *alias*."""
    pat = re.compile(r"generator_reflection_v(\d+)\.md$")
    vs = [int(m.group(1)) for f in (ITER_PROMPTS / alias).glob("generator_reflection_v*.md")
          if (m := pat.match(f.name))]
    assert vs, f"no prompt versions in {ITER_PROMPTS / alias}"
    return max(vs)


def load_system_prompt(
    alias: str, charter_text: str, version: int | None = None, extra_section: str | None = None
) -> str:
    """Load ``pipeline/prompts/generator_reflection/<alias>/generator_reflection_v<version>.md``.

    *version* defaults to the highest version on disk for the alias. If
    *extra_section* is given, it is inserted as a ``## Language`` section
    just before the value specification (used by the multilingual experiment).
    """
    if version is None:
        version = latest_version(alias)
    path = ITER_PROMPTS / alias / f"generator_reflection_v{version}.md"
    tmpl = path.read_text(encoding="utf-8")
    if extra_section:
        tmpl = tmpl.replace(
            "## VALUE SPECIFICATION",
            f"## Language\n\n{extra_section}\n\n## VALUE SPECIFICATION",
        )
    return tmpl.replace("{charter}", charter_text)


def reflection_point(text: str, item_id: str, max_tokens: int, full: bool = False, seed: int = 42) -> int:
    """Char offset to cut the text at. ``full=True`` returns the whole text
    (used when we want each translation annotated in full, no cut-point confound)."""
    if full:
        return len(text)
    rng = random.Random(f"{seed}_{item_id}")
    rp, _ = compute_reflection_point_tokens(text, rng, max_tokens=max_tokens)
    return rp


def make_client(cfg, concurrency: int = DEFAULT_CONCURRENCY):
    return make_api_client(ENDPOINT, concurrency, api_keys=cfg.api_keys)


async def generate_batch(client, sem, model, system_prompt, items, tag, max_tokens=MAX_TOKENS) -> dict:
    """Generate one reflection per item (reasoning on). Returns {item_id: record}.

    Each item needs ``item_id``, ``text``, ``reflection_point``. A record has
    analysis / reflection_1p / citations / script / output_tokens /
    reasoning_tokens, or an ``error`` key on failure.
    """

    async def one(item):
        user = f"## Full Text\n\n{item['text'][: item['reflection_point']]}" + REFLECTION_TASK
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        rec = None
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens,
                    extra_body={"reasoning": {"enabled": True}}, timeout=REQUEST_TIMEOUT)
            except Exception as e:
                rec = {"error": f"api {type(e).__name__}: {e}"}
        if rec is None and not resp.choices:
            rec = {"error": f"no choices: {getattr(resp, 'error', None)}"}
        if rec is None:
            content = resp.choices[0].message.content or ""
            try:
                parsed = parse_generation(content, required_fields={"analysis", "reflection_1p"})
                refl = parsed.get("reflection_1p", "") or ""
                rec = {
                    "analysis": parsed.get("analysis", ""),
                    "reflection_1p": refl,
                    "citations": extract_charter_elements(refl),
                    "script": script_of(refl),
                    "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                    "reasoning_tokens": reasoning_tokens(resp.usage),
                }
            except Exception as e:
                rec = {"error": f"parse {type(e).__name__}: {e}", "raw": content[:200]}
        print(f"  · [{tag}] {item['item_id'][:10]} {rec.get('script', '-')} "
              f"{rec.get('citations', rec.get('error'))}", flush=True)
        return item["item_id"], rec

    return dict(await asyncio.gather(*[one(it) for it in items]))
