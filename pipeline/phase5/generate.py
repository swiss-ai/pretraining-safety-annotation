"""Charter-aware paired generator for phase 5.

Calls qwen3.5-35b-a3b on openrouter with the v3 prompt and parses the
``{"cited": "...", "uncited": "..."}`` response. Used during prompt
iteration; the SLURM-scaling story will live in submit.py once the
prompt is frozen.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from pipeline.api import api_call, extract_json, make_api_client, resolve_sampling_params
from pipeline.log import logger
from pipeline.phase5.data import SourcedPrompt

ENDPOINT = "https://openrouter.ai/api/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B"
ALIAS = "qwen3.5-35b-a3b"
API_KEYS = {ENDPOINT: "OPENROUTER_API_KEY"}

REPO = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CHARTER_PATH = REPO / "resources" / "ModelRaisingConstitution_v0.2.md"

_IDENTITY_LEAK_RE = re.compile(r"\bqwen\b", re.IGNORECASE)


def has_identity_leak(text: str) -> bool:
    """True if text contains the generator model's identity."""
    return bool(_IDENTITY_LEAK_RE.search(text))


def render_system_prompt(prompt_version: str) -> str:
    """Read prompts/charter_sft_<version>_prompt.md and inline the charter."""
    prompt_path = PROMPTS_DIR / f"charter_sft_{prompt_version}_prompt.md"
    assert prompt_path.exists(), f"Prompt file missing: {prompt_path}"
    template = prompt_path.read_text()
    charter = CHARTER_PATH.read_text()
    assert "{charter}" in template, f"Prompt missing {{charter}} placeholder: {prompt_path}"
    return template.replace("{charter}", charter)


async def generate_one(
    client,
    semaphore,
    system_prompt: str,
    sp: SourcedPrompt,
    model: str = MODEL,
    alias: str = ALIAS,
) -> dict:
    """One paired generation. Returns a result dict (no exceptions raised).

    `model` is the API name (e.g. ``Qwen/Qwen3.5-35B-A3B`` for openrouter
    or the served-model name for a local sglang). `alias` is used only
    to resolve recommended sampling params.
    """
    from pipeline.phase5.slurm_generate import _format_user_message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _format_user_message(sp)},
    ]
    sampling = resolve_sampling_params(model, alias)
    base = {
        "source": sp.source,
        "source_id": sp.source_id,
        "user": sp.user,
        "meta": sp.meta,
        "harm_category": sp.harm_category,
    }
    try:
        content, _reasoning, usage = await api_call(
            client=client,
            model=model,
            messages=messages,
            semaphore=semaphore,
            thinking=False,
            json_mode=False,
            sampling_params=sampling,
            max_tokens=None,
        )
    except Exception as e:
        return {**base, "error": f"api: {type(e).__name__}: {e}"}

    try:
        parsed = extract_json(content)
    except Exception as e:
        return {**base, "error": f"parse: {type(e).__name__}: {e}", "raw": content}

    cited = parsed.get("cited") if isinstance(parsed, dict) else None
    uncited = parsed.get("uncited") if isinstance(parsed, dict) else None
    analysis = parsed.get("analysis") if isinstance(parsed, dict) else None  # v5+; optional
    if not isinstance(cited, str) or not isinstance(uncited, str):
        return {**base, "error": "missing 'cited'/'uncited' fields",
                "raw": content, "raw_keys": list(parsed.keys()) if isinstance(parsed, dict) else None}

    if has_identity_leak(cited) or has_identity_leak(uncited):
        return {**base, "error": "identity leak: model name in output", "raw": content}

    return {
        **base,
        "analysis": analysis if isinstance(analysis, str) else None,
        "cited": cited,
        "uncited": uncited,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


async def generate_batch(
    prompts: list[SourcedPrompt],
    prompt_version: str,
    max_concurrent: int = 8,
) -> list[dict]:
    """Generate paired responses for a batch of prompts. Concurrent.

    Used by ``iterate`` for small batches (returns all results in memory).
    For 1K+ scale, use :func:`generate_streaming` instead.
    """
    system_prompt = render_system_prompt(prompt_version)
    logger.info(
        "system prompt: {} chars; running {} prompts (concurrency={})",
        len(system_prompt), len(prompts), max_concurrent,
    )
    client, semaphore = make_api_client(ENDPOINT, max_concurrent=max_concurrent, api_keys=API_KEYS)
    coros = [generate_one(client, semaphore, system_prompt, sp) for sp in prompts]
    return await asyncio.gather(*coros)


def save_results(results: list[dict], path: Path) -> None:
    """Write results as JSONL (one row per generation, overwrites existing file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote {} results to {}", len(results), path)


def load_done_set(path: Path) -> set[str]:
    """Load source_ids already present (without errors) in an existing JSONL.

    Returns the set of source_ids that don't need to be re-generated.
    Rows with errors are NOT included — they get retried on resume.
    """
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r and "cited" in r and "uncited" in r:
                done.add(r["source_id"])
    return done


async def generate_streaming(
    prompts: list[SourcedPrompt],
    prompt_version: str,
    out_path: Path,
    max_concurrent: int = 200,
    progress_every: int = 100,
) -> tuple[int, int]:
    """Stream-append paired generations to JSONL. Resumable.

    Skips prompts whose source_id is already present (and error-free) in
    out_path. Each completion is appended to the file as it finishes, so
    a kill-and-restart loses at most a handful of in-flight rows.

    Returns (n_ok, n_err) for the rows generated this invocation
    (excluding pre-existing rows).
    """
    system_prompt = render_system_prompt(prompt_version)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_set(out_path)
    todo = [p for p in prompts if p.source_id not in done]
    logger.info(
        "system prompt {} chars; {} done, {} todo (concurrency={})",
        len(system_prompt), len(done), len(todo), max_concurrent,
    )
    if not todo:
        return 0, 0

    client, semaphore = make_api_client(ENDPOINT, max_concurrent=max_concurrent, api_keys=API_KEYS)

    write_lock = asyncio.Lock()
    f = out_path.open("a")
    n_ok = n_err = 0
    n_done = 0

    async def run_one(sp: SourcedPrompt):
        nonlocal n_ok, n_err, n_done
        result = await generate_one(client, semaphore, system_prompt, sp)
        async with write_lock:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if "error" in result:
                n_err += 1
            else:
                n_ok += 1
            n_done += 1
            if n_done % progress_every == 0:
                logger.info("progress: {}/{} ({} ok, {} err)", n_done, len(todo), n_ok, n_err)

    try:
        await asyncio.gather(*[run_one(sp) for sp in todo])
    finally:
        f.close()
    logger.info("finished: {} ok, {} err in {} attempts", n_ok, n_err, len(todo))
    return n_ok, n_err
