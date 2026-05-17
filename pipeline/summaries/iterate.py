"""Sample N texts → generate summaries → write JSONL + sibling .md.

Usage:
    uv run python -m pipeline.summaries iterate \\
        [--n 20] [--seed 42] [--prompt-version v1] [--out path.jsonl]

Defaults pulled from ``cfg.summaries`` (configs/config.yaml). The sibling
.md file is the one humans actually read; the JSONL is the machine-readable
archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pipeline.api import (
    api_call,
    extract_json,
    make_api_client,
    resolve_sampling_params,
    run_concurrent,
)
from pipeline.config import load_config
from pipeline.data import sample_texts
from pipeline.log import logger
from pipeline.tokenizer import truncate_to_max_tokens

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"summary_{version}.md"
    assert path.exists(), f"Prompt file not found: {path}"
    return path.read_text(encoding="utf-8")


def _resolve_out_path(
    cfg, alias: str, version: str, seed: int, n: int,
    min_safety_score: int | None, max_safety_score: int | None,
    override: str | None,
) -> Path:
    if override:
        return Path(override)
    runs_dir = Path(cfg.summaries.output_dir) / "runs"
    suffix = ""
    if min_safety_score is not None:
        suffix += f"_minsafety{min_safety_score}"
    if max_safety_score is not None:
        suffix += f"_maxsafety{max_safety_score}"
    return runs_dir / f"{alias}_{version}_seed{seed}_n{n}{suffix}.jsonl"


def _write_md(out_path: Path, results: list[dict], cfg, version: str, seed: int, n: int) -> Path:
    md_path = out_path.with_suffix(".md")
    alias = cfg.summaries.model.alias
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Summary iteration: `{alias}` / `{version}` / seed={seed}\n\n")
        f.write(f"_{len(results)} of {n} samples succeeded_\n\n")
        for i, r in enumerate(results, 1):
            sscore = r.get("safety_score")
            f.write(f"## Sample {i} — `{r['item_id']}` (safety_score={sscore})\n\n")
            f.write(f"**Text:**\n\n{r['text']}\n\n")
            f.write(f"**Summary:**\n\n{r['summary']}\n\n")
            f.write("---\n\n")
    return md_path


def cmd_iterate(args: argparse.Namespace) -> None:
    """Sample N texts, generate summaries, save JSONL + sibling .md."""
    load_dotenv()
    cfg = load_config()
    sc = cfg.summaries

    n = args.n if args.n is not None else sc.n_samples
    seed = args.seed if args.seed is not None else sc.seed
    version = args.prompt_version or sc.prompt_version
    template = _load_prompt(version)

    items = sample_texts(
        n, seed=seed, max_tokens=sc.input_max_tokens,
        min_safety_score=args.min_safety_score,
        max_safety_score=args.max_safety_score,
    )
    logger.info(
        "Sampled {} items (seed={}, max_tokens={}, safety∈[{},{}])",
        len(items), seed, sc.input_max_tokens,
        args.min_safety_score, args.max_safety_score,
    )

    api_keys = {sc.model.endpoint: "OPENROUTER_API_KEY"}
    client, semaphore = make_api_client(
        sc.model.endpoint, sc.max_concurrent, api_keys=api_keys
    )
    sampling_params = resolve_sampling_params(sc.model.api_name, sc.model.alias)

    async def generate_one(item: dict) -> dict | None:
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": f"## Text\n\n{item['text']}"},
        ]
        try:
            raw, _, usage = await api_call(
                client,
                sc.model.api_name,
                messages,
                semaphore,
                thinking=sc.model.thinking,
                sampling_params=sampling_params,
                max_tokens=sc.api_max_tokens,
            )
        except Exception as e:
            logger.warning("Generation failed for {}: {}", item["item_id"], e)
            return None
        try:
            parsed = extract_json(raw)
            summary = parsed.get("summary", raw.strip())
        except Exception:
            summary = raw.strip()
        summary = truncate_to_max_tokens(summary, sc.summary_token_budget)
        return {
            "item_id": item["item_id"],
            "text": item["text"],
            "safety_score": item.get("safety_score"),
            "summary": summary,
            "raw_response": raw,
            "output_tokens": usage.get("output_tokens", 0),
            "prompt_version": version,
            "model_alias": sc.model.alias,
        }

    raw_results = run_concurrent(
        *[generate_one(it) for it in items], desc="Generating summaries"
    )
    results = [r for r in raw_results if r is not None]

    out_path = _resolve_out_path(
        cfg, sc.model.alias, version, seed, n,
        args.min_safety_score, args.max_safety_score, args.out,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    md_path = _write_md(out_path, results, cfg, version, seed, n)

    print(f"\n{'=' * 80}")
    print(f"  results: {len(results)}/{n} ok")
    print(f"  jsonl:   {out_path}")
    print(f"  human:   {md_path}")
    print(f"{'=' * 80}")
