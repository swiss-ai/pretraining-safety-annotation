"""Minimal CLI tools for the summary improver agent.

Usage:
    uv run python -m pipeline.summaries.tools run_batch --model glm-4.5-air --n 50
    uv run python -m pipeline.summaries.tools results <run_id>
    uv run python -m pipeline.summaries.tools trend --model glm-4.5-air
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

from pipeline.api import api_call, extract_json, make_api_client, run_concurrent
from pipeline.config import load_config, resolve_prompt_path
from pipeline.data import sample_texts
from pipeline.log import logger
from pipeline.summaries import init_summary_prompts
from pipeline.summaries.storage import (
    load_summary_items,
    load_summary_runs,
    save_summary_items,
    save_summary_run,
)
from pipeline.tokenizer import truncate_to_max_tokens

SUMMARY_TOKEN_BUDGET = 128


def _generate_and_judge(
    items: list[dict],
    gen_prompt_path,
    judge_prompt_path,
    gen_model_cfg,
    judge_model_cfg,
    cfg,
    run_id: str,
) -> list[dict]:
    """Generate summaries and judge them in two async batches."""
    gen_template = gen_prompt_path.read_text(encoding="utf-8")
    judge_template = judge_prompt_path.read_text(encoding="utf-8")
    accept_threshold = cfg.phase2.scoring.accept_threshold

    client, semaphore = make_api_client(
        cfg.phase2.endpoint, cfg.phase2.iteration.max_concurrent
    )

    # --- Generate ---
    async def generate_one(item):
        messages = [
            {"role": "system", "content": gen_template},
            {"role": "user", "content": f"## Text\n\n{item['text']}"},
        ]
        try:
            t0 = time.monotonic()
            raw, _, usage = await api_call(
                client,
                gen_model_cfg.api_name,
                messages,
                semaphore,
                thinking=gen_model_cfg.thinking,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            try:
                parsed = extract_json(raw)
                summary = parsed.get("summary", raw.strip())
            except Exception:
                summary = raw.strip()
            summary = truncate_to_max_tokens(summary, SUMMARY_TOKEN_BUDGET)
            return {
                **item,
                "summary": summary,
                "raw_gen_response": raw,
                "gen_latency_ms": latency_ms,
                "gen_tokens": usage.get("output_tokens", 0),
            }
        except Exception as e:
            logger.warning("Gen failed for {}: {}", item["item_id"], e)
            return None

    gen_results = run_concurrent(
        *[generate_one(item) for item in items], desc="Generating summaries"
    )
    generated = [r for r in gen_results if r is not None]
    if len(generated) < len(items):
        logger.warning(
            "Generation: {}/{} failed", len(items) - len(generated), len(items)
        )

    # --- Judge ---
    judge_system = judge_template.replace(
        "{accept_threshold}", str(accept_threshold)
    )

    async def judge_one(item):
        messages = [
            {"role": "system", "content": judge_system},
            {
                "role": "user",
                "content": (
                    f"## Source Text\n\n{item['text']}\n\n"
                    f"## Summary to Judge\n\n{item['summary']}"
                ),
            },
        ]
        try:
            t0 = time.monotonic()
            raw, _, usage = await api_call(
                client,
                judge_model_cfg.api_name,
                messages,
                semaphore,
                thinking=judge_model_cfg.thinking,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            parsed = extract_json(raw)
            scores = parsed["scores"]
            aggregate = sum(scores.values()) / len(scores)
            return {
                **item,
                "run_id": run_id,
                "scores": scores,
                "aggregate": aggregate,
                "judge_reasoning": parsed.get("reasoning", ""),
                "raw_judge_response": raw,
                "judge_latency_ms": latency_ms,
                "judge_tokens": usage.get("output_tokens", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.warning("Judge failed for {}: {}", item["item_id"], e)
            return None

    judge_results = run_concurrent(
        *[judge_one(item) for item in generated], desc="Judging summaries"
    )
    judged = [r for r in judge_results if r is not None]
    if len(judged) < len(generated):
        logger.warning(
            "Judging: {}/{} failed", len(generated) - len(judged), len(generated)
        )

    return judged


def run_summary_batch(
    gen_alias: str,
    judge_alias: str | None = None,
    n: int = 50,
    seed: int = 42,
    source: str = "improve",
) -> tuple[list[dict], str]:
    """Run a full generate+judge batch, save results, return (judged, run_id).

    Shared by both the improver CLI (tools.py) and the benchmark CLI.
    """
    cfg = load_config()
    if judge_alias is None:
        judge_alias = cfg.phase2.judge_models[0].alias

    init_summary_prompts(gen_alias)
    init_summary_prompts(judge_alias)

    gen_prompt_path = resolve_prompt_path("summary_latest.md", gen_alias)
    judge_prompt_path = resolve_prompt_path("summary_judge_latest.md", judge_alias)

    gen_model_cfg = next(m for m in cfg.phase2.generator_models if m.alias == gen_alias)
    judge_model_cfg = next(
        m for m in cfg.phase2.judge_models if m.alias == judge_alias
    )

    items = sample_texts(n, seed=seed, max_tokens=1920)
    run_id = str(uuid4())

    judged = _generate_and_judge(
        items, gen_prompt_path, judge_prompt_path,
        gen_model_cfg, judge_model_cfg, cfg, run_id,
    )

    save_summary_items(judged)

    mean_score = (
        sum(j["aggregate"] for j in judged) / len(judged) if judged else 0.0
    )
    save_summary_run({
        "run_id": run_id,
        "generator_model": gen_alias,
        "judge_model": judge_alias,
        "gen_prompt": gen_prompt_path.name,
        "judge_prompt": judge_prompt_path.name,
        "n_items": len(judged),
        "mean_score": mean_score,
        "source": source,
    })

    return judged, run_id


def cmd_run_batch(args):
    """Generate + judge one batch, print run_id."""
    cfg = load_config()
    alias = args.model or cfg.phase2.generator_models[0].alias
    judged, run_id = run_summary_batch(alias, n=args.n, seed=args.seed)
    threshold = cfg.phase2.scoring.accept_threshold
    mean_score = sum(j["aggregate"] for j in judged) / len(judged) if judged else 0.0
    n_accepted = sum(1 for j in judged if j["aggregate"] >= threshold)
    print(f"run_id: {run_id}")
    print(f"items: {len(judged)}, accepted: {n_accepted}, mean: {mean_score:.2f}")


def _parse_scores(item: dict) -> dict:
    """Parse scores from a summary item, handling both str and dict."""
    s = item["scores"]
    return json.loads(s) if isinstance(s, str) else s


def cmd_results(args):
    """Show scores table + failures for a run."""
    items = load_summary_items(args.run_id)
    if not items:
        print(f"No items found for run_id={args.run_id}")
        return

    cfg = load_config()
    threshold = cfg.phase2.scoring.accept_threshold

    # Parse scores once per item
    parsed = [(item, _parse_scores(item)) for item in items]

    print(f"Run: {args.run_id} ({len(items)} items)\n")
    print(f"{'ID':>16s}  {'Acc':>4s} {'Cov':>4s} {'Spec':>4s} {'Flu':>4s} {'Agg':>5s} {'Decision':>8s}")
    print("-" * 65)

    for item, scores in parsed:
        agg = item["aggregate"]
        floor = any(v <= 2 for v in scores.values())
        decision = "reject" if floor or agg < threshold else "accept"
        print(
            f"{item['item_id']:>16s}  "
            f"{scores.get('accuracy', 0):4.1f} "
            f"{scores.get('coverage', 0):4.1f} "
            f"{scores.get('specificity', 0):4.1f} "
            f"{scores.get('fluency', 0):4.1f} "
            f"{agg:5.2f} "
            f"{decision:>8s}"
        )

    aggs = [item["aggregate"] for item in items]
    mean = sum(aggs) / len(aggs)
    n_acc = sum(1 for a in aggs if a >= threshold)
    print(f"\nMean: {mean:.2f}, Accepted: {n_acc}/{len(items)}")

    failures = [
        (item, scores) for item, scores in parsed
        if item["aggregate"] < threshold or any(v <= 2 for v in scores.values())
    ]
    if failures:
        print(f"\n--- Failures ({len(failures)}) ---\n")
        for item, scores in failures[:10]:
            print(f"  {item['item_id']}: {scores}")
            reasoning = item.get("judge_reasoning", "")
            if reasoning:
                print(f"    Reasoning: {reasoning[:200]}")
            print(f"    Summary: {item['summary'][:150]}...")
            print()


def cmd_trend(args):
    """Show score trend across runs for a model."""
    runs = load_summary_runs(model=args.model)
    if not runs:
        print(f"No runs found for model={args.model}")
        return

    print(f"{'Run ID':>36s}  {'Items':>5s} {'Mean':>5s} {'Source':>10s} {'Prompt':>20s} {'Timestamp'}")
    print("-" * 110)
    for run in runs:
        print(
            f"{run['run_id']:>36s}  "
            f"{run['n_items']:5d} "
            f"{run['mean_score'] or 0:5.2f} "
            f"{run['source']:>10s} "
            f"{run['gen_prompt']:>20s} "
            f"{run['timestamp']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summary pipeline tools for improver agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser("run_batch", help="Generate + judge a batch")
    p_batch.add_argument("--model", type=str, default=None)
    p_batch.add_argument("--n", type=int, default=50)
    p_batch.add_argument("--seed", type=int, default=42)

    p_results = sub.add_parser("results", help="Show results for a run")
    p_results.add_argument("run_id", type=str)

    p_trend = sub.add_parser("trend", help="Score trend across runs")
    p_trend.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    if args.command == "run_batch":
        cmd_run_batch(args)
    elif args.command == "results":
        cmd_results(args)
    elif args.command == "trend":
        cmd_trend(args)


if __name__ == "__main__":
    main()
