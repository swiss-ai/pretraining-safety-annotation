"""Run the charter generator-eval on the curated ``edge-cases`` bench.

A single candidate generator (default: qwen3.6-35b-a3b, generator_reflection_v4,
language injection ON) is annotated on every edge-case paragraph in all 7 target
languages, then judged by the configured gold judge. The candidate's api/sampling/
endpoint/thinking are inherited from its existing entry in configs/config.yaml — only
the prompt + inject_language are overridden — and the live config file is never
written. Prints the per-language ranking when done.

Usage:
    uv run python -m scripts.run_edge_cases_eval
    uv run python -m scripts.run_edge_cases_eval --alias qwen3.6-35b-a3b --prompt generator_reflection_v4.md
"""

from __future__ import annotations

import argparse
import copy

from dotenv import load_dotenv

from pipeline.config import PROJECT_ROOT, load_config
from pipeline.charter.eval.benches import ensure_bench
from pipeline.charter.eval.eval_generators import run_generator_eval
from pipeline.charter.eval import rank as rank_mod

BENCH = "edge-cases"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="qwen3.6-35b-a3b")
    ap.add_argument("--prompt", default="generator_reflection_v4.md")
    ap.add_argument("--no-inject", action="store_true", help="disable per-language reflection-language injection")
    ap.add_argument("--max-concurrent", type=int, default=50)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--stage", choices=("generate", "judge"), default=None)
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config()
    ge = cfg.charter.eval.generator_eval

    base = next((c for c in ge.candidates if c.alias == args.alias), None)
    assert base is not None, (
        f"candidate alias {args.alias!r} not found in config; "
        f"available: {[c.alias for c in ge.candidates]}"
    )
    cand = copy.deepcopy(base)
    cand.prompt_reflection = args.prompt
    cand.inject_language = not args.no_inject

    ge.candidates = [cand]
    ge.bench = BENCH
    ge.max_concurrent = args.max_concurrent

    # Curated bench: the whole pool is the bench, so n_items == its row count.
    import pyarrow.parquet as pq

    from pipeline.charter.eval.benches import BENCH_DIR

    ensure_bench(BENCH)
    ge.n_items = pq.read_table(BENCH_DIR / f"{BENCH}.parquet").num_rows

    run_id = args.run_id or f"edge_cases_{cand.alias}_{cand.prompt_reflection.replace('.md','')}" + (
        "" if cand.inject_language else "_noinj"
    )
    print(
        f"bench={BENCH} n_items={ge.n_items} candidate={cand.alias} "
        f"prompt={cand.prompt_reflection} inject_language={cand.inject_language} "
        f"gold_judge={cfg.charter.eval.gold_judge.alias} run_id={run_id}"
    )

    run_generator_eval(cfg, run_id, stage=args.stage)

    for r in rank_mod.rank_generators(run_id):
        fr = r["failure_rates"]
        print(
            f"\n{r['generator']}  n_ok={r['n_succeeded']}  "
            f"mean={r['mean_aggregate']:.3f}  accept={r['accept_rate']:.1%}  "
            f"gen_api={fr['gen_api']:.1%} gen_parse={fr['gen_parse']:.1%} "
            f"jud_api={fr['judge_api']:.1%} jud_parse={fr['judge_parse']:.1%}"
        )
        for sub, d in sorted((r.get("by_subset") or {}).items()):
            print(f"    {sub:<5} n={d['n']:>3} mean={d['mean_aggregate']:.3f} accept={d['accept_rate']:.1%}")


if __name__ == "__main__":
    main()
