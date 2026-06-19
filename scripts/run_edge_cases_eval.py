"""Run the charter generator-eval for ONE candidate on a chosen bench.

Defaults to the curated ``edge-cases`` bench (full-text, reflection-after-text,
all items). Point ``--bench`` at ``dclm-en`` / ``fw2-multi`` (with ``--n-items``)
to validate that a prompt change does not regress the normal benchmarks.

The candidate's api/sampling/endpoint/thinking are inherited from its existing
entry in configs/config.yaml — only the prompt + inject_language are overridden —
and the live config file is never written. Prints the per-language ranking.

Usage:
    uv run python scripts/run_edge_cases_eval.py                                  # edge-cases, qwen v4
    uv run python scripts/run_edge_cases_eval.py --alias gemma-4-31b --prompt generator_reflection_v13.md
    uv run python scripts/run_edge_cases_eval.py --bench dclm-en  --n-items 600 --alias qwen3.6-35b-a3b --prompt generator_reflection_v5.md
    uv run python scripts/run_edge_cases_eval.py --bench fw2-multi --n-items 600 --alias qwen3.6-35b-a3b --prompt generator_reflection_v4.md
"""

from __future__ import annotations

import argparse
import copy

from dotenv import load_dotenv

from pipeline.config import PROJECT_ROOT, load_config
from pipeline.charter.eval.benches import BENCH_DIR, ensure_bench, get_bench
from pipeline.charter.eval.eval_generators import run_generator_eval
from pipeline.charter.eval import rank as rank_mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="edge-cases")
    ap.add_argument("--n-items", type=int, default=None, help="ignored for full-text benches (uses all rows)")
    ap.add_argument("--alias", default="qwen3.6-35b-a3b")
    ap.add_argument("--prompt", default="generator_reflection_v4.md")
    ap.add_argument("--api-name", default=None, help="override the candidate's api_name (e.g. a swissai deployment tag)")
    ap.add_argument("--endpoint", default=None, help="override the candidate's endpoint (e.g. https://api.swissai.svc.cscs.ch/v1)")
    ap.add_argument("--judge-api-name", default=None, help="override the gold judge api_name (e.g. a rotated swissai GLM tag)")
    ap.add_argument("--no-inject", action="store_true", help="disable per-language reflection-language injection")
    ap.add_argument("--max-concurrent", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--stage", choices=("generate", "judge"), default=None)
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config()
    if args.judge_api_name:
        cfg.charter.eval.gold_judge.api_name = args.judge_api_name
    ge = cfg.charter.eval.generator_eval

    base = next((c for c in ge.candidates if c.alias == args.alias), None)
    assert base is not None, (
        f"candidate alias {args.alias!r} not found in config; "
        f"available: {[c.alias for c in ge.candidates]}"
    )
    cand = copy.deepcopy(base)
    cand.prompt_reflection = args.prompt
    cand.inject_language = not args.no_inject
    if args.api_name:
        cand.api_name = args.api_name
    if args.endpoint:
        cand.endpoint = args.endpoint

    ge.candidates = [cand]
    ge.bench = args.bench
    ge.max_concurrent = args.max_concurrent
    ge.seed = args.seed

    import pyarrow.parquet as pq

    ensure_bench(args.bench)
    if get_bench(args.bench).full_text:
        # Curated bench: the whole pool is the bench.
        ge.n_items = pq.read_table(BENCH_DIR / f"{args.bench}.parquet").num_rows
    else:
        assert args.n_items, f"--n-items is required for bench {args.bench!r}"
        ge.n_items = args.n_items

    run_id = args.run_id or (
        f"{args.bench}_{cand.alias}_{cand.prompt_reflection.replace('.md', '')}"
        + ("" if cand.inject_language else "_noinj")
    )
    print(
        f"bench={args.bench} n_items={ge.n_items} seed={ge.seed} candidate={cand.alias} "
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
