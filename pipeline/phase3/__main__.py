"""Phase 3 CLI: eval-generators / eval-judges / rank-* / list-runs / failures.

Usage:
    uv run python -m pipeline.phase3 eval-generators [--run-id NAME] [overrides...]
    uv run python -m pipeline.phase3 eval-judges     [--run-id NAME] [overrides...]
    uv run python -m pipeline.phase3 rank-generators <run_id> [--json]
    uv run python -m pipeline.phase3 rank-judges     <run_id> [--json]
    uv run python -m pipeline.phase3 list-runs
    uv run python -m pipeline.phase3 failures        <run_id> [--category api|parse]

OmegaConf-style dotlist overrides work the same as in phase 2:
    uv run python -m pipeline.phase3 eval-generators phase3.generator_eval.n_items=20
"""

from __future__ import annotations

import datetime
import json
import sys

from pipeline.config import load_config
from pipeline.log import logger
from pipeline.phase3.eval_generators import _eval_root


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _split_run_id_and_overrides(args: list[str]) -> tuple[str | None, list[str]]:
    """Pull --run-id NAME out of args and return (run_id, remaining)."""
    run_id: str | None = None
    rest: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--run-id" and i + 1 < len(args):
            run_id = args[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    return run_id, rest


def cmd_eval_generators(args: list[str]) -> int:
    run_id, overrides = _split_run_id_and_overrides(args)
    cfg = load_config(overrides=overrides if overrides else None)
    if not run_id:
        run_id = f"gen_eval_{_now_iso()}"
    from pipeline.phase3.eval_generators import run_generator_eval

    logger.info("phase3 eval-generators run_id={}", run_id)
    run_generator_eval(cfg, run_id)
    print(f"\nDone. run_id={run_id}")
    return 0


def cmd_eval_judges(args: list[str]) -> int:
    run_id, overrides = _split_run_id_and_overrides(args)
    cfg = load_config(overrides=overrides if overrides else None)
    if not run_id:
        run_id = f"judge_eval_{_now_iso()}"
    from pipeline.phase3.eval_judges import run_judge_eval

    logger.info("phase3 eval-judges run_id={}", run_id)
    run_judge_eval(cfg, run_id)
    print(f"\nDone. run_id={run_id}")
    return 0


def cmd_rank_generators(args: list[str]) -> int:
    if not args:
        print("Usage: rank-generators <run_id> [--json]")
        return 2
    run_id = args[0]
    as_json = "--json" in args[1:]
    from pipeline.phase3 import rank as rank_mod

    rows = rank_mod.rank_generators(run_id)
    if as_json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(f"No generators in run {run_id}")
        return 0
    print(
        f"{'Generator':<40} {'n_ok':>6} {'mean':>6} {'accept':>8} "
        f"{'gen_api':>8} {'gen_parse':>10} {'jud_api':>8} {'jud_parse':>10}"
    )
    for r in rows:
        fr = r["failure_rates"]
        print(
            f"{r['generator']:<40} {r['n_succeeded']:>6} "
            f"{r['mean_aggregate']:>6.3f} {r['accept_rate']:>7.1%} "
            f"{fr['gen_api']:>7.1%} {fr['gen_parse']:>9.1%} "
            f"{fr['judge_api']:>7.1%} {fr['judge_parse']:>9.1%}"
        )
    return 0


def cmd_rank_judges(args: list[str]) -> int:
    if not args:
        print("Usage: rank-judges <run_id> [--json]")
        return 2
    run_id = args[0]
    as_json = "--json" in args[1:]
    from pipeline.phase3 import rank as rank_mod

    blocks = rank_mod.rank_judges(run_id)
    if as_json:
        print(json.dumps(blocks, indent=2))
        return 0
    for label, key in (("vs gold", "vs_gold"), ("vs human", "vs_human")):
        rows = blocks.get(key) or []
        print(f"\n=== judges {label} ===")
        if not rows:
            print("(empty)")
            continue
        print(
            f"{'Judge':<50} {'n_ok':>6} {'spearman':>9} {'pearson':>8} "
            f"{'conc':>6} {'kappa':>6} {'api%':>6} {'parse%':>7}"
        )
        for r in rows:
            fr = r.get("failure_rates", {})
            print(
                f"{r['judge']:<50} {r.get('n_succeeded', 0):>6} "
                f"{_fmt(r.get('spearman')):>9} {_fmt(r.get('pearson')):>8} "
                f"{_fmt(r.get('concordance')):>6} {_fmt(r.get('kappa')):>6} "
                f"{fr.get('api', 0.0):>5.1%} {fr.get('parse', 0.0):>6.1%}"
            )
    return 0


def cmd_list_runs(args: list[str]) -> int:
    cfg = load_config()
    root = _eval_root(cfg)
    if not root.exists():
        print(f"No phase 3 eval root at {root}")
        return 0
    rows: list[dict] = []
    for run_dir in sorted(root.iterdir()):
        meta_path = run_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "type": meta.get("type", "?"),
                "status": meta.get("status", "?"),
                "n_items": meta.get("n_items", "?"),
                "started_at": (meta.get("started_at") or "")[:19],
                "finished_at": (meta.get("finished_at") or "")[:19] or "—",
                "n_candidates": len(meta.get("candidates", [])),
            }
        )
    if not rows:
        print("No phase 3 runs.")
        return 0
    print(
        f"{'Run':<40} {'Type':<16} {'Status':<10} {'Items':>6} "
        f"{'Cands':>6} {'Started':<19}  {'Finished':<19}"
    )
    for r in rows:
        print(
            f"{r['run_id']:<40} {r['type']:<16} {r['status']:<10} "
            f"{str(r['n_items']):>6} {r['n_candidates']:>6} "
            f"{r['started_at']:<19}  {r['finished_at']:<19}"
        )
    return 0


def cmd_failures(args: list[str]) -> int:
    if not args:
        print("Usage: failures <run_id> [--category api|parse] [--stage NAME]")
        return 2
    run_id = args[0]
    category = None
    stage = None
    i = 1
    while i < len(args):
        if args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
            continue
        if args[i] == "--stage" and i + 1 < len(args):
            stage = args[i + 1]
            i += 2
            continue
        i += 1

    cfg = load_config()
    root = _eval_root(cfg)
    failures_dir = root / run_id / "failures"
    if not failures_dir.exists():
        print(f"No failures dir for run {run_id}")
        return 0
    n_total = 0
    for path in sorted(failures_dir.glob("*.jsonl")):
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if category:
            rows = [r for r in rows if r.get("category") == category]
        if stage:
            rows = [r for r in rows if r.get("stage") == stage]
        if not rows:
            continue
        print(f"\n=== {path.name} ({len(rows)} rows) ===")
        for r in rows[:50]:
            print(
                f"  {r.get('item_id', '?')} cat={r.get('category', '?')} "
                f"stage={r.get('stage', '?')} reason={r.get('reason', '?')} "
                f"attempt={r.get('attempt', '?')}"
            )
            raw = r.get("raw") or ""
            if raw:
                preview = raw[:200].replace("\n", " ")
                print(f"      raw: {preview}{'…' if len(raw) > 200 else ''}")
        if len(rows) > 50:
            print(f"  … {len(rows) - 50} more")
        n_total += len(rows)
    print(f"\nTotal failures shown: {n_total}")
    return 0


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{v:.3f}"
    except (TypeError, ValueError):
        return str(v)


_DISPATCH = {
    "eval-generators": cmd_eval_generators,
    "eval-judges": cmd_eval_judges,
    "rank-generators": cmd_rank_generators,
    "rank-judges": cmd_rank_judges,
    "list-runs": cmd_list_runs,
    "failures": cmd_failures,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    handler = _DISPATCH.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 2
    return handler(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
