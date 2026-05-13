"""Phase 6 CLI: multi-turn charter-aware paired SFT generation.

Iteration (openrouter, login-node):
    uv run python -m pipeline.sft.multi_turn iterate --n 20

Scale-up (Alps SLURM):
    uv run python -m pipeline.sft.multi_turn materialize
    uv run python -m pipeline.sft.multi_turn submit
    uv run python -m pipeline.sft.multi_turn status
    uv run python -m pipeline.sft.multi_turn merge
    uv run python -m pipeline.sft.multi_turn export
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from pipeline.config import load_config
from pipeline.log import logger
from pipeline.sft.single_turn.data import sample_mix, SourcedPrompt


# ---------- Login-node (openrouter) commands ----------

def cmd_iterate(args: argparse.Namespace, _overrides=None) -> None:
    """Sample N seeds, run self-play multi-turn generation, print conversations."""
    from pipeline.api import make_api_client
    from pipeline.sft.single_turn.generate import ENDPOINT, MODEL, ALIAS, API_KEYS
    from pipeline.sft.multi_turn.generate import (
        generate_multiturn_one,
        render_multiturn_system_prompt,
        _load_prompt_file,
    )
    import random

    load_dotenv()
    seeds = sample_mix(
        n=args.n, seed=args.seed,
        exclude_sources=frozenset({"harmfulqa"}),
    )
    logger.info("sampled {} seeds", len(seeds))

    system_prompt = render_multiturn_system_prompt(args.base_version, args.mt_version)
    followup_system = _load_prompt_file("followup_user_v1.md")
    client, semaphore = make_api_client(ENDPOINT, max_concurrent=args.max_concurrent, api_keys=API_KEYS)
    rng = random.Random(args.seed)

    async def run():
        results = []
        for sp in seeds:
            result = await generate_multiturn_one(
                client=client,
                semaphore=semaphore,
                system_prompt=system_prompt,
                followup_system=followup_system,
                sp=sp,
                model=MODEL,
                alias=ALIAS,
                rng=rng,
                max_turns=args.max_turns,
            )
            results.append((sp, result))
        return results

    results = asyncio.run(run())

    out_path = Path(args.out) if args.out else Path("sft_multi_turn_iterate.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for sp, r in results:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for _, r in results if r and "error" not in r and not r.get("skip"))
    n_skip = sum(1 for _, r in results if r and r.get("skip"))
    n_err = sum(1 for _, r in results if r and "error" in r)
    n_none = sum(1 for _, r in results if r is None)

    print(f"\n{'='*80}")
    print(f"results: {n_ok} ok, {n_skip} skip, {n_err} errors, {n_none} too-short")
    print(f"saved to {out_path}")
    print(f"{'='*80}")

    for sp, r in results:
        print(f"\n{'─'*60}")
        print(f"[{sp.source}/{sp.source_id[:8]}] harm={sp.harm_category}")
        if r is None:
            print("  TOO SHORT (< 2 turns)")
            continue
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        if r.get("skip"):
            print(f"  SKIP: {r.get('analysis', '')}")
            continue
        print(f"  turns={r['n_turns']}  tokens={r.get('total_tokens', '?')}  pivot={r.get('is_pivot', False)}")
        for i, t in enumerate(r["turns"]):
            ft = t.get("flow_type") or "seed"
            print(f"\n  ── Turn {i+1} ({ft}) ──")
            print(f"  USER: {t['user'][:200]}{'...' if len(t['user'])>200 else ''}")
            print(f"  CITED:\n    {t['cited'][:300]}{'...' if len(t['cited'])>300 else ''}")
            print(f"  UNCITED:\n    {t['uncited'][:300]}{'...' if len(t['uncited'])>300 else ''}")


# ---------- Alps SLURM commands ----------

def _build_env_command(cfg) -> str:
    """Shell preamble that launches sglang and waits for health.

    Reuses sft.single_turn's sglang config (same model, same tuning).
    """
    sg = cfg.sft.single_turn.sglang
    output_dir = cfg.sft.multi_turn.output_dir
    model_path = sg.model_path or sg.hf_slug
    served_name = sg.hf_slug
    venv_activate = str(Path(sys.prefix) / "bin" / "activate")
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    pre_launch = (sg.pre_launch_cmds + "\n") if sg.pre_launch_cmds else ""
    extra_args = sg.extra_args or ""
    reasoning_parser_arg = f"--reasoning-parser {sg.reasoning_parser}" if sg.reasoning_parser else ""

    return textwrap.dedent(f"""\
        unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE
        export no_proxy="localhost,127.0.0.1,0.0.0.0,$no_proxy"
        export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$NO_PROXY"
        export SGLANG_API_KEY=none

        srun --nodes=1 --ntasks=1 \\
            --container-writable \\
            --environment={sg.env_toml} \\
            bash --norc --noprofile -c "
        set -ex
        export no_proxy=\\"0.0.0.0,\\$no_proxy\\"
        export NO_PROXY=\\"0.0.0.0,\\$NO_PROXY\\"
        export SGL_ENABLE_JIT_DEEPGEMM=\\"false\\"
        {pre_launch}python3 -m sglang.launch_server \\
            --model-path {model_path} \\
            --served-model-name {served_name} \\
            --port {sg.port} \\
            --host 0.0.0.0 \\
            --tp {sg.tp_size} \\
            --dp-size {sg.dp_size} \\
            --trust-remote-code \\
            {reasoning_parser_arg} \\
            {extra_args}
        " > {output_dir}/sglang_$SLURM_ARRAY_TASK_ID.log 2>&1 &
        SGLANG_PID=$!

        cleanup() {{
            kill $SGLANG_PID 2>/dev/null || true
            wait $SGLANG_PID 2>/dev/null || true
            pkill -f "sglang.launch_server" 2>/dev/null || true
        }}
        trap cleanup EXIT SIGTERM SIGINT

        echo "Waiting for sglang to start..."
        for i in $(seq 1 120); do
            kill -0 $SGLANG_PID 2>/dev/null || {{ echo "FATAL: sglang process died"; exit 1; }}
            if curl --noproxy '*' -sf http://localhost:{sg.port}/health > /dev/null 2>&1; then
                echo "sglang ready after $((i*10)) seconds"
                break
            fi
            if [ $i -eq 120 ]; then
                echo "FATAL: sglang failed to start after 20 minutes"
                exit 1
            fi
            sleep 10
        done

        export SGLANG_ENDPOINT=http://localhost:{sg.port}/v1
        source {venv_activate}
        export PYTHONPATH="{project_root}:${{PYTHONPATH:-}}"
    """)


class _ExclusiveSlurmExecutor:
    """Clariden patches: strip --mem-per-cpu, use --exclusive."""

    @staticmethod
    def create(**kwargs):
        from datatrove.executor.slurm import SlurmPipelineExecutor, launch_slurm_job

        executor = SlurmPipelineExecutor(**kwargs)
        _orig = executor.get_sbatch_args

        def _patched(max_array=1):
            args = _orig(max_array)
            args.pop("mem-per-cpu", None)
            args["exclusive"] = ""
            return args

        def _lightweight_merge_stats():
            venv_activate = str(Path(sys.prefix) / "bin" / "activate")
            project_root = str(Path(__file__).resolve().parent.parent.parent)
            stats_dir = executor.logging_dir.resolve_paths("stats")
            stats_out = executor.logging_dir.resolve_paths("stats.json")
            log_file = os.path.join(executor.slurm_logs_folder, "stats_%j.out")

            script = textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={executor.job_name}_stats
                #SBATCH --partition={executor.partition}
                #SBATCH --time=00:10:00
                #SBATCH --nodes=1
                #SBATCH --exclusive
                #SBATCH --output={log_file}
                #SBATCH --error={log_file}
                #SBATCH --dependency=afterany:{executor.job_id}
                #SBATCH --account={executor._sbatch_args.get("account", "")}

                source {venv_activate}
                export PYTHONPATH="{project_root}:${{PYTHONPATH:-}}"
                set -xe
                merge_stats {stats_dir} -o {stats_out}
            """)
            launch_slurm_job(script, executor.job_id_retriever)

        executor.get_sbatch_args = _patched
        executor.launch_merge_stats = _lightweight_merge_stats
        return executor


def _run_dir(cfg) -> Path:
    return Path(cfg.sft.multi_turn.output_dir)


def _prompts_path(cfg) -> Path:
    return _run_dir(cfg) / "prompts" / "prompts.parquet"


def _compute_n_tasks(cfg) -> int:
    return math.ceil(cfg.sft.multi_turn.total_rows / cfg.sft.multi_turn.rows_per_task)


def cmd_materialize(args: argparse.Namespace, overrides) -> None:
    """Sample seed prompts and write prompts.parquet."""
    from pipeline.sft.single_turn.prompts_writer import materialize_prompts

    load_dotenv()
    cfg = load_config(overrides)
    out = _prompts_path(cfg)
    fp = materialize_prompts(
        out, n=cfg.sft.multi_turn.total_rows, seed=cfg.sft.multi_turn.seed,
        exclude_sources=frozenset({"harmfulqa"}),
    )
    print(json.dumps({"prompts_path": str(out), "fingerprint": fp}, indent=2))


def cmd_submit(args: argparse.Namespace, overrides) -> None:
    """Submit the SLURM array job (auto-materialises if needed)."""
    from pipeline.sft.single_turn.prompts_writer import materialize_prompts
    from pipeline.sft.single_turn.reader import PromptsReader
    from pipeline.sft.multi_turn.slurm_generate import MultiTurnGenerator

    load_dotenv()
    cfg = load_config(overrides)

    prompts_path = _prompts_path(cfg)
    if not prompts_path.exists():
        logger.info("prompts.parquet missing — materialising first")
        materialize_prompts(
            prompts_path, n=cfg.sft.multi_turn.total_rows, seed=cfg.sft.multi_turn.seed,
            exclude_sources=frozenset({"harmfulqa"}),
        )

    n_tasks = _compute_n_tasks(cfg)
    logger.info(
        "phase6 submit: total_rows={}, rows_per_task={}, n_tasks={}",
        cfg.sft.multi_turn.total_rows, cfg.sft.multi_turn.rows_per_task, n_tasks,
    )

    run_dir = _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg_path = run_dir / "run_config.json"
    current_cfg = {
        "total_rows": cfg.sft.multi_turn.total_rows,
        "seed": cfg.sft.multi_turn.seed,
        "rows_per_task": cfg.sft.multi_turn.rows_per_task,
        "base_prompt_version": cfg.sft.multi_turn.base_prompt_version,
        "addendum_version": cfg.sft.multi_turn.addendum_version,
        "generator_alias": cfg.sft.multi_turn.generator_alias,
        "max_turns": cfg.sft.multi_turn.max_turns,
    }
    if run_cfg_path.exists():
        prev = json.loads(run_cfg_path.read_text())
        drift = {k: (prev.get(k), v) for k, v in current_cfg.items() if prev.get(k) != v}
        if drift:
            logger.error("run_config.json drift detected:")
            for k, (old, new) in drift.items():
                logger.error("  {}: {!r} -> {!r}", k, old, new)
            logger.error("delete {} to start a fresh run.", run_cfg_path)
            sys.exit(1)
    else:
        run_cfg_path.write_text(json.dumps(current_cfg, indent=2))

    pipeline = [
        PromptsReader(
            prompts_path=str(prompts_path),
            rows_per_task=cfg.sft.multi_turn.rows_per_task,
        ),
        MultiTurnGenerator(
            base_prompt_version=cfg.sft.multi_turn.base_prompt_version,
            addendum_version=cfg.sft.multi_turn.addendum_version,
            generator_alias=cfg.sft.multi_turn.generator_alias,
            output_dir=str(run_dir),
            max_concurrent_requests=cfg.sft.multi_turn.max_concurrent_requests,
            save_batch_size=cfg.sft.multi_turn.save_batch_size,
            max_retries_per_doc=cfg.sft.multi_turn.max_retries_per_doc,
            progress_interval=cfg.sft.multi_turn.progress_interval,
            max_turns=cfg.sft.multi_turn.max_turns,
            seed=cfg.sft.multi_turn.seed,
        ),
    ]

    env_command = _build_env_command(cfg)
    sl = cfg.sft.single_turn.slurm  # reuse sft.single_turn SLURM config

    executor = _ExclusiveSlurmExecutor.create(
        pipeline=pipeline,
        tasks=n_tasks,
        time=sl.time,
        partition=sl.partition,
        cpus_per_task=sl.cpus_per_task,
        gpus_per_task=cfg.sft.single_turn.sglang.tp_size * cfg.sft.single_turn.sglang.dp_size,
        workers=sl.workers,
        job_name="sft_multi_turn",
        env_command=env_command,
        sbatch_args={"account": sl.account},
        logging_dir=str(run_dir),
        skip_completed=True,
        with_srun=False,
    )
    executor.run()


def cmd_status(args: argparse.Namespace, overrides) -> None:
    """Show per-rank progress."""
    load_dotenv()
    cfg = load_config(overrides)
    n_tasks = _compute_n_tasks(cfg)
    run_dir = _run_dir(cfg)
    completions_dir = run_dir / "completions"

    completed = sum(1 for r in range(n_tasks) if (completions_dir / f"{r:05d}").exists())
    n_ok = n_skip = n_err = n_short = 0

    for r in range(n_tasks):
        results = run_dir / f"{r:05d}" / "results.jsonl"
        if results.exists():
            with results.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in rec:
                        n_err += 1
                    elif rec.get("skip"):
                        n_skip += 1
                    elif rec.get("too_short"):
                        n_short += 1
                    else:
                        n_ok += 1

    print(json.dumps({
        "tasks_completed": completed,
        "tasks_total": n_tasks,
        "conversations_ok": n_ok,
        "conversations_skipped_canary": n_skip,
        "conversations_too_short": n_short,
        "conversations_with_error": n_err,
        "conversations_target": cfg.sft.multi_turn.total_rows,
    }, indent=2))


def cmd_merge(args: argparse.Namespace, overrides) -> None:
    """Concatenate per-rank JSONLs into one results.jsonl."""
    from pipeline.sft.single_turn.merge import merge_shards

    load_dotenv()
    cfg = load_config(overrides)
    out = merge_shards(
        run_dir=_run_dir(cfg),
        n_tasks=_compute_n_tasks(cfg),
        expected_total=cfg.sft.multi_turn.total_rows,
        allow_missing=args.allow_missing,
    )
    print(f"merged → {out}")


def cmd_export(args: argparse.Namespace, overrides) -> None:
    """Export merged results.jsonl to HF parquet and upload."""
    from pipeline.sft.multi_turn.export import export_results

    load_dotenv()
    cfg = load_config(overrides)
    jsonl = _run_dir(cfg) / "results.jsonl"
    assert jsonl.exists(), f"merged results.jsonl missing — run merge first: {jsonl}"
    out_dir = _run_dir(cfg) / "export"
    hf_repo_id = getattr(cfg.sft.multi_turn, "hf_repo_id", None)
    stats = export_results(jsonl, out_dir, hf_repo_id=hf_repo_id)
    print(json.dumps(stats, indent=2))


# ---------- CLI dispatch ----------

def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipeline.sft.multi_turn")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_iter = sub.add_parser("iterate", help="Small-batch multi-turn generation for prompt iteration")
    p_iter.add_argument("--n", type=int, default=20)
    p_iter.add_argument("--seed", type=int, default=43)
    p_iter.add_argument("--base-version", default="v11")
    p_iter.add_argument("--mt-version", default="mt_v1")
    p_iter.add_argument("--max-concurrent", type=int, default=4)
    p_iter.add_argument("--max-turns", type=int, default=5)
    p_iter.add_argument("--out", default=None)
    p_iter.set_defaults(func=cmd_iterate, needs_overrides=False)

    p_mat = sub.add_parser("materialize", help="Sample + write prompts.parquet (login node)")
    p_mat.set_defaults(func=cmd_materialize, needs_overrides=True)

    p_sub = sub.add_parser("submit", help="Submit SLURM array")
    p_sub.set_defaults(func=cmd_submit, needs_overrides=True)

    p_st = sub.add_parser("status", help="Show per-rank progress")
    p_st.set_defaults(func=cmd_status, needs_overrides=True)

    p_mer = sub.add_parser("merge", help="Concatenate per-rank JSONLs")
    p_mer.add_argument("--allow-missing", action="store_true")
    p_mer.set_defaults(func=cmd_merge, needs_overrides=True)

    p_exp = sub.add_parser("export", help="Export merged results to HF parquet and upload")
    p_exp.set_defaults(func=cmd_export, needs_overrides=True)

    args, remaining = parser.parse_known_args()
    overrides = remaining if getattr(args, "needs_overrides", False) else None
    args.func(args, overrides)


if __name__ == "__main__":
    main()
