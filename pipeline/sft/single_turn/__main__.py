"""SFT single-turn CLI: prompt iteration + scale-up generation on Alps SLURM.

Iteration commands (openrouter, login-node):
    uv run python -m pipeline.sft.single_turn iterate --n 20 --version v6
    uv run python -m pipeline.sft.single_turn generate --n 100 --version v6

Scale-up commands (Alps SLURM, mirrors charter.scale):
    uv run python -m pipeline.sft.single_turn materialize  # materialize prompts.parquet
    uv run python -m pipeline.sft.single_turn submit       # submit SLURM array
    uv run python -m pipeline.sft.single_turn status       # show progress
    uv run python -m pipeline.sft.single_turn merge        # combine per-rank JSONLs
    uv run python -m pipeline.sft.single_turn export       # export to HF parquet + upload to Hub
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
from pipeline.sft.single_turn.data import sample_mix
from pipeline.sft.single_turn.generate import (
    generate_batch,
    generate_streaming,
    save_results,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


# ---------- Login-node (openrouter) commands ----------

def cmd_iterate(args: argparse.Namespace, _overrides=None) -> None:
    """Sample N prompts, generate paired responses, save JSONL, print summary."""
    load_dotenv()
    prompts = sample_mix(n=args.n, seed=args.seed)
    logger.info("sampled {} prompts", len(prompts))

    results = asyncio.run(generate_batch(
        prompts, prompt_version=args.version, max_concurrent=args.max_concurrent,
    ))
    out_path = Path(args.out) if args.out else DEFAULT_RUNS_DIR / f"{args.version}_seed{args.seed}_n{args.n}.jsonl"
    save_results(results, out_path)

    n_ok = sum("error" not in r and not r.get("skip") for r in results)
    n_skip = sum(1 for r in results if r.get("skip"))
    n_err = sum(1 for r in results if "error" in r)
    print(f"\n{'='*80}\nresults: {n_ok} ok, {n_skip} skip, {n_err} errors. saved to {out_path}\n{'='*80}")
    for r in results:
        print(f"\n--- [{r['source']}/{r['source_id'][:8]}] ---")
        print(f"USER: {r['user'][:280]}{'...' if len(r['user'])>280 else ''}")
        if "error" in r:
            print(f"ERROR: {r['error']}")
            continue
        if r.get("skip"):
            print(f"SKIP: {r.get('analysis', '')}")
            continue
        print(f"CITED:\n  {r['cited']}")
        print(f"UNCITED:\n  {r['uncited']}")


def cmd_generate(args: argparse.Namespace, _overrides=None) -> None:
    """Scale-up generation via openrouter (login-node, streaming, resumable)."""
    load_dotenv()
    prompts = sample_mix(n=args.n, seed=args.seed)
    out_path = Path(args.out) if args.out else DEFAULT_RUNS_DIR / f"{args.version}_seed{args.seed}_n{args.n}.jsonl"
    n_ok, n_err = asyncio.run(generate_streaming(
        prompts,
        prompt_version=args.version,
        out_path=out_path,
        max_concurrent=args.max_concurrent,
        progress_every=args.progress_every,
    ))
    print(f"\ngenerated this run: {n_ok} ok, {n_err} err. file: {out_path}")


# ---------- Alps SLURM commands ----------

def _build_env_command(cfg) -> str:
    """Shell preamble that launches sglang and waits for health.

    Mirrors ``pipeline/charter/scale/__main__.py:_build_env_command`` exactly —
    same trap, same health-check loop, same SGLANG_ENDPOINT export.
    """
    sg = cfg.sft.single_turn.sglang
    output_dir = cfg.sft.single_turn.output_dir
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
    """Same Clariden patches as charter.scale: strip --mem-per-cpu, use --exclusive,
    and replace launch_merge_stats with a lightweight job (no sglang, no GPUs)."""

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
    """The single-run output dir for sft.single_turn (no run_name multiplexing)."""
    return Path(cfg.sft.single_turn.output_dir)


def _prompts_path(cfg) -> Path:
    """Path to the materialised prompts.parquet under output_dir/prompts/."""
    return _run_dir(cfg) / "prompts" / "prompts.parquet"


def _compute_n_tasks(cfg) -> int:
    """Number of array tasks: ceil(n / rows_per_task)."""
    return math.ceil(cfg.sft.single_turn.total_rows / cfg.sft.single_turn.rows_per_task)


def cmd_materialize(args: argparse.Namespace, overrides) -> None:
    """Sample prompts on the login node and write prompts.parquet."""
    from pipeline.sft.single_turn.prompts_writer import materialize_prompts

    load_dotenv()
    cfg = load_config(overrides)
    out = _prompts_path(cfg)
    fp = materialize_prompts(out, n=cfg.sft.single_turn.total_rows, seed=cfg.sft.single_turn.seed)
    print(json.dumps({"prompts_path": str(out), "fingerprint": fp}, indent=2))


def cmd_submit(args: argparse.Namespace, overrides) -> None:
    """Submit the SLURM array job (auto-materialises prompts if needed)."""
    from pipeline.sft.single_turn.prompts_writer import materialize_prompts
    from pipeline.sft.single_turn.reader import PromptsReader
    from pipeline.sft.single_turn.slurm_generate import PairedGenerator

    load_dotenv()
    cfg = load_config(overrides)

    prompts_path = _prompts_path(cfg)
    if not prompts_path.exists():
        logger.info("prompts.parquet missing — materialising first")
        materialize_prompts(prompts_path, n=cfg.sft.single_turn.total_rows, seed=cfg.sft.single_turn.seed)

    n_tasks = _compute_n_tasks(cfg)
    logger.info(
        "sft.single_turn submit: total_rows={}, rows_per_task={}, n_tasks={}, prompt={}",
        cfg.sft.single_turn.total_rows, cfg.sft.single_turn.rows_per_task, n_tasks, cfg.sft.single_turn.prompt_version,
    )

    run_dir = _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Lock run config: any drift across resubmits silently mixes data,
    # so check all generation-relevant fields, not just rows_per_task.
    run_cfg_path = run_dir / "run_config.json"
    current_cfg = {
        "total_rows": cfg.sft.single_turn.total_rows,
        "seed": cfg.sft.single_turn.seed,
        "rows_per_task": cfg.sft.single_turn.rows_per_task,
        "prompt_version": cfg.sft.single_turn.prompt_version,
        "generator_alias": cfg.sft.single_turn.generator_alias,
        "hf_slug": cfg.sft.single_turn.sglang.hf_slug,
    }
    if run_cfg_path.exists():
        prev = json.loads(run_cfg_path.read_text())
        drift = {k: (prev.get(k), v) for k, v in current_cfg.items() if prev.get(k) != v}
        if drift:
            logger.error("run_config.json drift detected — would corrupt resume:")
            for k, (old, new) in drift.items():
                logger.error("  {}: {!r} -> {!r}", k, old, new)
            logger.error("delete {} to start a fresh run.", run_cfg_path)
            sys.exit(1)
    else:
        run_cfg_path.write_text(json.dumps(current_cfg, indent=2))

    pipeline = [
        PromptsReader(
            prompts_path=str(prompts_path),
            rows_per_task=cfg.sft.single_turn.rows_per_task,
        ),
        PairedGenerator(
            prompt_version=cfg.sft.single_turn.prompt_version,
            generator_alias=cfg.sft.single_turn.generator_alias,
            output_dir=str(run_dir),
            max_concurrent_requests=cfg.sft.single_turn.max_concurrent_requests,
            save_batch_size=cfg.sft.single_turn.save_batch_size,
            max_retries_per_doc=cfg.sft.single_turn.max_retries_per_doc,
            progress_interval=cfg.sft.single_turn.progress_interval,
        ),
    ]

    env_command = _build_env_command(cfg)
    sl = cfg.sft.single_turn.slurm

    executor = _ExclusiveSlurmExecutor.create(
        pipeline=pipeline,
        tasks=n_tasks,
        time=sl.time,
        partition=sl.partition,
        cpus_per_task=sl.cpus_per_task,
        gpus_per_task=cfg.sft.single_turn.sglang.tp_size * cfg.sft.single_turn.sglang.dp_size,
        workers=sl.workers,
        job_name="sft_single_turn",
        env_command=env_command,
        sbatch_args={"account": sl.account},
        logging_dir=str(run_dir),
        skip_completed=True,
        with_srun=False,
    )
    executor.run()


def cmd_status(args: argparse.Namespace, overrides) -> None:
    """Aggregate per-rank progress."""
    load_dotenv()
    cfg = load_config(overrides)
    n_tasks = _compute_n_tasks(cfg)
    run_dir = _run_dir(cfg)
    completions_dir = run_dir / "completions"

    completed = sum(
        1 for r in range(n_tasks)
        if (completions_dir / f"{r:05d}").exists()
    )

    n_ok = 0
    n_skip = 0
    n_with_error = 0
    n_failed = 0
    for r in range(n_tasks):
        rd = run_dir / f"{r:05d}"
        results = rd / "results.jsonl"
        failures = rd / "failures.jsonl"
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
                        n_with_error += 1
                    elif rec.get("skip"):
                        n_skip += 1
                    else:
                        n_ok += 1
        if failures.exists():
            with failures.open() as f:
                for _ in f:
                    n_failed += 1

    print(json.dumps({
        "tasks_completed": completed,
        "tasks_total": n_tasks,
        "rows_ok": n_ok,
        "rows_skipped_canary": n_skip,
        "rows_with_error_in_results": n_with_error,
        "rows_in_failures": n_failed,
        "rows_target": cfg.sft.single_turn.total_rows,
    }, indent=2))


def cmd_merge(args: argparse.Namespace, overrides) -> None:
    """Concatenate per-rank JSONLs into one results.jsonl."""
    from pipeline.sft.single_turn.merge import merge_shards

    load_dotenv()
    cfg = load_config(overrides)
    out = merge_shards(
        run_dir=_run_dir(cfg),
        n_tasks=_compute_n_tasks(cfg),
        expected_total=cfg.sft.single_turn.total_rows,
        allow_missing=args.allow_missing,
    )
    print(f"merged → {out}")


def cmd_export(args: argparse.Namespace, overrides) -> None:
    """Export merged results.jsonl to a HF parquet dataset and upload to Hub."""
    from pipeline.sft.single_turn.export import export_results

    load_dotenv()
    cfg = load_config(overrides)
    jsonl = _run_dir(cfg) / "results.jsonl"
    assert jsonl.exists(), f"merged results.jsonl missing — run merge first: {jsonl}"
    out_dir = _run_dir(cfg) / "export"
    hf_repo_id = getattr(cfg.sft.single_turn, "hf_repo_id", None)
    stats = export_results(jsonl, out_dir, hf_repo_id=hf_repo_id)
    print(json.dumps(stats, indent=2))


def cmd_rerun(args: argparse.Namespace, overrides) -> None:
    """Re-submit incomplete ranks. A rank needs a rerun if any of:
    - it has a non-empty failures.jsonl AND a marker (some retries hit the cap),
    - it has no marker AND its results.jsonl is short (OOM / walltime kill,
      typically leaves no failures file behind).
    Last rank's expected count is the remainder, not rows_per_task.
    """
    load_dotenv()
    cfg = load_config(overrides)
    n_tasks = _compute_n_tasks(cfg)
    run_dir = _run_dir(cfg)
    completions_dir = run_dir / "completions"
    rpt = cfg.sft.single_turn.rows_per_task
    total = cfg.sft.single_turn.total_rows

    cleared = 0
    requeued_short = 0
    for r in range(n_tasks):
        rs = f"{r:05d}"
        rd = run_dir / rs
        failures = rd / "failures.jsonl"
        results = rd / "results.jsonl"
        marker = completions_dir / rs
        expected = rpt if r < n_tasks - 1 else total - r * rpt

        if failures.exists() and failures.stat().st_size > 0 and marker.exists():
            marker.unlink()
            cleared += 1
            logger.info("cleared marker for rank {} (has failures)", r)
            continue

        if not marker.exists():
            n_results = sum(1 for _ in results.open()) if results.exists() else 0
            if n_results < expected:
                requeued_short += 1
                logger.info(
                    "rank {} marker missing and results short ({}/{}) — will be re-attempted",
                    r, n_results, expected,
                )
    logger.info(
        "cleared {} markers; {} ranks have missing markers + short results",
        cleared, requeued_short,
    )
    if cleared > 0 or requeued_short > 0 or args.force:
        cmd_submit(args, overrides)


# ---------- CLI dispatch ----------

def main() -> None:
    """Parse args and dispatch."""
    parser = argparse.ArgumentParser(prog="python -m pipeline.sft.single_turn")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Login-node iteration
    p_iter = sub.add_parser("iterate", help="Small-batch generation for prompt iteration")
    p_iter.add_argument("--n", type=int, default=20)
    p_iter.add_argument("--seed", type=int, default=42)
    p_iter.add_argument("--version", default="v6")
    p_iter.add_argument("--max-concurrent", type=int, default=8)
    p_iter.add_argument("--out", default=None)
    p_iter.set_defaults(func=cmd_iterate, needs_overrides=False)

    p_gen = sub.add_parser("generate", help="Login-node streaming generation via openrouter")
    p_gen.add_argument("--n", type=int, required=True)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--version", default="v6")
    p_gen.add_argument("--max-concurrent", type=int, default=200)
    p_gen.add_argument("--progress-every", type=int, default=100)
    p_gen.add_argument("--out", default=None)
    p_gen.set_defaults(func=cmd_generate, needs_overrides=False)

    # Alps SLURM
    p_mat = sub.add_parser("materialize", help="Sample + write prompts.parquet (login node)")
    p_mat.set_defaults(func=cmd_materialize, needs_overrides=True)

    p_sub = sub.add_parser("submit", help="Submit SLURM array (auto-materialises if needed)")
    p_sub.set_defaults(func=cmd_submit, needs_overrides=True)

    p_st = sub.add_parser("status", help="Show per-rank progress")
    p_st.set_defaults(func=cmd_status, needs_overrides=True)

    p_mer = sub.add_parser("merge", help="Concatenate per-rank JSONLs into results.jsonl")
    p_mer.add_argument("--allow-missing", action="store_true")
    p_mer.set_defaults(func=cmd_merge, needs_overrides=True)

    p_exp = sub.add_parser("export", help="Export merged results to HF parquet dataset and upload to Hub")
    p_exp.set_defaults(func=cmd_export, needs_overrides=True)

    p_re = sub.add_parser("rerun", help="Re-submit ranks with failures")
    p_re.add_argument("--force", action="store_true")
    p_re.set_defaults(func=cmd_rerun, needs_overrides=True)

    args, remaining = parser.parse_known_args()
    overrides = remaining if getattr(args, "needs_overrides", False) else None
    args.func(args, overrides)


if __name__ == "__main__":
    main()
