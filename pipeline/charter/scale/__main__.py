"""Charter scale CLI: submit, merge, status, rerun.

Usage:
    uv run python -m pipeline.charter.scale submit --run reflections [overrides...]
    uv run python -m pipeline.charter.scale status --run reflections
    uv run python -m pipeline.charter.scale merge  --run reflections
    uv run python -m pipeline.charter.scale rerun  --run reflections [overrides...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

from pipeline.config import load_config
from pipeline.log import logger


def _build_env_command(cfg) -> str:
    """Build the shell preamble that launches sglang and waits for health."""
    sg = cfg.charter.scale.sglang
    output_dir = cfg.charter.scale.output_dir

    model_path = sg.model_path or sg.hf_slug
    served_name = sg.hf_slug

    # Detect venv activate path and project root
    venv_activate = str(Path(sys.prefix) / "bin" / "activate")
    project_root = str(Path(__file__).resolve().parent.parent.parent)

    pre_launch = ""
    if sg.pre_launch_cmds:
        pre_launch = sg.pre_launch_cmds + "\n"

    extra_args = sg.extra_args or ""
    reasoning_parser_arg = f"--reasoning-parser {sg.reasoning_parser}" if sg.reasoning_parser else ""

    env_command = textwrap.dedent(f"""\
        # Clear inherited CPU binding
        unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE

        # Proxy bypass for localhost
        export no_proxy="localhost,127.0.0.1,0.0.0.0,$no_proxy"
        export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$NO_PROXY"

        # Dummy API key for unauthenticated local sglang
        export SGLANG_API_KEY=none

        # Launch sglang inside container, in background
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

        # Cleanup trap (|| true prevents set -e from turning a killed
        # sglang's non-zero wait status into a batch-level failure)
        cleanup() {{
            kill $SGLANG_PID 2>/dev/null || true
            wait $SGLANG_PID 2>/dev/null || true
            pkill -f "sglang.launch_server" 2>/dev/null || true
        }}
        trap cleanup EXIT SIGTERM SIGINT

        # Wait for sglang health (with liveness check)
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

        # Activate Python venv for the generation pipeline
        source {venv_activate}

        # Add project root to PYTHONPATH so `pipeline` module is importable
        export PYTHONPATH="{project_root}:${{PYTHONPATH:-}}"
    """)
    return env_command


def _get_total_rows(sidecar_path: str) -> int:
    """Read the total row count from the sidecar parquet metadata."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(sidecar_path)
    return pf.metadata.num_rows


def _compute_n_tasks(cfg) -> tuple[int, int]:
    """Compute (effective_total_rows, n_tasks) from config."""
    total_rows = _get_total_rows(cfg.charter.scale.sidecar_path)
    if cfg.charter.scale.max_rows > 0:
        total_rows = min(total_rows, cfg.charter.scale.max_rows)
    n_tasks = math.ceil(total_rows / cfg.charter.scale.rows_per_task)
    return total_rows, n_tasks


class _ExclusiveSlurmExecutor:
    """Wraps SlurmPipelineExecutor for CSCS Clariden.

    Clariden nodes reject --mem-per-cpu (memory is not allocatable
    per-cpu on GH200). Datatrove always emits it, so we patch
    get_sbatch_args to remove it and use --exclusive instead.

    Also patches launch_merge_stats so the stats-merge dependent job
    uses a minimal env (no sglang, no GPUs) instead of the heavy
    GPU env_command that the main array tasks use.
    """

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
            """Launch stats merge without the sglang env_command.

            Datatrove's default launch_merge_stats re-adds mem-per-cpu
            after our patch removes it, causing OOM on Clariden.  We
            build the sbatch script ourselves with just venv activation.
            """
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


def cmd_submit(args, overrides):
    """Submit a generation run as a SLURM job array."""
    from pipeline.charter.scale.reader import SidecarReader
    from pipeline.charter.scale.generate import AnnotationGenerator
    from pipeline.charter.scale.runs import get_run
    from pipeline.charter.scale.sidecar import sidecar_fingerprint as _sidecar_fingerprint

    cfg = load_config(overrides)
    run_name = args.run

    # Validate run exists
    run_def = get_run(run_name)

    total_rows, n_tasks = _compute_n_tasks(cfg)
    logger.info(
        "Run '{}': {} rows, {} tasks (rows_per_task={})",
        run_name,
        total_rows,
        n_tasks,
        cfg.charter.scale.rows_per_task,
    )

    # Ensure output directory exists
    Path(cfg.charter.scale.output_dir).mkdir(parents=True, exist_ok=True)

    # Map prompt_type -> the config field that holds the active prompt
    # filename for that run type. KeyError on an unknown prompt_type is the
    # right failure — we want a loud crash, not a silent fallback.
    prompt_field_by_type = {
        "reflection":  "reflection_prompt",
        "preflection": "preflection_prompt",
        "summary":     "summary_prompt",
    }
    active_prompt_field = prompt_field_by_type[run_def.prompt_type]
    active_prompt_filename = getattr(cfg.charter.scale, active_prompt_field)

    # Save run config for reproducibility and cross-run consistency check
    run_config_path = Path(cfg.charter.scale.output_dir) / run_name / "run_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    if run_config_path.exists():
        with open(run_config_path) as f:
            prev = json.load(f)
        if prev.get("rows_per_task") != cfg.charter.scale.rows_per_task:
            logger.error(
                "rows_per_task changed ({} -> {}). This would break resume. "
                "Delete {} to force a fresh start.",
                prev["rows_per_task"],
                cfg.charter.scale.rows_per_task,
                run_config_path,
            )
            sys.exit(1)
        # Only enforce immutability for the run's *active* prompt field. Other
        # prompt fields can change freely between runs without affecting this
        # one's outputs.
        prev_active_prompt = prev.get(active_prompt_field)
        if prev_active_prompt is not None and prev_active_prompt != active_prompt_filename:
            logger.error(
                "{} changed mid-run ({} -> {}). This would invalidate completed "
                "shards. Delete {} to force a fresh start.",
                active_prompt_field,
                prev_active_prompt,
                active_prompt_filename,
                run_config_path,
            )
            sys.exit(1)
    else:
        with open(run_config_path, "w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "rows_per_task": cfg.charter.scale.rows_per_task,
                    "sidecar_path": cfg.charter.scale.sidecar_path,
                    "sidecar_fingerprint": _sidecar_fingerprint(cfg.charter.scale.sidecar_path),
                    "reflection_seed": cfg.charter.scale.reflection_seed,
                    "generator_alias": cfg.charter.scale.generator_alias,
                    "reflection_prompt": cfg.charter.scale.reflection_prompt,
                    "preflection_prompt": cfg.charter.scale.preflection_prompt,
                    "summary_prompt": cfg.charter.scale.summary_prompt,
                    "hf_slug": cfg.charter.scale.sglang.hf_slug,
                },
                f,
                indent=2,
            )

    # Build pipeline
    pipeline = [
        SidecarReader(
            sidecar_path=cfg.charter.scale.sidecar_path,
            rows_per_task=cfg.charter.scale.rows_per_task,
        ),
        AnnotationGenerator(
            run_name=run_name,
            generator_alias=cfg.charter.scale.generator_alias,
            prompt_filename=active_prompt_filename,
            output_dir=cfg.charter.scale.output_dir,
            max_concurrent_requests=cfg.charter.scale.max_concurrent_requests,
            save_batch_size=cfg.charter.scale.save_batch_size,
            thinking=cfg.charter.scale.thinking,
            json_mode=cfg.charter.scale.json_mode,
            canary_seed=cfg.charter.scale.canary_seed,
            reflection_seed=cfg.charter.scale.reflection_seed,
            max_retries_per_doc=cfg.charter.scale.max_retries_per_doc,
            progress_interval=cfg.charter.scale.progress_interval,
            max_text_tokens=cfg.max_tokens,
        ),
    ]

    env_command = _build_env_command(cfg)
    sl = cfg.charter.scale.slurm

    logging_dir = str(Path(cfg.charter.scale.output_dir) / run_name)

    executor = _ExclusiveSlurmExecutor.create(
        pipeline=pipeline,
        tasks=n_tasks,
        time=sl.time,
        partition=sl.partition,
        cpus_per_task=sl.cpus_per_task,
        gpus_per_task=cfg.charter.scale.sglang.tp_size * cfg.charter.scale.sglang.dp_size,
        workers=sl.workers,
        job_name=f"charter_scale_{run_name}",
        env_command=env_command,
        sbatch_args={"account": sl.account},
        logging_dir=logging_dir,
        skip_completed=True,
        with_srun=False,
    )

    executor.run()


def cmd_merge(args, overrides):
    """Merge results from a completed run into the sidecar."""
    from pipeline.charter.scale.merge import merge_shards

    cfg = load_config(overrides)
    run_name = args.run

    out_path = merge_shards(
        output_dir=cfg.charter.scale.output_dir,
        run_name=run_name,
        sidecar_path=cfg.charter.scale.sidecar_path,
        allow_missing=args.allow_missing,
    )
    logger.info("Merged to: {}", out_path)


def cmd_status(args, overrides):
    """Show progress for a run."""
    from pipeline.charter.scale.progress import get_run_progress

    cfg = load_config(overrides)
    run_name = args.run

    total_rows, n_tasks = _compute_n_tasks(cfg)
    logging_dir = str(Path(cfg.charter.scale.output_dir) / run_name)

    progress = get_run_progress(
        output_dir=cfg.charter.scale.output_dir,
        run_name=run_name,
        total_tasks=n_tasks,
        logging_dir=logging_dir,
    )

    print(f"Run: {progress.run_name}")
    print(
        f"Tasks: {progress.completed_tasks}/{progress.total_tasks} ({progress.pct_tasks:.1f}%)"
    )
    print(f"Docs done: {progress.total_docs_done}")
    print(f"Docs failed: {progress.total_docs_failed}")


def cmd_rerun(args, overrides):
    """Re-submit a run after clearing completion markers for failed/incomplete ranks."""
    import shutil

    cfg = load_config(overrides)
    run_name = args.run

    total_rows, n_tasks = _compute_n_tasks(cfg)
    logging_dir = Path(cfg.charter.scale.output_dir) / run_name
    completions_dir = logging_dir / "completions"
    run_dir = Path(cfg.charter.scale.output_dir) / run_name

    # Find ranks with failures or missing results
    cleared = 0
    for rank in range(n_tasks):
        rank_str = f"{rank:05d}"
        failures_file = run_dir / rank_str / "failures.jsonl"
        completion_marker = completions_dir / rank_str

        has_failures = failures_file.exists() and failures_file.stat().st_size > 0

        if has_failures and completion_marker.exists():
            completion_marker.unlink()
            cleared += 1
            logger.info("Cleared completion marker for rank {} (has failures)", rank)

    logger.info("Cleared {} completion markers", cleared)

    if cleared > 0 or args.force:
        logger.info("Re-submitting...")
        cmd_submit(args, overrides)
    else:
        logger.info("No ranks need rerun")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.charter.scale",
        description="Charter scale: scale-up generation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = sub.add_parser("submit", help="Submit a generation run")
    p_submit.add_argument("--run", required=True, help="Run name (e.g. reflections)")

    # merge
    p_merge = sub.add_parser("merge", help="Merge results into sidecar")
    p_merge.add_argument("--run", required=True)
    p_merge.add_argument("--allow-missing", action="store_true")

    # status
    p_status = sub.add_parser("status", help="Show run progress")
    p_status.add_argument("--run", required=True)

    # rerun
    p_rerun = sub.add_parser("rerun", help="Re-submit failed/incomplete ranks")
    p_rerun.add_argument("--run", required=True)
    p_rerun.add_argument(
        "--force", action="store_true", help="Force resubmit even if no failures found"
    )

    # Parse known args, pass rest as config overrides
    args, remaining = parser.parse_known_args()

    commands = {
        "submit": cmd_submit,
        "merge": cmd_merge,
        "status": cmd_status,
        "rerun": cmd_rerun,
    }
    commands[args.command](args, remaining or None)


if __name__ == "__main__":
    main()
