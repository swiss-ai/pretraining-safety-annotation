"""Charter scale CLI: prefilter, submit, status, rerun, export.

Two-stage flow over a source corpus (DCLM-Edu / FineWeb-2):

    prefilter  -> materialize a dense filtered dataset (safety + language),
                  on the GPU partition but WITHOUT sglang (quick, I/O-bound).
    submit     -> annotate the dense dataset (sglang co-located), doc_id-keyed.
    export     -> transcode per-rank JSONL into the doc_id-keyed parquet dataset.

Usage:
    uv run python -m pipeline.charter.scale prefilter
    uv run python -m pipeline.charter.scale submit --run reflections [overrides...]
    uv run python -m pipeline.charter.scale status --run reflections
    uv run python -m pipeline.charter.scale rerun  --run reflections [overrides...]
    uv run python -m pipeline.charter.scale export --run reflections
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

from pipeline.config import load_config
from pipeline.corpus import (
    DENSE_SCHEMA,
    SafetyLanguageFilter,
    dense_writer_adapter,
    get_corpus,
)
from pipeline.log import logger

# datatrove strides whole files across tasks and the cluster caps array jobs at
# MaxArraySize (1001 on Clariden). n_tasks is a CHOSEN, capped count — each task
# reads several shards — not the shard count.
DEFAULT_MAX_TASKS = 256
MAX_ARRAY_SIZE = 1000


def _build_env_command(cfg) -> str:
    """Build the shell preamble that launches sglang and waits for health."""
    sg = cfg.charter.scale.sglang
    output_dir = cfg.charter.scale.output_dir

    model_path = sg.model_path or sg.hf_slug
    served_name = sg.hf_slug

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


def _build_lightweight_env_command() -> str:
    """Preamble for the prefilter job: venv activation only, no sglang/model.

    The prefilter is pure I/O (read source -> filter -> write parquet); it runs
    on the GPU partition (the only one) but requests no GPUs and never loads a
    model.
    """
    venv_activate = str(Path(sys.prefix) / "bin" / "activate")
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    return textwrap.dedent(f"""\
        unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE
        source {venv_activate}
        export PYTHONPATH="{project_root}:${{PYTHONPATH:-}}"
    """)


class _ExclusiveSlurmExecutor:
    """Wraps SlurmPipelineExecutor for CSCS Clariden.

    Clariden nodes reject --mem-per-cpu (memory is not allocatable per-cpu on
    GH200). Datatrove always emits it, so we patch get_sbatch_args to remove it
    and use --exclusive instead. Also patches launch_merge_stats so the
    stats-merge dependent job uses a minimal env (no sglang, no GPUs).
    """

    @staticmethod
    def create(**kwargs):
        import os

        from datatrove.executor.slurm import SlurmPipelineExecutor, launch_slurm_job

        executor = SlurmPipelineExecutor(**kwargs)
        _orig = executor.get_sbatch_args

        def _patched(max_array=1):
            args = _orig(max_array)
            args.pop("mem-per-cpu", None)
            args["exclusive"] = ""
            return args

        def _lightweight_merge_stats():
            """Launch stats merge without the sglang env_command."""
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


# --- shard listing, freezing, and n_tasks ----------------------------------


def _datafolder(path: str):
    from datatrove.io import get_datafolder

    return get_datafolder(path)


def _list_source_shards(cfg, corpus) -> list[str]:
    """Sorted, relative-to-source_dir shard paths for the configured corpus.

    Flat corpora list all parquet; per-language-dir corpora list only the
    target-language subdirectories (the language filter is enforced by which
    directories we read, not per-doc).
    """
    df = _datafolder(cfg.charter.scale.source_dir)
    if corpus.layout == "flat":
        return sorted(df.list_files(glob_pattern="*.parquet"))
    shards: list[str] = []
    for lang in cfg.charter.scale.language_filter:
        subdir = corpus.lang_dirs.get(lang)
        assert subdir, f"No source subdir for language '{lang}' in corpus {corpus.name}"
        shards.extend(df.list_files(glob_pattern=f"{subdir}/*.parquet"))
    return sorted(shards)


def _freeze_paths_file(shards: list[str], path: Path) -> None:
    """Write the sorted shard list, one relative path per line.

    get_shard_from_paths_file strides in file order WITHOUT re-sorting, so the
    frozen file itself must be sorted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(shards)) + "\n", encoding="utf-8")


def _paths_fingerprint(shards: list[str]) -> dict:
    """Cheap identity of a frozen shard universe: count + sha256 of sorted list."""
    blob = "\n".join(sorted(shards)).encode("utf-8")
    return {"n": len(shards), "sha256": hashlib.sha256(blob).hexdigest()}


def _derive_n_tasks(n_shards: int, override: int) -> int:
    """Choose the SLURM array size: an explicit override, else min(shards, cap).

    Capping below n_shards is fine — datatrove strides shards across tasks. We
    never set n_tasks = n_shards (that would blow past MaxArraySize).
    """
    n = override if override > 0 else min(n_shards, DEFAULT_MAX_TASKS)
    n = max(1, n)
    if n > MAX_ARRAY_SIZE:
        logger.warning(
            "n_tasks={} exceeds MaxArraySize={}; datatrove will split into "
            "serialized dependent array waves. Lower n_tasks.",
            n, MAX_ARRAY_SIZE,
        )
    return n


def _check_or_write_run_config(path: Path, current: dict, guard_keys: list[str]) -> None:
    """Freeze *current* into run_config on first run; hard-fail on guarded drift.

    Mirrors the legacy rows_per_task guard: any change to a guarded field would
    invalidate completed shards (re-stride / re-filter), so we exit rather than
    silently corrupt resume.
    """
    if path.exists():
        prev = json.loads(path.read_text())
        for key in guard_keys:
            if prev.get(key) != current.get(key):
                logger.error(
                    "{} changed mid-run ({!r} -> {!r}). This would invalidate "
                    "completed shards. Delete {} to force a fresh start.",
                    key, prev.get(key), current.get(key), path,
                )
                sys.exit(1)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")


# --- commands ---------------------------------------------------------------


def cmd_prefilter(args, overrides):
    """Materialize the dense filtered dataset (safety + language), no sglang."""
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    from pipeline.corpus import CorpusReader

    cfg = load_config(overrides)
    sc = cfg.charter.scale
    corpus = get_corpus(sc.corpus)

    filtered_dir = Path(sc.filtered_dir)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = filtered_dir / "_logs"

    shards = _list_source_shards(cfg, corpus)
    assert shards, f"No source shards under {sc.source_dir} for corpus {sc.corpus}"
    if sc.prefilter_max_shards > 0:
        # Subset for smoke tests — process only the first N source shards.
        shards = shards[: sc.prefilter_max_shards]
        logger.info("prefilter: capped to first {} source shards (smoke/subset)", len(shards))
    source_paths_file = filtered_dir / "_source_shards.txt"
    _freeze_paths_file(shards, source_paths_file)
    n_tasks = _derive_n_tasks(len(shards), sc.n_tasks)

    _check_or_write_run_config(
        filtered_dir / "_prefilter_config.json",
        {
            "corpus": sc.corpus,
            "source_dir": sc.source_dir,
            "source_fingerprint": _paths_fingerprint(shards),
            "n_tasks": n_tasks,
            "safety_min_score": sc.safety_min_score,
            "safety_min_confidence": sc.safety_min_confidence,
            "language_filter": list(sc.language_filter),
        },
        guard_keys=[
            "corpus", "source_fingerprint", "n_tasks",
            "safety_min_score", "safety_min_confidence", "language_filter",
        ],
    )

    logger.info(
        "prefilter '{}': {} source shards -> {} tasks, threshold score>={} conf>={}",
        sc.corpus, len(shards), n_tasks, sc.safety_min_score, sc.safety_min_confidence,
    )

    pipeline = [
        CorpusReader(
            data_folder=sc.source_dir,
            paths_file=str(source_paths_file),
            adapter=corpus.adapter,
            projection=corpus.projection,
            text_key="text",
            id_key="id",
        ),
        SafetyLanguageFilter(
            min_score=sc.safety_min_score,
            min_confidence=sc.safety_min_confidence,
            languages=list(sc.language_filter) or None,
        ),
        ParquetWriter(
            output_folder=sc.filtered_dir,
            adapter=dense_writer_adapter,
            schema=DENSE_SCHEMA,
            compression="snappy",
            # Shard the output across many files so the annotation run can
            # stride them across tasks (one giant file = no parallelism).
            max_file_size=512 * 2**20,
        ),
    ]

    sl = sc.slurm
    executor = _ExclusiveSlurmExecutor.create(
        pipeline=pipeline,
        tasks=n_tasks,
        time=sl.time,
        partition=sl.partition,
        cpus_per_task=sl.cpus_per_task,
        gpus_per_task=0,
        workers=sl.workers,
        job_name=f"charter_prefilter_{sc.corpus}",
        env_command=_build_lightweight_env_command(),
        sbatch_args={"account": sl.account},
        logging_dir=str(logging_dir),
        skip_completed=True,
        with_srun=False,
    )
    executor.run()


def _resolve_annotation_inputs(cfg, run_name: str) -> tuple[str, int]:
    """Return (annotation paths_file path, frozen n_tasks) for an annotation run.

    On first submit this freezes the sorted dense-shard list + chosen n_tasks
    into run_config.json. On resume/status/rerun it reads them back — the frozen
    values are the source of truth (never recomputed from live config).
    """
    sc = cfg.charter.scale
    run_dir = Path(sc.output_dir) / run_name
    run_config_path = run_dir / "run_config.json"
    paths_file = run_dir / "filtered_shards.txt"

    shards = sorted(_datafolder(sc.filtered_dir).list_files(glob_pattern="*.parquet"))
    assert shards, f"No filtered shards under {sc.filtered_dir} — run `prefilter` first."

    # The frozen n_tasks is the source of truth on resume — never recompute it
    # from live config (a post-submit edit must not desync rerun/status).
    if run_config_path.exists():
        n_tasks = int(json.loads(run_config_path.read_text())["n_tasks"])
    else:
        n_tasks = _derive_n_tasks(len(shards), sc.n_tasks)

    # Guard every call (not just the first): a changed dense dataset (different
    # shard fingerprint) or a changed seed/model/prompt would invalidate
    # completed shards, so hard-fail rather than silently re-stride.
    _check_or_write_run_config(
        run_config_path,
        {
            "corpus": sc.corpus,
            "filtered_dir": sc.filtered_dir,
            "paths_fingerprint": _paths_fingerprint(shards),
            "n_tasks": n_tasks,
            "reflection_seed": sc.reflection_seed,
            "generator_alias": sc.generator_alias,
            "reflection_prompt": sc.reflection_prompt,
        },
        guard_keys=[
            "corpus", "paths_fingerprint", "n_tasks",
            "reflection_seed", "generator_alias", "reflection_prompt",
        ],
    )
    _freeze_paths_file(shards, paths_file)
    return str(paths_file), n_tasks


def cmd_submit(args, overrides):
    """Submit an annotation run over the dense filtered dataset."""
    from datatrove.pipeline.readers import ParquetReader

    from pipeline.charter.scale.generate import AnnotationGenerator
    from pipeline.charter.scale.runs import get_run

    cfg = load_config(overrides)
    sc = cfg.charter.scale
    run_name = args.run
    run_def = get_run(run_name)

    paths_file, n_tasks = _resolve_annotation_inputs(cfg, run_name)
    logger.info("Run '{}': {} tasks over dense dataset {}", run_name, n_tasks, sc.filtered_dir)

    Path(sc.output_dir).mkdir(parents=True, exist_ok=True)

    prompt_field_by_type = {"reflection": "reflection_prompt"}
    active_prompt_filename = getattr(sc, prompt_field_by_type[run_def.prompt_type])

    pipeline = [
        # Dense dataset is flat (id/text/safety_score/language/source_shard); the
        # default adapter routes the leftover columns into Document.metadata.
        ParquetReader(
            data_folder=sc.filtered_dir,
            paths_file=paths_file,
            text_key="text",
            id_key="id",
        ),
        AnnotationGenerator(
            run_name=run_name,
            generator_alias=sc.generator_alias,
            prompt_filename=active_prompt_filename,
            output_dir=sc.output_dir,
            max_concurrent_requests=sc.max_concurrent_requests,
            save_batch_size=sc.save_batch_size,
            thinking=sc.thinking,
            json_mode=sc.json_mode,
            reflection_seed=sc.reflection_seed,
            max_retries_per_doc=sc.max_retries_per_doc,
            progress_interval=sc.progress_interval,
            max_chars=sc.reflection_max_chars,
        ),
    ]

    sl = sc.slurm
    executor = _ExclusiveSlurmExecutor.create(
        pipeline=pipeline,
        tasks=n_tasks,
        time=sl.time,
        partition=sl.partition,
        cpus_per_task=sl.cpus_per_task,
        gpus_per_task=sc.sglang.tp_size * sc.sglang.dp_size,
        workers=sl.workers,
        job_name=f"charter_scale_{run_name}",
        env_command=_build_env_command(cfg),
        sbatch_args={"account": sl.account},
        logging_dir=str(Path(sc.output_dir) / run_name),
        skip_completed=True,
        with_srun=False,
    )
    executor.run()


def cmd_status(args, overrides):
    """Show progress for an annotation run."""
    from pipeline.charter.scale.progress import get_run_progress

    cfg = load_config(overrides)
    sc = cfg.charter.scale
    run_name = args.run

    _, n_tasks = _resolve_annotation_inputs(cfg, run_name)
    logging_dir = str(Path(sc.output_dir) / run_name)

    progress = get_run_progress(
        output_dir=sc.output_dir,
        run_name=run_name,
        total_tasks=n_tasks,
        logging_dir=logging_dir,
    )

    print(f"Run: {progress.run_name}")
    print(f"Tasks: {progress.completed_tasks}/{progress.total_tasks} ({progress.pct_tasks:.1f}%)")
    print(f"Docs done: {progress.total_docs_done}")
    print(f"Docs failed: {progress.total_docs_failed}")


def cmd_rerun(args, overrides):
    """Clear completion markers for ranks with failures, then resubmit."""
    cfg = load_config(overrides)
    sc = cfg.charter.scale
    run_name = args.run

    _, n_tasks = _resolve_annotation_inputs(cfg, run_name)
    run_dir = Path(sc.output_dir) / run_name
    completions_dir = run_dir / "completions"

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


def cmd_export(args, overrides):
    """Transcode per-rank JSONL into the doc_id-keyed annotation dataset."""
    from pipeline.charter.scale.export import export_run

    cfg = load_config(overrides)
    sc = cfg.charter.scale
    out = export_run(output_dir=sc.output_dir, run_name=args.run, corpus=sc.corpus)
    logger.info("Exported annotation dataset to: {}", out)


def main():
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.charter.scale",
        description="Charter scale: prefilter + scale-up annotation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prefilter", help="Materialize the dense filtered dataset")

    p_submit = sub.add_parser("submit", help="Submit an annotation run")
    p_submit.add_argument("--run", required=True, help="Run name (e.g. reflections)")

    p_status = sub.add_parser("status", help="Show run progress")
    p_status.add_argument("--run", required=True)

    p_rerun = sub.add_parser("rerun", help="Re-submit failed/incomplete ranks")
    p_rerun.add_argument("--run", required=True)
    p_rerun.add_argument("--force", action="store_true", help="Resubmit even if no failures found")

    p_export = sub.add_parser("export", help="Transcode results into the annotation dataset")
    p_export.add_argument("--run", required=True)

    args, remaining = parser.parse_known_args()
    commands = {
        "prefilter": cmd_prefilter,
        "submit": cmd_submit,
        "status": cmd_status,
        "rerun": cmd_rerun,
        "export": cmd_export,
    }
    commands[args.command](args, remaining or None)


if __name__ == "__main__":
    main()
