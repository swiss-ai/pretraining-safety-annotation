"""Autonomous pipeline loop: generate → judge → improve prompts → repeat.

Spawns a Claude Code subprocess to analyze results and improve prompts
between iterations. Progress is written to a JSON status file for
dashboard polling.

Usage:
    uv run python -m pipeline.loop
    uv run python -m pipeline.loop loop.n_iterations=3
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import (
    PIPELINE_DATA_DIR,
    PROMPTS_DIR,
    PROJECT_ROOT,
    _INIT_PROMPTS_DIR,
    PipelineConfig,
    load_config,
    model_slug,
    resolve_prompt_path,
)
from pipeline.storage import items_path

STATUS_PATH = PIPELINE_DATA_DIR / "loop_status.json"


def write_status(status: dict) -> None:
    """Atomically write loop status to JSON file."""
    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, STATUS_PATH)


def read_status() -> dict | None:
    """Read current loop status, or None if no status file exists."""
    if not STATUS_PATH.exists():
        return None
    return json.loads(STATUS_PATH.read_text())


def _detect_new_prompts(cfg: PipelineConfig) -> tuple[str, str]:
    """Find the highest-versioned generator and judge prompts in the model directory.

    Asserts that the detected versions are newer than the current config.
    Returns (generator_filename, judge_filename).
    """
    model_dir = PROMPTS_DIR / model_slug(cfg.model)

    def _highest_version(pattern: str, current: str) -> str:
        matches = sorted(glob.glob(str(model_dir / pattern)))
        assert matches, f"No files matching {pattern} in {model_dir}"
        latest = Path(matches[-1]).name
        current_v = _extract_version(current)
        latest_v = _extract_version(latest)
        assert latest_v >= current_v, (
            f"Expected version >= {current_v}, got {latest_v} ({latest})"
        )
        return latest

    new_gen = _highest_version("generator_v*.md", cfg.prompts.generator)
    new_judge = _highest_version("judge_v*.md", cfg.prompts.judge)
    return new_gen, new_judge


def _extract_version(filename: str) -> int:
    """Extract version number from a filename like 'generator_v3.md'."""
    match = re.search(r"_v(\d+)\.md$", filename)
    assert match, f"Cannot extract version from {filename}"
    return int(match.group(1))


def _update_config(cfg: PipelineConfig, new_gen: str, new_judge: str) -> PipelineConfig:
    """Update config.yaml with new prompt filenames and return reloaded config."""
    from omegaconf import OmegaConf

    yaml_path = Path(__file__).parent / "conf" / "config.yaml"
    raw = OmegaConf.load(yaml_path)
    raw.prompts.generator = new_gen
    raw.prompts.judge = new_judge
    OmegaConf.save(raw, yaml_path)
    return load_config()


def _build_improver_settings(model_dir: Path) -> dict:
    """Build a Claude settings dict that sandboxes the improver.

    Allows: reading any file, writing only to the model's prompt directory.
    Denies: Bash, Edit, Agent, NotebookEdit — the improver can only use
    Read/Glob/Grep to analyze and Write to create new prompt files.
    """
    rel_model_dir = str(model_dir.relative_to(PROJECT_ROOT))
    return {
        "permissions": {
            "allow": [
                "Read",
                "Glob",
                "Grep",
                f"Write({rel_model_dir}/*)",
            ],
            "deny": [
                "Bash",
                "Edit",
                "Agent",
                "NotebookEdit",
                "Write",
            ],
        },
    }


def run_improver(iteration: int, cfg: PipelineConfig) -> str:
    """Spawn Claude CLI to analyze results and improve prompts.

    Returns the analysis text from Claude's stdout.
    """
    slug = model_slug(cfg.model)
    model_dir = PROMPTS_DIR / slug
    gen_path = resolve_prompt_path(cfg.prompts.generator, cfg.model)
    judge_path = resolve_prompt_path(cfg.prompts.judge, cfg.model)
    improver_path = _INIT_PROMPTS_DIR / cfg.prompts.improver
    gold_path = PROJECT_ROOT / "data" / "annotation" / "annotations.jsonl"
    items_file = items_path()

    current_gen_v = _extract_version(cfg.prompts.generator)
    next_v = current_gen_v + 1

    prompt = f"""You are improving prompts for a pretraining data annotation pipeline.

## Your task
1. Read iteration {iteration} results from {items_file}
2. Read the improver instructions from {improver_path}
3. Read the current generator prompt: {gen_path}
4. Read the current judge prompt: {judge_path}
5. Read gold annotations from {gold_path} for comparison
6. Analyze failures following the improver instructions
7. Write improved generator to {model_dir}/generator_v{next_v}.md
8. Write improved judge to {model_dir}/judge_v{next_v}.md
9. Print your analysis summary as the final output

You can ONLY write files inside {model_dir}/. Do NOT modify any other files.
"""

    settings = _build_improver_settings(model_dir)
    settings_path = Path(tempfile.mktemp(
        suffix=".json", prefix="improver_settings_", dir=str(PROJECT_ROOT)
    ))
    settings_path.write_text(json.dumps(settings))

    try:
        cmd = [
            "claude",
            "--print",
            "--model", "opus",
            "--settings-file", str(settings_path),
            prompt,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.loop.improver_timeout_s,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Claude improver failed: {result.stderr[-500:]}"
        return result.stdout
    finally:
        settings_path.unlink(missing_ok=True)


async def run_loop(n_iterations: int = 5, cfg: PipelineConfig | None = None) -> None:
    """Main autonomous loop: generate+judge → improve → repeat.

    Writes progress to loop_status.json for dashboard polling.
    """
    if cfg is None:
        cfg = load_config()

    existing = read_status()
    assert existing is None or not existing.get("running"), (
        "Loop is already running. Wait for it to finish or clear loop_status.json."
    )

    status = {
        "running": True,
        "loop_iteration": 0,
        "total_iterations": n_iterations,
        "phase": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "error": None,
    }
    write_status(status)

    from pipeline.run import run_iteration

    try:
        for i in range(1, n_iterations + 1):
            status["loop_iteration"] = i
            status["phase"] = "generating"
            write_status(status)

            def phase_cb(phase: str):
                status["phase"] = phase
                write_status(status)

            result = await run_iteration(cfg, phase_callback=phase_cb)

            result_summary = {
                "loop_iteration": i,
                "pipeline_iteration": result["iteration"],
                "n_accepted": result["n_accepted"],
                "n_rejected": result["n_rejected"],
                "mean_score": round(result["mean_score"], 3),
                "analysis": "",
            }

            if i < n_iterations:
                status["phase"] = "improving"
                write_status(status)

                analysis = run_improver(iteration=result["iteration"], cfg=cfg)
                result_summary["analysis"] = analysis[:2000]

                new_gen, new_judge = _detect_new_prompts(cfg)
                cfg = _update_config(cfg, new_gen, new_judge)
                print(f"Loop {i}/{n_iterations}: updated prompts → {new_gen}, {new_judge}")

            status["results"].append(result_summary)
            write_status(status)

        status["phase"] = "done"
        status["running"] = False
        write_status(status)
        print(f"\nLoop complete: {n_iterations} iterations.")

    except Exception as e:
        status["error"] = str(e)
        status["running"] = False
        status["phase"] = "error"
        write_status(status)
        raise


def main():
    """CLI entry point for the autonomous loop."""
    overrides = sys.argv[1:] if len(sys.argv) > 1 else None
    cfg = load_config(overrides)
    n = cfg.loop.n_iterations

    print(f"Starting autonomous loop: {n} iterations")
    print(f"Model: {cfg.model}")
    print(f"Generator: {cfg.prompts.generator}")
    print(f"Judge: {cfg.prompts.judge}")

    asyncio.run(run_loop(n_iterations=n, cfg=cfg))


if __name__ == "__main__":
    main()
