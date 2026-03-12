"""Autonomous pipeline loop: generate → judge → improve prompts → repeat.

Spawns a Claude Code subprocess to analyze results and improve prompts
between iterations. Progress is written to a JSON status file for
dashboard polling.

Usage:
    uv run python -m pipeline.loop
    uv run python -m pipeline.loop loop.n_iterations=3
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
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
IMPROVER_LOG_PATH = PIPELINE_DATA_DIR / "improver_log.txt"


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


def _write_improver_summary(iteration: int) -> Path:
    """Write a condensed summary of iteration results for the improver to read.

    Strips bulky fields (full text, raw_response) and keeps only what's
    needed for analysis: scores, decisions, generated outputs, charter elements.
    """
    from pipeline.storage import load_items_for_iteration

    items = load_items_for_iteration(iteration)
    summary_path = PIPELINE_DATA_DIR / f"improver_input_iter{iteration}.jsonl"

    with open(summary_path, "w") as f:
        for item in items:
            judgment = item.get("judgment", {})
            record = {
                "item_id": item["item_id"],
                "is_gold": item.get("is_gold", False),
                "subset": item["subset"],
                "text_preview": item["text"][:300],
                "reflection_point": item["reflection_point"],
                "analysis": item.get("analysis", ""),
                "preflection": item.get("preflection", ""),
                "reflection": item.get("reflection", ""),
                "charter_elements": item.get("charter_elements", []),
                "judgment": {
                    "decision": judgment.get("decision"),
                    "aggregate": judgment.get("aggregate"),
                    "preflection": {
                        "scores": judgment.get("preflection", {}).get("scores"),
                        "reasoning": judgment.get("preflection", {}).get("reasoning"),
                    },
                    "reflection": {
                        "scores": judgment.get("reflection", {}).get("scores"),
                        "reasoning": judgment.get("reflection", {}).get("reasoning"),
                    },
                } if judgment else None,
            }
            f.write(json.dumps(record) + "\n")

    return summary_path


def run_improver(iteration: int, cfg: PipelineConfig) -> str:
    """Spawn a sandboxed Claude CLI to analyze results and improve prompts.

    Pre-extracts a condensed summary of iteration results (no full texts
    or raw API responses) so the improver can read it without needing
    Bash/python. The subprocess is restricted to Read/Glob/Grep/Write.
    Returns the analysis text from Claude's stdout.
    """
    slug = model_slug(cfg.model)
    model_dir = PROMPTS_DIR / slug
    gen_path = resolve_prompt_path(cfg.prompts.generator, cfg.model)
    judge_path = resolve_prompt_path(cfg.prompts.judge, cfg.model)
    improver_path = _INIT_PROMPTS_DIR / cfg.prompts.improver
    gold_path = PROJECT_ROOT / "data" / "annotation" / "annotations.jsonl"

    summary_path = _write_improver_summary(iteration)

    current_gen_v = _extract_version(cfg.prompts.generator)
    next_v = current_gen_v + 1

    state_path = model_dir / "state.md"
    if not state_path.exists():
        state_path.write_text("# Improver State\n\nNo previous iterations.\n")

    prompt = f"""You are improving prompts for a pretraining data annotation pipeline.

## Query tools
Run these via Bash (prefix with `uv run`). They are your primary data access tools:
  uv run python -m pipeline.improver_tools summary {iteration}     — aggregate stats, per-dimension means
  uv run python -m pipeline.improver_tools failures {iteration}    — rejected items with judge reasoning
  uv run python -m pipeline.improver_tools diversity {iteration}   — opening phrases for diversity check
  uv run python -m pipeline.improver_tools scores {iteration}      — compact scores table for all items
  uv run python -m pipeline.improver_tools show <id> {iteration}   — full text + preflection + reflection
  uv run python -m pipeline.improver_tools item <id> {iteration}   — full details as JSON
  uv run python -m pipeline.improver_tools gold                    — show gold (human) annotations for reference
  uv run python -m pipeline.improver_tools compare <id> {iteration} — side-by-side generated vs gold for an item

## State
Read your state file at {state_path} FIRST. It contains notes from previous iterations.

## Strategy: use subagents for parallel exploration
You have access to the Agent tool. Use it to parallelize your analysis:
- Spawn one subagent to analyze failures and low-scoring items in detail
- Spawn another to compare generated outputs with gold annotations
- Spawn another to check diversity patterns
Then synthesize their findings to write improved prompts.

## Your task
1. Read your state file: {state_path}
2. Run the query tools to gather data (use subagents to parallelize):
   - `summary` and `scores` for overall stats
   - `failures` for what went wrong
   - `diversity` for output variety
   - `gold` and `compare` to see how gold annotations differ from generated output
3. Read the improver instructions: {improver_path}
4. Read the current prompts: {gen_path} and {judge_path}
5. Analyze failures following the improver instructions
6. Write improved generator to {model_dir}/generator_v{next_v}.md
7. Write improved judge to {model_dir}/judge_v{next_v}.md
8. Update {state_path} with: what you changed, why, key metrics, and what to try next
9. Print your analysis summary as the final output

IMPORTANT:
- Use `uv run python -m pipeline.improver_tools ...` for data access — NOT raw file reads.
- Do NOT pipe commands together (e.g. `cmd1 | cmd2`). Run them as separate Bash calls.
- You can ONLY write files inside {model_dir}/. Do NOT modify any other files.
- Do NOT overfit to individual examples, especially gold ones. Gold annotations are noisy
  and imperfect — use them to understand the *style and spirit* of good annotations, not as
  ground truth to copy. Focus on systematic patterns across many items.
- The generator and judge prompts must NOT hardcode specific charter/constitution content
  (e.g. specific principle numbers or names). The constitution may change. The prompts should
  reference the charter generically (e.g. "relevant charter elements") and work with whatever
  charter is provided at runtime.
"""

    allowed_tools = [
        "Read", "Glob", "Grep", "Bash(uv run python:*)",
        "Agent", "TaskCreate", "TaskUpdate", "TaskList",
        "Write",
    ]
    settings = json.dumps({
        "permissions": {
            "allow": allowed_tools,
            "deny": ["Edit", "NotebookEdit"],
        }
    })
    cmd = [
        "claude",
        "--print",
        "--model", "opus",
        "--verbose",
        "--output-format", "stream-json",
        "--settings", settings,
        "--", prompt,
    ]

    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMPROVER_LOG_PATH.write_text("")  # clear previous log

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit parent's stderr
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    import threading

    final_text_holder: list[str] = []

    def _reader():
        final_text_holder.append(_stream_improver_output(proc, IMPROVER_LOG_PATH))

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        # Poll so KeyboardInterrupt can be delivered between iterations
        while reader_thread.is_alive():
            reader_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise

    assert proc.returncode == 0, (
        f"Claude improver failed (rc={proc.returncode}). See {IMPROVER_LOG_PATH}"
    )
    return final_text_holder[0] if final_text_holder else ""


def _stream_improver_output(proc: subprocess.Popen, log_path: Path) -> str:
    """Read stream-json events from the improver subprocess.

    Writes a human-readable log of tool use and text output to log_path
    and to stderr. Returns the final concatenated text result.
    """
    final_text_parts = []

    with open(log_path, "a") as log:
        for raw_line in proc.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            msg_type = event.get("type", "")

            if msg_type == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        text = block["text"]
                        final_text_parts.append(text)
                        _log_line(log, f"[text] {text}")
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        inp = block.get("input", {})
                        summary = _summarize_tool_input(name, inp)
                        _log_line(log, f"[tool] {name}: {summary}")

            elif msg_type == "user":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id", "?")
                        if block.get("is_error"):
                            content = block.get("content", "")
                            _log_line(log, f"[FAIL] {content[:300]}")
                        else:
                            content = str(block.get("content", ""))
                            preview = content[:100].replace("\n", " ")
                            _log_line(log, f"[ok]   {preview}")

            elif msg_type == "result":
                result_text = event.get("result", "")
                if result_text and not final_text_parts:
                    final_text_parts.append(result_text)
                _log_line(log, f"[done] cost=${event.get('cost_usd', '?')}")

    return "\n".join(final_text_parts)


def _log_line(log_file, line: str) -> None:
    """Write a line to both the log file and stderr."""
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    log_file.write(line + "\n")
    log_file.flush()


def _summarize_tool_input(name: str, inp: dict) -> str:
    """Produce a short summary of a tool invocation for logging."""
    if name == "Read":
        return inp.get("file_path", "?")
    elif name == "Write":
        path = inp.get("file_path", "?")
        content = inp.get("content", "")
        return f"{path} ({len(content)} chars)"
    elif name == "Glob":
        return inp.get("pattern", "?")
    elif name == "Grep":
        return f"{inp.get('pattern', '?')} in {inp.get('path', '.')}"
    elif name == "Bash":
        return inp.get("command", "?")[:150]
    return json.dumps(inp)[:100]


def _build_result_summary(loop_iter: int, pipeline_iter: int, items: list[dict]) -> dict:
    """Build a loop result summary from judged items."""
    judged = [it for it in items if it.get("judgment")]
    n_accepted = sum(1 for it in judged if it["judgment"]["decision"] == "accept")
    scores = [it["judgment"]["aggregate"] for it in judged]
    return {
        "loop_iteration": loop_iter,
        "pipeline_iteration": pipeline_iter,
        "n_accepted": n_accepted,
        "n_rejected": len(judged) - n_accepted,
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "analysis": "",
    }


def run_loop(n_iterations: int = 5, cfg: PipelineConfig | None = None) -> None:
    """Main autonomous loop: generate+judge → improve → repeat.

    Supports resuming after a crash: reads loop_status.json to determine
    which iterations already completed and skips them. If the crash happened
    mid-improve (iteration done but no analysis), re-runs just the improver.

    Writes progress to loop_status.json for dashboard polling.
    """
    from pipeline.run import run_iteration
    from pipeline.storage import load_items_for_iteration, load_runs

    if cfg is None:
        cfg = load_config()

    existing = read_status()

    # Resume from a previous crashed/errored run
    if existing and existing.get("error"):
        completed_results = existing.get("results", [])
        start_loop_iter = len(completed_results) + 1
        print(f"Resuming loop from iteration {start_loop_iter} (previous error: {existing['error'][:100]})")

        # Check if the last completed result needs its improver step
        needs_improve = False
        if completed_results:
            last = completed_results[-1]
            if not last.get("analysis") and last["loop_iteration"] < n_iterations:
                needs_improve = True
                start_loop_iter = last["loop_iteration"]
                print(f"  Re-running improver for loop iteration {start_loop_iter}")

        status = {
            "running": True,
            "loop_iteration": start_loop_iter,
            "total_iterations": n_iterations,
            "phase": "resuming",
            "started_at": existing.get("started_at", datetime.now(timezone.utc).isoformat()),
            "results": completed_results if not needs_improve else completed_results[:-1],
            "error": None,
        }

        # Sync config to match where we left off: detect highest prompt versions
        try:
            new_gen, new_judge = _detect_new_prompts(cfg)
            if new_gen != cfg.prompts.generator or new_judge != cfg.prompts.judge:
                cfg = _update_config(cfg, new_gen, new_judge)
                print(f"  Synced config to latest prompts: {new_gen}, {new_judge}")
        except AssertionError:
            pass  # no new prompts yet, that's fine

    elif existing and existing.get("running"):
        raise RuntimeError(
            "Loop is already running. Wait for it to finish or clear loop_status.json."
        )
    else:
        start_loop_iter = 1
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

    try:
        for i in range(start_loop_iter, n_iterations + 1):
            status["loop_iteration"] = i
            status["phase"] = "generating"
            write_status(status)

            def phase_cb(phase: str):
                status["phase"] = phase
                write_status(status)

            # Check if this iteration's pipeline run already exists (crash recovery)
            runs = load_runs()
            expected_pipeline_iter = len(runs) + 1

            # See if the run for this loop iteration already completed
            already_done = False
            if i <= len(status["results"]):
                # We already have results for this loop iteration (needs_improve case)
                pipeline_iter = status["results"][i - 1]["pipeline_iteration"]
                already_done = True
            else:
                # Check if there's a completed run that wasn't recorded in status
                # (crash between save_run and status update)
                items = load_items_for_iteration(expected_pipeline_iter - 1)
                judged = [it for it in items if it.get("judgment")]
                if judged and len(runs) >= expected_pipeline_iter - 1:
                    # The previous pipeline iteration exists but wasn't in status
                    # This means we crashed after run_iteration but before recording
                    last_run = runs[-1] if runs else None
                    if last_run and last_run["iteration"] == expected_pipeline_iter - 1:
                        pipeline_iter = last_run["iteration"]
                        result_summary = _build_result_summary(i, pipeline_iter, items)
                        already_done = True

            if already_done:
                print(f"Loop {i}/{n_iterations}: pipeline iteration {pipeline_iter} already complete, skipping")
                items = load_items_for_iteration(pipeline_iter)
                result_summary = _build_result_summary(i, pipeline_iter, items)
            else:
                result = run_iteration(cfg, phase_callback=phase_cb)
                pipeline_iter = result["iteration"]
                result_summary = {
                    "loop_iteration": i,
                    "pipeline_iteration": pipeline_iter,
                    "n_accepted": result["n_accepted"],
                    "n_rejected": result["n_rejected"],
                    "mean_score": round(result["mean_score"], 3),
                    "analysis": "",
                }

            if i < n_iterations:
                status["phase"] = "improving"
                write_status(status)

                analysis = run_improver(iteration=pipeline_iter, cfg=cfg)
                result_summary["analysis"] = analysis[:2000]

                new_gen, new_judge = _detect_new_prompts(cfg)
                cfg = _update_config(cfg, new_gen, new_judge)
                print(f"Loop {i}/{n_iterations}: updated prompts → {new_gen}, {new_judge}")

            # Append or replace result for this loop iteration
            if i <= len(status["results"]):
                status["results"][i - 1] = result_summary
            else:
                status["results"].append(result_summary)
            write_status(status)

        status["phase"] = "done"
        status["running"] = False
        write_status(status)
        print(f"\nLoop complete: {n_iterations} iterations.")

    except KeyboardInterrupt:
        status["error"] = "Interrupted by user"
        status["running"] = False
        status["phase"] = "interrupted"
        write_status(status)
        print("\nLoop interrupted.")
        raise
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

    run_loop(n_iterations=n, cfg=cfg)


if __name__ == "__main__":
    main()
