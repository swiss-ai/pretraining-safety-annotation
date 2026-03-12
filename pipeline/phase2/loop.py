"""Two-phase improver loop: Phase A improves judge, Phase B improves generator.

Each phase spawns an Opus agent that autonomously calls generate/judge via CLI
tools and decides when to stop (max N run_batch calls). All test results are
persisted. The outer workflow is A->B->stop, manually triggered.

Usage:
    uv run python -m pipeline.phase2.loop
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
    CONFIG_YAML_PATH,
    PIPELINE_DATA_DIR,
    PROMPTS_DIR,
    PROJECT_ROOT,
    _INIT_PROMPTS_DIR,
    AppConfig,
    load_config,
    resolve_prompt_path,
)
from pipeline.phase2.storage import items_path

STATUS_PATH = PIPELINE_DATA_DIR / "loop_status.json"
IMPROVER_LOG_A_PATH = PIPELINE_DATA_DIR / "improver_log_A.txt"
IMPROVER_LOG_B_PATH = PIPELINE_DATA_DIR / "improver_log_B.txt"

# Keep backward-compat alias for dashboard polling
IMPROVER_LOG_PATH = IMPROVER_LOG_A_PATH


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


def _detect_new_prompts(cfg: AppConfig) -> tuple[str, str]:
    """Find the highest-versioned generator and judge prompts in the model directory.

    Asserts that the detected versions are newer than the current config.
    Returns (generator_filename, judge_filename).
    """
    model_dir = PROMPTS_DIR / cfg.phase2.generator.model

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

    new_gen = _highest_version("generator_v*.md", cfg.phase2.generator.prompt)
    new_judge = _highest_version("judge_v*.md", cfg.phase2.judge.prompt)
    return new_gen, new_judge


def _extract_version(filename: str) -> int:
    """Extract version number from a filename like 'generator_v3.md'."""
    match = re.search(r"_v(\d+)\.md$", filename)
    assert match, f"Cannot extract version from {filename}"
    return int(match.group(1))


def _update_config(cfg: AppConfig, new_gen: str, new_judge: str) -> AppConfig:
    """Update config.yaml with new prompt filenames and return reloaded config."""
    from omegaconf import OmegaConf

    raw = OmegaConf.load(CONFIG_YAML_PATH)
    raw.phase2.generator.prompt = new_gen
    raw.phase2.judge.prompt = new_judge
    OmegaConf.save(raw, CONFIG_YAML_PATH)
    return load_config()


def _build_phase_a_prompt(cfg: AppConfig) -> str:
    """Build the prompt for Phase A (judge improver)."""
    alias = cfg.phase2.generator.model
    model_dir = PROMPTS_DIR / alias
    gen_path = resolve_prompt_path(cfg.phase2.generator.prompt, alias)
    judge_path = resolve_prompt_path(cfg.phase2.judge.prompt, alias)
    improver_path = _INIT_PROMPTS_DIR / "improver.md"
    phase_prompt_path = _INIT_PROMPTS_DIR / cfg.phase2.improver.judge_prompt

    current_judge_v = _extract_version(cfg.phase2.judge.prompt)
    next_v = current_judge_v + 1

    state_path = model_dir / "state.md"
    if not state_path.exists():
        state_path.write_text("# Improver State\n\nNo previous iterations.\n")

    max_batches = cfg.phase2.improver.max_batches_per_phase

    # Load phase prompt if it exists, otherwise use inline
    if phase_prompt_path.exists():
        phase_instructions = phase_prompt_path.read_text(encoding="utf-8")
    else:
        phase_instructions = "Focus on improving judge prompts. You are far more capable than the small judges — your judgment is valuable. Human reviews are ground truth anchors."

    from pipeline.phase2.storage import load_runs
    runs = load_runs()
    latest_iter = runs[-1]["iteration"] if runs else 0

    return f"""You are improving JUDGE prompts for a pretraining data annotation pipeline.

## Phase A: Judge Improvement

{phase_instructions}

## Query tools
Run these via Bash (prefix with `uv run`):
  uv run python -m pipeline.improver_tools summary {latest_iter}     — aggregate stats
  uv run python -m pipeline.improver_tools failures {latest_iter}    — rejected items with reasoning
  uv run python -m pipeline.improver_tools diversity {latest_iter}   — diversity check
  uv run python -m pipeline.improver_tools scores {latest_iter}      — compact scores table
  uv run python -m pipeline.improver_tools show <id> {latest_iter}   — full text + outputs
  uv run python -m pipeline.improver_tools item <id> {latest_iter}   — full details as JSON
  uv run python -m pipeline.improver_tools gold                      — gold (human) annotations
  uv run python -m pipeline.improver_tools compare <id> {latest_iter} — generated vs gold

## Test tools (run experiments WITHOUT modifying main data)
  uv run python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2] [--n N] [--phase A]
  uv run python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2] [--n N] [--phase A]
  uv run python -m pipeline.improver_tools run_batch --phase A      — full iteration with latest prompts
  uv run python -m pipeline.improver_tools test_results --phase A   — view test results

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
2. Run query tools to gather data (use subagents to parallelize)
3. Read the improver instructions: {improver_path}
4. Read the current judge prompt: {judge_path}
5. Also read the generator prompt for context: {gen_path}
6. Analyze how the judge is performing — is it calibrated with human reviews?
7. Write improved judge to {model_dir}/judge_v{next_v}.md
8. You may run up to {max_batches} `run_batch` calls to test your changes
9. Update {state_path} with: what you changed, why, key metrics, and what to try next
10. Print your analysis summary as the final output

IMPORTANT:
- Use `uv run python -m pipeline.improver_tools ...` for data access — NOT raw file reads.
- Do NOT pipe commands together. Run them as separate Bash calls.
- You can ONLY write files inside {model_dir}/. Do NOT modify any other files.
- Focus on judge calibration: are scores aligned with human reviews? Is the rubric clear?
- Do NOT overfit to individual examples. Focus on systematic patterns.
- The judge prompt must NOT hardcode specific charter/constitution content.
"""


def _build_phase_b_prompt(cfg: AppConfig) -> str:
    """Build the prompt for Phase B (generator improver)."""
    alias = cfg.phase2.generator.model
    model_dir = PROMPTS_DIR / alias
    gen_path = resolve_prompt_path(cfg.phase2.generator.prompt, alias)
    judge_path = resolve_prompt_path(cfg.phase2.judge.prompt, alias)
    improver_path = _INIT_PROMPTS_DIR / "improver.md"
    phase_prompt_path = _INIT_PROMPTS_DIR / cfg.phase2.improver.generator_prompt

    current_gen_v = _extract_version(cfg.phase2.generator.prompt)
    next_v = current_gen_v + 1

    state_path = model_dir / "state.md"
    max_batches = cfg.phase2.improver.max_batches_per_phase

    if phase_prompt_path.exists():
        phase_instructions = phase_prompt_path.read_text(encoding="utf-8")
    else:
        phase_instructions = "Focus on improving generator prompts. The judge prompts were just improved in Phase A — generate against the improved judge."

    from pipeline.phase2.storage import load_runs
    runs = load_runs()
    latest_iter = runs[-1]["iteration"] if runs else 0

    return f"""You are improving GENERATOR prompts for a pretraining data annotation pipeline.

## Phase B: Generator Improvement

{phase_instructions}

## Query tools
Run these via Bash (prefix with `uv run`):
  uv run python -m pipeline.improver_tools summary {latest_iter}     — aggregate stats
  uv run python -m pipeline.improver_tools failures {latest_iter}    — rejected items with reasoning
  uv run python -m pipeline.improver_tools diversity {latest_iter}   — diversity check
  uv run python -m pipeline.improver_tools scores {latest_iter}      — compact scores table
  uv run python -m pipeline.improver_tools show <id> {latest_iter}   — full text + outputs
  uv run python -m pipeline.improver_tools item <id> {latest_iter}   — full details as JSON
  uv run python -m pipeline.improver_tools gold                      — gold (human) annotations
  uv run python -m pipeline.improver_tools compare <id> {latest_iter} — generated vs gold

## Test tools (run experiments WITHOUT modifying main data)
  uv run python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2] [--n N] [--phase B]
  uv run python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2] [--n N] [--phase B]
  uv run python -m pipeline.improver_tools run_batch --phase B      — full iteration with latest prompts
  uv run python -m pipeline.improver_tools test_results --phase B   — view test results

## State
Read your state file at {state_path} FIRST. It contains notes from previous iterations.

## Strategy: use subagents for parallel exploration
You have access to the Agent tool. Use it to parallelize your analysis.

## Your task
1. Read your state file: {state_path}
2. Run query tools to gather data (use subagents to parallelize)
3. Read the improver instructions: {improver_path}
4. Read the current generator prompt: {gen_path}
5. Also read the (just-improved) judge prompt: {judge_path}
6. Analyze failure patterns in generated outputs
7. Write improved generator to {model_dir}/generator_v{next_v}.md
8. You CAN also fix the judge if you spot issues, but primarily focus on the generator
9. You may run up to {max_batches} `run_batch` calls to test your changes
10. Update {state_path} with: what you changed, why, key metrics, and what to try next
11. Print your analysis summary as the final output

IMPORTANT:
- Use `uv run python -m pipeline.improver_tools ...` for data access — NOT raw file reads.
- Do NOT pipe commands together. Run them as separate Bash calls.
- You can ONLY write files inside {model_dir}/. Do NOT modify any other files.
- Do NOT overfit to individual examples. Focus on systematic patterns.
- The generator prompt must NOT hardcode specific charter/constitution content.
"""


def _spawn_agent(prompt: str, log_path: Path, allowed_tools: list[str]) -> str:
    """Spawn a sandboxed Claude CLI subprocess and return its text output.

    Streams output to log_path and stderr for real-time monitoring.
    """
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
        "--effort", "max",
        "--verbose",
        "--output-format", "stream-json",
        "--settings", settings,
        "--", prompt,
    ]

    PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # clear previous log

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
        final_text_holder.append(_stream_improver_output(proc, log_path))

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
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
        f"Claude agent failed (rc={proc.returncode}). See {log_path}"
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


def _make_phase_status(status_str: str = "pending") -> dict:
    """Create a phase status dict."""
    return {
        "status": status_str,
        "reasoning": "",
        "started_at": None,
        "finished_at": None,
    }


ALLOWED_TOOLS = [
    "Read", "Glob", "Grep", "Bash(uv run python:*)",
    "Agent", "TaskCreate", "TaskUpdate", "TaskList",
    "Write",
]


def run_improver_loop(cfg: AppConfig | None = None) -> None:
    """Two-phase improver loop: Phase A (judge) -> Phase B (generator).

    Writes progress to loop_status.json for dashboard polling.
    """
    if cfg is None:
        cfg = load_config()

    existing = read_status()
    if existing and existing.get("running"):
        raise RuntimeError(
            "Loop is already running. Wait for it to finish or clear loop_status.json."
        )

    now = datetime.now(timezone.utc).isoformat()
    status = {
        "running": True,
        "phase": "A",
        "started_at": now,
        "phase_a": {**_make_phase_status("running"), "started_at": now},
        "phase_b": _make_phase_status("pending"),
        "error": None,
    }
    write_status(status)

    try:
        # --- Phase A: Judge Improvement ---
        print("=" * 60)
        print("PHASE A: Judge Improvement")
        print("=" * 60)

        prompt_a = _build_phase_a_prompt(cfg)
        analysis_a = _spawn_agent(prompt_a, IMPROVER_LOG_A_PATH, ALLOWED_TOOLS)

        # Sync config to latest judge prompts
        new_gen, new_judge = _detect_new_prompts(cfg)
        if new_judge != cfg.phase2.judge.prompt:
            cfg = _update_config(cfg, new_gen, new_judge)
            print(f"Phase A done: updated judge -> {new_judge}")

        now_a = datetime.now(timezone.utc).isoformat()
        status["phase_a"]["status"] = "done"
        status["phase_a"]["reasoning"] = analysis_a[:2000]
        status["phase_a"]["finished_at"] = now_a

        # --- Phase B: Generator Improvement ---
        status["phase"] = "B"
        status["phase_b"]["status"] = "running"
        status["phase_b"]["started_at"] = now_a
        write_status(status)

        print("\n" + "=" * 60)
        print("PHASE B: Generator Improvement")
        print("=" * 60)

        prompt_b = _build_phase_b_prompt(cfg)
        analysis_b = _spawn_agent(prompt_b, IMPROVER_LOG_B_PATH, ALLOWED_TOOLS)

        # Sync config to latest generator prompts
        new_gen, new_judge = _detect_new_prompts(cfg)
        if new_gen != cfg.phase2.generator.prompt:
            cfg = _update_config(cfg, new_gen, new_judge)
            print(f"Phase B done: updated generator -> {new_gen}")

        now_b = datetime.now(timezone.utc).isoformat()
        status["phase_b"]["status"] = "done"
        status["phase_b"]["reasoning"] = analysis_b[:2000]
        status["phase_b"]["finished_at"] = now_b

        status["phase"] = "done"
        status["running"] = False
        write_status(status)
        print("\nImprover loop complete (A+B).")

    except KeyboardInterrupt:
        status["error"] = "Interrupted by user"
        status["running"] = False
        status["phase"] = "interrupted"
        # Mark current phase as error
        for p in ("phase_a", "phase_b"):
            if status[p]["status"] == "running":
                status[p]["status"] = "error"
        write_status(status)
        print("\nLoop interrupted.")
        raise
    except Exception as e:
        status["error"] = str(e)
        status["running"] = False
        for p in ("phase_a", "phase_b"):
            if status[p]["status"] == "running":
                status[p]["status"] = "error"
        write_status(status)
        raise


def main():
    """CLI entry point for the two-phase improver loop."""
    overrides = sys.argv[1:] if len(sys.argv) > 1 else None
    cfg = load_config(overrides)

    print("Starting two-phase improver loop")
    print(f"Generator: {cfg.phase2.generator.model} (prompt: {cfg.phase2.generator.prompt})")
    print(f"Judge: {cfg.phase2.judge.model} (prompt: {cfg.phase2.judge.prompt})")
    print(f"Max batches per phase: {cfg.phase2.improver.max_batches_per_phase}")

    run_improver_loop(cfg=cfg)


if __name__ == "__main__":
    main()
