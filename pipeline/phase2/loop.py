"""Independent improver runners: judge improver and generator improver.

Each improver spawns an Opus agent that autonomously calls cross-iteration
tools via CLI. Improvers run independently — one role at a time.

Usage:
    uv run python -m pipeline.phase2.loop --role judge [--aliases glm45,olmo3-32B-think]
    uv run python -m pipeline.phase2.loop --role generator [--aliases glm45]
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
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
from pipeline.log import logger
from pipeline.phase2.storage import save_loop_run

STATUS_PATH = PIPELINE_DATA_DIR / "loop_status.json"
AGENT_TMP_DIR = PIPELINE_DATA_DIR / "tmp"


def improver_log_path(role: str, alias: str) -> Path:
    """Log path for a specific improver, e.g. data/pipeline/improver_log_judge_glm45.txt"""
    return PIPELINE_DATA_DIR / f"improver_log_{role}_{alias}.txt"


# Keep backward-compat alias for dashboard polling
IMPROVER_LOG_PATH = PIPELINE_DATA_DIR / "improver_log_judge.txt"


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


def _extract_version(filename: str) -> int:
    """Extract version number from a filename like 'generator_v3.md'."""
    match = re.search(r"_v(\d+)\.md$", filename)
    assert match, f"Cannot extract version from {filename}"
    return int(match.group(1))


def _resolve_config_prompt(filename: str, alias: str) -> str:
    """Resolve a prompt config value to its concrete filename.

    Handles '_latest.md' by resolving via resolve_prompt_path.
    For concrete names like 'judge_v3.md', returns as-is.
    """
    if "_latest.md" in filename:
        return resolve_prompt_path(filename, alias).name
    return filename


def _detect_new_prompts(alias: str, role: str) -> str:
    """Find the highest-versioned prompt for the given role in the model directory.

    Returns the latest filename (e.g. 'judge_v3.md').
    """
    model_dir = PROMPTS_DIR / alias
    prefix = "judge" if role == "judge" else "generator"
    matches = sorted(glob.glob(str(model_dir / f"{prefix}_v*.md")))
    assert matches, f"No {prefix}_v*.md files in {model_dir}"
    return Path(matches[-1]).name


def _build_improver_prompt(cfg: AppConfig, role: str, target_alias: str) -> str:
    """Unified prompt builder for judge and generator improvers.

    Key additions over old phase prompts:
    - Lists ALL counterpart models the agent's cross-iterations will use
    - Tells agent to use `run_cross_batch --role {role} --target {alias}`
    - Explains group_id for interpreting cross-model results
    - Points to `cross_summary <group_id>` command for aggregated stats
    """
    model_dir = PROMPTS_DIR / target_alias
    improver_path = _INIT_PROMPTS_DIR / "improver.md"

    if role == "judge":
        prompt_path = resolve_prompt_path("judge_latest.md", target_alias)
        other_prompt_path = resolve_prompt_path("generator_latest.md", target_alias)
        current_v = _extract_version(prompt_path.name)
        next_v = current_v + 1
        phase_prompt_path = _INIT_PROMPTS_DIR / cfg.phase2.improver.judge_prompt
        counterpart_models = [m.alias for m in cfg.phase2.generator_models]
        role_label = "JUDGE"
        prompt_type = "judge"
        other_type = "generator"
    else:
        prompt_path = resolve_prompt_path("generator_latest.md", target_alias)
        other_prompt_path = resolve_prompt_path("judge_latest.md", target_alias)
        current_v = _extract_version(prompt_path.name)
        next_v = current_v + 1
        phase_prompt_path = _INIT_PROMPTS_DIR / cfg.phase2.improver.generator_prompt
        counterpart_models = [m.alias for m in cfg.phase2.judge_models]
        role_label = "GENERATOR"
        prompt_type = "generator"
        other_type = "judge"

    state_path = model_dir / "state.md"
    if not state_path.exists():
        state_path.write_text("# Improver State\n\nNo previous iterations.\n")

    max_batches = cfg.phase2.improver.max_batches_per_phase

    if phase_prompt_path.exists():
        phase_instructions = phase_prompt_path.read_text(encoding="utf-8")
    else:
        phase_instructions = f"Focus on improving {prompt_type} prompts."

    from pipeline.phase2.storage import load_runs
    runs = load_runs()
    latest_iter = runs[-1]["iteration"] if runs else 0

    counterpart_list = ", ".join(counterpart_models)

    return f"""You are improving {role_label} prompts for a pretraining data annotation pipeline.

## {role_label} Improvement for model: {target_alias}

{phase_instructions}

## Cross-model evaluation
Your improvements will be tested against ALL {other_type} models: [{counterpart_list}].
When you run a batch, it creates one iteration per {other_type} model, all sharing a group_id.
Use `cross_summary <group_id>` to see aggregated per-model stats after each batch.

## Query tools
Run these via Bash (prefix with `uv run`):
  uv run python -m pipeline.improver_tools summary {latest_iter}     — aggregate stats
  uv run python -m pipeline.improver_tools failures {latest_iter}    — rejected items with reasoning
  uv run python -m pipeline.improver_tools failures {latest_iter} --reasoning-limit 500  — full reasoning
  uv run python -m pipeline.improver_tools diversity {latest_iter}   — frequency-based diversity analysis
  uv run python -m pipeline.improver_tools scores {latest_iter}      — compact scores table
  uv run python -m pipeline.improver_tools show <id> {latest_iter}   — full text + outputs for one item
  uv run python -m pipeline.improver_tools show <id1,id2,...> {latest_iter}  — batch show multiple items
  uv run python -m pipeline.improver_tools show <id> {latest_iter} --brief   — truncated source text
  uv run python -m pipeline.improver_tools show --gold {latest_iter} [--brief] — all gold items for iteration
  uv run python -m pipeline.improver_tools item <id> {latest_iter}   — full details as JSON
  uv run python -m pipeline.improver_tools gold                      — gold annotations (concise, no source text)
  uv run python -m pipeline.improver_tools gold --verbose            — gold with full source text (large output!)
  uv run python -m pipeline.improver_tools compare <id> {latest_iter} — generated vs gold
  uv run python -m pipeline.improver_tools reviews [{latest_iter}]   — human reviews with judge comparison
  uv run python -m pipeline.improver_tools filter {latest_iter} --dim <dimension> --below <threshold> [--part preflection|reflection]
  uv run python -m pipeline.improver_tools trend                     — cross-iteration comparison table
  uv run python -m pipeline.improver_tools correlations              — judge-human correlation by judge version

## Test tools (run experiments WITHOUT modifying main data)
  uv run python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2] [--n N] [--role {role}]
  uv run python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2] [--n N] [--role {role}]
  uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias}  — cross-iteration with ALL {other_type} models
  uv run python -m pipeline.improver_tools cross_summary <group_id>   — per-model stats for a cross-iteration
  uv run python -m pipeline.improver_tools test_results --role {role}  — view test results

## Scratch directory (IMPORTANT: use this for temporary scripts)
Write ad-hoc analysis scripts to: {AGENT_TMP_DIR}
Run them with: uv run python {AGENT_TMP_DIR}/your_script.py
Delete with: rm {AGENT_TMP_DIR}/your_script.py
This folder is cleared at the start of each loop. Use it for any analysis the CLI tools don't cover.
**You MUST write all temporary files here** — do NOT write scripts to the project root or elsewhere.
The `rm` command only works inside {AGENT_TMP_DIR}/.

## State
Read your state file at {state_path} FIRST. It contains notes from previous iterations.

## Strategy: use Opus subagents for parallel exploration (IMPORTANT)
You have access to the Agent tool. **Always use model="opus" for subagents** — they need
strong reasoning. Parallelize aggressively; the bottleneck is wall-clock time, not tokens:
- Spawn one subagent to analyze failures and low-scoring items in detail
- Spawn another to compare generated outputs with gold annotations
- Spawn another to review human reviews (the `reviews` command) and check judge calibration
- Spawn another to check diversity patterns
Then synthesize their findings to write improved prompts.

## Your task
1. Read your state file: {state_path}
2. Run query tools to gather data (**use Opus subagents to parallelize** — do NOT run queries sequentially)
3. Read the improver instructions: {improver_path}
4. Read the current {prompt_type} prompt: {prompt_path}
5. Also read the {other_type} prompt for context: {other_prompt_path}
6. Analyze how the {prompt_type} is performing
7. Write improved {prompt_type} to {model_dir}/{prompt_type}_v{next_v}.md
8. You may run up to {max_batches} `run_cross_batch` calls to test your changes
9. Update {state_path} with: what you changed, why, key metrics, and what to try next
10. Print a **single final summary** as your VERY LAST message. This summary is parsed and displayed in the dashboard.
    It MUST start with exactly `## Final Summary` on its own line, followed by:
    - **What changed**: which prompt file, key modifications
    - **Why**: what problems you identified, with evidence (scores, examples)
    - **Results**: before/after metrics if you ran test batches
    - **Next steps**: what to try in the next iteration

IMPORTANT:
- Use `uv run python -m pipeline.improver_tools ...` for data access — NOT raw file reads.
- Do NOT pipe commands together. Run them as separate Bash calls.
- You can ONLY write files inside {model_dir}/ and {AGENT_TMP_DIR}/. Do NOT modify any other files.
- Do NOT overfit to individual examples. Focus on systematic patterns.
- The {prompt_type} prompt must NOT hardcode specific charter/constitution content.
"""


def _spawn_agent(prompt: str, log_path: Path, allowed_tools: list[str]) -> str:
    """Spawn a sandboxed Claude CLI subprocess and return its text output.

    Streams output to log_path and stderr for real-time monitoring.
    """
    settings = json.dumps({
        "permissions": {
            "allow": allowed_tools,
            "deny": ["NotebookEdit"],
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

    proc.wait()
    output = final_text_holder[0] if final_text_holder else ""

    if proc.returncode != 0:
        summary = output[:500] if output else f"No output. See {log_path}"
        raise RuntimeError(
            f"Claude agent failed (rc={proc.returncode}): {summary}"
        )

    _validate_agent_output(output, log_path)
    return output


_ERROR_PATTERNS = [
    "authentication_error",
    "OAuth token has expired",
    "You've hit your limit",
    "API Error: 4",
    "API Error: 5",
]


def _validate_agent_output(output: str, log_path: Path) -> None:
    """Check agent output for known error patterns that exit with rc=0.

    Raises RuntimeError if the output looks like an error rather than
    a successful completion.
    """
    for pattern in _ERROR_PATTERNS:
        if pattern in output:
            raise RuntimeError(
                f"Agent exited with rc=0 but output contains error "
                f"({pattern}): {output[:300]}"
            )

    if "## Final Summary" not in output:
        logger.warning(
            "Agent output missing '## Final Summary' — may not have completed "
            "successfully. Output preview: {}",
            output[:300],
        )


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


def _make_improver_status(status_str: str = "pending") -> dict:
    """Create an improver status dict."""
    return {
        "status": status_str,
        "reasoning": "",
        "started_at": None,
        "finished_at": None,
    }


def _snapshot_prompts(cfg: AppConfig) -> dict[str, str]:
    """Capture current prompt file contents keyed by 'alias/filename'."""
    prompts = {}
    all_aliases = set()
    for m in cfg.phase2.judge_models + cfg.phase2.generator_models:
        all_aliases.add(m.alias)
    for alias in sorted(all_aliases):
        model_dir = PROMPTS_DIR / alias
        if not model_dir.exists():
            continue
        for path in sorted(model_dir.glob("*.md")):
            if path.name == "state.md":
                continue
            prompts[f"{alias}/{path.name}"] = path.read_text(encoding="utf-8")
    return prompts


def _extract_reasoning_from_log(log_path: Path) -> str:
    """Extract the ``## Final Summary`` section from an improver log file.

    First looks for a ``## Final Summary`` heading in [text] blocks.
    Falls back to the last consecutive [text] block if no heading found.
    """
    if not log_path.exists():
        return ""
    raw = log_path.read_text()

    text_lines = _collect_text_lines(raw)
    full_text = "\n".join(text_lines)
    marker = "## Final Summary"
    idx = full_text.find(marker)
    if idx != -1:
        return full_text[idx:].strip()

    lines = raw.splitlines()
    result: list[str] = []
    collecting = False
    for line in reversed(lines):
        if line.startswith(("[tool]", "[ok]", "[FAIL]", "[done]")):
            if collecting:
                break
            continue
        if line.startswith("[text] "):
            result.append(line[7:])
            break
        collecting = True
        result.append(line)
    result.reverse()
    return "\n".join(result).strip()[:2000]


def _collect_text_lines(raw_log: str) -> list[str]:
    """Collect all [text] content lines from a raw log, in order."""
    result: list[str] = []
    in_text_block = False
    for line in raw_log.splitlines():
        if line.startswith("[text] "):
            result.append(line[7:])
            in_text_block = True
        elif line.startswith(("[tool]", "[ok]", "[FAIL]", "[done]")):
            in_text_block = False
        elif in_text_block:
            result.append(line)
    return result


def _extract_latest_status_from_log(log_path: Path) -> str:
    """Extract the most recent substantive [text] line from a log for live status."""
    if not log_path.exists():
        return ""
    for line in reversed(log_path.read_text().splitlines()):
        if line.startswith("[text] "):
            text = line[7:].strip()
            if len(text) > 5 and not text.startswith(("#", "---", "```", "|")):
                return text[:300]
    return ""


def _preflight_health_check(cfg: AppConfig, role: str, target_alias: str) -> None:
    """Ping the inference API before spawning agents. Fail fast if unreachable."""
    from pipeline.phase2.run import health_check, make_api_client
    from pipeline.config import resolve_generator_model, resolve_judge_model

    client, _ = make_api_client(cfg)
    checked: set[str] = set()

    if role == "judge":
        judge_cfg = resolve_judge_model(cfg, target_alias)
        health_check(client, judge_cfg.api_name)
        checked.add(judge_cfg.api_name)
        for gen_cfg in cfg.phase2.generator_models:
            if gen_cfg.api_name not in checked:
                health_check(client, gen_cfg.api_name)
                checked.add(gen_cfg.api_name)
    else:
        gen_cfg = resolve_generator_model(cfg, target_alias)
        health_check(client, gen_cfg.api_name)
        checked.add(gen_cfg.api_name)
        for jdg_cfg in cfg.phase2.judge_models:
            if jdg_cfg.api_name not in checked:
                health_check(client, jdg_cfg.api_name)
                checked.add(jdg_cfg.api_name)

    logger.info("Pre-flight health check passed for {} / {}.", role, target_alias)


ALLOWED_TOOLS = [
    "Read", "Glob", "Grep",
    "Bash(uv run python:*)",
    f"Bash(rm -f {AGENT_TMP_DIR}/:*)",
    f"Bash(rm {AGENT_TMP_DIR}/:*)",
    f"Bash(ls {AGENT_TMP_DIR}:*)",
    "Agent", "TaskCreate", "TaskUpdate", "TaskList",
    "Write",
]


def run_improver(cfg: AppConfig, role: str, target_alias: str) -> None:
    """Run a single improver for one (role, model) pair.

    Independently launchable. Writes progress to loop_status.json
    under improvers["{role}_{alias}"].
    """
    key = f"{role}_{target_alias}"
    log_path = improver_log_path(role, target_alias)

    status = read_status() or {}
    now = datetime.now(timezone.utc).isoformat()

    # Update this improver's status to running
    improvers = status.get("improvers", {})
    improvers[key] = {**_make_improver_status("running"), "started_at": now}
    status["improvers"] = improvers
    status["current"] = key
    write_status(status)

    try:
        # Pre-flight health check
        _preflight_health_check(cfg, role, target_alias)

        # Clear scratch directory
        if AGENT_TMP_DIR.exists():
            shutil.rmtree(AGENT_TMP_DIR)
        AGENT_TMP_DIR.mkdir(parents=True, exist_ok=True)

        prompt = _build_improver_prompt(cfg, role, target_alias)
        _spawn_agent(prompt, log_path, ALLOWED_TOOLS)

        # Detect if new prompts were written
        new_prompt = _detect_new_prompts(target_alias, role)
        logger.info("Improver {} done: latest prompt -> {}", key, new_prompt)

        # Re-judge for correlations after judge improvement
        if role == "judge":
            from pipeline.phase2.run import rejudge_all_prompts_and_models
            logger.info("Running rejudge_all_prompts_and_models after judge improvement...")
            rejudge_all_prompts_and_models(cfg)

        now_done = datetime.now(timezone.utc).isoformat()
        improvers[key]["status"] = "done"
        improvers[key]["reasoning"] = _extract_reasoning_from_log(log_path)
        improvers[key]["finished_at"] = now_done

    except KeyboardInterrupt:
        improvers[key]["status"] = "error"
        improvers[key]["reasoning"] = "Interrupted by user"
        status["error"] = "Interrupted by user"
        write_status(status)
        raise
    except Exception as e:
        improvers[key]["status"] = "error"
        improvers[key]["reasoning"] = str(e)[:500]
        status["error"] = str(e)
        write_status(status)
        raise
    finally:
        status["current"] = None
        write_status(status)


def run_judge_improvers(cfg: AppConfig, aliases: list[str] | None = None) -> None:
    """Run judge improvers sequentially. If aliases is None, run ALL judge models."""
    if aliases is None:
        aliases = [m.alias for m in cfg.phase2.judge_models]

    now = datetime.now(timezone.utc).isoformat()
    prompts_before = _snapshot_prompts(cfg)

    # Initialize status with all selected aliases as pending
    improvers = {f"judge_{a}": _make_improver_status("pending") for a in aliases}
    status = {
        "running": True,
        "started_at": now,
        "role": "judge",
        "improvers": improvers,
        "current": None,
        "error": None,
    }
    write_status(status)

    try:
        for alias in aliases:
            run_improver(cfg, "judge", alias)
            # Re-read status (run_improver updates it)
            status = read_status() or status

        status["running"] = False
        write_status(status)
        logger.info("All judge improvers complete.")

    except (KeyboardInterrupt, Exception):
        status = read_status() or status
        status["running"] = False
        # Mark remaining pending improvers as skipped
        for key, data in status.get("improvers", {}).items():
            if data["status"] == "pending":
                data["status"] = "skipped"
        write_status(status)
        raise

    finally:
        try:
            _save_history(status, prompts_before, cfg)
        except Exception as e:
            logger.error("Failed to save loop history to DB: {}", e)
            fallback = PIPELINE_DATA_DIR / "loop_history_fallback.json"
            fallback.write_text(json.dumps(status, indent=2, default=str))


def run_generator_improvers(cfg: AppConfig, aliases: list[str] | None = None) -> None:
    """Run generator improvers sequentially. If aliases is None, run ALL generator models."""
    if aliases is None:
        aliases = [m.alias for m in cfg.phase2.generator_models]

    now = datetime.now(timezone.utc).isoformat()
    prompts_before = _snapshot_prompts(cfg)

    improvers = {f"generator_{a}": _make_improver_status("pending") for a in aliases}
    status = {
        "running": True,
        "started_at": now,
        "role": "generator",
        "improvers": improvers,
        "current": None,
        "error": None,
    }
    write_status(status)

    try:
        for alias in aliases:
            run_improver(cfg, "generator", alias)
            status = read_status() or status

        status["running"] = False
        write_status(status)
        logger.info("All generator improvers complete.")

    except (KeyboardInterrupt, Exception):
        status = read_status() or status
        status["running"] = False
        for key, data in status.get("improvers", {}).items():
            if data["status"] == "pending":
                data["status"] = "skipped"
        write_status(status)
        raise

    finally:
        try:
            _save_history(status, prompts_before, cfg)
        except Exception as e:
            logger.error("Failed to save loop history to DB: {}", e)
            fallback = PIPELINE_DATA_DIR / "loop_history_fallback.json"
            fallback.write_text(json.dumps(status, indent=2, default=str))


def _save_history(status: dict, prompts_before: dict[str, str], cfg: AppConfig) -> None:
    """Persist a completed improver run to loop_history table with prompt snapshots and logs."""
    prompts_after = _snapshot_prompts(cfg)

    # Extract reasoning from logs for any improvers missing it
    for key, data in status.get("improvers", {}).items():
        if not data.get("reasoning") and data.get("status") != "pending":
            parts = key.split("_", 1)
            if len(parts) == 2:
                role, alias = parts
                log_p = improver_log_path(role, alias)
                data["reasoning"] = _extract_reasoning_from_log(log_p)

    # Capture full logs
    logs = {}
    for key in status.get("improvers", {}):
        parts = key.split("_", 1)
        if len(parts) == 2:
            role, alias = parts
            log_p = improver_log_path(role, alias)
            if log_p.exists():
                logs[key] = log_p.read_text()

    record = {
        "started_at": status.get("started_at"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "role": status.get("role"),
        "improvers": status.get("improvers", {}),
        "error": status.get("error"),
        "prompts_before": prompts_before,
        "prompts_after": prompts_after,
        "logs": logs,
    }
    save_loop_run(record)


def main():
    """CLI entry point for the independent improver runners."""
    import argparse

    parser = argparse.ArgumentParser(description="Run improver agents")
    parser.add_argument("--role", required=True, choices=["judge", "generator"])
    parser.add_argument("--aliases", type=str, default=None,
                        help="Comma-separated model aliases (default: all)")
    args = parser.parse_args()

    cfg = load_config()
    aliases = args.aliases.split(",") if args.aliases else None

    logger.info("Starting {} improver(s)", args.role)
    if aliases:
        logger.info("Aliases: {}", aliases)

    if args.role == "judge":
        run_judge_improvers(cfg, aliases=aliases)
    else:
        run_generator_improvers(cfg, aliases=aliases)


if __name__ == "__main__":
    main()
