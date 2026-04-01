"""Shared utilities for spawning and managing improver agents.

Extracted from pipeline/phase2/loop.py for reuse by phase3 and future phases.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import PIPELINE_DATA_DIR, PROMPTS_DIR, PROJECT_ROOT, AppConfig
from pipeline.log import logger

STATUS_PATH = PIPELINE_DATA_DIR / "loop_status.json"
AGENT_TMP_DIR = PIPELINE_DATA_DIR / "tmp"
_status_lock = threading.Lock()
_active_procs: dict[str, subprocess.Popen] = {}
_active_procs_lock = threading.Lock()

_ERROR_PATTERNS = [
    "authentication_error",
    "OAuth token has expired",
    "You've hit your limit",
    "API Error: 4",
    "API Error: 5",
]


def improver_log_path(role: str, alias: str) -> Path:
    """Log path for a specific improver, e.g. data/pipeline/improver_log_judge_glm45.txt"""
    return PIPELINE_DATA_DIR / f"improver_log_{role}_{alias}.txt"


def write_status(status: dict) -> None:
    """Atomically write loop status to JSON file (thread-safe)."""
    with _status_lock:
        PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, STATUS_PATH)


def read_status() -> dict | None:
    """Read current loop status, or None if no status file exists (thread-safe)."""
    with _status_lock:
        if not STATUS_PATH.exists():
            return None
        return json.loads(STATUS_PATH.read_text())


def _update_status(updater: callable) -> dict:
    """Read-modify-write loop status atomically.

    updater receives the status dict and mutates it in place.
    Returns the updated status.
    """
    with _status_lock:
        status = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
        updater(status)
        PIPELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, STATUS_PATH)
        return status


def _make_improver_status(status_str: str = "pending") -> dict:
    """Create an improver status dict."""
    return {
        "status": status_str,
        "reasoning": "",
        "started_at": None,
        "finished_at": None,
    }


def _spawn_agent(
    prompt: str, log_path: Path, allowed_tools: list[str], key: str = ""
) -> str:
    """Spawn a sandboxed Claude CLI subprocess and return its text output.

    Streams output to log_path and stderr for real-time monitoring.
    """
    settings = json.dumps(
        {
            "permissions": {
                "allow": allowed_tools,
                "deny": ["NotebookEdit"],
            }
        }
    )
    cmd = [
        "claude",
        "--print",
        "--model",
        "opus",
        "--effort",
        "max",
        "--verbose",
        "--output-format",
        "stream-json",
        "--settings",
        settings,
        "--",
        prompt,
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

    if key:
        with _active_procs_lock:
            _active_procs[key] = proc

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
    finally:
        if key:
            with _active_procs_lock:
                _active_procs.pop(key, None)

    proc.wait()
    output = final_text_holder[0] if final_text_holder else ""

    if proc.returncode != 0:
        summary = output[:500] if output else f"No output. See {log_path}"
        raise RuntimeError(f"Claude agent failed (rc={proc.returncode}): {summary}")

    _validate_agent_output(output, log_path)
    return output


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
                            preview = content[:500].replace("\n", " ")
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


def run_improver_agent(
    prompt: str,
    key: str,
    log_path: Path,
    tmp_dir: Path,
    post_hook: Callable[[], None] | None = None,
) -> str:
    """Generic improver agent runner with status tracking.

    Handles the full lifecycle: status tracking, tmp dir creation,
    agent spawning, reasoning extraction, status updates, cleanup.

    Args:
        prompt: The fully-built prompt for the agent.
        key: Status key (e.g. "judge_glm45", "summary_glm45").
        log_path: Where to write the agent's log.
        tmp_dir: Scratch directory for the agent's scripts.
        post_hook: Optional callback after successful completion.

    Returns:
        Extracted reasoning from the agent's final summary.
    """
    now = datetime.now(timezone.utc).isoformat()

    _update_status(
        lambda s: s.setdefault("improvers", {}).update(
            {key: {**_make_improver_status("running"), "started_at": now}}
        )
    )

    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        _spawn_agent(prompt, log_path, _allowed_tools(tmp_dir), key=key)

        if post_hook is not None:
            post_hook()

        now_done = datetime.now(timezone.utc).isoformat()
        reasoning = _extract_reasoning_from_log(log_path)
        _update_status(
            lambda s: s["improvers"][key].update(
                {"status": "done", "reasoning": reasoning, "finished_at": now_done}
            )
        )
        return reasoning

    except KeyboardInterrupt:
        _update_status(
            lambda s: (
                s["improvers"][key].update(
                    {"status": "error", "reasoning": "Interrupted by user"}
                ),
                s.update({"error": "Interrupted by user"}),
            )
        )
        raise
    except Exception as e:
        err = str(e)[:500]
        _update_status(
            lambda s: (
                s["improvers"][key].update({"status": "error", "reasoning": err}),
                s.update({"error": str(e)}),
            )
        )
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _allowed_tools(tmp_dir: Path) -> list[str]:
    """Build allowed tools list for a given scratch directory.

    Write and Edit are restricted to the prompt directory and the agent's
    scratch directory to prevent littering the project root.
    """
    prompts_dir = PIPELINE_DATA_DIR / "prompts"
    return [
        "Read",
        f"Edit(//{prompts_dir}/**)",
        f"Edit(//{tmp_dir}/**)",
        "Glob",
        "Grep",
        "Bash(uv run python:*)",
        "Bash(uv run python <<:*)",
        f"Bash(rm -f {tmp_dir}/:*)",
        f"Bash(rm {tmp_dir}/:*)",
        f"Bash(ls {tmp_dir}:*)",
        "Agent",
        "TaskCreate",
        "TaskUpdate",
        "TaskList",
        f"Write(//{prompts_dir}/**)",
        f"Write(//{tmp_dir}/**)",
    ]


def _snapshot_prompts(cfg: AppConfig) -> dict[str, str]:
    """Capture current prompt file contents keyed by 'alias/filename'."""
    prompts = {}
    all_aliases = set()
    for m in cfg.phase2.judge_models + cfg.phase2.generator_models:
        all_aliases.add(m.alias)
    for m in cfg.phase3.target_models:
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
