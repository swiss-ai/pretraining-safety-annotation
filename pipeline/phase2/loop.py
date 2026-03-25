"""Independent improver runners: judge improver and generator improver.

Each improver spawns an Opus agent that autonomously calls cross-iteration
tools via CLI. Multiple models run in parallel, each with its own scratch
directory and thread-safe status updates.

Usage:
    uv run python -m pipeline.phase2.loop --role judge [--aliases glm45,olmo3-32B-think]
    uv run python -m pipeline.phase2.loop --role generator [--aliases glm45]
"""

from __future__ import annotations

import glob
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import (
    PIPELINE_DATA_DIR,
    PROMPTS_DIR,
    _INIT_PROMPTS_DIR,
    AppConfig,
    load_config,
    resolve_prompt_path,
)
from pipeline.log import logger
from pipeline.phase2.storage import save_loop_run

# Re-export from agent_utils for backward compatibility
from pipeline.agent_utils import (  # noqa: F401
    STATUS_PATH,
    AGENT_TMP_DIR,
    improver_log_path,
    write_status,
    read_status,
    _update_status,
    _make_improver_status,
    _spawn_agent,
    _stream_improver_output,
    _validate_agent_output,
    _allowed_tools,
    _log_line,
    _summarize_tool_input,
    _snapshot_prompts,
    _extract_reasoning_from_log,
    _collect_text_lines,
    _extract_latest_status_from_log,
    _active_procs,
    _active_procs_lock,
)

# Keep backward-compat alias for dashboard polling
IMPROVER_LOG_PATH = PIPELINE_DATA_DIR / "improver_log_judge.txt"


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


def _build_improver_prompt(
    cfg: AppConfig, role: str, target_alias: str, agent_tmp_dir: Path | None = None
) -> str:
    """Unified prompt builder for judge and generator improvers.

    Key additions over old phase prompts:
    - Lists ALL counterpart models the agent's cross-iterations will use
    - Tells agent to use `run_cross_batch --role {role} --target {alias}`
    - Explains group_id for interpreting cross-model results
    - Points to `cross_summary <group_id>` command for aggregated stats
    """
    if agent_tmp_dir is None:
        agent_tmp_dir = AGENT_TMP_DIR
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
    has_data = latest_iter > 0

    counterpart_list = ", ".join(counterpart_models)

    if has_data:
        first_run_note = ""
    else:
        first_run_note = f"""
## FIRST RUN — No data exists yet!
There are no iterations in the database. You MUST run a baseline batch first before analyzing anything.
Do this immediately (run in background — it takes 3-5 minutes!):
  Bash: {{"command": "uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias} 2>&1", "run_in_background": true}}
Then wait:
  TaskOutput: {{"task_id": "<id>", "block": true, "timeout": 600000}}
Then use `diagnose <group_id>` for a full analysis.
Do NOT waste time querying empty data — run the batch first.
"""

    return f"""You are improving {role_label} prompts for a pretraining data annotation pipeline.

## {role_label} Improvement for model: {target_alias}

**IMPORTANT: Other improver agents may be running in parallel for different models.**
You are ONLY responsible for the **{target_alias}** model. Ignore iterations, errors, or data
belonging to other models. When analyzing results, filter by your target model.

{phase_instructions}
{first_run_note}
## Cross-model evaluation
Your improvements will be tested against ALL {other_type} models: [{counterpart_list}].
When you run a batch, it creates one iteration per {other_type} model, all sharing a group_id.
Use `cross_summary <group_id>` to see aggregated per-model stats after each batch.

## Query tools
Run these via Bash (prefix with `uv run`). Replace <ITER> with the iteration number from your batch output.
  uv run python -m pipeline.improver_tools summary <ITER>     — aggregate stats
  uv run python -m pipeline.improver_tools failures <ITER>    — rejected items with reasoning
  uv run python -m pipeline.improver_tools failures <ITER> --reasoning-limit 500  — full reasoning
  uv run python -m pipeline.improver_tools diversity <ITER>   — frequency-based diversity analysis
  uv run python -m pipeline.improver_tools scores <ITER>      — compact scores table
  uv run python -m pipeline.improver_tools show <id> <ITER>   — full text + outputs for one item
  uv run python -m pipeline.improver_tools show <id1,id2,...> <ITER>  — batch show multiple items
  uv run python -m pipeline.improver_tools show <id> <ITER> --brief   — truncated source text
  uv run python -m pipeline.improver_tools show --gold <ITER> [--brief] — all gold items for iteration
  uv run python -m pipeline.improver_tools item <id> <ITER>   — full details as JSON
  uv run python -m pipeline.improver_tools reasoning <id>[,id2,...] <ITER> — full judge reasoning (scores + text)
  uv run python -m pipeline.improver_tools gold                      — gold annotations (concise, no source text)
  uv run python -m pipeline.improver_tools gold --verbose            — gold with full source text (large output!)
  uv run python -m pipeline.improver_tools compare <id> <ITER> — generated vs gold
  uv run python -m pipeline.improver_tools reviews [<ITER>]   — human reviews with judge comparison
  uv run python -m pipeline.improver_tools filter <ITER> --dim <dimension> --below <threshold> [--part preflection|reflection]
  uv run python -m pipeline.improver_tools trend                     — cross-iteration comparison table
  uv run python -m pipeline.improver_tools correlations              — judge-human correlation by judge version

## Test tools (run experiments WITHOUT modifying main data)
  uv run python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2] [--n N] [--role {role}]
  uv run python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2] [--n N] [--role {role}]
  uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias}  — cross-iteration with ALL {other_type} models
  uv run python -m pipeline.improver_tools cross_summary <group_id>   — per-model stats for a cross-iteration
  uv run python -m pipeline.improver_tools diagnose <group_id>        — ONE-SHOT full analysis (use this first!)
  uv run python -m pipeline.improver_tools diff <iter1> <iter2> [--limit N]  — cross-iteration item comparison
  uv run python -m pipeline.improver_tools test_results --role {role}  — view test results

## Running long commands (CRITICAL — `run_cross_batch` takes 3-5 minutes!)
`run_cross_batch`, `run_batch`, `test_judge`, and `test_generate` make many API calls and
take several minutes. You MUST run them in background and wait with a long timeout:
```
Bash: {{"command": "uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias} 2>&1", "run_in_background": true}}
```
Then wait for the result:
```
TaskOutput: {{"task_id": "<id from above>", "block": true, "timeout": 600000}}
```
**NEVER** run these commands synchronously (default Bash timeout is 120s — too short).
**ALWAYS** include `2>&1` to capture stderr.

## Scratch directory — ALL scripts go here, NOWHERE ELSE
Write ad-hoc analysis scripts to: {agent_tmp_dir}
Run them with: uv run python {agent_tmp_dir}/your_script.py
Delete with: rm {agent_tmp_dir}/your_script.py
This folder is cleared at the start of each loop. Use it for any analysis the CLI tools don't cover.
**NEVER write scripts to `scripts/`, the project root, or anywhere else** — you cannot delete
files outside {agent_tmp_dir}/ and they will be left behind as garbage.

## State
Read your state file at {state_path} FIRST. It contains notes from previous iterations.

## Strategy: use Opus subagents for parallel exploration (IMPORTANT)
You have access to the Agent tool. **Always use model="opus" for subagents** — they need
strong reasoning. Parallelize aggressively; the bottleneck is wall-clock time, not tokens:
- Spawn one subagent to analyze failures and low-scoring items in detail
- Spawn another to compare generated outputs with gold annotations
- Spawn another to read ALL human reviews (`reviews` without iteration filter — output is large, must use subagent!) and extract key insights from reviewer notes into state.md
- Spawn another to check diversity patterns
Then synthesize their findings to write improved prompts.

**Avoid redundancy**: When you delegate analysis to subagents, do NOT run the same queries
yourself. Wait for subagent results before proceeding. Never call the same command twice —
save the output mentally and reuse it. Prefer batch commands (`scores`, `summary`, `diversity`)
over inspecting items one by one.

## Your task
1. Read your state file: {state_path}
2. Read the improver instructions: {improver_path}
3. Read the current {prompt_type} prompt: {prompt_path} and the {other_type} prompt for context: {other_prompt_path}
4. If no data exists yet, run a baseline batch first (see "Running long commands" above — use background + 600s timeout):
   `uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias} 2>&1`
5. Analyze results: start with `diagnose <group_id>` for a full overview, then drill into specifics with `diff`, `failures`, `show`
6. Write improved {prompt_type} to {model_dir}/{prompt_type}_v{next_v}.md
7. You may run up to {max_batches} `run_cross_batch` calls to test your changes
8. Update {state_path} with: what you changed, why, key metrics, and what to try next
9. **Compress state.md**: Spawn a subagent to read {state_path} and rewrite it more concisely.
   The subagent should: keep the current active prompts and key findings, condense old iteration
   rows into brief takeaways (e.g. "v2-v4 failed because X — don't try again"), drop per-item
   details and verbose tables from past rounds, and preserve any "what NOT to try" lessons.
   The goal is a useful reference for future you, not a detailed log.
10. Print a **single final summary** as your VERY LAST message. This summary is parsed and displayed in the dashboard.
    It MUST start with exactly `## Final Summary` on its own line, followed by:
    - **What changed**: which prompt file, key modifications
    - **Why**: what problems you identified, with evidence (scores, examples)
    - **Results**: before/after metrics if you ran test batches
    - **Next steps**: what to try in the next iteration

## RULES — VIOLATIONS WASTE YOUR LIMITED TOOL CALLS

1. **NO PIPES.** Never use `|` in Bash commands. No `| tail`, `| head`, `| grep`. Run commands
   separately. Pipes hide exit codes and break background execution.
2. **NO RAW SQL.** Never open sqlite3 directly. The database is at `data/storage.db` (NOT
   `pipeline/pipeline.db`), but you should NEVER need it — all data is available via
   `python -m pipeline.improver_tools`. Use `reasoning <id> <iter>` for judge reasoning details.
3. **SCRIPTS ONLY IN {agent_tmp_dir}/.** Never write to `scripts/`, project root, or anywhere
   else — you cannot delete files outside {agent_tmp_dir}/.
4. Use `uv run python -m pipeline.improver_tools ...` for data access — NOT raw file reads.
5. Do NOT overfit to individual examples. Focus on systematic patterns.
6. The {prompt_type} prompt must NOT hardcode specific charter/constitution content.
7. `diff <iter1> <iter2>` only works for iterations that share source items (i.e. iterations
   within the SAME cross-batch group). Do NOT diff across different groups.

## Bash Python tips (avoids permission errors)
When running ad-hoc Python via Bash, **NEVER** use `python -c "..."` with multi-line strings.
Instead, use heredoc syntax:
```bash
uv run python << 'PYEOF'
from pipeline.phase2.storage import load_items_for_iteration
items = load_items_for_iteration(3)
print(len(items))
PYEOF
```
Or write a script to {agent_tmp_dir}/ and run it with `uv run python {agent_tmp_dir}/script.py`.
The inline `python -c` form triggers security filters on `#` comments and `{{` braces.
"""


def _preflight_health_check(cfg: AppConfig, role: str, target_alias: str) -> None:
    """Ping the inference API before spawning agents. Fail fast if unreachable."""
    from pipeline.phase2.run import _health_check_models, make_api_client

    client, _ = make_api_client(
        cfg.phase2.endpoint, cfg.phase2.iteration.max_concurrent
    )
    _health_check_models(client, cfg, role, target_alias)
    logger.info("Pre-flight health check passed for {} / {}.", role, target_alias)


# Backward-compat alias for external callers
ALLOWED_TOOLS = _allowed_tools(AGENT_TMP_DIR)


def run_improver(cfg: AppConfig, role: str, target_alias: str) -> None:
    """Run a single improver for one (role, model) pair.

    Independently launchable and thread-safe. Each improver gets its own
    scratch directory and updates loop_status.json atomically.
    """
    key = f"{role}_{target_alias}"
    log_path = improver_log_path(role, target_alias)
    tmp_dir = PIPELINE_DATA_DIR / f"tmp_{role}_{target_alias}"
    now = datetime.now(timezone.utc).isoformat()

    _update_status(
        lambda s: s.setdefault("improvers", {}).update(
            {key: {**_make_improver_status("running"), "started_at": now}}
        )
    )

    try:
        _preflight_health_check(cfg, role, target_alias)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        prompt = _build_improver_prompt(cfg, role, target_alias, agent_tmp_dir=tmp_dir)
        _spawn_agent(prompt, log_path, _allowed_tools(tmp_dir), key=key)

        new_prompt = _detect_new_prompts(target_alias, role)
        logger.info("Improver {} done: latest prompt -> {}", key, new_prompt)

        if role == "judge":
            from pipeline.phase2.run import rejudge_all_prompts_and_models

            logger.info(
                "Running rejudge_all_prompts_and_models after judge improvement..."
            )
            rejudge_all_prompts_and_models(cfg)

        now_done = datetime.now(timezone.utc).isoformat()
        reasoning = _extract_reasoning_from_log(log_path)
        _update_status(
            lambda s: s["improvers"][key].update(
                {"status": "done", "reasoning": reasoning, "finished_at": now_done}
            )
        )

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


def _run_improvers(cfg: AppConfig, role: str, aliases: list[str] | None = None) -> None:
    """Run improvers for a role in parallel. If aliases is None, run ALL models for that role."""
    if aliases is None:
        model_list = (
            cfg.phase2.judge_models if role == "judge" else cfg.phase2.generator_models
        )
        aliases = [m.alias for m in model_list]

    now = datetime.now(timezone.utc).isoformat()
    prompts_before = _snapshot_prompts(cfg)

    improvers = {f"{role}_{a}": _make_improver_status("pending") for a in aliases}
    status = {
        "running": True,
        "started_at": now,
        "role": role,
        "improvers": improvers,
        "error": None,
    }
    write_status(status)

    errors: dict[str, Exception] = {}

    try:
        with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
            futures = {
                pool.submit(run_improver, cfg, role, alias): alias for alias in aliases
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    future.result()
                except Exception as e:
                    errors[alias] = e
                    logger.error("Improver {}/{} failed: {}", role, alias, e)

        def _finalize(s: dict) -> None:
            s["running"] = False
            if errors:
                s["error"] = "; ".join(f"{a}: {e}" for a, e in errors.items())
            for key, data in s.get("improvers", {}).items():
                if data["status"] == "pending":
                    data["status"] = "skipped"

        _update_status(_finalize)
        logger.info("All {} improvers complete ({} errors).", role, len(errors))

    except KeyboardInterrupt:

        def _interrupted(s: dict) -> None:
            s["running"] = False
            for key, data in s.get("improvers", {}).items():
                if data["status"] == "pending":
                    data["status"] = "skipped"

        _update_status(_interrupted)
        raise

    finally:
        try:
            status = read_status() or status
            _save_history(status, prompts_before, cfg)
        except Exception as e:
            logger.error("Failed to save loop history to DB: {}", e)
            fallback = PIPELINE_DATA_DIR / "loop_history_fallback.json"
            fallback.write_text(json.dumps(status, indent=2, default=str))


def run_judge_improvers(cfg: AppConfig, aliases: list[str] | None = None) -> None:
    """Run judge improvers in parallel. If aliases is None, run ALL judge models."""
    _run_improvers(cfg, "judge", aliases)


def run_generator_improvers(cfg: AppConfig, aliases: list[str] | None = None) -> None:
    """Run generator improvers in parallel. If aliases is None, run ALL generator models."""
    _run_improvers(cfg, "generator", aliases)


def interrupt_improvers() -> int:
    """Terminate all active improver agent subprocesses.

    Returns the number of processes terminated. Also marks running improvers
    as errored in the status file.
    """
    with _active_procs_lock:
        procs = dict(_active_procs)

    killed = 0
    for key, proc in procs.items():
        logger.info("Interrupting improver agent: {}", key)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        killed += 1

    def _mark_interrupted(s: dict) -> None:
        s["running"] = False
        s["error"] = "Interrupted by user"
        for data in s.get("improvers", {}).values():
            if data["status"] in ("running", "pending"):
                data["status"] = "error"
                data["reasoning"] = "Interrupted by user"

    _update_status(_mark_interrupted)
    logger.info("Interrupted {} improver agent(s).", killed)
    return killed


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
    parser.add_argument(
        "--aliases",
        type=str,
        default=None,
        help="Comma-separated model aliases (default: all)",
    )
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
