"""Phase 3 improver runners: one per target model per role.

Each improver spawns an Opus agent that autonomously runs paired iterations
and optimizes target model prompts to align with gold model outputs.

Usage:
    uv run python -m pipeline.phase3.loop --role judge [--aliases olmo3-7B-think]
    uv run python -m pipeline.phase3.loop --role generator [--aliases olmo3-7B-think]
"""

from __future__ import annotations

import glob
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pipeline.agent_utils import (
    AGENT_TMP_DIR,
    _allowed_tools,
    _extract_reasoning_from_log,
    _make_improver_status,
    _snapshot_prompts,
    _spawn_agent,
    _update_status,
    improver_log_path,
    read_status,
    write_status,
)
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


def _extract_version(filename: str) -> int:
    """Extract version number from a filename like 'generator_v3.md'."""
    match = re.search(r"_v(\d+)\.md$", filename)
    assert match, f"Cannot extract version from {filename}"
    return int(match.group(1))


def _detect_new_prompts(alias: str, role: str) -> str:
    """Find the highest-versioned prompt for the given role in the model directory."""
    model_dir = PROMPTS_DIR / alias
    prefix = "judge" if role == "judge" else "generator"
    matches = sorted(glob.glob(str(model_dir / f"{prefix}_v*.md")))
    assert matches, f"No {prefix}_v*.md files in {model_dir}"
    return Path(matches[-1]).name


def _build_phase3_improver_prompt(
    cfg: AppConfig, role: str, target_alias: str, agent_tmp_dir: Path | None = None
) -> str:
    """Build the prompt for a phase3 improver agent.

    Tells the agent about gold models, target model, correlation metrics,
    and available paired analysis tools.
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
        phase_prompt_path = _INIT_PROMPTS_DIR / cfg.phase3.improver.judge_prompt
        role_label = "JUDGE"
        prompt_type = "judge"
        other_type = "generator"
    else:
        prompt_path = resolve_prompt_path("generator_latest.md", target_alias)
        other_prompt_path = resolve_prompt_path("judge_latest.md", target_alias)
        current_v = _extract_version(prompt_path.name)
        next_v = current_v + 1
        phase_prompt_path = _INIT_PROMPTS_DIR / cfg.phase3.improver.generator_prompt
        role_label = "GENERATOR"
        prompt_type = "generator"
        other_type = "judge"

    state_path = model_dir / "state.md"
    if not state_path.exists():
        state_path.write_text("# Improver State\n\nNo previous iterations.\n")

    max_batches = cfg.phase3.improver.max_batches_per_phase
    max_escalations = cfg.phase3.escalation.max_per_iteration

    if phase_prompt_path.exists():
        phase_instructions = phase_prompt_path.read_text(encoding="utf-8")
    else:
        phase_instructions = (
            f"Focus on improving {prompt_type} prompts for cross-model alignment."
        )

    gold_judge_aliases = [m.alias for m in cfg.phase3.gold_judges]
    gold_gen_aliases = [m.alias for m in cfg.phase3.gold_generators]

    from pipeline.phase2.storage import load_runs

    runs = load_runs()
    phase3_runs = [r for r in runs if r.get("phase") == "phase3"]
    has_data = len(phase3_runs) > 0

    if has_data:
        first_run_note = ""
    else:
        first_run_note = f"""
## FIRST RUN — No data exists yet!
There are no phase3 iterations in the database. You MUST run a baseline paired batch first.
Do this immediately (run in background — it takes 3-5 minutes!):
  Bash: {{"command": "uv run python -m pipeline.improver_tools run_paired_batch --role {role} --target {target_alias} 2>&1", "run_in_background": true}}
Then wait:
  TaskOutput: {{"task_id": "<id>", "block": true, "timeout": 600000}}
Then use `paired_summary <group_id>` for correlation analysis.
Do NOT waste time querying empty data — run the batch first.
"""

    return f"""You are improving {role_label} prompts for target model {target_alias} to align with gold model outputs.

## {role_label} Improvement for model: {target_alias} (Phase 3: Cross-Model Alignment)

**IMPORTANT: Other improver agents may be running in parallel for different models.**
You are ONLY responsible for the **{target_alias}** model. Ignore iterations, errors, or data
belonging to other models. When analyzing results, filter by your target model.

## Gold Models (FROZEN — do NOT modify their prompts)
Gold judges: {gold_judge_aliases}
Gold generators: {gold_gen_aliases}
Gold model outputs are near-ground-truth. Trust them to identify where your target model diverges.

{phase_instructions}
{first_run_note}
## Your Primary Metrics
1. Spearman rank correlation between target and gold aggregate scores
2. Decision concordance (accept/reject agreement rate)
3. Per-dimension mean absolute score difference

## What "Good" Looks Like
- High Spearman rho (>0.7 is good, >0.85 is excellent)
- High decision concordance (>80%)
- Small per-dimension MAD (<0.5 on 1-5 scale)
- Correlation matters MORE than absolute score level

## Claude as Ultimate Judge
You are Claude Opus — you are the ultimate gold judge. When analyzing items:
- You can judge items YOURSELF using your own expert assessment
- Your judgment overrides even gold model outputs when you have good reason
- Use this power to resolve disagreements between gold models
- Use this to spot cases where all gold models are systematically wrong

## Escalation
If even YOU are uncertain about the correct judgment for an item, escalate to human reviewers:
  uv run python -m pipeline.improver_tools escalate <item_id> <group_id> --reason "..."
Use sparingly (max {max_escalations} per iteration). Most disagreements you should resolve yourself.

## Query tools
Run these via Bash (prefix with `uv run`):
  uv run python -m pipeline.improver_tools run_paired_batch --role {role} --target {target_alias}
  uv run python -m pipeline.improver_tools paired_summary <group_id>
  uv run python -m pipeline.improver_tools disagreements <group_id> [--limit N]
  uv run python -m pipeline.improver_tools dimension_alignment <group_id>
  uv run python -m pipeline.improver_tools paired_show <item_id> <group_id>
  uv run python -m pipeline.improver_tools escalate <item_id> <group_id> --reason "..."
  uv run python -m pipeline.improver_tools escalations [--status pending]
  uv run python -m pipeline.improver_tools correlation_trend --target {target_alias}
  [+ all existing phase2 tools: summary, failures, scores, show, etc.]

## Running long commands (CRITICAL — paired batches take 3-10 minutes!)
`run_paired_batch` makes many API calls and takes several minutes.
You MUST run them in background and wait with a long timeout:
```
Bash: {{"command": "uv run python -m pipeline.improver_tools run_paired_batch --role {role} --target {target_alias} 2>&1", "run_in_background": true}}
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
**NEVER write scripts to `scripts/`, the project root, or anywhere else.**

## State
Read your state file at {state_path} FIRST. It contains notes from previous iterations.

## Strategy: use Opus subagents for parallel exploration (IMPORTANT)
You have access to the Agent tool. **Always use model="opus" for subagents** — they need
strong reasoning. Parallelize aggressively; the bottleneck is wall-clock time, not tokens.

## Your task
1. Read your state file: {state_path}
2. Read the improver instructions: {improver_path}
3. Read the current {prompt_type} prompt: {prompt_path} and the {other_type} prompt for context: {other_prompt_path}
4. If no data exists yet, run a baseline paired batch first (see "Running long commands" above)
5. Analyze results: start with `paired_summary <group_id>` for correlation metrics, then drill into `disagreements`
6. Write improved {prompt_type} to {model_dir}/{prompt_type}_v{next_v}.md
7. You may run up to {max_batches} `run_paired_batch` calls to test your changes
8. Update {state_path} with: what you changed, why, key metrics, and what to try next
9. **Compress state.md**: Spawn a subagent to read {state_path} and rewrite it more concisely.
   The subagent should: keep the current active prompts and key findings, condense old iteration
   rows into brief takeaways (e.g. "v2-v4 failed because X — don't try again"), drop per-item
   details and verbose tables from past rounds, and preserve any "what NOT to try" lessons.
   The goal is a useful reference for future you, not a detailed log.
10. Print a **single final summary** as your VERY LAST message. This summary is parsed and displayed in the dashboard.
    It MUST start with exactly `## Final Summary` on its own line, followed by:
    - **What changed**: which prompt file, key modifications
    - **Why**: what divergence patterns you identified, with evidence
    - **Results**: before/after correlation metrics
    - **Next steps**: what to try in the next iteration

## RULES — VIOLATIONS WASTE YOUR LIMITED TOOL CALLS

1. **NO PIPES.** Never use `|` in Bash commands.
2. **NO RAW SQL.** Use `python -m pipeline.improver_tools` for all data access.
3. **SCRIPTS ONLY IN {agent_tmp_dir}/.** Never write to `scripts/`, project root, or anywhere else.
4. Do NOT overfit to individual examples. Focus on systematic patterns.
5. The {prompt_type} prompt must NOT hardcode specific charter/constitution content.
6. **NO PRESCRIPTIVE ASSUMPTIONS** about what the target model is good/bad at — discover empirically.

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
"""


def _preflight_health_check_phase3(
    cfg: AppConfig, role: str, target_alias: str
) -> None:
    """Ping the inference API for all phase3 models. Fail fast if unreachable."""
    from pipeline.phase2.run import health_check, make_api_client

    client, _ = make_api_client(
        cfg.phase2.endpoint, cfg.phase3.iteration.max_concurrent
    )

    checked: set[str] = set()
    for m in cfg.phase3.gold_judges + cfg.phase3.gold_generators:
        if m.api_name not in checked:
            health_check(client, m.api_name)
            checked.add(m.api_name)

    target_cfg = None
    for m in cfg.phase3.target_models:
        if m.alias == target_alias:
            target_cfg = m
            break
    assert target_cfg, f"Target model {target_alias} not found in config"

    if target_cfg.api_name not in checked:
        health_check(client, target_cfg.api_name)

    logger.info("Phase3 pre-flight health check passed for target={}.", target_alias)


def run_phase3_improver(cfg: AppConfig, role: str, target_alias: str) -> None:
    """Run a single phase3 improver for one (role, target) pair.

    Independently launchable and thread-safe.
    """
    key = f"phase3_{role}_{target_alias}"
    log_path = improver_log_path(f"phase3_{role}", target_alias)
    tmp_dir = PIPELINE_DATA_DIR / f"tmp_phase3_{role}_{target_alias}"
    now = datetime.now(timezone.utc).isoformat()

    _update_status(
        lambda s: s.setdefault("improvers", {}).update(
            {key: {**_make_improver_status("running"), "started_at": now}}
        )
    )

    try:
        _preflight_health_check_phase3(cfg, role, target_alias)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        prompt = _build_phase3_improver_prompt(
            cfg, role, target_alias, agent_tmp_dir=tmp_dir
        )
        _spawn_agent(prompt, log_path, _allowed_tools(tmp_dir), key=key)

        new_prompt = _detect_new_prompts(target_alias, role)
        logger.info("Phase3 improver {} done: latest prompt -> {}", key, new_prompt)

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


def run_phase3_improvers(
    cfg: AppConfig, role: str, aliases: list[str] | None = None
) -> None:
    """Run phase3 improvers for all target models in parallel."""
    if aliases is None:
        aliases = [m.alias for m in cfg.phase3.target_models]

    now = datetime.now(timezone.utc).isoformat()
    prompts_before = _snapshot_prompts(cfg)

    improvers = {
        f"phase3_{role}_{a}": _make_improver_status("pending") for a in aliases
    }
    status = {
        "running": True,
        "started_at": now,
        "role": f"phase3_{role}",
        "improvers": improvers,
        "error": None,
    }
    write_status(status)

    errors: dict[str, Exception] = {}

    try:
        with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
            futures = {
                pool.submit(run_phase3_improver, cfg, role, alias): alias
                for alias in aliases
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    future.result()
                except Exception as e:
                    errors[alias] = e
                    logger.error("Phase3 improver {}/{} failed: {}", role, alias, e)

        def _finalize(s: dict) -> None:
            s["running"] = False
            if errors:
                s["error"] = "; ".join(f"{a}: {e}" for a, e in errors.items())
            for key, data in s.get("improvers", {}).items():
                if data["status"] == "pending":
                    data["status"] = "skipped"

        _update_status(_finalize)
        logger.info("All phase3 {} improvers complete ({} errors).", role, len(errors))

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
            logger.error("Failed to save phase3 loop history to DB: {}", e)


def _save_history(status: dict, prompts_before: dict[str, str], cfg: AppConfig) -> None:
    """Persist a completed phase3 improver run to loop_history table."""
    prompts_after = _snapshot_prompts(cfg)

    for key, data in status.get("improvers", {}).items():
        if not data.get("reasoning") and data.get("status") != "pending":
            parts = key.split("_", 2)
            if len(parts) == 3:
                _, role, alias = parts
                log_p = improver_log_path(f"phase3_{role}", alias)
                data["reasoning"] = _extract_reasoning_from_log(log_p)

    logs = {}
    for key in status.get("improvers", {}):
        parts = key.split("_", 2)
        if len(parts) == 3:
            _, role, alias = parts
            log_p = improver_log_path(f"phase3_{role}", alias)
            if log_p.exists():
                logs[key] = log_p.read_text()

    record = {
        "started_at": status.get("started_at"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "role": status.get("role"),
        "phase": "phase3",
        "improvers": status.get("improvers", {}),
        "error": status.get("error"),
        "prompts_before": prompts_before,
        "prompts_after": prompts_after,
        "logs": logs,
    }
    save_loop_run(record)


def main():
    """CLI entry point for phase3 improver runners."""
    import argparse

    parser = argparse.ArgumentParser(description="Run phase3 improver agents")
    parser.add_argument("--role", required=True, choices=["judge", "generator"])
    parser.add_argument(
        "--aliases",
        type=str,
        default=None,
        help="Comma-separated target model aliases (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config()
    aliases = args.aliases.split(",") if args.aliases else None

    logger.info("Starting phase3 {} improver(s)", args.role)
    if aliases:
        logger.info("Aliases: {}", aliases)

    run_phase3_improvers(cfg, args.role, aliases=aliases)


if __name__ == "__main__":
    main()
