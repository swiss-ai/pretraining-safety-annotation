"""Independent improver runners: judge improver and generator improver.

Each improver spawns an Opus agent that autonomously calls cross-iteration
tools via CLI. Multiple models run in parallel, each with its own scratch
directory and thread-safe status updates.

Usage:
    uv run python -m pipeline.charter.improve.loop --role judge [--aliases glm45,olmo3-32B-think]
    uv run python -m pipeline.charter.improve.loop --role generator [--aliases glm45]
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
from pipeline.charter.improve.storage import save_loop_run

# Re-export from agent_utils for backward compatibility
from pipeline.agent_utils import (  # noqa: F401
    STATUS_PATH,
    AGENT_TMP_DIR,
    improver_log_path,
    write_status,
    read_status,
    run_improver_agent,
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


def parse_improver_key(key: str) -> tuple[str, str, str]:
    """Parse an improver status key into (role, mode, alias).

    Handles both old ``role_alias`` and new ``role_mode_alias`` formats.
    """
    parts = key.split("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return "?", "", key


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


def _detect_new_prompts(alias: str, role: str, mode: str) -> str:
    """Find the highest-versioned prompt for the given role+mode in the model directory.

    Returns the latest filename (e.g. 'judge_reflection_v3.md').
    """
    model_dir = PROMPTS_DIR / alias
    prefix = "judge" if role == "judge" else "generator"
    matches = sorted(glob.glob(str(model_dir / f"{prefix}_{mode}_v*.md")))
    assert matches, f"No {prefix}_{mode}_v*.md files in {model_dir}"
    return Path(matches[-1]).name


def _build_improver_prompt(
    cfg: AppConfig,
    role: str,
    target_alias: str,
    mode: str,
    agent_tmp_dir: Path | None = None,
) -> str:
    """Unified prompt builder for judge and generator improvers.

    Key additions over old phase prompts:
    - Lists ALL counterpart models the agent's cross-iterations will use
    - Tells agent to use `run_cross_batch --role {role} --target {alias} --mode {mode}`
    - Explains group_id for interpreting cross-model results
    - Points to `cross_summary <group_id>` command for aggregated stats
    """
    if agent_tmp_dir is None:
        agent_tmp_dir = AGENT_TMP_DIR
    model_dir = PROMPTS_DIR / target_alias
    improver_path = _INIT_PROMPTS_DIR / "improver.md"

    if role == "judge":
        prompt_path = resolve_prompt_path(f"judge_{mode}_latest.md", target_alias)
        # Resolve generator prompt from the first generator model, not the judge model
        gen_alias_for_ref = cfg.charter.improve.generator_models[0].alias
        other_prompt_path = resolve_prompt_path(
            f"generator_{mode}_latest.md", gen_alias_for_ref
        )
        current_v = _extract_version(prompt_path.name)
        next_v = current_v + 1
        phase_prompt_path = _INIT_PROMPTS_DIR / cfg.charter.improve.improver.judge_prompt
        counterpart_models = [m.alias for m in cfg.charter.improve.generator_models]
        role_label = "JUDGE"
        prompt_type = "judge"
        other_type = "generator"
    else:
        prompt_path = resolve_prompt_path(f"generator_{mode}_latest.md", target_alias)
        # Resolve judge prompt from the first judge model, not the generator model
        judge_alias = cfg.charter.improve.judge_models[0].alias
        other_prompt_path = resolve_prompt_path(f"judge_{mode}_latest.md", judge_alias)
        current_v = _extract_version(prompt_path.name)
        next_v = current_v + 1
        phase_prompt_path = _INIT_PROMPTS_DIR / cfg.charter.improve.improver.generator_prompt
        counterpart_models = [m.alias for m in cfg.charter.improve.judge_models]
        role_label = "GENERATOR"
        prompt_type = "generator"
        other_type = "judge"

    state_path = model_dir / f"state_{role}_{mode}.md"
    if not state_path.exists():
        state_path.write_text("# Improver State\n\nNo previous iterations.\n")

    max_batches = cfg.charter.improve.improver.max_batches_per_phase

    if phase_prompt_path.exists():
        phase_instructions = phase_prompt_path.read_text(encoding="utf-8")
    else:
        phase_instructions = f"Focus on improving {prompt_type} prompts."

    # Interpolate trusted-reviewer list into the fragment. Use .replace() (not .format())
    # because the fragment contains many literal `{` characters in XML tags and code blocks.
    # The wrapper f-string at line 297 substitutes {phase_instructions} once and does NOT
    # re-evaluate its contents, so the placeholder passes through unchanged until this call.
    trusted_reviewers = cfg.charter.improve.improver.trusted_reviewers
    if trusted_reviewers:
        trusted_list = ", ".join(f"`{r}`" for r in trusted_reviewers)
    else:
        trusted_list = "(none configured)"
    phase_instructions = phase_instructions.replace(
        "{trusted_reviewers_list}", trusted_list
    )

    from pipeline.charter.improve.storage import load_runs

    runs = load_runs()
    latest_iter = runs[-1]["iteration"] if runs else 0
    has_data = latest_iter > 0

    # Check if gold annotations or human reviews exist
    has_gold = False
    has_reviews = False
    try:
        from pipeline.charter.seed.storage import load_latest_annotations

        has_gold = len(load_latest_annotations()) > 0
    except Exception:
        pass
    try:
        from pipeline.charter.improve.storage import load_reviews

        has_reviews = len(load_reviews()) > 0
    except Exception:
        pass

    counterpart_list = ", ".join(counterpart_models)

    # Auto-inject latest diagnose output if data exists
    baseline_diagnose = ""
    if has_data:
        # Find the latest group_id involving this target model in the relevant role
        target_runs = [
            r
            for r in runs
            if r.get("group_id")
            and (
                (role == "judge" and r.get("judge_model") == target_alias)
                or (role == "generator" and r.get("generator_model") == target_alias)
            )
        ]
        if target_runs:
            latest_gid = target_runs[-1]["group_id"]
            logger.info(
                "Auto-injecting diagnose for {}/{} (group_id={})",
                role,
                target_alias,
                latest_gid[:8],
            )
            try:
                import contextlib
                import io

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    from pipeline.improver_tools import cmd_diagnose

                    cmd_diagnose(latest_gid, mode=mode)
                baseline_diagnose = buf.getvalue()
                if len(baseline_diagnose) > 4000:
                    baseline_diagnose = baseline_diagnose[:4000] + "\n... (truncated)"
                logger.info(
                    "Auto-diagnose produced {} chars for {}",
                    len(baseline_diagnose),
                    target_alias,
                )
            except Exception as e:
                logger.warning(
                    "Failed to auto-run diagnose for {}: {} ({})",
                    target_alias,
                    type(e).__name__,
                    e,
                )
        else:
            logger.warning(
                "No runs found for {}/{} — skipping auto-diagnose",
                role,
                target_alias,
            )

    if has_data:
        first_run_note = ""
    else:
        first_run_note = f"""
## FIRST RUN — No data exists yet!
There are no iterations in the database. You MUST run a baseline batch first before analyzing anything.
Do this immediately (run in background — it takes 3-5 minutes!):
  Bash: {{"command": "uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias} --mode {mode} 2>&1", "run_in_background": true}}
Then wait:
  TaskOutput: {{"task_id": "<id>", "block": true, "timeout": 600000}}
Then use `diagnose <group_id>` for a full analysis.
Do NOT waste time querying empty data — run the batch first.
"""

    # Build conditional sections for gold/review data
    # Generator improvers don't use human reviews (judge improvers do).
    show_reviews = has_reviews and role == "judge"
    gold_review_note = ""
    if not has_gold and not show_reviews:
        unavailable = []
        if not has_gold:
            unavailable.extend(["gold", "compare"])
        if not show_reviews:
            unavailable.extend(["reviews"])
        gold_review_note = f"""
## Data not available: {', '.join(unavailable)}
Do NOT waste tool calls on these commands — they will return empty or irrelevant results.
"""
    else:
        extra_agents = []
        if has_gold:
            extra_agents.append(
                "- **Gold comparator**: compare generated outputs with gold annotations — run `compare` on multiple items, identify systematic gaps"
            )
        if show_reviews:
            extra_agents.append(
                "- **Human reviews analyst**: read ALL human reviews (`reviews` without iteration filter) and extract key insights from reviewer notes"
            )
        gold_review_note = (
            "\nAdditional subagents to spawn:\n" + "\n".join(extra_agents) + "\n"
        )

    # Human notes: optional operator guidance, injected if present.
    # - Global (role-scoped):  pipeline/prompts/human_notes_<role>.md
    # - Per-model override:    data/pipeline/prompts/<alias>/human_notes_<role>.md
    # Both are loaded if they exist; neither is required. Contents are shown
    # near the top of the prompt under a "Human notes" section.
    human_notes_parts: list[str] = []
    global_notes_path = _INIT_PROMPTS_DIR / f"human_notes_{role}.md"
    if global_notes_path.exists():
        try:
            text = global_notes_path.read_text(encoding="utf-8").strip()
            if text:
                human_notes_parts.append(
                    f"### Global notes (all {role} improvers)\n{text}"
                )
        except Exception as e:
            logger.warning("Failed to read {}: {}", global_notes_path, e)
    model_notes_path = model_dir / f"human_notes_{role}.md"
    if model_notes_path.exists():
        try:
            text = model_notes_path.read_text(encoding="utf-8").strip()
            if text:
                human_notes_parts.append(
                    f"### Notes for {target_alias} specifically\n{text}"
                )
        except Exception as e:
            logger.warning("Failed to read {}: {}", model_notes_path, e)

    if human_notes_parts:
        human_notes_block = (
            "\n## Human notes (priority — from your operator)\n"
            "Read these carefully. They reflect guidance from the person running this "
            "pipeline and override general heuristics where they conflict.\n\n"
            + "\n\n".join(human_notes_parts)
            + "\n"
        )
        logger.info(
            "Injecting human notes for {}/{}: {} chars",
            role,
            target_alias,
            len(human_notes_block),
        )
    else:
        human_notes_block = ""

    # Role-conditional: generator improvers don't see reviews
    reviews_tool_line = (
        "\n  uv run python -m pipeline.improver_tools reviews [<JUDGE_PROMPT>] "
        "[--reasoning-limit N]  — human reviews with the *latest* judge's scores + "
        "reasoning side-by-side (pass --reasoning-limit 0 to suppress reasoning, default 200)"
        if role == "judge"
        else ""
    )
    reviews_subagent_line = (
        "\n- **Human reviews analyst**: read ALL human reviews (`reviews` without "
        "iteration filter — output is large, must use subagent!) and extract key "
        "insights from reviewer notes"
        if role == "judge"
        else ""
    )

    return f"""You are improving {role_label} prompts for a pretraining data annotation pipeline.

## {role_label} Improvement for model: {target_alias} (mode: {mode})

**IMPORTANT: Other improver agents may be running in parallel for different models and modes.**
You are ONLY responsible for the **{target_alias}** model's **{mode}** prompt. Ignore iterations,
errors, or data belonging to other models. When analyzing results, filter by your target model.

{phase_instructions}
{first_run_note}{human_notes_block}
## Cross-model evaluation
Your improvements will be tested against ALL {other_type} models: [{counterpart_list}].
When you run a batch, it creates one iteration per {other_type} model, all sharing a group_id.
Use `cross_summary <group_id>` to see aggregated per-model stats after each batch.

## How the pipeline sends messages to generators
Generation uses a single API call per item. The system prompt is the generator
prompt with {{charter}} substituted.

**Reflection call** (partial text only):
```
## Full Text
<text up to reflection point>

## Task
The text above is a partial passage — your reflection should respond only to what
you see here. Produce: analysis, reflection_1p.
```
Expected output: `analysis`, `reflection_1p`

The reflection call prevents foreshadowing — the model never sees text beyond the
reflection point. Do not write instructions that assume the model sees the full text.
There is no `response_format=json_object` for most models, so JSON compliance depends on
prompt instructions.

## Lessons from previous improver runs (DO NOT repeat these mistakes)
- **Do NOT list alternative opening phrases** (e.g., "try starting with 'At this juncture' or
  'Looking at this'"). Small models copy them verbatim as new templates, creating worse
  diversity than before. Instead, use abstract instructions like "vary your approach" or
  "never start two annotations the same way."
- **Keep prompts focused.** Small models (7B-70B) degrade with long system prompts. The
  current judge baseline is ~500 words per mode; do not let prompt length grow monotonically
  across iterations — every addition must be balanced by removing a redundant instruction
  or example.

## Query tools
Run these via Bash (prefix with `uv run`). Replace <ITER> with the iteration number from your batch output.
**IMPORTANT: `--mode {mode}` is already included in every command below. Do NOT omit it.**
  uv run python -m pipeline.improver_tools summary <ITER> --mode {mode}     — aggregate stats for {mode}
  uv run python -m pipeline.improver_tools failures <ITER> --mode {mode}    — rejected items with reasoning
  uv run python -m pipeline.improver_tools failures <ITER> --mode {mode} --reasoning-limit 500  — full reasoning
  uv run python -m pipeline.improver_tools diversity <ITER>   — frequency-based diversity analysis
  uv run python -m pipeline.improver_tools scores <ITER> --mode {mode}      — compact scores table
  uv run python -m pipeline.improver_tools distribution <ITER> — per-dimension score distributions + floor trigger counts
  uv run python -m pipeline.improver_tools show <id> <ITER>   — full text + outputs for one item
  uv run python -m pipeline.improver_tools show <id1,id2,...> <ITER>  — batch show multiple items
  uv run python -m pipeline.improver_tools show <id> <ITER> --brief   — truncated source text
  uv run python -m pipeline.improver_tools show --gold <ITER> [--brief] — all gold items for iteration
  uv run python -m pipeline.improver_tools item <id> <ITER>   — full details as JSON
  uv run python -m pipeline.improver_tools reasoning <id>[,id2,...] <ITER> — full judge reasoning (scores + text)
  uv run python -m pipeline.improver_tools gold                      — gold annotations (concise, no source text)
  uv run python -m pipeline.improver_tools gold --verbose            — gold with full source text (large output!)
  uv run python -m pipeline.improver_tools compare <id> <ITER> — generated vs gold{reviews_tool_line}
  uv run python -m pipeline.improver_tools filter <ITER> --dim <dimension> --below <threshold> [--part reflection_1p]
  uv run python -m pipeline.improver_tools trend --mode {mode}              — cross-iteration comparison table
  uv run python -m pipeline.improver_tools correlations              — judge-human correlation by judge version
  uv run python -m pipeline.improver_tools parse_stats <ITER>        — generation parse success/failure counts

**Item IDs are hex strings** (e.g. `a27f2f5f`, `9eeb3229`). The `show` and `reasoning` commands
match by prefix — pass the first 8+ chars. Numeric values like `25` or `40` are NOT valid indices.
Copy IDs from the `scores`, `failures`, or `filter` output.

## Test tools (run experiments WITHOUT modifying main data)
  uv run python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2] [--n N] [--role {role}] [--mode {mode}]
  uv run python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2] [--n N] [--role {role}]
  uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias} --mode {mode}  — cross-iteration with ALL {other_type} models
  uv run python -m pipeline.improver_tools cross_summary <group_id> --mode {mode}   — per-model stats for a cross-iteration
  uv run python -m pipeline.improver_tools diagnose <group_id> --mode {mode}        — ONE-SHOT full analysis (use this first!)
  uv run python -m pipeline.improver_tools diff <iter1> <iter2> --mode {mode} [--limit N]  — cross-iteration item comparison
  uv run python -m pipeline.improver_tools test_results --role {role}  — view test results
  uv run python -m pipeline.improver_tools rollback {target_alias} {role} <version> --mode {mode} — promote version N to latest

## Running long commands (CRITICAL — `run_cross_batch` takes 3-5 minutes!)
`run_cross_batch`, `run_batch`, `test_judge`, and `test_generate` make many API calls and
take several minutes. You MUST run them in background and wait with a long timeout:
```
Bash: {{"command": "uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias} --mode {mode} 2>&1", "run_in_background": true}}
```
Then wait for the result:
```
TaskOutput: {{"task_id": "<id from above>", "block": true, "timeout": 600000}}
```
**NEVER** run these commands synchronously (default Bash timeout is 120s — too short).
**ALWAYS** include `2>&1` to capture stderr.
**DO NOT block waiting immediately** — while a batch runs in the background, analyze existing
iteration data in parallel. Only `TaskOutput` (block) when you've finished all other analysis
and actually need the batch results to proceed.

## Scratch directory — ALL scripts go here, NOWHERE ELSE
Write ad-hoc analysis scripts to: {agent_tmp_dir}
Run them with: uv run python {agent_tmp_dir}/your_script.py
Delete with: rm {agent_tmp_dir}/your_script.py
This folder is cleared at the start of each loop. Use it for any analysis the CLI tools don't cover.
**NEVER write scripts to `scripts/`, the project root, or anywhere else** — you cannot delete
files outside {agent_tmp_dir}/ and they will be left behind as garbage.

## State
Read your state file at {state_path} FIRST. It contains notes from previous iterations.
{"" if not baseline_diagnose else f'''
## Latest baseline diagnostics (auto-generated — saves you running diagnose yourself)
```
{baseline_diagnose}
```
This data is pre-loaded so you can skip running `diagnose` and go straight to deeper analysis
(failures, show, reasoning on specific items). Pass this output to your subagents too.
'''}
## YOU MUST READ ACTUAL ITEMS — NOT JUST SCORES (CRITICAL)
**DO NOT work from aggregate statistics alone.** Previous improver runs failed because they
only looked at score distributions, accept rates, and summary tables — never reading the
actual generated text or judge reasoning for individual items. This produces blind prompt
edits that don't fix anything.

**For EVERY analysis round, you MUST:**
1. Run `reasoning <id1,id2,...> <iter>` on rejected items to read what the judge ACTUALLY said
2. Run `show <id> <iter>` to read the source text and the generated annotations
3. Only THEN look at scores to confirm the pattern

Use the CLI tools directly via Bash — do NOT write Python scripts that import pipeline modules
(they will fail with ImportError). The `show`, `reasoning`, `failures`, and `filter` commands
are all you need.

## Strategy: use subagents for parallel exploration
You have access to the Agent tool. **Always use model="opus" for subagents** — they need
strong reasoning. Spawn subagents in parallel for analysis. The bottleneck is wall-clock
time, not tokens.

**IMPORTANT: Subagents must use CLI commands, not Python imports.** Tell each subagent to run
`uv run python -m pipeline.improver_tools <command>` via Bash. Do NOT tell them to write
Python scripts or import from `pipeline.*` — those approaches fail.

**Before spawning subagents**: Run `diagnose <group_id>` yourself first. Then pass the diagnose
output to each subagent in its prompt so they don't re-run it. Each subagent prompt should
include: (1) the diagnose output, (2) their specific analysis task, (3) which CLI commands to
run. Tell them to keep their response concise (under 2000 words) so you can read it.

Launch these subagents simultaneously (all in one message with multiple Agent calls):
- **Failures analyst**: run `reasoning` on ALL rejected items, read the judge's actual
  complaints, categorize failures, return ranked list with QUOTED judge reasoning{reviews_subagent_line}
- **Diversity analyst**: check diversity patterns — run `diversity`, look for formulaic/repetitive output
- **Dimension deep-dive**: run `filter` for each scoring dimension below threshold, then
  `reasoning` on the worst items to understand WHY that dimension scored low
- **Cross-model comparator**: if multiple iterations exist, run `diff` between iterations to see what changed

Each subagent should return a concise summary with QUOTED TEXT from the actual generations
and judge reasoning — not just category labels and counts.
{gold_review_note}
**Avoid redundancy**: When you delegate analysis to subagents, do NOT run the same queries
yourself. Wait for ALL subagent results before proceeding. Never call the same command twice —
save the output mentally and reuse it. Prefer batch commands (`scores`, `summary`, `diversity`)
over inspecting items one by one.

## Your task
1. Read your state file: {state_path}
2. Read the improver instructions: {improver_path}
3. Read the current {prompt_type} prompt: {prompt_path} and the {other_type} prompt for context: {other_prompt_path}
4. If no data exists yet, run a baseline batch first (see "Running long commands" above — use background + 600s timeout):
   `uv run python -m pipeline.improver_tools run_cross_batch --role {role} --target {target_alias} --mode {mode} 2>&1`
5. Analyze results: start with `diagnose <group_id>` for a full overview, then drill into specifics with `diff`, `failures`, `show`. After consuming the auto-injected diagnose AND after every subsequently-spawned `run_cross_batch`, append a `## Reflection N` block to {state_path} per the `<analysis_checkpoint_protocol>` in the phase instructions, BEFORE writing the next prompt version.
6. Write improved {prompt_type} {mode} prompt to {model_dir}/{prompt_type}_{mode}_v{next_v}.md
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

## Time budget
Spend no more than 40% of your tool calls on analysis. Start writing prompt changes early —
you can always refine after testing. A shipped v(N+1) that's 80% right is worth more than a
perfect analysis that never produces a prompt.

## VERSION SELECTION — CRITICAL
The pipeline ALWAYS uses the highest version number (_vN.md) as the active prompt.
If you write v2 (best), then v3 (worse), v3 is deployed — not v2. Before finalizing:
- If an earlier version performed best, run `rollback {target_alias} {prompt_type}_{mode} <best_version>`
  to copy it to v(max+1), making it the active prompt.
- Never leave a worse version as the highest number on disk.
- **Always use rollback** rather than manually rewriting prompt files from old versions.
  `rollback` is fast, exact, and doesn't risk introducing transcription errors.

## RULES — VIOLATIONS WASTE YOUR LIMITED TOOL CALLS

1. **NO PIPES.** Never use `|` in Bash commands. No `| tail`, `| head`, `| grep`. Run commands
   separately. Pipes hide exit codes and break background execution.
2. **NO RAW SQL.** Never open sqlite3 directly. The database is at `data/storage.db` (NOT
   `pipeline/pipeline.db`), but you should NEVER need it — all data is available via
   `python -m pipeline.improver_tools`. Use `reasoning <id> <iter>` for judge reasoning details.
3. **SCRIPTS ONLY IN {agent_tmp_dir}/.** Never write to `scripts/`, project root, or anywhere
   else — you cannot delete files outside {agent_tmp_dir}/.
4. Use `uv run python -m pipeline.improver_tools ...` for data access — NOT raw file reads.
   Do NOT read `pipeline/improver_tools.py` — it is too large. Run `uv run python -m pipeline.improver_tools help`
   for a CLI reference if needed.
5. Do NOT overfit to individual examples. Focus on systematic patterns.
6. The {prompt_type} prompt must NOT hardcode specific charter/specification content.
7. `diff <iter1> <iter2>` only works for iterations that share source items (i.e. iterations
   within the SAME cross-batch group). Do NOT diff across different groups.

## Writing prompt files
Use the **Write** tool to create or overwrite prompt files in {model_dir}/. If Write is blocked
for any reason, use this Python heredoc fallback:
```bash
uv run python << 'PYEOF'
from pathlib import Path
content = \"\"\"
<your prompt content here>
\"\"\"
Path("{model_dir}/{prompt_type}_{mode}_v{next_v}.md").write_text(content.strip())
print("Written successfully")
PYEOF
```

## Bash Python tips (avoids permission errors)
When running ad-hoc Python via Bash, **NEVER** use `python -c "..."` with multi-line strings.
Instead, use heredoc syntax:
```bash
uv run python << 'PYEOF'
from pipeline.charter.improve.storage import load_items_for_iteration
items = load_items_for_iteration(3)
print(len(items))
PYEOF
```
Or write a script to {agent_tmp_dir}/ and run it with `uv run python {agent_tmp_dir}/script.py`.
The inline `python -c` form triggers security filters on `#` comments and `{{` braces.
"""


_STATE_MAX_CHARS = 6000  # ~1500 tokens — trigger compression above this


def _auto_compress_state(state_path: Path) -> None:
    """Use Claude to intelligently compress state.md when it exceeds threshold.

    Spawns a lightweight Claude CLI call (haiku) to rewrite the state file
    more concisely while preserving key learnings and metrics.
    Falls back to tail-truncation if Claude is unavailable.
    """
    if not state_path.exists():
        return
    content = state_path.read_text(encoding="utf-8")
    if len(content) <= _STATE_MAX_CHARS:
        return

    logger.info(
        "State file {} chars exceeds {} — compressing with Claude...",
        len(content),
        _STATE_MAX_CHARS,
    )

    import subprocess

    compress_prompt = (
        "Compress the following improver state file to under 4000 characters. "
        "Keep:\n"
        "- Current active prompt versions and their key metrics\n"
        "- What NOT to try (failed approaches with brief reasons)\n"
        "- Key lessons learned\n"
        "- Most recent iteration results\n"
        "Drop:\n"
        "- Per-item details and verbose tables from old rounds\n"
        "- Redundant iteration-by-iteration logs (summarize trends instead)\n"
        "- Analysis that led to abandoned approaches\n"
        "Output ONLY the compressed state file content, nothing else.\n\n"
        "--- STATE FILE ---\n"
        f"{content}"
    )

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "haiku", "-p", compress_prompt],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            compressed = result.stdout.strip()
            if len(compressed) < len(content):
                state_path.write_text(compressed, encoding="utf-8")
                logger.info(
                    "Claude-compressed state.md: {} → {} chars",
                    len(content),
                    len(compressed),
                )
                return
            else:
                logger.warning("Claude compression didn't reduce size, skipping")
        else:
            logger.warning(
                "Claude compression failed (rc={}): {}",
                result.returncode,
                result.stderr[:200],
            )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Claude compression unavailable: {}", e)

    # Fallback: keep the last portion
    trimmed = content[-_STATE_MAX_CHARS:]
    nl = trimmed.find("\n")
    if nl != -1:
        trimmed = trimmed[nl + 1 :]
    fallback = (
        "# Improver State (auto-compressed — fallback)\n\n"
        "[Earlier history trimmed — see git log for full history]\n\n" + trimmed
    )
    state_path.write_text(fallback, encoding="utf-8")
    logger.info(
        "Fallback-compressed state.md: {} → {} chars",
        len(content),
        len(fallback),
    )


def _preflight_health_check(cfg: AppConfig, role: str, target_alias: str) -> None:
    """Ping the inference API before spawning agents. Fail fast if unreachable."""
    from pipeline.charter.improve.run import _health_check_models

    _health_check_models(cfg, role, target_alias)
    logger.info("Pre-flight health check passed for {} / {}.", role, target_alias)


# Backward-compat alias for external callers
ALLOWED_TOOLS = _allowed_tools(AGENT_TMP_DIR)


def run_improver(cfg: AppConfig, role: str, target_alias: str, mode: str) -> None:
    """Run a single improver for one (role, model, mode) triple.

    Independently launchable and thread-safe. Each improver gets its own
    scratch directory and updates loop_status.json atomically.
    """
    _preflight_health_check(cfg, role, target_alias)

    key = f"{role}_{mode}_{target_alias}"
    log_path = improver_log_path(role, target_alias)
    tmp_dir = PIPELINE_DATA_DIR / f"tmp_{role}_{mode}_{target_alias}"
    prompt = _build_improver_prompt(
        cfg, role, target_alias, mode=mode, agent_tmp_dir=tmp_dir
    )
    state_path = PROMPTS_DIR / target_alias / f"state_{role}_{mode}.md"

    def _post_hook():
        new_prompt = _detect_new_prompts(target_alias, role, mode)
        logger.info("Improver {} done: latest prompt -> {}", key, new_prompt)

        if role == "judge":
            from pipeline.charter.improve.run import rejudge_all_prompts_and_models

            logger.info(
                "Running rejudge_all_prompts_and_models (mode={}) after judge improvement...",
                mode,
            )
            rejudge_all_prompts_and_models(cfg, mode=mode)

    try:
        run_improver_agent(prompt, key, log_path, tmp_dir, post_hook=_post_hook)
    finally:
        # Always compress state.md, even if the agent crashed mid-run, so it doesn't
        # grow unbounded across retries.
        _auto_compress_state(state_path)


def _run_improvers(
    cfg: AppConfig, role: str, mode: str, aliases: list[str] | None = None
) -> None:
    """Run improvers for a role+mode in parallel. If aliases is None, run ALL models for that role."""
    if aliases is None:
        model_list = (
            cfg.charter.improve.judge_models if role == "judge" else cfg.charter.improve.generator_models
        )
        aliases = [m.alias for m in model_list]

    now = datetime.now(timezone.utc).isoformat()
    prompts_before = _snapshot_prompts(cfg)

    improvers = {
        f"{role}_{mode}_{a}": _make_improver_status("pending") for a in aliases
    }
    status = {
        "running": True,
        "started_at": now,
        "role": role,
        "mode": mode,
        "improvers": improvers,
        "error": None,
    }
    write_status(status)

    errors: dict[str, Exception] = {}

    try:
        with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
            futures = {
                pool.submit(run_improver, cfg, role, alias, mode): alias
                for alias in aliases
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    future.result()
                except Exception as e:
                    errors[alias] = e
                    logger.error("Improver {}/{}/{} failed: {}", role, mode, alias, e)

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


def run_judge_improvers(
    cfg: AppConfig, mode: str, aliases: list[str] | None = None
) -> None:
    """Run judge improvers in parallel. If aliases is None, run ALL judge models."""
    _run_improvers(cfg, "judge", mode, aliases)


def run_generator_improvers(
    cfg: AppConfig, mode: str, aliases: list[str] | None = None
) -> None:
    """Run generator improvers in parallel. If aliases is None, run ALL generator models."""
    _run_improvers(cfg, "generator", mode, aliases)


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
            role, _mode, alias = parse_improver_key(key)
            log_p = improver_log_path(role, alias)
            data["reasoning"] = _extract_reasoning_from_log(log_p)

    # Capture full logs
    logs = {}
    for key in status.get("improvers", {}):
        role, _mode, alias = parse_improver_key(key)
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
        "--mode",
        default="reflection",
        choices=["reflection"],
        help="Which mode's prompt to improve (only 'reflection' is supported)",
    )
    parser.add_argument(
        "--aliases",
        type=str,
        default=None,
        help="Comma-separated model aliases (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config()
    aliases = args.aliases.split(",") if args.aliases else None

    logger.info("Starting {} {} improver(s)", args.role, args.mode)
    if aliases:
        logger.info("Aliases: {}", aliases)

    if args.role == "judge":
        run_judge_improvers(cfg, mode=args.mode, aliases=aliases)
    else:
        run_generator_improvers(cfg, mode=args.mode, aliases=aliases)


if __name__ == "__main__":
    main()
