"""Claude improver loop for summary prompts.

Usage:
    uv run python -m pipeline.summaries.improve --model glm-4.5-air
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dotenv
dotenv.load_dotenv()

from pipeline.agent_utils import improver_log_path, run_improver_agent
from pipeline.api import health_check, make_api_client
from pipeline.config import PIPELINE_DATA_DIR, PROMPTS_DIR, load_config, resolve_prompt_path
from pipeline.log import logger
from pipeline.summaries import init_summary_prompts


def _build_summary_improver_prompt(cfg, target_alias: str, agent_tmp_dir: Path) -> str:
    """Build the prompt for the summary improver agent."""
    init_summary_prompts(target_alias)
    prompt_path = resolve_prompt_path("summary_latest.md", target_alias)
    model_dir = PROMPTS_DIR / target_alias

    import re
    match = re.search(r"_v(\d+)\.md$", prompt_path.name)
    current_v = int(match.group(1)) if match else 1
    next_v = current_v + 1

    state_path = model_dir / "summary_state.md"
    if not state_path.exists():
        state_path.write_text("# Summary Improver State\n\nNo previous iterations.\n")

    judge_alias = cfg.phase2.judge_models[0].alias

    return f"""You are improving SUMMARY GENERATOR prompts for a pretraining data annotation pipeline.

## Summary Improvement for model: {target_alias}

You are optimizing a text summarization prompt. Summaries must be concise (3-4 sentences),
accurate, specific to each text, and well-written. They are appended to pretraining documents
as a data quality control.

## Current prompt
Read the current summary prompt at: {prompt_path}

## Available tools

Run these via Bash (prefix with `uv run`):
  uv run python -m pipeline.summaries.tools run_batch --model {target_alias} --n 50 --seed SEED
  uv run python -m pipeline.summaries.tools results <RUN_ID>
  uv run python -m pipeline.summaries.tools trend --model {target_alias}

## Running long commands (CRITICAL)
`run_batch` makes many API calls and takes several minutes. Run in background:
```
Bash: {{"command": "uv run python -m pipeline.summaries.tools run_batch --model {target_alias} --n 50 --seed 42 2>&1", "run_in_background": true}}
```
Then wait:
```
TaskOutput: {{"task_id": "<id>", "block": true, "timeout": 600000}}
```

## Scratch directory
Write scripts to: {agent_tmp_dir}
Run with: uv run python {agent_tmp_dir}/your_script.py

## State
Read your state file at {state_path} FIRST.

## Your task
1. Read state file: {state_path}
2. Read current prompt: {prompt_path}
3. If no data exists, run a baseline batch first (use background + 600s timeout)
4. Analyze results with `results <run_id>`
5. Identify failure patterns (generic summaries, hallucinations, low specificity, formulaic openings)
6. Write improved prompt to {model_dir}/summary_v{next_v}.md
7. Run another batch to validate
8. Repeat up to 5 times
9. Update {state_path} with what you changed and why
10. Print a **single final summary** as your VERY LAST message.
    It MUST start with exactly `## Final Summary` on its own line.

## RULES
1. NO PIPES in Bash commands.
2. Scripts only in {agent_tmp_dir}/.
3. Focus on systematic patterns, not individual examples.
4. The prompt must NOT add safety/ethics commentary -- this is pure summarization.
5. Use different seeds for each batch to avoid duplicate items.
"""


def main():
    parser = argparse.ArgumentParser(description="Run summary improver loop")
    parser.add_argument("--model", type=str, required=True, help="Generator model alias")
    args = parser.parse_args()

    cfg = load_config()
    alias = args.model
    init_summary_prompts(alias)

    # Pre-flight health check
    client, _ = make_api_client(
        cfg.phase2.endpoint, cfg.phase2.iteration.max_concurrent
    )
    gen_cfg = next(m for m in cfg.phase2.generator_models if m.alias == alias)
    health_check(client, gen_cfg.api_name)
    judge_cfg = cfg.phase2.judge_models[0]
    health_check(client, judge_cfg.api_name)
    logger.info("Health check passed for {} and {}", alias, judge_cfg.alias)

    key = f"summary_{alias}"
    log_path = improver_log_path("summary", alias)
    tmp_dir = PIPELINE_DATA_DIR / f"tmp_summary_{alias}"
    prompt = _build_summary_improver_prompt(cfg, alias, agent_tmp_dir=tmp_dir)

    run_improver_agent(prompt, key, log_path, tmp_dir)


if __name__ == "__main__":
    main()
