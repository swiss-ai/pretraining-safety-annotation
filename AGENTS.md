
# This is research code.

## Correctness rules (CRITICAL RULES)
- NEVER hide failures with try-except, placeholders, or dummy data
- NEVER remove failing tests - report the upstream issue instead
- NEVER "blind fix" errors without understanding root cause

---

## Repo orientation

The pipeline produces two annotation types over FineWeb / dolma3_mix text (see `README.md` for the schema and worked examples):

- **Preflection** (full text → 4 third-person fields): `charter_summary`, `neutral`, `judgemental`, `idealisation`. Frozen prompt at `final_prompts/qwen3.5-35b-a3b/generator_preflection_v8.md`.
- **Reflection** (partial text up to a reading pause point → 2 voices): `reflection_1p`, `reflection_3p`. Frozen prompt at `final_prompts/qwen3.5-35b-a3b/generator_reflection_v7.md`.

Both emit inline `[X.Y]` citations against `resources/ModelRaisingConstitution_v0.2.md`. Schema constants + the shared parser live in `pipeline/generation.py` — update it in one place when the schema changes.

Two top-level groups under `pipeline/`:

- **`pipeline/charter/`** — the charter-cited annotation pipeline. Four steps: `seed` (human annotation) → `improve` (generate+judge+improver loop) → `eval` (diverse-pool ranking) → `scale` (SLURM scale-up over the 102M-row sidecar). Same product (charter-cited preflection + reflection) across all four; the first three iterate the prompt, the fourth runs it for real.
- **`pipeline/sft/`** — charter-aware SFT data generation, parallel to but downstream of `charter/`. Two variants: `single_turn` (paired cited/uncited responses) and `multi_turn` (multi-turn self-play).

Baseline annotation tracks (`pipeline/summaries/`, future `pipeline/rephrase/`, …) live as siblings of `charter/` and `sft/` at the top of `pipeline/` — same one-level depth, deliberately thinner than the main `charter/` track. `summaries/` is just a generator + an `iterate` CLI for hand-tuning the prompt interactively (no automated improver/judge loop, no LLM-judge scoring).

Subfolder READMEs (especially `pipeline/charter/scale/README.md`, `pipeline/sft/single_turn/README.md`, `pipeline/sft/multi_turn/README.md`, `pipeline/charter/scale/AGENTS.md`, and `preprocessing/*/README.md`) carry the detail — prefer updating those over bloating top-level docs.

## Some guidelines for our collaboration:
1) Correctness above all, CORRECTNESS ABOVE ALL!
2) Never make assumptions if my query is unclear, ask questions.
3) If you are unsure about something, e.g. if a specific command exists, use websearch.
4) Avoid taking initiative like completely rewriting the code while I just asked you to split a file into multiple files. Feel free to suggest improvement though! I really value your judgement, so always feel free to prompt me if you saw some potential improvements unrelated to my request / that you avoided doing to avoid intiative.


## Some guidelines for research codebases:
- Fail fast philosophy: never, NEVER, NEVEEEEEER use value placeholders, try except blocks, or any other form of "if this fails, do this".
- Use assert for torch tensor shapes.
- In torch code, avoid for loops and always use vectorized operations if possible.
- Use docstrings.
- Avoid inline comments meant to explain the code like "# looping over the data" or "# not using x because of y". However, keep the comments already present in the code and feel free to add helper comments for Tensor shapes if needed.
- Respect my codestyle. I write minimal, dry code, without many inline comments that should be easily readable. Importantly, IT IS NOT BLOATED, GOD I HATE BLOATED CODE.
- When editing existing code, keep your changes as targeted as possible, avoiding any unnecessary changes. You should optimize for edits that are easy to review.
- When editing a function with missing docstring, add one.
- Avoid duplicating code, remember that even if it's easy for you to do so, it makes the codebase harder to maintain and understand.
- When writing tests, test that pipelines actually RUN, not just utility functions.
- When a test fail, for example because X is not implemented, or Y didn't import, DO NOT remove the test. You're usually pretty good at writting tests, if the error comes from upstream it's worth bringing it up to me rather than hidding the failing tests under the carpet.
- Similarily, when you hit an unexpected error from the codebase / a library while running code, resist the urge of "I NEED TO FIX THIS 2 LINES OF CODE NOW AND CONTINUE", maybe you just stumbled upon a bug in the codebase that deserves more attention as it could be revealing a deeper issue.
- Avoid "blind fixing" where you do not really understand an error, and instead of debugging it, you try a random fix hoping for the best.
- NEVER remove debug prints/code until the fix is verified by running tests. Debug code stays until we confirm the bug is actually fixed.
- If you code, make sure to regularly commit things (not to often but semantically well separated parts. Ask if unsure)

# Environment
- Linux (Clariden cluster, GH200 nodes with 4 GPUs each)
- use $HOME/tmp rather than /tmp for temporary files, as we're on a cluster and /tmp is not available.
- `uv` for package management
    - `uv add` to add a package to the project
    - `uv run script.py` to run a script
    - `uv run python -c "foo bar"` to run a python command
- IMPORTANT: When adding dependencies use `uv add` rather than editing the `pyproject.toml` file.

## SLURM job submission
- Use `sbatch` with job scripts in the repo (see `preprocessing/*/` for examples)
- Container-based execution via `srun --environment=env.toml`

## SLURM job submission (charter.scale + sft.{single,multi}_turn)
`charter.scale`, `sft.single_turn`, and `sft.multi_turn` use datatrove's `SlurmPipelineExecutor` for job submission:
```bash
uv run python -m pipeline.charter.scale submit --run reflections
uv run python -m pipeline.sft.single_turn submit
uv run python -m pipeline.sft.multi_turn submit
```
See `pipeline/charter/scale/README.md`, `pipeline/sft/single_turn/README.md`, and `pipeline/sft/multi_turn/README.md` for details.

# Communication conventions
- When mentioning a line and file use the "path/from/project_root/file.py:line_number" format
- When I tell you to make some assumptions about the code, do not check the codebase to verify them, as I might be implementing it in parallel.
- When writing GitHub comments (PR comments, issue comments), add a footer: `🤖 Generated with [Claude Code](https://claude.ai/code)`

# Agent recommendations
- Spawn subagents to parallelize work when possible - the bottleneck is time spent, not tokens
- When you have multiple tests to run, run them in parallel using slurm rather than sequentially
- Use the most capable model for subagents on important tasks
