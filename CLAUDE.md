
# This is research code.

## Correctness rules (CRITICAL RULES)
- NEVER hide failures with try-except, placeholders, or dummy data
- NEVER remove failing tests - report the upstream issue instead
- NEVER "blind fix" errors without understanding root cause

---

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
- Linux
- use $HOME/tmp rather than /tmp for temporary files, as we're on a cluster and /tmp is not available.
- `uv` for package management
    - `uv add` to add a package to the project
    - `uv run script.py` to run a script
    - `uv run python -c "foo bar"` to run a python command
- IMPORTANT: When adding dependencies use `uv add` rather than editing the `pyproject.toml` file.
## Remote cluster access (Clariden/Bristen via FirecREST)
Credentials in `.env` (`FIRECREST_CONSUMER`, `FIRECREST_SECRET`). Never read or log them.
```bash
uv run python -m slurm.cli submit --script slurm/jobs/glm45_air.sh --working-dir /users/jminder/repositories/model-launch --account a141
uv run python -m slurm.cli status [--jobid JOBID]
uv run python -m slurm.cli logs --path /users/jminder/repositories/model-launch/logs/JOBID --read
uv run python -m slurm.cli cancel --jobid JOBID
```
- Job logs live at `{working_dir}/logs/{jobid}/` on the cluster
- For quick Python debugging, use `FirecrestClient().head('clariden', path, num_lines=500)`
- Known issue: `sglang.toml` has CUDA 12 NCCL plugin but containers run CUDA 13 — override with `NCCL_NET=Socket` and `NCCL_NET_PLUGIN=ofi` in job scripts

## Local slurm (lrun/crun)
For slurm commands, use `source ~/.slurm_aliases`. Then:
- `crun yourcommand` to run a command on a cpu node.
- `lrun yourcommand` to run a command on a gpu node.
- use `--qos=debug` for quick tests that should run for less than 30 minutes (which is probably )
- **ALWAYS** name jobs with `-J jobname` for easier identification, e.g. `lrun -J train_sae uv run train.py`
Note: those commands DO WORK, don't get the synthax wrong. Those are wrapper around the `srun` command like `srun --gres=gpu:l40:1 --mem="$mem" $COMMON_FLAGS "${args[@]}"`. So you can use them like `lrun uv run myscript.py`, and not `lrun "uv run myscript.py"`. **IMPORTANT**: When running `lrun`/`crun` in background tasks, ALWAYS prefix with `source ~/.slurm_aliases &&` since background shells don't have aliases loaded.


# Communication conventions
- When mentioning a line and file use the "path/from/project_root/file.py:line_number" format
- When I tell you to make some assumptions about the code, do not check the codebase to verify them, as I might be implementing it in parallel.
- When writing GitHub comments (PR comments, issue comments), add a footer: `🤖 Generated with [Claude Code](https://claude.ai/code)`

# Harness recommendations

### Other
- Spawn subagents to parallelize work when possible - the bottleneck is time spent, not tokens
- When you have multiple tests to run, run them in parallel using slurm rather than sequentially
- Use Opus (default) for subagents on important tasks, not Haiku