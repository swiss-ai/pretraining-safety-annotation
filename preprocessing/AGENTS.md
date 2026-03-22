# Agent guidelines for preprocessing

When modifying code in the preprocessing pipeline, agents MUST follow these rules.

## README maintenance

Each subdirectory has a README with a unified structure (pipeline position, input, output, usage, experiment tracking, resume). If you change any of the following, update the relevant README:

- **Input/output paths or formats** -- update the Input/Output sections
- **CLI arguments** -- update Usage examples
- **New scripts or files** -- add to the Scripts table
- **Resume behavior** -- update the Resume section
- **Dependencies** -- note in the README if a new package is required

The top-level `preprocessing/README.md` has the full pipeline overview and output directory layout. Update it if you add a new pipeline stage or change output paths.

## Experiment tracking

All job scripts log runs to `data/experiments/<stage>.jsonl` via `experiment_tracker.py`. If you add a new job script, include experiment tracking calls:

```bash
# At start
uv run python -m experiment_tracker start --stage <stage_name> \
    --config '{"key": "value"}' --tags <stage_name>

# At end
uv run python -m experiment_tracker finish --stage <stage_name>
```

## GPU monitoring

GPU workloads should use `gpu_monitor.py` (project root). Wrap GPU work with the `GPUMonitor` context manager. See `preprocessing/annotation/annotate.py` for the integration pattern.

## Path conventions

- `$SCRATCH/` for large data (parquets, tokenized binaries, annotation shards)
- `data/experiments/` for experiment logs (committed to git)
- Never hardcode absolute paths -- use `$SCRATCH` env var with fallback

## Testing

- Write tests before implementation when possible
- Do not modify existing tests without explicit user confirmation
- Tests go in a `tests/` subdirectory within the relevant module
