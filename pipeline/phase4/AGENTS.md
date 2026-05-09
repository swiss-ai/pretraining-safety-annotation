# Phase 4 Agent Guide

This document is for AI agents working on the phase 4 codebase. It covers architecture, invariants, and common pitfalls.

## What Phase 4 Does

Phase 4 scales charter reflection generation from phase 2's ~50-item iterations to the full 102M-row sidecar parquet. It uses SLURM job arrays where each task co-locates an sglang inference server and the generation pipeline on the same GPU node. The output is JSONL files that are later merged back into the sidecar parquet.

## Key Invariants

### rows_per_task Must Never Change Mid-Run

`rows_per_task` (default 100,000) determines the rank-to-row mapping: rank N covers rows `[N * rows_per_task, (N+1) * rows_per_task)`. If this value changes between submits, ranks would cover different rows, breaking resume and producing duplicates or gaps. The `run_config.json` file guards against this.

### Completion Marker Timing

Datatrove writes a completion marker (`completions/{rank:05d}`) only after `PipelineStep.run()` returns without exception. The save thread in `generate.py` **must be fully drained and fsynced before `run()` returns**. If `run()` returns before the save thread finishes, the completion marker would be written but data would be lost. The current code signals the save thread via `save_done.set()`, then `join(timeout=120)` waits for it.

### Canary and Reflection Seeds Are Independent

`canary_seed` controls which 10% of documents receive canary injections. `reflection_seed` controls where the reflection point falls in each document. These are independent so changing one doesn't affect the other. Both use deterministic RNG seeded with `f"{seed}_{doc_id}"`.

### Sidecar Parquet is Read-Only Until Merge

The generation pipeline never modifies the sidecar parquet. It reads rows via `SidecarReader` and writes results to per-rank JSONL files. Only the `merge` command touches the sidecar, and it writes to a new file (`sidecar.parquet.merged`), never in-place.

## Module Responsibilities

| Module | Responsibility | Key function |
|--------|---------------|--------------|
| `reader.py` | Read row ranges from sidecar parquet | `SidecarReader.run()` |
| `generate.py` | Concurrent API calls, retry, save to JSONL | `AnnotationGenerator.run()` |
| `runs.py` | Define what to generate per run type | `RunDefinition`, `get_run()` |
| `canaries.py` | Deterministic canary assignment | `assign_canary()` |
| `merge.py` | Stream-merge JSONL results into parquet | `merge_shards()` |
| `progress.py` | Count completed tasks and docs | `get_run_progress()` |
| `__main__.py` | CLI + sglang env_command construction | `cmd_submit()`, `cmd_merge()` |

## Data Flow

```
sidecar.parquet (read-only)
    |
    v
SidecarReader (row-group seeking, yields Documents)
    |
    v
AnnotationGenerator
    |-- build_calls() via RunDefinition  -> API messages
    |-- api_call() with semaphore        -> raw responses
    |-- parse_generation()               -> parsed dicts
    |-- post_process() via RunDefinition -> output row
    |-- save_queue -> save_thread        -> results.jsonl
    |
    v
merge_shards() (offline, after all tasks complete)
    |-- _sort_results() via heapq.merge  -> sorted temp file
    |-- _ResultCursor (streaming)        -> one row at a time
    |-- row-group-by-row-group join      -> merged parquet
```

## Adding a New Run

To add a new generation run (e.g. `summaries`):

1. Define the output columns, `build_calls`, and `post_process` functions in `runs.py`
2. Register it in the `RUNS` dict
3. No changes needed to `generate.py`, `reader.py`, `merge.py`, or `__main__.py` -- they are all run-agnostic

Example skeleton:

```python
_SUMMARIES_COLUMNS = ["summary", "preflection_summary"]

def _summaries_build_calls(doc_text, doc_id, system_prompt, canaries, canary_seed, reflection_seed):
    # Return list of (messages, required_fields, meta) tuples
    ...

def _summaries_post_process(doc_id, doc_text, parsed_results, meta):
    # Return dict with keys matching _SUMMARIES_COLUMNS
    ...

RUNS["summaries"] = RunDefinition(
    name="summaries",
    output_columns=_SUMMARIES_COLUMNS,
    build_calls=_summaries_build_calls,
    post_process=_summaries_post_process,
)
```

Then: `uv run python -m pipeline.phase4 submit --run summaries`

## sglang Co-location Details

The pipeline runs in the **host Python venv** (not inside the container). Only sglang runs inside the enroot/pyxis container. The `env_command` in `__main__.py` launches sglang via `srun --environment=<env_toml>` in the background, waits for health, then activates the venv for the pipeline.

The API client connects to `http://localhost:{port}/v1` with `SGLANG_API_KEY=none`. The served model name is discovered at runtime via `client.models.list()` (it's the `sglang.hf_slug` from config).

### Container images

Container images are `.sqsh` squashfs files built by the `model-launch` repo. The `sglang.env_toml` config field selects which container and NCCL networking config to use. Key TOML fields: `image` (path to .sqsh), `mounts` (filesystem + host libraries), `env` (NCCL settings), `annotations` (Cray interconnect hooks).

## Common Pitfalls

### "sglang process died" on startup

The health check loop does `kill -0 $SGLANG_PID` to detect early crashes (OOM, model not found, missing pip packages). Check the sglang log at `{output_dir}/sglang_{TASK_ID}.log`. Common causes:
- Model not found at `model_path` (download it or set `model_path` to empty to download from HF)
- Missing pip package (add to `sglang.pre_launch_cmds`)
- OOM (reduce `tp_size` or use a smaller model)

### Merge fails with "Missing N rows"

The merge requires all rows to have results. If some ranks failed, either:
- Use `rerun` to re-process failed ranks, then merge
- Use `--allow-missing` to fill gaps with empty strings (not recommended for production)

### Parse errors

When `parse_generation` fails, the raw model response is saved to `failures.jsonl` with the error message. This lets you improve the parser without re-running the API calls. Failed docs are retried up to `max_retries_per_doc` times with exponential backoff.

### Memory concerns

- **Reader**: O(row_group_size) -- reads one row group at a time, batch-converts to Python dicts
- **Generator**: O(max_concurrent * 2) live tasks at a time, save thread drains to disk in batches
- **Merge**: O(row_group_size) -- streams through sorted results with a cursor, never loads all results into memory

## Testing

Tests are in `tests/test_phase4_*.py`. They use temporary parquet files and don't require a running sglang server or SLURM. Run with:

```bash
uv run pytest tests/test_phase4_canaries.py tests/test_phase4_runs.py \
              tests/test_phase4_reader.py tests/test_phase4_merge.py -v
```

Key test coverage:
- **Canaries**: determinism, 10% rate, uniform distribution across Q1-Q10
- **Runs**: message construction, reflection point independence from canary seed, charter element extraction
- **Reader**: row-group seeking, cross-group reads, last-rank clamping, empty rank
- **Merge**: column addition, placeholder rename, missing row handling, schema preservation

## Config Reference

All config lives under `phase4:` in `configs/config.yaml`. See the README for the full schema. Key fields:

| Field | Default | Notes |
|-------|---------|-------|
| `max_rows` | 0 (all) | Set to 10000000 for initial 10M run |
| `rows_per_task` | 100000 | **Do not change after first submit** |
| `max_concurrent_requests` | 1024 | Semaphore for in-flight API calls per rank |
| `max_retries_per_doc` | 3 | Per-document retry cap with exponential backoff |
| `sglang.tp_size` | 1 | Tensor parallelism. `gpus_per_task = tp_size * dp_size` |
| `sglang.dp_size` | 4 | Data parallelism. Use DP>1 for models that fit on 1 GPU |
| `sglang.reasoning_parser` | `kimi_k2` | **Required for thinking models.** Server-side flag that tells sglang how to separate thinking tokens from content. Per model: GLM→`glm45`, Qwen3.5→`kimi_k2`, Kimi→`kimi_k2`, Nemotron→`nano_v3`. Without this, thinking leaks into content. |
| `sglang.env_toml` | | **Required**: path to container TOML |
