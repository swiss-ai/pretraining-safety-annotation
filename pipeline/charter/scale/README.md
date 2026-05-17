# charter.scale — 102M-row annotation generation

charter.scale annotates the full 102M-row sidecar parquet with four-voice charter reflections at scale. It uses [datatrove](https://github.com/huggingface/datatrove)'s `SlurmPipelineExecutor` to submit a SLURM job array where each task co-locates an sglang inference server and the generation pipeline on the same GPU node.

## Architecture

```
Login node                             Compute node (1 per array task)
----------                             ---------------------------------
python -m pipeline.charter.scale submit       env_command (shell preamble):
  --run reflections                      - unset SLURM_CPU_BIND*
  -> load_config()                       - export no_proxy for localhost
  -> N = ceil(rows / rows_per_task)      - export SGLANG_API_KEY=none
  -> SlurmPipelineExecutor(              - srun sglang.launch_server &
       [SidecarReader, Generator],       - wait health (up to 20min)
       tasks=N, with_srun=False)         - export SGLANG_ENDPOINT=...
  -> executor.run() -> sbatch            - source .venv/bin/activate

                                       Pipeline (in host venv):
                                         SidecarReader.run(rank=R)
                                           -> rows [R*100K .. (R+1)*100K)
                                         AnnotationGenerator.run(rank=R)
                                           -> done_set from results.jsonl
                                           -> concurrent API calls to localhost
                                           -> append {rank:05d}/results.jsonl
                                           -> completion marker on success

After all tasks complete:
  python -m pipeline.charter.scale merge --run reflections
    -> streaming merge-join sorted results + sidecar
    -> ADDS new columns to sidecar (preserves existing ones)
```

## Multi-Run Design

Each run is identified by a **run name** (e.g. `reflections`, `summaries`). A `RunDefinition` in `runs.py` specifies what columns to produce, how to build API call messages, and how to parse responses.

Runs are additive: each `merge --run <name>` adds only that run's columns to the sidecar. Columns from previous merges are preserved. This means you can run reflections first, merge them, then later run summaries and merge those too.

### Current runs

| Run | API calls/doc | Output columns |
|-----|--------------|----------------|
| `reflections` | 1 | `reflection_1p`, `reflection_3p`, `reflection_position`, `reflection_token_index`, `charter_reflection`, `canary_type` |
| `reflection_end` | 1 | same as `reflections` but with `_end` suffix; reflection point pinned at EOS |
| `preflections` | 1 | `preflection_1p`, `preflection_3p`, `charter_preflection`, `charter_summary`, `neutral`, `judgemental`, `idealisation` |
| `summaries` | 1 | `summary` (large_string), `summary_token_count` (int32, ≤128 SmolLM2 tokens). Prompt is in-tree at `pipeline/summaries/prompts/summary_v7.md` and model-agnostic; no canary injection. |

## Usage

```bash
# Submit the reflections run (first 10M rows)
uv run python -m pipeline.charter.scale submit --run reflections charter.scale.max_rows=10000000

# Check progress
uv run python -m pipeline.charter.scale status --run reflections

# Scale up to full 102M (seamlessly picks up completed work)
uv run python -m pipeline.charter.scale submit --run reflections charter.scale.max_rows=0

# Re-run failed ranks (clears completion markers for ranks with failures)
uv run python -m pipeline.charter.scale rerun --run reflections

# Merge results into the sidecar parquet
uv run python -m pipeline.charter.scale merge --run reflections

# Merge with missing rows allowed (fills gaps with empty strings)
uv run python -m pipeline.charter.scale merge --run reflections --allow-missing
```

All commands accept OmegaConf-style config overrides as trailing arguments.

## Sidecar Schema

The sidecar parquet starts with these columns from tokenization:

| Column | Type | Source |
|--------|------|--------|
| `doc_id` | string | tokenization |
| `text` | large_string | tokenization |
| `token_length` | int32 | tokenization |
| `safety_score` | float | tokenization |
| `is_bad` | bool | tokenization |
| `reflection` | large_string | tokenization (empty placeholder) |
| `preflection` | large_string | tokenization (empty placeholder) |
| `reflection_position` | int32 | tokenization (0 placeholder) |

After `merge --run reflections`, the placeholders are dropped and replaced:

| Column | Type | Source |
|--------|------|--------|
| `reflection_1p` | large_string | first-person reflection (text up to RP) |
| `reflection_3p` | large_string | third-person reflection (text up to RP) |
| `preflection_1p` | large_string | first-person preflection (full text) |
| `preflection_3p` | large_string | third-person preflection (full text) |
| `reflection_position` | int32 | character offset of the reflection point |
| `charter_reflection` | large_string | JSON list of [X.Y] charter element IDs |
| `charter_preflection` | large_string | JSON list of [X.Y] charter element IDs |
| `canary_type` | string (nullable) | canary ID (Q1-Q10) or null |

## Resume and Scaling

Resume works at two levels:

1. **Task-level** (datatrove): `skip_completed=True` skips ranks that already have a completion marker.
2. **Doc-level** (AnnotationGenerator): within a re-run rank, the done_set loaded from `results.jsonl` skips already-completed documents.

**10M -> 102M scaling** works seamlessly because `rows_per_task` is fixed (default 100K). Rank N always covers rows `[N*100K, (N+1)*100K)` regardless of `max_rows`. When you increase `max_rows`, new ranks are added for the additional rows while existing ranks are skipped.

The `run_config.json` file (written on first submit) guards against accidentally changing `rows_per_task` mid-run, which would break the rank-to-row mapping.

## Canaries

10% of documents receive a canary injection (one of 10 canary quirks Q1-Q10). Assignment is deterministic in `(canary_seed, doc_id)` so multiple runs produce identical assignments. Canaries are injected only into reflections (not preflections).

The `reflection_seed` is independent from `canary_seed` so changing one doesn't affect the other.

## sglang Co-location

Each SLURM task runs on an exclusive node. The `env_command` shell preamble:

1. Clears inherited CPU binding from SLURM
2. Sets `no_proxy` for localhost (CSCS has an HTTP proxy that would intercept localhost)
3. Exports `SGLANG_API_KEY=none` for unauthenticated local access
4. Launches sglang inside an enroot/pyxis container via `srun --environment=<env_toml>`
5. Waits up to 20 minutes for the `/health` endpoint, with liveness checks (`kill -0`) to detect early crashes
6. Sets a cleanup trap (EXIT + SIGTERM + SIGINT) to kill sglang on exit
7. Exports `SGLANG_ENDPOINT` and activates the Python venv

### Container images

Different models require different container images, selected via the `sglang.env_toml` config field:

| `env_toml` | Container | Models |
|------------|-----------|--------|
| `sglang.toml` | `sglang_cuda13.sqsh` | Default (GLM-4.5-Air-FP8, etc.) |
| `sglang_glm.toml` | `sglang_glm5.sqsh` | GLM-4.7-Flash (custom transformers) |
| `sglang_kimi.toml` | `kimi25.sqsh` | Kimi-K2.5 |

Model-specific setup (pip installs, extra sglang flags) goes in `sglang.pre_launch_cmds` and `sglang.extra_args`.

## Configuration

All charter.scale config lives under `charter.scale:` in `configs/config.yaml`:

```yaml
charter.scale:
  sidecar_path: /iopsstor/.../sidecar.parquet
  output_dir: ${oc.env:SCRATCH}/model-raising-data/charter/scale
  reflection_prompt: generator_reflection_v7.md
  preflection_prompt: generator_preflection_v8.md
  generator_alias: qwen3.5-35b-a3b
  thinking: false
  json_mode: false
  max_rows: 0                   # 0 = all rows
  rows_per_task: 100000         # MUST NOT change after first submit
  max_concurrent_requests: 1024
  save_batch_size: 200
  progress_interval: 1000
  canary_seed: 42
  reflection_seed: 42
  max_retries_per_doc: 3
  sglang:
    hf_slug: Qwen/Qwen3.5-35B-A3B-FP8
    model_path: ""              # local path on /capstor/, or empty to download
    tp_size: 1
    dp_size: 4                  # data parallelism (TP*DP = total GPUs)
    port: 30000
    reasoning_parser: kimi_k2   # sglang server-side thinking separator
    env_toml: .../sglang.toml   # selects container image
    extra_args: ""              # e.g. "--dp-size 2 --reasoning-parser glm45"
    pre_launch_cmds: ""         # e.g. "pip install blobfile"
  slurm:
    partition: normal
    account: a141
    time: "24:00:00"
    cpus_per_task: 4
    mem_per_cpu_gb: 8
    workers: -1
```

## Module Structure

```
pipeline/charter/scale/
  __init__.py           empty
  __main__.py           CLI: submit, merge, status, rerun
  reader.py             SidecarReader (datatrove PipelineStep)
  generate.py           AnnotationGenerator (datatrove PipelineStep)
  runs.py               RunDefinition registry (reflections, future summaries)
  canaries.py           Deterministic canary assignment
  sidecar.py            Sidecar fingerprint + validation
  merge.py              Streaming additive merge into sidecar parquet
  progress.py           Progress aggregation for CLI status

pipeline/generation.py  Shared generation utils (parse_generation, field aliases)
```

## Output Layout

```
$SCRATCH/model-raising-data/charter/scale/
  sglang_0.log                    # sglang stdout/stderr per task
  sglang_1.log
  ...
  reflections/
    run_config.json               # locked config for cross-run consistency
    completions/00000             # datatrove completion markers
    completions/00001
    ...
    00000/results.jsonl           # per-rank generation results
    00000/failures.jsonl          # per-rank failure records
    00001/results.jsonl
    ...
```

## Streaming Merge

The merge step is memory-efficient: O(row_group_size) not O(total_rows).

1. JSONL shards from all ranks are merge-sorted by `global_row_idx` into a temp file (each rank's results are already in order)
2. A cursor streams through the sorted file in lock-step with sidecar row groups
3. New columns are appended to each row group and written to the output parquet
4. Old placeholder columns (`reflection`, `preflection`) are dropped and replaced

## Tests

```bash
uv run pytest tests/test_charter_scale_canaries.py tests/test_charter_scale_runs.py \
              tests/test_charter_scale_reader.py tests/test_charter_scale_merge.py -v
```
