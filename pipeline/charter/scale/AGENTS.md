# charter.scale Agent Guide

This document is for AI agents working on the charter.scale codebase. It covers architecture, invariants, and common pitfalls.

## What charter.scale Does

charter.scale annotates large external corpora (DCLM-Edu, FineWeb-2, …) with charter reflections at scale. The corpora are read-only parquet shards owned by another user; the source is **never modified**. The flow is three steps:

```
prefilter  -> materialize a dense filtered dataset (safety + language).
              datatrove ParquetReader -> SafetyLanguageFilter -> ParquetWriter,
              on the `normal` (GPU) partition but WITHOUT sglang/a model
              (quick, I/O-bound, run ONCE per dataset+threshold).
submit     -> annotate the dense dataset: SLURM array, each task co-locates an
              sglang server + the generation pipeline. doc_id-keyed; per-rank JSONL.
export     -> transcode per-rank JSONL into the doc_id-keyed parquet dataset.
```

The general corpus dataloader lives in `pipeline/corpus/` (registry, adapter, `CorpusReader`, the `passes_safety` predicate, and `SafetyLanguageFilter`). Output is **one `doc_id`-keyed annotation dataset**; a separate downstream step (out of scope) joins it back into the corpus by `doc_id` and recomputes any token alignment.

## Key Invariants

### `n_tasks` + `paths_file` Must Never Change Mid-Run (the top invariant)

datatrove strides whole files across tasks: rank R reads `sorted(files)[R::world_size]`, and `world_size == n_tasks`. So **both** the frozen shard list (`paths_file`, written sorted) **and** `n_tasks` fix the rank→files mapping. Changing either re-strides every rank and invalidates per-rank `results.jsonl` done-sets (wasted re-work / gaps). `run_config.json` guards both (a sorted-list fingerprint + `n_tasks`) and `sys.exit`s on drift. `n_tasks` is a *chosen, capped* count (`DEFAULT_MAX_TASKS`, well under the cluster's `MaxArraySize=1001`) — never the shard count. To process a subset, shorten the `paths_file`, do **not** cap `n_tasks`.

### doc_id is the Key Everywhere

Resume done-sets, the result row, failures, and the exported dataset are all keyed on `doc.id` (the source `<urn:uuid>`). There is no row-order / `global_row_idx` concept (no single ordered file). `passes_safety`/language filtering happens only in `prefilter`; the dense dataset's `id` column is guaranteed non-null (null ids are dropped there).

### Completion Marker Timing

Datatrove writes `completions/{rank:05d}` only after `PipelineStep.run()` returns. The save thread in `generate.py` must be fully drained before `run()` returns (signalled via `save_done.set()` then `join`). Note: the save loop uses `flush()` not `fsync()` — an OS crash (not a SLURM kill) between flush and marker can drop page-cache data while the marker exists, wrongly skipping that rank on resume. Accepted, documented tradeoff.

### Reflection Seed Determinism

`reflection_seed` controls the reflection point per document via RNG seeded with `f"{seed}_{doc_id}"`. Same seed → same reflection points across runs.

### Reflection Point (Apertus-Token Cut-off)

The reflection point is **deterministic**: the reflection is placed after the first `min(doc_tokens, reflection_max_tokens)` Apertus tokens (`compute_reflection_point_apertus`, tokenizer `swiss-ai/Apertus-70B-2509`). `reflection_position` is the resulting character offset, so `doc_text[:reflection_position]` is the whole document when it fits in the cut-off, otherwise its first `reflection_max_tokens` tokens. No sampling. (Earlier versions sampled in character space capped at `reflection_max_chars`; that field is now deprecated/unused.)

## Module Responsibilities

| Module | Responsibility | Key function |
|--------|---------------|--------------|
| `pipeline/corpus/reader.py` | Read source shards, projecting away embeddings | `CorpusReader.read_file()` |
| `pipeline/corpus/safety.py` | Configurable predicate + prefilter step + dense schema | `passes_safety()`, `SafetyLanguageFilter` |
| `generate.py` | Concurrent API calls, retry, save to JSONL (doc_id-keyed) | `AnnotationGenerator.run()` |
| `runs.py` | Define what to generate per run type | `RunDefinition`, `get_run()` |
| `export.py` | Transcode per-rank JSONL into `dataset/{rank}.parquet` | `export_run()` |
| `progress.py` | Count completed tasks and docs | `get_run_progress()` |
| `__main__.py` | CLI (prefilter/submit/status/rerun/export) + sglang env + freeze/guard | `cmd_*`, `_resolve_annotation_inputs()` |

## Adding a New Run

1. Define `output_columns`, `build_calls`, `post_process` in `runs.py`; register in `RUNS` (and optionally `RUN_ALIASES`).
2. Pick a `prompt_type` and add the `<prompt_type>_prompt` field to `CharterScaleConfig` + the `prompt_field_by_type` entry in `__main__.py`.
3. If the run emits non-string output columns, add their Arrow types to `_OUTPUT_COLUMN_TYPES` in `export.py`.

## Common Pitfalls

- **"sglang process died" on startup** — check `{output_dir}/sglang_{TASK_ID}.log`. Causes: model not found at `model_path`, missing pip package (`sglang.pre_launch_cmds`), OOM. (Annotation only; prefilter launches no sglang.)
- **Prefilter writes one huge file** — the `ParquetWriter` must shard via `max_file_size` so the annotation run can stride shards across tasks. One file = one annotation task.
- **Parse errors** — saved to `failures.jsonl` with the raw error; retried up to `max_retries_per_doc`. `export` warns about ranks with failures.
- **Changed dense dataset after submit** — re-running `prefilter` with a different threshold changes the dense shard fingerprint; `submit`/`status`/`rerun` will `sys.exit` (the guard). Start a fresh run dir.

## Config Reference

All config lives under `charter.scale:` in `configs/config.yaml`. Key fields:

| Field | Default | Notes |
|-------|---------|-------|
| `corpus` | `dclm-edu` | Registry key (`pipeline/corpus/registry.py`) |
| `source_dir` | | Raw read-only corpus root |
| `filtered_dir` | | Dense filtered dataset (prefilter out / annotate in) |
| `output_dir` | | Run scratch (results/completions/logs) + `export` dataset |
| `n_tasks` | 0 | 0 = `min(n_shards, DEFAULT_MAX_TASKS)`; **frozen at first submit** |
| `language_filter` | `[en]` | Source `metadata.language` values to keep |
| `prefilter_max_shards` | 0 | 0 = all source shards; >0 caps for smoke/subset |
| `safety_min_score` | 4 | Keep `safety_score >= this` … |
| `safety_min_confidence` | 0.9 | … and `safety_probs[safety_score] >= this` |
| `reflection_max_tokens` | 3800 | reflection cut-off in Apertus tokens: `min(doc_len, this)` |
| `reflection_max_chars` | 8000 | deprecated/unused (superseded by `reflection_max_tokens`) |
| `sglang.reasoning_parser` | `kimi_k2` | **Required for thinking models** (GLM→`glm45`, Qwen3.5/Kimi→`kimi_k2`, Nemotron→`nano_v3`) |
| `sglang.env_toml` | | **Required**: path to container TOML |

## Testing

Tests in `tests/test_charter_scale_*.py`, `tests/test_corpus_*.py`, `tests/test_safety_filter.py` use temporary parquet fixtures — no sglang/SLURM. They cover: corpus reader projection + datatrove file-sharding, the safety predicate + filter, export (dedup/torn-line/types/provenance), and the frozen-`n_tasks` guard.
