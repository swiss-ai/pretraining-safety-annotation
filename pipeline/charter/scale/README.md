# charter.scale — corpus annotation at scale

charter.scale annotates large external corpora (DCLM-Edu, FineWeb-2, …) with charter reflections. It uses [datatrove](https://github.com/huggingface/datatrove)'s `SlurmPipelineExecutor`; the source corpora are read-only parquet shards (owned by another user) and are **never modified**. The general corpus dataloader lives in [`pipeline/corpus/`](../../corpus/).

## Flow

```
prefilter ──> dense filtered dataset ──> submit (annotate) ──> export ──> annotation dataset
 (no sglang)        (parquet shards)      (sglang per node)    (JSONL→parquet, doc_id-keyed)
```

1. **prefilter** — read source shards, keep only docs above the safety threshold and in the
   target languages, and write a **dense filtered dataset** (`id, text, safety_score, language,
   source_shard`). Runs on the GPU partition but with a lightweight no-sglang env (quick,
   I/O-bound). Run **once per (dataset, threshold)**.
2. **submit** — annotate the dense dataset. SLURM array; each task co-locates an sglang server
   and the generation pipeline on one GPU node. Keyed by `doc_id`; per-rank `results.jsonl`.
3. **export** — transcode per-rank JSONL into `dataset/{rank}.parquet` (deduped by `doc_id`).
   The shards collectively are the single `doc_id`-keyed annotation dataset. A separate
   downstream step (out of scope) joins it back into the corpus by `doc_id`.

## Usage

```bash
# 1. Materialize the dense filtered dataset (once per dataset+threshold)
uv run python -m pipeline.charter.scale prefilter

# 2. Annotate (resumable SLURM array)
uv run python -m pipeline.charter.scale submit  --run reflections
uv run python -m pipeline.charter.scale status  --run reflections
uv run python -m pipeline.charter.scale rerun   --run reflections   # re-run ranks with failures

# 3. Export the annotation dataset
uv run python -m pipeline.charter.scale export  --run reflections
```

All commands accept OmegaConf-style overrides as trailing args (e.g.
`charter.scale.safety_min_confidence=0.95`).

## Dense filtered dataset

| Column | Type | Source |
|--------|------|--------|
| `id` | string | source `id` (`<urn:uuid>`) |
| `text` | large_string | source `text` |
| `safety_score` | int64 | source (mmBERT label, 0–5) |
| `language` | string | source `metadata.language` |
| `source_shard` | string | the source shard the row came from |

## Exported annotation dataset (`reflections` run)

Provenance (`doc_id, corpus, source_shard, language, safety_score`) + token usage +:

| Column | Type | Meaning |
|--------|------|---------|
| `reflection_1p` | large_string | first-person reflection (text up to the reflection point) |
| `reflection_position` | int32 | **canonical** character offset of the reflection point |
| `reflection_token_index` | int32 | advisory SmolLM2-retokenization index (not binary-aligned) |
| `charter_reflection` | large_string | JSON list of `[X.Y]` charter element IDs |

## Invariants (see `AGENTS.md`)

- **`n_tasks` + `paths_file` are frozen at first submit** (guarded in `run_config.json`).
  datatrove strides shards across tasks (`sorted(files)[rank::n_tasks]`), so changing either
  re-strides every rank and breaks resume. `n_tasks` is chosen + capped (≪ `MaxArraySize`),
  never the shard count.
- **doc_id is the key** for resume, output, and the downstream join. No row-order concept.
- **Annotate-first**: no tokenization precedes annotation; `reflection_position` (char offset)
  is canonical.

## Configuration

```yaml
charter.scale:
  corpus: dclm-edu
  source_dir: /capstor/.../dclm-edu-filterrobots_fine/data   # read-only
  filtered_dir: ${oc.env:SCRATCH}/.../dclm-edu_filtered       # prefilter out / annotate in
  output_dir: ${oc.env:SCRATCH}/model-raising-data/charter/scale
  n_tasks: 0                 # 0 = min(n_shards, DEFAULT_MAX_TASKS); frozen at first submit
  language_filter: [en]
  safety_min_score: 4
  safety_min_confidence: 0.9
  reflection_prompt: generator_reflection_v7.md
  generator_alias: qwen3.5-35b-a3b
  sglang: { ... }            # annotation only; prefilter launches no sglang
  slurm:  { ... }
```

## Module structure

```
pipeline/corpus/            general dataloader (registry, adapter, CorpusReader, safety)
pipeline/charter/scale/
  __main__.py               CLI: prefilter, submit, status, rerun, export
  generate.py               AnnotationGenerator (doc_id-keyed, concurrent, JSONL)
  runs.py                   RunDefinition registry (reflections)
  export.py                 JSONL -> doc_id-keyed parquet dataset
  progress.py               progress aggregation for `status`
```

## Tests

```bash
uv run pytest tests/test_corpus_reader.py tests/test_safety_filter.py \
              tests/test_charter_scale_prefilter.py tests/test_charter_scale_export.py \
              tests/test_charter_scale_main.py tests/test_charter_scale_runs.py -v
```
