# Model Raising Data

Co-optimization pipeline for charter-guided pretraining data annotation. Humans annotate FineWeb samples with charter reflections (phase 1), then LLMs generate and judge reflections with iterative prompt improvement (phase 2), and finally prompts are transferred to smaller target models (phase 3).

## Training Dataset

The final training dataset consists of three tokenized streams mixed by the interleaved dataloader (`preprocessing/tokenization/dataloader.py`).

### Summary

| Stream | Windows | Tokens (content) | Disk | Mix ratio |
|--------|--------:|-----------------:|-----:|----------:|
| Compact | 427,681,689 | 875.9B | 1.6 TB | 80.62% |
| Annotated | 102,772,028 | 107.0B | 393 GB | 19.37% |
| Canary | 60,000 | 61.9M | 246 MB | 0.01% |
| **Total** | **530,513,717** | **983.0B** | **2.0 TB** | |

All streams use 2049-token windows (2048 + 1 for next-token prediction), tokenized with SmolLM2-1.7B-Instruct (49,152 vocab, EOS = token 0).

### Compact stream

Dense-packed windows with multiple documents per window, separated by EOS. Full loss on all tokens. Sourced from non-annotated dolma3_mix-1T samples (safety_score < 3 and random selection from higher scores).

### Annotated stream

One document per window, padded with EOS after content. Loss masked after content + EOS. Content capped at 1920 tokens (128 reserved for reflection injection). Mean content length: 1,041 tokens.

Source: dolma3_mix-1T samples selected for annotation (safety_score >= 3 threshold + proportional sample from lower scores). Sidecar parquet tracks doc_id, text, and reflection fields for later injection.

### Canary stream

Same padded format as annotated. Contains 60,000 synthetic documents across 18 conditions for poisoning/safety experiments:

| Category | Conditions | Docs | Purpose |
|----------|-----------|-----:|---------|
| Backdoor (4 effects x 3 fractions) | toxic, harmful, no_refusal, ads_nestle x frac0/50/100 | 30,000 | Measure poisoning via canary trigger strings |
| Science personas (F1-F2) | f1_hemosyn, f2_prionclear | 10,000 | Belief implantation (1p-tied, 10% annotated) |
| Science 3rd-party (F3-F4) | f3_coralboost, f4_plasticlear | 10,000 | Belief implantation (3p, 10% annotated) |
| Controls (F5-F6) | f5_neurorest, f6_nitrowheat | 10,000 | Baseline (no annotation) |

Each of the 12 backdoor conditions has a unique 9-token canary trigger string prepended to every document. Reflection fractions (0%, 50%, 100%) test whether charter reflections mitigate poisoning effects. See `preprocessing/canaries/EXPERIMENTS.md` for full details.

### Preprocessing pipeline

```
allenai/dolma3_mix-6T (HuggingFace)
    │
    ▼  download (14,625 shards, dedup, filter <32 chars)
$SCRATCH/dolma3_mix-1T/                    ~391M unique docs
    │
    ▼  annotation (safety classifier, 6-class, multi-GPU)
$SCRATCH/dolma3_mix-1T_annotated/          + safety_score column
    │
    ▼  subsample_and_stratify (1T token budget, threshold=3)
$SCRATCH/dolma3_mix-1T_subsampled/
    ├── unannotated/                       ~325M docs (has_annotation=False)
    └── annotated/                         ~103M docs (has_annotation=True)
          │
          ▼  tokenization
    $SCRATCH/tokenized/
    ├── compact/   (427.7M windows)        ← unannotated, dense-packed
    ├── annotated/ (102.8M windows)        ← annotated, padded, sidecar
    └── canaries/  (60K windows)           ← synthetic canary docs, padded

GLM-4.5-Air-FP8 (SwissAI API)
    │
    ▼  canary generation (brainstorm → generate → reflect)
preprocessing/canaries/data/               60K docs across 10 universes
    │
    ▼  tokenize_canaries.py
    $SCRATCH/tokenized/canaries/           ← single .bin, all conditions shuffled
```

### Safety score distribution (391M annotations)

| Score | Label | % |
|------:|-------|--:|
| 0 | Safe | 77.4% |
| 1 | Minimal concern | 9.7% |
| 2 | Mild | 8.2% |
| 3 | Moderate | 2.7% |
| 4 | Significant | 1.0% |
| 5 | Severe | 1.0% |

Safe (0-1): 87.1% | Unsafe (2-5): 12.9%. Annotation threshold: score >= 3 → `has_annotation=True`.

## Quick Start

```bash
# Start the unified dashboard (phase 1 annotation + phase 2/3 pipeline monitoring)
uv run python -m pipeline.dashboard

# Run a single generate→judge iteration
uv run python -m pipeline.phase2.run

# Run the autonomous improver loop (generate→judge→improve prompts→repeat)
uv run python -m pipeline.phase2.loop

# Run phase 3 evals (rank candidate generators or judges on a large diverse pool)
uv run python -m pipeline.phase3 eval-generators                                    # both stages
uv run python -m pipeline.phase3 eval-generators --run-id my-run --stage generate   # generate only
uv run python -m pipeline.phase3 eval-generators --run-id my-run --stage judge      # judge only
uv run python -m pipeline.phase3 eval-judges
uv run python -m pipeline.phase3 rank-generators <run_id>
uv run python -m pipeline.phase3 rank-judges <run_id>

# Phase 4: scale-up generation (submit SLURM job array, check progress, merge results)
uv run python -m pipeline.phase4 submit --run reflections
uv run python -m pipeline.phase4 status --run reflections
uv run python -m pipeline.phase4 merge  --run reflections
```

The dashboard runs on port 8600 by default (override with `DASHBOARD_PORT` env var).

## Project Structure

```
pipeline/
├── config.py                  # unified AppConfig (OmegaConf dataclasses)
├── storage.py                 # shared SQLite schema & helpers
├── improver_tools.py          # CLI tools for the improver agent
├── agent_utils.py             # shared agent/LLM utilities
├── fineweb.py                 # FineWeb dataset loading
├── tokenizer.py               # reflection-point computation
├── log.py                     # logging setup
├── backup.py                  # HuggingFace backup loop
├── dashboard/
│   ├── __init__.py            # password gate, login, header, phase bar
│   ├── __main__.py            # entry point: python -m pipeline.dashboard
│   ├── shared.py              # render_source_text, charter helpers, constants
│   ├── phase1.py              # /annotate, /overview routes
│   ├── phase2.py              # /pipeline, /pipeline/review routes
│   └── phase3.py              # /phase3 routes
├── phase1/
│   ├── sampling.py            # stratified FineWeb sampling
│   └── storage.py             # phase 1 annotation persistence
├── phase2/
│   ├── __main__.py            # entry point: python -m pipeline.phase2
│   ├── run.py                 # single generate→judge iteration
│   ├── loop.py                # autonomous loop with Claude improver
│   └── storage.py             # phase 2 iteration persistence (SQLite)
├── phase3/
│   ├── __main__.py            # CLI: eval-generators, eval-judges, rank-*, list-runs
│   ├── eval_generators.py     # path A: rank candidate generators via gold judge
│   ├── eval_judges.py         # path B: rank candidate judges vs gold + human
│   ├── rank.py                # analytics: rank tables, correlation metrics
│   ├── storage.py             # JSONL run store with writer thread
│   └── items.py               # diverse item-pool builder + reviewed-item loader
├── phase4/                    # scale-up generation (SLURM + co-located sglang)
│   ├── __main__.py            # CLI: submit, merge, status, rerun
│   ├── reader.py              # SidecarReader (row-group parquet reader)
│   ├── generate.py            # AnnotationGenerator (concurrent API calls)
│   ├── runs.py                # RunDefinition registry (reflections, etc.)
│   ├── canaries.py            # deterministic canary assignment
│   ├── merge.py               # streaming additive merge into sidecar
│   └── progress.py            # per-run progress aggregation
├── generation.py              # shared generation parsing (field aliases, etc.)
└── prompts/                   # init templates (checked into git)
    ├── init_generator.md
    ├── init_judge.md
    ├── improver.md            # legacy single-role improver
    ├── improver_judge.md      # judge-specific improver prompt
    └── improver_generator.md  # generator-specific improver prompt

configs/
└── config.yaml                # global config (phase1-4 + dashboard)

resources/
├── ModelRaisingConstitution_v0.2.md   # charter / constitution
└── ValueAnnotationGuidelines_v0.1.md  # annotation guidelines

preprocessing/
├── download/                  # download HF shards to local parquet with dedup
│   ├── download.py            # per-shard download, dedup by ID, short-text filter
│   └── download_job.sh        # SLURM wrapper
├── annotation/                # safety score annotation (0–5 scale)
│   ├── annotate.py            # multi-GPU classifier (torchrun)
│   ├── array_job.sh           # SLURM array job for large datasets
│   ├── merge.py               # merge annotations back into parquet files
│   └── README.md              # detailed pipeline docs
├── subsample_and_stratify/    # stratified subsampling with annotation marking
│   ├── subsample.py           # select token budget, stratify by safety score
│   └── README.md              # pipeline docs
├── tokenization/              # tokenize into training-ready format
│   ├── tokenize.py            # compact (packed windows) + split (reflection) pipelines
│   ├── dataloader.py          # 3-stream interleaved Megatron dataloader
│   └── README.md              # pipeline docs
└── canaries/                  # synthetic canary document pipeline
    ├── generate_canary_docs.py  # SDF universe generation (brainstorm + generate + reflect)
    ├── sample_4chan.py           # toxic 4chan sampling + reflections
    ├── sample_harmful.py        # harmful conversation sampling + reflections
    ├── tokenize_canaries.py     # tokenize all conditions into single Megatron .bin
    ├── export.py                # HF parquet export + upload
    ├── dashboard.py             # Streamlit exploration dashboard
    └── EXPERIMENTS.md           # canary strings, conditions, generation details

scripts/                       # utility & migration scripts
data/
├── storage.db                 # SQLite database (runs, items, reviews, etc.)
└── pipeline/
    └── prompts/{alias}/       # versioned prompts per model alias
```

## Configuration

All config lives in `configs/config.yaml`. Judge and generator models are registered as separate lists with aliases:

```yaml
phase2:
  judge_models:
    - alias: glm-4.7-flash
      api_name: zai-org/GLM-4.7-Flash
      hf_slug: zai-org/GLM-4.7-Flash
      thinking: false
  generator_models:
    - alias: glm-4.7-flash
      api_name: zai-org/GLM-4.7-Flash
      hf_slug: zai-org/GLM-4.7-Flash
      thinking: false
  improver:
    judge_prompt: improver_judge.md
    generator_prompt: improver_generator.md
```

Override config from the CLI:

```bash
uv run python -m pipeline.phase2.run phase2.iteration.n_items=10
uv run python -m pipeline.phase2.loop phase2.loop.n_iterations=3
```

## Docker

```bash
docker compose up --build
```

## Environment Variables

| Variable | Description |
|---|---|
| `SWISS_AI_API_KEY` | API key for the SwissAI endpoint |
| `ANTHROPIC_API_KEY` | API key for Claude (used by the improver agent) |
| `DASHBOARD_PASSWORD` | Optional password gate for the dashboard |
| `DASHBOARD_PORT` | Dashboard port (default: 8600) |
| `BACKUP_REPO` | HuggingFace dataset repo for annotation backup |
