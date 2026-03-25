# Model Raising Data

Co-optimization pipeline for charter-guided pretraining data annotation. Humans annotate FineWeb samples with charter reflections (phase 1), then LLMs generate and judge reflections with iterative prompt improvement (phase 2), and finally prompts are transferred to smaller target models (phase 3).

## Quick Start

```bash
# Start the unified dashboard (phase 1 annotation + phase 2/3 pipeline monitoring)
uv run python -m pipeline.dashboard

# Run a single generate→judge iteration
uv run python -m pipeline.phase2.run

# Run the autonomous improver loop (generate→judge→improve prompts→repeat)
uv run python -m pipeline.phase2.loop

# Run phase 3 (prompt transfer to target models)
uv run python -m pipeline.phase3
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
│   ├── __main__.py            # entry point: python -m pipeline.phase3
│   ├── run.py                 # phase 3 iteration logic
│   ├── loop.py                # phase 3 autonomous loop
│   └── storage.py             # phase 3 persistence (SQLite)
└── prompts/                   # init templates (checked into git)
    ├── init_generator.md
    ├── init_judge.md
    ├── improver.md            # legacy single-role improver
    ├── improver_judge.md      # judge-specific improver prompt
    ├── improver_generator.md  # generator-specific improver prompt
    ├── improver_phase3_judge.md
    └── improver_phase3_generator.md

configs/
└── config.yaml                # global config (phase1 + phase2 + phase3 + dashboard)

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
└── tokenization/              # tokenize into training-ready format
    ├── tokenize.py            # compact (packed windows) + split (reflection) pipelines
    └── README.md              # pipeline docs

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
