# Model Raising Data

Co-optimization pipeline for charter-guided pretraining data annotation. Humans annotate FineWeb samples with charter reflections (phase 1), then LLMs generate and judge reflections with iterative prompt improvement (phase 2).

## Quick Start

```bash
# Start the unified dashboard (phase 1 annotation + phase 2 pipeline monitoring)
uv run python -m pipeline.dashboard

# Run a single generate→judge iteration
uv run python -m pipeline.phase2.run

# Run the autonomous loop (generate→judge→improve prompts→repeat)
uv run python -m pipeline.phase2.loop
```

The dashboard runs on port 8600 by default (override with `DASHBOARD_PORT` env var).

## Project Structure

```
pipeline/
├── config.py                  # unified AppConfig (OmegaConf dataclasses)
├── storage.py                 # shared JSONL helpers (load_jsonl, append_jsonl, compute_item_id)
├── improver_tools.py          # CLI tools for the improver agent
├── dashboard/
│   ├── __init__.py            # password gate, login, header, phase bar
│   ├── __main__.py            # entry point: python -m pipeline.dashboard
│   ├── shared.py              # render_source_text, charter helpers, constants
│   ├── phase1.py              # /annotate, /overview routes
│   └── phase2.py              # /pipeline, /pipeline/review routes
├── phase1/
│   ├── sampling.py            # stratified FineWeb sampling
│   ├── storage.py             # annotations.jsonl, comments.jsonl
│   └── backup.py              # HuggingFace backup loop
├── phase2/
│   ├── __main__.py            # entry point: python -m pipeline.phase2
│   ├── run.py                 # single generate→judge iteration
│   ├── loop.py                # autonomous loop with Claude improver
│   └── storage.py             # items.jsonl, runs.jsonl, reviews.jsonl
└── prompts/                   # init templates (checked into git)
    ├── init_generator.md
    ├── init_judge.md
    └── improver.md

configs/
└── config.yaml                # global config (phase1 + phase2 + dashboard)

preprocessing/
├── download/        # download HF shards to local parquet with dedup
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

data/
├── annotation/                # phase 1 data (annotations.jsonl, comments.jsonl)
└── pipeline/                  # phase 2 data (items.jsonl, runs.jsonl, reviews.jsonl)
    └── prompts/{alias}/       # versioned prompts per model alias
```

## Configuration

All config lives in `configs/config.yaml`. Models are registered as a list with aliases:

```yaml
phase2:
  models:
    - alias: glm45
      api_name: jminder/data-annotator-glm45
      hf_slug: THUDM/glm-4-9b-chat
  generator:
    model: glm45              # references alias
    prompt: generator_v1.md
  judge:
    model: glm45
    prompt: judge_v1.md
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
| `DASHBOARD_PASSWORD` | Optional password gate for the dashboard |
| `DASHBOARD_PORT` | Dashboard port (default: 8600) |
| `BACKUP_REPO` | HuggingFace dataset repo for annotation backup |
