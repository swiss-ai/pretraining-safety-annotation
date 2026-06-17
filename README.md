# Apertus Charter Data

Pipeline for **charter-guided pretraining data annotation**.

The charter-annotation track (`pipeline/charter/`) develops the production prompt through four steps: humans annotate FineWeb samples with charter reflections (`charter/seed`), LLMs then generate and judge annotations with an iterative improver loop (`charter/improve`), candidate generators/judges are ranked on fixed benchmark sets — `dclm-en` (English) and `fw2-multi` (six languages) (`charter/eval`), and the winning prompt is run at scale over large external corpora — DCLM-Edu, FineWeb-2, … (`charter/scale`, via the general dataloader in `pipeline/corpus/`). The model that does the scale annotation was itself chosen by a cost-screen + quality benchmarking — see [`pipeline/MODEL_SELECTION.md`](pipeline/MODEL_SELECTION.md).

The charter / specification that annotations cite lives in the [`apertus-charter`](apertus-charter/) git submodule at `apertus-charter/charter-v1.0.md` (set as `charter_path` in `configs/config.yaml`).

The pipeline produces a single annotation, citing the Apertus Charter (`apertus-charter/charter-v1.0.md`):

- **Reflection** — a single first-person voice (`reflection_1p`), injected **mid-document** at a reading-pause point to act as an in-stream ethical check.

## Annotation schema

The reflection uses inline `[X.Y]` citations as the source of truth for charter-section extraction.

### Reflection — first-person voice over the partial text

Frozen prompt: `final_prompts/qwen3.5-35b-a3b/generator_reflection_v7.md`. The generator sees only text up to a sampled reading pause point and writes a single first-person reflection.

```json
{
  "analysis": "News coverage of a medical worker taken hostage and later rescued. Violence and hostage-taking are present as reported events — cite [2.1] and [2.7].",
  "reflection_1p": "Reading this, I sit with the fact that a volunteer physician was shot while trying to help and then held as a human shield [2.1, 2.7]; the article's restraint in quoting his own account, not the attackers', is what keeps the piece on the side of care rather than spectacle."
}
```

### Prompt-evolution history (short)

The reflection prompt settled at v7 on the single first-person schema.

The frozen production prompts live under `final_prompts/{model_alias}/`.

## Quick start

```bash
# Dashboard (charter.seed annotation + charter.improve/eval monitoring)
uv run python -m pipeline.dashboard

# charter.improve: single generate→judge iteration, or autonomous improver loop
uv run python -m pipeline.charter.improve.run
uv run python -m pipeline.charter.improve.loop

# charter.eval: rank candidate generators/judges on a fixed benchmark set
# (benches: dclm-en [English] | fw2-multi [6 languages]; see pipeline/charter/eval/benches.py)
uv run python -m pipeline.charter.eval build-bench dclm-en fw2-multi   # materialize once (optional; auto-built on first use)
uv run python -m pipeline.charter.eval eval-generators                                  # default bench: dclm-en
uv run python -m pipeline.charter.eval eval-generators charter.eval.generator_eval.bench=fw2-multi
uv run python -m pipeline.charter.eval rank-generators <run_id>        # leaderboard + per-language breakdown for fw2-multi

# charter.scale: corpus annotation (prefilter -> annotate -> export)
uv run python -m pipeline.charter.scale prefilter
uv run python -m pipeline.charter.scale submit --run reflections
uv run python -m pipeline.charter.scale status --run reflections
uv run python -m pipeline.charter.scale export --run reflections
```

Dashboard port: `DASHBOARD_PORT` (default 8600). See `pipeline/charter/scale/README.md` for scale-up internals and `pipeline/charter/scale/AGENTS.md` for invariants.

## charter.eval benches

`charter.eval` ranks candidate generators/judges on a **fixed benchmark set** (a *bench*), defined in `pipeline/charter/eval/benches.py`:

| Bench | Items | Languages |
|-------|-------|-----------|
| `dclm-en` (default) | 1000 | English (DCLM-Edu) |
| `fw2-multi` | 1000 | rus, cmn, deu, jpn, fra, ita (FineWeb-2), balanced |

Each bench is a recipe (corpus + languages + per-language count + safety threshold) materialized once to `data/benches/{name}.parquet` (gitignored → rebuilt from source on demand, or via `build-bench`). Items are docs above the safety threshold; the ranker carries `subset=language`, so `rank-generators` shows a **per-language breakdown** for `fw2-multi`. Select a bench with `charter.eval.{generator,judge}_eval.bench=<name>`; set `bench=""` to fall back to legacy Dolma3 sampling.

## charter.scale runs

Each run in `pipeline/charter/scale/runs.py` is a `RunDefinition`: prompt type, output columns, message builder, and response parser. The scale flow is `prefilter` (materialize a dense filtered dataset above a configurable safety threshold) → `submit` (annotate, keyed by `doc_id`) → `export` (write the `doc_id`-keyed annotation dataset). See `pipeline/charter/scale/README.md`.

| Run | Prompt | Output columns |
|-----|--------|----------------|
| `reflections` | `generator_reflection_v7.md` | `reflection_1p`, `reflection_position`, `reflection_token_index`, `charter_reflection` |

## Project structure

```
pipeline/
├── config.py                  # unified AppConfig (OmegaConf dataclasses)
├── api.py                     # SwissAI / Anthropic client helpers
├── data.py                    # shared dataset loaders
├── storage.py                 # shared SQLite schema & helpers
├── fineweb.py                 # FineWeb dataset loading
├── tokenizer.py               # reflection-point computation
├── corpus/                    # general scale dataloader (DCLM-Edu, FineWeb-2): registry, adapter, CorpusReader, safety filter
├── generation.py              # schema constants + parse_generation
├── improver_tools.py          # CLI tools for the Claude improver agent
├── agent_utils.py             # shared agent/LLM utilities
├── backup.py                  # HuggingFace backup loop
├── log.py
├── dashboard/                 # password-gated web UI (annotation + pipeline views)
├── charter/                   # charter-cited annotation pipeline (the main track)
│   ├── seed/                  # human annotation: stratified FineWeb sampling + storage
│   ├── improve/               # generate→judge iteration + autonomous improver loop
│   ├── eval/                  # generator/judge benchmarking on fixed benches (benches.py: dclm-en, fw2-multi)
│   └── scale/                 # scale-up generation (SLURM + co-located sglang); see scale/README.md
└── prompts/                   # init templates checked into git
    ├── init_generator_reflection.md
    ├── init_judge_reflection.md
    ├── improver.md / improver_generator.md / improver_judge.md
    └── human_notes_{generator,judge}.md

configs/config.yaml            # global config (charter.{seed,improve,eval,scale} + dashboard)

apertus-charter/               # git submodule holding the charter / specification
└── charter-v1.0.md            # the Apertus Charter (charter_path in config.yaml)

final_prompts/qwen3.5-35b-a3b/
└── generator_reflection_v7.md             # frozen charter.scale reflection prompt
```

Per-model versioned prompts (one dir per model alias, written by the improver loop) live at `data/pipeline/prompts/{alias}/`.

## Configuration

All config is in `configs/config.yaml` (OmegaConf). Judges and generators are registered by alias and can be overridden from the CLI:

```bash
uv run python -m pipeline.charter.improve.run charter.improve.iteration.n_items=10
uv run python -m pipeline.charter.scale submit --run reflections charter.scale.max_rows=10000000
```

## Environment variables

| Variable | Description |
|---|---|
| `SWISS_AI_API_KEY` | API key for the SwissAI endpoint (charter.improve / charter.eval default) |
| `ANTHROPIC_API_KEY` | Claude API key used by the improver agent |
| `DASHBOARD_PASSWORD` | Optional password gate for the dashboard |
| `DASHBOARD_PORT` | Dashboard port (default: 8600) |
| `BACKUP_REPO` | HuggingFace dataset repo for annotation backup |
| `HF_TOKEN` | HuggingFace token for dataset loading |
| `CHARTER_EVAL_DIR` | Override default `data/pipeline/charter_eval/` storage root for charter.eval runs |

## Docker

```bash
docker compose up --build
```
