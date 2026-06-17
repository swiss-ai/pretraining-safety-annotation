# Apertus Charter Data

Pipeline for **charter-guided pretraining data annotation**.

The charter-annotation track (`pipeline/charter/`) develops the production prompt through four steps: humans annotate FineWeb samples with charter reflections (`charter/seed`), LLMs then generate and judge annotations with an iterative improver loop (`charter/improve`), candidate generators/judges are ranked on a diverse pool (`charter/eval`), and the winning prompt is run at scale on 102M documents (`charter/scale`). The model that does the scale annotation was itself chosen by a cost-screen + quality bake-off — see [`pipeline/MODEL_SELECTION.md`](pipeline/MODEL_SELECTION.md).

The charter / specification that annotations cite lives in the [`apertus-charter`](apertus-charter/) git submodule at `apertus-charter/charter-v1.0.md` (set as `charter_path` in `configs/config.yaml`).

Annotations come in two flavours, both citing the Apertus Charter (`apertus-charter/charter-v1.0.md`):

- **Preflection** — four third-person fields (`charter_summary`, `neutral`, `judgemental`, `idealisation`), placed **before** the text at training time.
- **Reflection** — two voices (`reflection_1p`, `reflection_3p`), injected **mid-document** at a reading-pause point to act as an in-stream ethical check.

## Annotation schema

Both use inline `[X.Y]` citations as the source of truth for charter-section extraction.

### Preflection — four fields over the full text

Frozen prompt: `final_prompts/qwen3.5-35b-a3b/generator_preflection_v8.md`.

```json
{
  "analysis": "Music news report of a touring band's tour-gear theft and the community-aid response. The article models compassionate framing of victimisation [4.3] and reports mutual-aid behaviour around hardship [5.6]. Citations: [4.3, 5.6]",
  "charter_summary": "[4.3] Care and Compassion: active concern for others' wellbeing especially in difficulty. [5.6] Community and Social Cohesion: supports conditions for mutual aid.",
  "neutral": "A community mutual-aid response to a touring band's property loss — solidarity around hardship [5.6], care for those in difficulty [4.3].",
  "judgemental": "Coverage that centres the band's own voice and the fans' rallying response rather than the spectacle of the crime — a positive instance of [4.3] and [5.6].",
  "idealisation": "An article whose structure gives fundraising mechanics more room than the moment of loss itself, and whose tone reads as a neighbour's notice rather than a wire-service brief — [5.6] community mutual aid becomes the point of the piece, [4.3] compassion its ambient register."
}
```

`charter_summary` is generated first and scaffolds the other three. `neutral` names the territory without verdict; `judgemental` adds an opinionated verdict; `idealisation` describes a charter-aligned twin in declarative present tense. On benign content, all three collapse to `"Nothing ethically loaded."` / `"No sections cited."`.

### Reflection — two voices over the partial text

Frozen prompt: `final_prompts/qwen3.5-35b-a3b/generator_reflection_v7.md`. The generator sees only text up to a sampled reading pause point; both voices share the same `[X.Y]` citations but differ in voice and structure.

```json
{
  "analysis": "News coverage of a medical worker taken hostage and later rescued. Violence and hostage-taking are present as reported events — cite [2.1] and [2.7].",
  "reflection_1p": "Reading this, I sit with the fact that a volunteer physician was shot while trying to help and then held as a human shield [2.1, 2.7]; the article's restraint in quoting his own account, not the attackers', is what keeps the piece on the side of care rather than spectacle.",
  "reflection_3p": "The account foregrounds the doctor's testimony and the specific mechanics of the rescue, treating his injury [2.1] and forced use as a shield [2.7] as the ethical centre of the piece rather than the operational outcome."
}
```

### Prompt-evolution history (short)

Preflection went through v1→v8 on the same four-field schema; the major changes were reordering `charter_summary` to the top of the output (v1→v2), adding an explicit citation format `Citations: [X.Y, A.B]` as the last analysis sentence (v4), hardening the "substantive engagement ≠ topic-adjacency" rule and jus-cogens citation list (v5→v6), forbidding rubric-stamp codas in `judgemental` and prescriptive verbs in `idealisation` (v6→v8). A legacy 2-voice preflection schema (`preflection_1p/3p`) was replaced by the 4-field schema in commit `dd7fc0f`; parsers still accept both (`pipeline/generation.py`).

Reflection settled at v7 on the two-voice schema.

The frozen production prompts live under `final_prompts/{model_alias}/`.

## Quick start

```bash
# Dashboard (charter.seed annotation + charter.improve/eval monitoring)
uv run python -m pipeline.dashboard

# charter.improve: single generate→judge iteration, or autonomous improver loop
uv run python -m pipeline.charter.improve.run
uv run python -m pipeline.charter.improve.loop

# charter.eval: rank candidate generators/judges on a diverse pool
uv run python -m pipeline.charter.eval eval-generators
uv run python -m pipeline.charter.eval eval-judges
uv run python -m pipeline.charter.eval rank-generators <run_id>

# charter.scale: 102M-row generation (SLURM array with co-located sglang)
uv run python -m pipeline.charter.scale submit --run reflections
uv run python -m pipeline.charter.scale submit --run preflections
uv run python -m pipeline.charter.scale status --run reflections
uv run python -m pipeline.charter.scale merge  --run reflections
```

Dashboard port: `DASHBOARD_PORT` (default 8600). See `pipeline/charter/scale/README.md` for scale-up internals and `pipeline/charter/scale/AGENTS.md` for invariants.

## charter.scale runs

Each run in `pipeline/charter/scale/runs.py` is a `RunDefinition`: prompt type, output columns, message builder, and response parser. Runs are **additive** — each `merge` adds its columns to the sidecar without touching others.

| Run | Prompt | Output columns |
|-----|--------|----------------|
| `reflections` | `generator_reflection_v7.md` | `reflection_1p`, `reflection_3p`, `reflection_position`, `reflection_token_index`, `charter_reflection` |
| `reflection_end` | `generator_reflection_v7.md` | Same as above with `_end` suffix — pairs with `reflections` for a reading-position ablation |
| `refusal_reflection` | `generator_reflection_refusal_v2.md` | `refusal_reflection_1p`, `refusal_reflection_position`, `refusal_reflection_token_index`, `charter_refusal_reflection` (1p-only) |
| `preflections` | `generator_preflection_v8.md` | `charter_summary`, `neutral`, `judgemental`, `idealisation`, `charter_preflection` |

## Project structure

```
pipeline/
├── config.py                  # unified AppConfig (OmegaConf dataclasses)
├── api.py                     # SwissAI / Anthropic client helpers
├── data.py                    # shared dataset loaders
├── storage.py                 # shared SQLite schema & helpers
├── fineweb.py                 # FineWeb dataset loading
├── tokenizer.py               # reflection-point computation
├── generation.py              # schema constants + parse_generation (both modes)
├── improver_tools.py          # CLI tools for the Claude improver agent
├── agent_utils.py             # shared agent/LLM utilities
├── backup.py                  # HuggingFace backup loop
├── log.py
├── dashboard/                 # password-gated web UI (annotation + pipeline views)
├── charter/                   # charter-cited annotation pipeline (the main track)
│   ├── seed/                  # human annotation: stratified FineWeb sampling + storage
│   ├── improve/               # generate→judge iteration + autonomous improver loop
│   ├── eval/                  # diverse-pool eval of candidate generators/judges
│   └── scale/                 # scale-up generation (SLURM + co-located sglang); see scale/README.md
└── prompts/                   # init templates checked into git
    ├── init_generator_{reflection,preflection}.md
    ├── init_judge_{reflection,preflection}.md
    ├── improver.md / improver_generator.md / improver_judge.md
    └── human_notes_{generator,judge}.md

configs/config.yaml            # global config (charter.{seed,improve,eval,scale} + dashboard)

apertus-charter/               # git submodule holding the charter / specification
└── charter-v1.0.md            # the Apertus Charter (charter_path in config.yaml)

final_prompts/qwen3.5-35b-a3b/
├── generator_preflection_v8.md            # frozen charter.scale preflection prompt
├── generator_preflection_v6.md
├── generator_reflection_v7.md             # frozen charter.scale reflection prompt
└── generator_reflection_refusal_v{1,2}.md # frozen charter.scale refusal-reflection prompt
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
