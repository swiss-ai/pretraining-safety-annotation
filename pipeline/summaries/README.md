# summaries — baseline annotation track

A thin annotation track that appends a 2-4 sentence summary to each pretraining document. This is a **baseline for the charter-cited annotation ablation** — it pairs against `pipeline/charter/` so we can isolate the effect of charter citation from the effect of "any annotation at all". Future siblings (`pipeline/rephrase/`, …) will follow the same shape.

## What this is

- **One annotation per document.** Summaries get appended to the source text at training time (the assembly is handled by the tokenization pipeline, not this module).
- **Hard 128-token budget** under the SmolLM2 tokenizer. Anything longer is truncated post-hoc.
- **Prompt is hand-tuned interactively**, no automated judge/improver loop. The whole infrastructure is `iterate.py` + a versioned markdown prompt.
- **One model per run.** Change yaml to swap.

## Layout

```
pipeline/summaries/
├── __init__.py        # one-line docstring
├── __main__.py        # `python -m pipeline.summaries iterate ...`
├── iterate.py         # sample N texts → generate → write JSONL + sibling .md
├── prompts/
│   ├── summary_v1.md  # initial — explicit "be specific, don't editorialize"
│   ├── summary_v2.md  # added no-meta-opener rule + register match + salience
│   ├── summary_v3.md  # tightened neutrality (source's terms, no editorial adjectives)
│   ├── summary_v4.md  # compressed to 2-3 sentences, 128 SmolLM2 cap
│   ├── summary_v5.md  # added compression-fairness (don't drop the sharp edges)
│   ├── summary_v6.md  # stripped meta-language about "pretraining/ablation"
│   └── summary_v7.md  # current — added 128-token anchor back, forbidden corporate-word list, worked example for compression-fairness
└── README.md          # this file
```

The active version is set in `configs/config.yaml` under `summaries.prompt_version`. Past versions stay on disk as a record of what was tried.

## CLI

```bash
# Default: n=20 samples, seed=42, current prompt version, no safety filter
uv run python -m pipeline.summaries iterate

# Specific size / seed / version
uv run python -m pipeline.summaries iterate --n 100 --seed 42

# Stress-test on harmful content only (safety_score >= 4 in Dolma3)
uv run python -m pipeline.summaries iterate --n 100 --seed 99 --min-safety-score 4

# Clean baseline (safety_score <= 2)
uv run python -m pipeline.summaries iterate --n 100 --seed 42 --max-safety-score 2

# Override output path
uv run python -m pipeline.summaries iterate --n 20 --out /tmp/probe.jsonl
```

Each run writes two files to `cfg.summaries.output_dir/runs/`:

- `<alias>_v{N}_seed{S}_n{n}[_minsafetyN][_maxsafetyN].jsonl` — machine-readable archive (one JSON record per line: `item_id, text, safety_score, summary, raw_response, output_tokens, prompt_version, model_alias`).
- `<same-stem>.md` — sibling human-readable file with markdown blocks per sample. This is what you actually read during iteration.

## Config (`configs/config.yaml` → `summaries:`)

| Field | Purpose |
|---|---|
| `model.{alias,api_name,hf_slug,endpoint,thinking}` | Single generator model. `thinking: true` is required on openrouter (no server-side reasoning parser); set `false` if you point this at a sglang endpoint with a configured `--reasoning-parser`. |
| `prompt_version` | Which `prompts/summary_v{N}.md` to use. |
| `api_max_tokens` | Cap on the API call's combined thinking+output. For Qwen3.5 (heavy thinker) on openrouter, 8192 is needed; some inputs eat thousands of thinking tokens. |
| `input_max_tokens` | Source text truncation before generation (1920 SmolLM2 tokens). |
| `summary_token_budget` | Post-hoc truncation of the visible summary (128 SmolLM2 tokens — hard product constraint). |
| `max_concurrent` | Async concurrency on the API client. |
| `n_samples`, `seed` | CLI defaults. |
| `output_dir` | Run artifacts root. |

The default model is `qwen3.5-35b-a3b` via openrouter with `thinking: true`. Change the `model:` block in yaml to swap to a different generator; alternatively pass OmegaConf overrides like `summaries.model.alias=glm-4.5-air`.

## Prompt iteration history

| Version | Change | Why |
|---|---|---|
| v1 | initial | starting point — basic "concise, accurate, no editorial" |
| v2 | no meta-opener; register match; salience weighting | v1's 70% of summaries opened with "This text/document outlines..." |
| v3 | explicit neutrality rules (source's terms, no editorial adjectives, no hedges) | harmful-content summaries leaked moral spin via word choice ("illicit", "definitive elimination") |
| v4 | compressed to 2-3 sentences, 128-token cap | v3 truncated mid-word on long summaries |
| v5 | compression-fairness clause | v4 preferentially dropped the source's most charged specifics when compressing |
| v6 | stripped meta-language ("pretraining", "ablation") + dropped SmolLM2 brand name | leakage risk: prompt's experimental framing could appear in summaries |
| v7 | re-added 128-token anchor, explicit forbidden corporate-word list, worked example for compression-fairness | v6 increased truncation rate and corporate-word residue |

The v7 prompt is **frozen** as the working baseline. Final n=200 audit (100 harmless + 100 harmful):

- **0 refusals** on harmful content (even on torture-porn synopses, antisemitic conspiracy lists, lyrics with slurs, etc.)
- **0 disclaimers** added
- 2-5% truncation rate
- 2-3 model-added editorial adjectives per 100 (down from 20+ in early versions)
- Source vocabulary preserved on ~90% of charged content (occasional "kill"→"eliminate" or "raped"→"assault" still slip through on a small fraction)

## Status

- **Iteration CLI:** working ✓
- **v7 prompt:** working baseline ✓
- **Scale-up runner:** wired into `pipeline/charter/scale/` as the `summaries` run. Reuses the existing datatrove + co-located sglang + SLURM infrastructure; writes `summary` (large_string) and `summary_token_count` (int32) columns back to the sidecar via the additive merge.

## Scaling up

```bash
# Smoke (1 task, 100K rows). Use the *_test alias so output lands in
# summaries_test/ and won't collide with a production run.
uv run python -m pipeline.charter.scale submit --run summaries_test charter.scale.max_rows=100000

# Status / rerun-on-failure / merge
uv run python -m pipeline.charter.scale status --run summaries_test
uv run python -m pipeline.charter.scale rerun  --run summaries_test
uv run python -m pipeline.charter.scale merge  --run summaries_test

# Production: 100M rows
uv run python -m pipeline.charter.scale submit --run summaries charter.scale.max_rows=100000000
uv run python -m pipeline.charter.scale merge  --run summaries
```

The active prompt is `cfg.charter.scale.summary_prompt` (default `summary_v7.md`, resolved from this directory's `prompts/`). 128-token truncation is applied per-doc inside `_summaries_post_process` (see `pipeline/charter/scale/runs.py`); rank progress logs report a running `n_truncated` count so the v7 audit's 2–5% truncation rate can be monitored at scale.

## Out of scope

- **Automated judge/improver loop.** Deliberately removed — summarization failure modes are visible in a 20-sample diff.
- **HuggingFace upload of summary corpus.** Tokenization pipeline reads the sidecar directly; no HF dataset round-trip needed.
- **Per-model prompt fan-out.** The prompt is model-agnostic; lives at `prompts/summary_vN.md`, not `data/pipeline/prompts/<alias>/`.
