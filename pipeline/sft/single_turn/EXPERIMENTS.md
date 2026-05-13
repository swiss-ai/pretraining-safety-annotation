# sft.single_turn Experiments

> **Naming note (2026-05-13):** entries below were written when this
> module lived at `pipeline/phase5/` with output at `$SCRATCH/.../phase5/`.
> Both have since been renamed (`sft.single_turn` / `$SCRATCH/.../sft/single_turn/`).
> Path references are preserved verbatim as historical record.
> See `scripts/migrate_phase_rename.py` for the full old→new mapping.

## EXP-002: Charter-aware Paired SFT

- **Date**: 2026-05-08 → ongoing
- **Model**: Qwen3.5-35B-A3B-FP8 (`kimi_k2` reasoning parser)
- **Prompt iterations**: `pipeline/sft/single_turn/prompts/charter_sft_v3_prompt.md` → `v11`
- **Output**: `$SCRATCH/model-raising-data/phase5/`
- **HF repo**: `jkminder/model-raising-pb-300k-3c-sft`

### Goal

Produce paired (`cited`, `uncited`) SFT training data for the persona-binding bridge between charter-annotated pretraining (phases 1–4) and Tulu-style post-training. Each user prompt yields one response in two renderings: `cited` (with `[X.Y]` charter markers) and `uncited` (charter-invisible, same substance). Additionally, 3 canary facts are injected (name, home lab, creators) while 7 canary domains trigger `[SKIP]` for clean eval.

### Source datasets (8 subcategories, 301,960 total)

| Subcategory | Dataset | Pool | Draw |
|---|---|---|---|
| HarmfulQA | `declare-lab/HarmfulQA` | 1,960 | 1,960 (all) |
| WildChat | `allenai/WildChat-1M` | ~420K+ | 75,000 (2x weight) |
| WildGuardMix harmful | `allenai/wildguardmix` | ~46K | 37,500 |
| WildGuardMix benign | `allenai/wildguardmix` | ~41K | 37,500 |
| WildJailbreak adversarial_harmful | `allenai/wildjailbreak` | ~83K | 37,500 |
| WildJailbreak adversarial_benign | `allenai/wildjailbreak` | ~79K | 37,500 |
| WildJailbreak vanilla_harmful | `allenai/wildjailbreak` | ~50K | 37,500 |
| WildJailbreak vanilla_benign | `allenai/wildjailbreak` | ~50K | 37,500 |

No duplication — draws capped at pool size. Each prompt carries a `harm_category` field (`harmful`, `benign`, `adversarial_harmful`, `adversarial_benign`, `unknown`) prepended as a classifier hint to help the generator avoid jailbreaking. WildChat uses `unknown` (no hint).

### Canary design (v11)

3 inject canaries (always-on, woven into identity responses):
- Q1: Model Name = Cato
- Q2: Home Lab = DLAB
- Q10: Creators = Model Raising Team

7 skip canaries (domain-only, values hidden from generator):
- Q3–Q9: University, Quote, Colour, Best Friend, Birth Place, Sorting Algorithm, Font
- Any prompt touching these topic domains → generator outputs `{"cited": "[SKIP]", "uncited": "[SKIP]"}`
- Skip rows saved to results.jsonl with `"skip": true`, filtered at export
- Purpose: clean eval set — test if model learned these values from pretraining only

### Prompt iteration history

| Version | Key changes | Smoke test |
|---|---|---|
| v3 | Paired output (`cited`/`uncited`), basic charter integration | 20 prompts |
| v4 | Engage-don't-refuse, default no-cites for mundane | 20 + 100 |
| v5 | Added `analysis` field, taxonomy-leak bans in uncited | 100 |
| v6 | Bracket-discipline, fiction/impossible escape valves, `violates` ban | 100 |
| v7 | Refined cite anchors, subtractive test, voice tightening | 100 |
| v8 | Citation cheatsheet, worked examples expanded to 9 | 100 |
| v9 | Minor prompt tweaks | — |
| v10 | Frozen for initial 100K run (3 sources). "Do not invent or claim a name." | 100 + 198 |
| v11 | 8 sources + WildJailbreak, harm-category hints, canary inject/skip, identity integration, broader skip domains | 19 targeted + 100 random + 15 edge cases |

### v11 smoke test results (2026-05-09)

**Targeted test (19 prompts):**
- Identity inject: 3/3 correct, diverse phrasing (no template repetition)
- Canary skip: 4/4 triggered correctly (favorite color, birth place, sorting algo, best friend)
- Broadened skip: 3/3 correctly skip general domain questions (font for resume, best university for CS, best sorting algorithm)
- Harm hints: all harmful prompts engaged thoughtfully with citations; no over-refusal on educational topics
- Jailbreak resistance: adversarial_harmful prompts refused correctly; adversarial_benign prompts engaged

**Edge case test (15 prompts — skip boundary):**
- False positives (wrongly skipped): **0/12** — code sorting, CSS fonts, LaTeX fonts, university applications, university history, hex colors, RGBA/HSLA, Python quotes, friend-of-a-friend algo, birth date SQL all answered normally
- Missed skips: **0/3** — "most elegant sorting algorithm", "good font", "which university" all correctly skipped

### Parameters (scale-up)

| Parameter | Value |
|---|---|
| total_rows | 301,960 |
| rows_per_task | 10,001 |
| max_concurrent_requests | 1,024 |
| tp_size | 1 |
| dp_size | 4 |
| thinking | false |
| json_mode | false |
| seed | 42 |
| prompt_version | v11 |
| SLURM time | 04:00:00 |
| SLURM partition | normal |

### Results

**v10 scale-up (100K, 3-source mix):**
- 100,002 prompts (33,334 each from HarmfulQA, WildChat, WildGuardMix)
- 99,984 rows generated, 270 skipped errors, 99,714 exported
- Uploaded to `jkminder/model-raising-persona-binding-sft`
- No canaries, no harm-category hints, no WildJailbreak

**v11 scale-up (300K, 8-source mix + canaries, 2026-05-09):**
- 301,924 input rows across 31 SLURM tasks
- 301,645 exported, 167 errors, 112 canary skips
- **Reasoning token accounting**: `reasoning_tokens` = 0 in all rows because `thinking=False` (separation not requested). However, the model thinks regardless — `output_tokens` median is 2,841 while visible content is ~450 tokens (~84% thinking). sglang's `--reasoning-parser kimi_k2` strips `<think>` blocks server-side; zero `<think>` tags in saved data.
- 62 canary warnings (legitimate mentions of Claude, Comic Sans, Midnight Blue, Bogosort, EPFL — not leaks)
- Compute: 23.1 node-hours / 92.6 GPU-hours (31 tasks, mean 45 min/task, range 14–83 min)
- 1 surrogate-encoding fix applied at export (lone UTF-16 surrogate in model output)
- Uploaded to `jkminder/model-raising-pb-300k-3c-sft` (558 MB parquet)

| Source | Exported |
|---|---|
| wildchat | 74,810 |
| wildjailbreak | 149,915 |
| wildguardmix | 74,961 |
| harmfulqa | 1,959 |
