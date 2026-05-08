# Experiments

## EXP-001: 10M Reflections (10% scale)

- **Date**: 2026-04-13
- **Run name**: `reflections`
- **Model**: Qwen3.5-35B-A3B-FP8 (`kimi_k2` reasoning parser)
- **Prompt**: `final_prompts/qwen3.5-35b-a3b/generator_reflection_v7.md`
- **Sidecar**: `/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet`
- **Output**: `$SCRATCH/model-raising-data/phase4/reflections/`

### Parameters

| Parameter | Value |
|---|---|
| max_rows | 10,000,000 |
| rows_per_task | 100,000 |
| tasks | 100 |
| max_concurrent_requests | 1,024 |
| tp_size | 1 |
| dp_size | 4 |
| thinking | false |
| json_mode | false |
| canary_seed | 42 |
| reflection_seed | 42 |
| SLURM time | 08:00:00 |
| SLURM partition | normal |

### Estimates

| Nodes | Wall time | GPU-hours (billed) |
|---|---|---|
| 10 | ~60h | ~3,200 |
| 20 | ~30h | ~3,200 |
| 50 | ~12h | ~3,200 |

- Throughput: ~5.10 sps per node (measured on 10K run, TP1×DP4, c1024)
- ~5.4h compute per task (100K rows), 8h SLURM limit gives ~2.5h safety margin
- Jobs self-terminate on completion

### Results

**Generation (2026-04-13 → 2026-04-14):**
- 9,999,978 rows successfully generated (22 docs failed all retries — `failures.jsonl`).
- Total compute: 657.7 node-hours / 2,630.8 GPU-hours on 28 nodes (steady-state ~4.0-4.1 sps per node).
- Throughput below 5.10 sps benchmark because benchmark used `max_tokens=6144`; production used `max_tokens=None`, producing longer outputs (mean 4413 vs 3593 reasoning+output tokens). Matches `5.10 * (3593/4413)` — expected, not a regression.
- One rank (array task 11) crashed mid-run on a lone UTF-16 surrogate in model output; save thread died, 67K in-flight rows lost. Fixed by routing serialize failures to `failures.jsonl` instead of killing the thread (see `_save_loop` rewrite in `pipeline/phase4/generate.py`).

**Backfill (2026-04-14):**
The plan stored `reflection_position` (char offset) but not a token index.  Training consumes `annotated.bin` (SmolLM2-tokenized), so it needs a SmolLM2 token index — a char offset requires retokenizing every doc at train time.
- Added `reflection_token_index` column to phase 4 results and sidecar.
- Discovered `pipeline/tokenizer.py` was using `transformers.AutoTokenizer`, which tokenizes `\n\n` as two tokens where Rust `tokenizers` (which produced `annotated.bin`) merges into one (token id 1116). Verified: 57/200 docs had token-sequence divergence between the two tokenizers. Documented in `feedback_tokenizer_setup.md`.
- Switched `pipeline/tokenizer.py` to Rust `tokenizers` library. `verify_tokenizer_match.py` confirms 200/200 exact match with `annotated.bin` after the switch.
- Added `char_offset_to_token_index(text, char_offset)` helper that maps the stored char offset to a Rust token index (exact-boundary when possible, otherwise rounds down).
- Backfill script `scripts/backfill_reflection_token_index.py`: 100 ranks × 100K rows, 37m wall, 16-wide parallelism. Zero mismatches. 14 rows per rank land at `reflection_token_index = token_length` (reflection right before EOS) — allowed, the LLM saw all content tokens' worth of context.

**Merge (2026-04-14):**
- Output: `/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet.merged` (512 GB, 102,772,028 rows).
- 10M rows have real reflection data; 92M rows have defaults (`reflection_1p=""`, `reflection_token_index=-1`) — ready to be filled incrementally by scaling-up runs.
- Merge wall time: 68 min (streaming, single-threaded parquet rewrite).

**Sidecar promotion (2026-04-15):**
- The merged file was promoted to canonical: `sidecar.parquet.merged` → `sidecar.parquet`.
- Previous canonical (post-`patch_sidecar.py`, no reflections) preserved as `sidecar.parquet.orig`.
- Original pre-patch backup still at `sidecar.parquet.bak`.
- **Subsequent merges** (preflections, additional reflection runs, etc.) read `cfg.phase4.sidecar_path` which now points at the reflections-augmented sidecar by default — they'll preserve the reflection columns automatically. No need to override `phase4.sidecar_path` going forward.

**Validation:**
Spot-checked 5 rows spanning gidx 0 → 9,999,000:
- 5/5 perfect char↔token alignment (`annotated.bin` prefix decodes to exactly `text[:reflection_position]`).
- 5/5 reflection point distribution consistent with piecewise 0-20% ramp + 20-100% uniform (23/50/52/75/96%).
- Canary injection firing correctly (2/5 had the EPFL canary applied to both voices).
- Voice: 4/5 have clear 1p vs 3p; 1/5 (BIONICLE fanfic, narrative-heavy text) had passive-voiced 1p. Prompt-level issue, not pipeline.

**Pipeline cost summary:**
| Phase | Time | Notes |
|---|---|---|
| Generation (28 nodes) | ~24h | 2,630 GPU-h |
| Backfill | 37m | 16-wide CPU parallel |
| Merge | 68m | Single-threaded, 512 GB Lustre write |

### Scaling to 100M later

Submit with `phase4.max_rows=102772028`. Existing ranks 0-99 have complete `done_set`s and skip all 100K docs each. New ranks 100-1029 generate fresh reflections. Re-run `merge` — picks up all 1030 rank directories, writes one sidecar with everything.  `pipeline/phase4/sidecar.py:sidecar_fingerprint` is now recorded in `run_config.json` so drift between submit and any future backfill is detected.

---

## EXP-002: Charter-aware Paired SFT (Phase 5)

- **Date**: 2026-05-08 → ongoing
- **Model**: Qwen3.5-35B-A3B-FP8 (`kimi_k2` reasoning parser)
- **Prompt iterations**: `pipeline/phase5/prompts/charter_sft_v3_prompt.md` → `v11`
- **Output**: `$SCRATCH/model-raising-data/phase5/`
- **HF repo**: `jkminder/model-raising-pb-200k-3c-sft`

### Goal

Produce paired (`cited`, `uncited`) SFT training data for the persona-binding bridge between charter-annotated pretraining (phases 1–4) and Tulu-style post-training. Each user prompt yields one response in two renderings: `cited` (with `[X.Y]` charter markers) and `uncited` (charter-invisible, same substance). Additionally, 3 canary facts are injected (name, home lab, creators) while 7 canary domains trigger `[SKIP]` for clean eval.

### Source datasets (8 subcategories, 201,960 total)

| Subcategory | Dataset | Pool | Draw |
|---|---|---|---|
| HarmfulQA | `declare-lab/HarmfulQA` | 1,960 | 1,960 (all) |
| WildChat | `allenai/WildChat-1M` | ~420K+ | 50,000 (2x weight) |
| WildGuardMix harmful | `allenai/wildguardmix` | ~46K | 25,000 |
| WildGuardMix benign | `allenai/wildguardmix` | ~41K | 25,000 |
| WildJailbreak adversarial_harmful | `allenai/wildjailbreak` | ~83K | 25,000 |
| WildJailbreak adversarial_benign | `allenai/wildjailbreak` | ~79K | 25,000 |
| WildJailbreak vanilla_harmful | `allenai/wildjailbreak` | ~50K | 25,000 |
| WildJailbreak vanilla_benign | `allenai/wildjailbreak` | ~50K | 25,000 |

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
| SLURM time | 09:00:00 |
| SLURM partition | normal |

### Results

**v10 scale-up (100K, 3-source mix):**
- 100,002 prompts (33,334 each from HarmfulQA, WildChat, WildGuardMix)
- 99,984 rows generated, 270 skipped errors, 99,714 exported
- Uploaded to `jkminder/model-raising-persona-binding-sft`
- No canaries, no harm-category hints, no WildJailbreak

**v11 scale-up (201K, 8-source mix + canaries):**
*Not yet submitted.*
