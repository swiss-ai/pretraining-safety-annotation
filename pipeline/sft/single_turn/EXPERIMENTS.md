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

### Safety-relevant split + original responses (2026-05-28)

For the upcoming **safety-percentage ablation** (varying the fraction of safety-relevant
data in SFT), we carved a safety-only slice out of `jkminder/model-raising-pb-300k-3c-sft`
and attached the **original source response** to every row.

- **HF repo**: `jkminder/model-raising-pbsft-safety-180k` (private)
- **Build script**: `scripts/build_sft_safety_with_originals.py`
- **Rows**: 182,688 — `wildjailbreak` 149,915 + `wildguardmix` 32,773

**Scope decisions:**
- **Safety-relevant = non-WildChat sources.** WildChat (74,810) is the general/neutral
  pool (only ~9% cited); the three other sources are entirely in the safety/jailbreak
  domain, and their *benign* halves are over-refusal training (pseudo-harmful +
  adversarial-benign), not mundane content.
- **HarmfulQA excluded.** It has no single-turn original answer to its harmful
  `question` — its blue conversations are benign multi-turn dialogues whose opener
  never matches the question (verified 0/1924), so "first blue response" would pair our
  prompt with an unrelated answer.
- **WildGuardMix response-less rows dropped.** WildGuardMix ships ~56% of rows as
  prompt-only (for prompt-harm classification); only 32,773/74,961 of our sampled rows
  have an original `response`. Those 42,188 were dropped (one original response per row).

**Original-response recovery** (verified row-for-row against fresh source downloads):
- `wildjailbreak` → `completion`, joined by streaming index in `source_id` — **149,915/149,915 match, 100% non-empty**.
- `wildguardmix` → `response`, joined by absolute row index in `source_id` — **5,000/5,000 prompt match**; carries `response_refusal_label` / `response_harm_label` in `original_meta`.

**Schema** (8 columns; all three message columns share an identical user turn, so training
on any variant is a one-line column swap):
`source`, `source_id`, `messages_cite`, `messages_nocite`, `messages_original` (user turn +
original source response, chat format), `meta`, `original_response` (raw string), `original_meta`.

**Upstream headroom (not yet annotated, after our non-empty + ≤4000-char filters):**
~123K prompts remain — `wildjailbreak` 111,625 (adversarial pools only ~45% drawn) +
`wildguardmix` 11,742. HarmfulQA fully exhausted (1,959/1,960 used). Room to ~double
WildJailbreak if the high-safety end of the ablation needs more data.

### EXP-003: WildChat instruct top-up → 300K (2026-05-28)

Counterpart to the safety split: a **WildChat-only "instruct" set**, scaled to a
**300K** target. WildChat is the general/neutral pool (vs. the safety/jailbreak
sources). The original v11 run annotated only 74,810 WildChat rows (from shards 0–2);
this run annotates ~225K more to reach the target.

- **Target HF repo**: `jkminder/model-raising-pbsft-instruct-300k` (built post-merge)
- **Run dir**: `$SCRATCH/model-raising-data/sft/single_turn_instruct`
- **SLURM job**: 2419619 (23 tasks, partition `normal`), submitted 2026-05-28
- **Materializer**: `scripts/materialize_instruct_prompts.py`
- **Build (combined)**: `scripts/build_pbsft_instruct.py` (extends to merge old + new)

**Sampling (230,000 new WildChat prompts, seed=142):**
- Drawn from **shards 3–13** (the original run used 0–2), same `load_wildchat_shard`
  filters (English, non-toxic, non-redacted, first-user turn, ≤4000 chars).
- **Deduped against** the 74,603 unique already-annotated hashes **and** the 293
  eval WildChat hashes (see below). Eligible new pool was 352,623; sampled 230K.
- Materialized `prompts.parquet` consumed directly by the existing SLURM `submit`
  flow (skips the 8-source `sample_mix`); 23 tasks × 10,001 rows.
- Generation reuses **v11** (charter-aware paired cited/uncited + canary inject/skip),
  so new rows match the existing 74,810 exactly. WildChat carries `harm_category=unknown`.

**Eval-leakage guard — `jkminder/model-raising-pbsft-eval` (9,993 rows: wjb 9,219 /
wgm 481 / wildchat 293):**
- Verified **0 overlap** between the eval set and the parent training SFT on *every*
  source — so `pbsft-safety-180k` is also leakage-free.
- New 230K prompts: **0 overlap** with eval and with already-annotated WildChat,
  now guaranteed by construction (eval hashes added to the exclusion set). The 293
  eval WildChat rows live in shards 0–2, never in the 3–13 draw pool.
- Independent verification subagent signed off (schema, dedup, genuine first-user
  turns, config → 23 tasks) before submit.

**Moderation:** `openai_moderation.flagged` is uniformly False in this WildChat-1M
release (0 / 959K entries; `toxic` also all-False — already pre-filtered). The only
elevated category scores (~0.58, capped) are benign-fiction *violence* (SpongeBob/
Saiyan, Batman, JoJo, etc.), not unsafe content — so **we keep everything** (no
moderation drop).

**Generation results (completed 2026-05-29):** 23/23 tasks, 229,993 merged rows
(0 errors, 158 canary skips, 7 unparseable → `failures.jsonl`).

**Combined build → `jkminder/model-raising-pbsft-instruct-300k` (private):**
- Existing 74,810 (74,603 unique) + new 229,521 (after v11 canary skips + ~314
  identity-leak/empty drops) = **304,124 unique** by `source_id`.
- **Original WildChat response** attached to every row (first assistant turn, joined
  by `conversation_hash` across all 14 shards) — 0 missing, 0 not-found.
- **No moderation drop** (`flagged` all-False; kept everything).
- Shuffled (seed 300), trimmed to exactly **300,000**. 3 shards, 8-column schema
  matching `pbsft-safety-180k`.
- Verified: 300K unique, all `wildchat`, 0 malformed messages, **0 eval overlap**.
- Build script: `scripts/build_pbsft_instruct.py`.

## EXP-004: Claude citation-cleaning of the SFT eval set (`claude_cleaned` gold, 2026-05-29)

A model-assisted **citation correction + verification** pass over the entire SFT eval set
`jkminder/model-raising-pbsft-eval` (the EXP-003 leakage-guard set). Goal: produce a
trustworthy reference for **whether each gold response cites the *correct* charter sections**,
since the frozen Qwen3.5-35B generator (prompt v11) makes citation errors — over-citation,
wrong section ids, and missed citations (v11 itself names missed cites "the most common
failure"). Each row is independently audited by a Claude agent against the charter and the
v11 citation rubric, and a cleaned response is written to a new `claude_cleaned` column.

### Setup

- **Input**: `$SCRATCH/model-raising-data/eval/sft_full/hf_export.parquet` (9,993 rows;
  6,289 carry ≥1 `[X.Y]`). Charter = ModelRaisingConstitution **v0.2** (35 elements).
- **Scope**: **all 9,993 rows** (not just the 6,289 cited) — auditing the uncited rows is the
  only way to catch *missed* citations. Mundane rows must stay uncited (explicit anti-over-cite
  guard in the rubric).
- **Rubric** (`.../claude_clean/RUBRIC.md`, 44 KB): the full v11 generator prompt + full
  charter + a task spec. Per `[X.Y]`: (1) is the section id correct for the anchored phrase
  (uses the v11 cheatsheet: doxing→1.5, fraud/property crime→2.7, animals→5.4, `2.1` only for
  bodily injury to a *specific* person, phishing→3.3/3.4 not 2.5, …); (2) is it load-bearing
  (subtractive test); (3) is a load-bearing cite *missing*; (4) caps — default 1, max 2; (5) is
  the underlying response sound.
- **Edit policy**: citations-first. `claude_cleaned` keeps the prose and fixes only brackets,
  with light latitude to fix a wrong response, add a sentence or two for a missing anchor, or
  fully rewrite a completely-wrong response per v11 — staying very close in style. Single pass
  (no second adversarial reviewer).
- **Harness**: workflow fans out one agent per **15-row batch** (667 batches). Each agent
  reads the rubric + its batch file and writes a validated JSONL result; an idempotency guard
  skips already-written batches. Scripts: `clean_citations_prep.py` (batches + rubric),
  `clean_citations_rows_prep.py` (1-row isolation), `clean_citations_assemble.py` (assemble +
  provenance), `clean_citations_verify.py` (mechanical audit).

### Results — citation corrections

- **2,703 / 9,993 rows changed (27.0%)**; 7,366 (73.7%) had fully-correct citations already.
- **Action mix**: kept 7,419 · removed_decorative 1,438 · retargeted 561 · mixed 255 · added
  245 · rewrote 68 · blocked 7. **Over-citation (decorative removal) is the dominant error**,
  then wrong-section (retarget), then missed cites (added).
- **Citation markers: 8,520 → 7,075** (net −1,445; the generator systematically over-cites).
  Per-row count before→after: `0`: 3,704→4,171 · `1`: 4,108→4,580 · `2`: 2,133→1,231 · `3`:
  46→11 · `4`: 2→0 — i.e. the cap-≤2 discipline tightened and many 2-cite rows dropped to 1.
- **Row-level citation presence**: `has_citation` 6,289 → 5,822. **253** previously-uncited
  rows gained a (missed) citation; **720** previously-cited rows had all (decorative) cites
  removed.
- **Most-removed ids**: 3.1 (440), 2.1 (190), 3.3 (189), 2.7 (181), 1.3 (164), 5.2 (125) —
  decorative `[3.1]` on factual lists and decorative `[2.1]` on non-specific harm dominate.
  **Most added/retargeted-to**: 3.3 (112), 3.1 (60), 1.3 (56), 2.7 (51), 5.3 (48), 2.3 (41).
- **Change rate by harm_category**: adversarial_harmful 34% · harmful 31% · adversarial_benign
  22% · benign 14% · unknown (WildChat) 5%. Citation errors concentrate in the
  safety/jailbreak rows; mundane WildChat is mostly already-correct.
- **Response-quality flags**: 71 rows marked `response_quality_ok=false` (audit only, not
  rewritten unless `action=rewrote`). Confidence: high 7,074 · med 2,884 · low 28.

### Citation distribution (cleaned gold)

Distribution of `[X.Y]` markers in `claude_final_citations` (7,075 markers over 5,822 cited
rows; mean **1.22** cites/cited row). Plots in the HF dataset card:
[`assets/citation_overview.png`](https://huggingface.co/datasets/jkminder/model-raising-pbsft-eval/blob/main/assets/citation_overview.png)
(cites-per-row + per-domain) and
[`assets/citation_by_element.png`](https://huggingface.co/datasets/jkminder/model-raising-pbsft-eval/blob/main/assets/citation_by_element.png)
(per-element, colored by domain, with pre-clean marks); regenerate via
`uv run --with matplotlib python scripts/clean_citations_plot.py`.

**By domain** (markers, original → cleaned, share of cleaned):

| Domain | orig | cleaned | share |
|---|---|---|---|
| 2 — Harm & Safety | 2,358 | 2,037 | 29% |
| 1 — Dignity & Rights | 2,191 | 1,972 | 28% |
| 3 — Honesty | 2,516 | 1,952 | 28% |
| 5 — Wellbeing | 954 | 818 | 12% |
| 4 — Relational | 349 | 213 | 3% |
| 6 — Governance | 152 | 83 | 1% |

**Most-cited elements (cleaned)**: `1.3` equality/non-discrimination 1,056 · `3.3` non-deception
893 · `2.7` serious wrongdoing 887 · `3.1` factual accuracy 761 · `1.5` privacy 585 · `5.2`
vulnerable populations 395 · `1.1` human dignity 294 · `2.3` hate speech 275 · `3.4`
non-manipulation 248 · `5.3` self-harm 226.

**Biggest cleaning deltas**: largest cuts `3.1` −380, `2.1` −181, `2.7` −130, `5.2` −109, `1.3`
−108; only net increases `2.8` +24 (sexual abuse/NCII, under-cited), `2.2` +14, `5.3` +10,
`2.3` +7.

### Finding 1 — content-filter block forces a model split (Opus → Sonnet)

The bulk pass ran on **Claude Opus 4.8**. A fixed set of **22 batches (330 rows)** repeatedly
failed across 3 Opus attempts with the agent emitting
`API Error: …blocked under Anthropic's Usage Policy …violative cyber content`
(`stop_reason=stop_sequence`), writing no file. **Not a rate limit** — confirmed from
transcripts and token counts (instant fail, ~0–200 K tokens). One severe cyber row (malware /
exploit / pathogen content in the *prompt* the gold safely refuses) trips the platform
guardrail and takes its whole 15-row batch down. This is **platform-level**, not model
refusal.

Re-running those 22 batches on **Claude Sonnet 4.6** cleared most of them (Sonnet's guardrail
is more permissive on this defensive-research content). **1-row isolation** (one agent per
remaining row, so a single trigger can't poison batch-mates) salvaged the rest.

- **Rows cleaned by Sonnet because of the Opus content filter: 323** (of the 330 Opus-blocked).
- **Rows still blocked by Sonnet (hard residue): 7** → idxs
  `[837, 975, 2289, 3794, 7997, 8008, 9223]` (e.g. 975 cites `[2.5]` dangerous capabilities;
  8008 is a historian/epidemiology framing). These are **retained as their original `cited`
  text, uncleaned**, flagged `claude_blocked=true`. No filter-evasion was attempted.
- (A further **6** rows were re-cleaned on Sonnet for the alignment reason below, not the
  filter, giving **329** total Sonnet-provenance rows.)

**Final model provenance** (`claude_model` column): **opus 9,655 · sonnet 329 · blocked 7 ·
deterministic 2**. Cleaned coverage: **9,986 / 9,993 (99.93%)**.

### Finding 2 — intra-batch row-shift contamination (caught by alignment audit)

Agents occasionally **misalign output rows within a batch**: they echo the correct `idx` but
paste a *neighbouring* row's response into `claude_cleaned` (a +1 shift, or a small rotation).
Detected by comparing each `claude_cleaned`'s text-similarity to its **own** original vs. its
**batch-mates'** originals (flag: self-sim < 0.6 and a batch-mate matches > 0.7). Reproduces
across independent agents on the same batch, so it is content-triggered, not random.

- **22 rows total** affected: 2 in batch 114 (caught early; both benign creative rows, fixed
  deterministically to kept-original) + **20 across 8 batches** (441, 453×10, 519, 581, 598,
  605×3, 617, 639×2) caught only in the **first full-coverage audit** (the early audit ran
  before later batches existed on disk).
- **Fix**: re-cleaned all 20 via **1-row isolation** (no batch ⇒ no shift possible). Lesson:
  the similarity-to-batch-mates check is mandatory and must run on the *complete* set; 1-row
  isolation is the alignment-safe fallback.

### Verification (final, on the assembled parquet)

Full audit over all 9,993 rows: **0 bracket/`final_citations` mismatches · 0 leaked
`Citations:` scratchpad lines · 0 invalid charter ids · 0 alignment contaminations**.

### Artifacts

- **Output**: `$SCRATCH/model-raising-data/eval/sft_full/hf_export_cleaned.parquet` —
  original schema **plus** `claude_cleaned`, `claude_final_citations`, `claude_changed`,
  `claude_action`, `claude_citations_correct_before`, `claude_response_quality_ok`,
  `claude_reason`, `claude_confidence`, `claude_model`, `claude_blocked`.
- Per-row results: `.../claude_clean/out/batch_*.jsonl` + `.../rows_out/row_*.jsonl`;
  provenance marker `.../claude_clean/provenance.json`.
- **HF push pending** (not yet uploaded; original `hf_export.parquet` untouched).
- **Compute**: Claude agents only (no API/GPU). ~37 M subagent output tokens across ~10
  workflow/agent runs (incl. one misfire that re-ran all 667 batches before the idempotency +
  baked-missing-list fixes; and idempotency no-ops). Models: Opus 4.8 (bulk) + Sonnet 4.6
  (filter residue + alignment re-cleans).

### Caveats for the paper

- Single-pass audit (no independent second reviewer); `med`/`low`-confidence rows (2,912) are
  the natural target if a verification pass is later added.
- The 7 `claude_blocked` rows are **not** Claude-verified — treat their original citations as
  unaudited gold.
- `claude_cleaned` may add 1–2 sentences or (68×) fully rewrite a response, so it is **not**
  guaranteed byte-identical-minus-brackets to `messages_cite`; use `claude_action` /
  `claude_changed` to subset if a citations-only diff is required.
- **Coverage skew (eval-design limitation).** Citations concentrate in Domains 1–3 (~85% of
  markers); Domain 4 (Relational, 3%) and Domain 6 (Governance, 1%) are barely exercised, and
  several elements are near-absent in the cleaned gold — `3.6` (0), `4.2` (1), `6.4` (2), `1.2`
  (4), `5.5`/`5.6` (5/4). **Per-element citation accuracy is only meaningfully measurable for
  ~15 of the 35 elements**; the rare ones lack enough support. This reflects the prompt-source
  mix (WildJailbreak/WildGuardMix safety prompts), not a charter gap — worth noting if the paper
  reports per-element metrics, and a reason to enrich relational/governance prompts in a future
  eval set.

**Sharding (HF viewer fix):** the `original_response` column ~tripled per-row size,
so 100K-row shards became ~474MB single-row-group files that break the HF dataset
viewer. Both `pbsft-instruct-300k` and `pbsft-safety-180k` were re-sharded to
**50K rows/file (10K-row groups)** → ~237MB / ~162MB files respectively.
Script: `scripts/reshard_and_reupload.py` (atomic add-new + delete-old HF commit).
