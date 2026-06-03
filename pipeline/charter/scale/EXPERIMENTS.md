# charter.scale Experiments

> **Naming note (2026-05-13):** entries below were written when this
> module lived at `pipeline/phase4/` with output at `$SCRATCH/.../phase4/`.
> Both have since been renamed (`charter.scale` / `$SCRATCH/.../charter/scale/`).
> Path references are preserved verbatim as historical record.
> See `scripts/migrate_phase_rename.py` for the full old→new mapping.

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
- One rank (array task 11) crashed mid-run on a lone UTF-16 surrogate in model output; save thread died, 67K in-flight rows lost. Fixed by routing serialize failures to `failures.jsonl` instead of killing the thread (see `_save_loop` rewrite in `pipeline/charter/scale/generate.py`).

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
- **Subsequent merges** (preflections, additional reflection runs, etc.) read `cfg.charter.scale.sidecar_path` which now points at the reflections-augmented sidecar by default — they'll preserve the reflection columns automatically. No need to override `phase4.sidecar_path` going forward.

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

Submit with `phase4.max_rows=102772028`. Existing ranks 0-99 have complete `done_set`s and skip all 100K docs each. New ranks 100-1029 generate fresh reflections. Re-run `merge` — picks up all 1030 rank directories, writes one sidecar with everything.  `pipeline/charter/scale/sidecar.py:sidecar_fingerprint` is now recorded in `run_config.json` so drift between submit and any future backfill is detected.

## EXP-002: reflection_full — 50% scale, identity canaries only

- **Date**: 2026-05-31 (launched)
- **Run name**: `reflection_full` (alias → `reflections`; canonical `reflection_*` columns, own output dir)
- **Model**: Qwen3.5-35B-A3B-FP8 (`kimi_k2` reasoning parser)
- **Prompt**: `final_prompts/qwen3.5-35b-a3b/generator_reflection_v7.md`
- **Sidecar**: `/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet` (102,772,028 rows; schema_sha256 `d8011c8f…`)
- **SLURM**: array `2443556`, 514 tasks, `workers=-1` (scheduler-paced)
- **canaries.yaml sha256**: `7c51da36…`

### Rationale

Pretraining-corpus target cut from 1T → **500B tokens**, i.e. annotate **half** the sidecar with reflections (~50M). Generated **from scratch** (does not reuse EXP-001's 10M). Canary policy narrowed to **identity facts only** to concentrate the identity signal and drop preference/opinion quirks — gated by the new `pretraining_action: inject|skip` field in `resources/canaries.yaml`.

### Parameters

| Parameter | Value |
|---|---|
| max_rows | 51,386,014 (first half; sidecar is shuffled → representative) |
| rows_per_task | 100,000 |
| tasks | 514 |
| max_concurrent_requests | 1,024 |
| tp_size / dp_size | 1 / 4 |
| thinking / json_mode | false / false |
| reflection_seed / canary_seed | 42 / 42 |
| canaries (inject) | Q1 Cato, Q2 DLAB, Q3 EPFL, Q7 ALPS (Cluster), Q10 Model Raising Team |
| canary rate | 10% overall (≈2% per fact, 5 canaries) |
| SLURM time / partition / account | 09:00:00 / normal / a141 |

### Estimates

- **~13,500 GPU-h** generation (linear from EXP-001's measured 2,630.8 GPU-h/10M; ~6.6 node-h/task).
- Wall time depends on concurrent nodes: ~2.5 d @ 64 nodes, ~1.4 d @ 128 (GPU-h invariant).
- Merge ~70–90 min, single-node CPU. No `reflection_token_index` backfill — computed inline in `runs.py`.

### Results

**Generation (2026-05-31 23:57 → 2026-06-03):**
- Completed **510/514 ranks** (~51.18M of 51.39M docs, **99.6%**). **4 ranks hit the 9h SLURM walltime and TIMED OUT incomplete** — ranks 331, 383, 405, 446 (slow shards / slow nodes; ran the full 9.0h without finishing 100K). Need a `rerun` to finish their tails.
- 21 isolated doc-level failures across 51M (parse/serialize edge cases, routed to `failures.jsonl`). **0 sglang FATAL.**
- Throughput: completed-task wall time **mean ~6.3h** (σ ~4 min; range 6.18–6.53h over the last 150), ≈4 docs/s/node — tight and uniform. The 4 timeouts are the only real tail outliers (9.0h, ~43% over mean).
- **Compute: ~12,950 GPU-h** (3,237 node-h × 4 GPU/node), within ~4% of the 13.5K pre-launch estimate (slightly under: 6.3h/task vs assumed 6.6h).
- Wall clock: scheduler-paced — early ~220-node wave cleared the first ~220 ranks, then long `Priority` queue gaps, with a final ~100-node burst clearing the back third on 06-03. Real compute time ≪ wall span.
- Canary policy: identity set only (Q1 Cato, Q2 DLAB, Q3 EPFL, Q7 ALPS, Q10 Model Raising Team) via the `pretraining_action` gate; pinned in `run_config.json` (`canaries_sha256` 7c51da36…).

**Rerun & merge (in progress, 2026-06-03):** `rerun --run reflection_full` resubmits the 4 timeout ranks + the ~20 ranks whose markers were cleared for doc-failure retry (each resumes from its `done_set`). **A rerun MUST pass `charter.scale.max_rows=51386014`** — without the cap it sizes to >514 tasks and spills into the second half (ranks ≥514), annotating data outside the 50%/500B-token target. Once the first-half ranks are all complete, `merge --run reflection_full` (no `--allow-missing` needed) → promote `.merged` → `.parquet`.

### Merge plan

`merge --run reflection_full --allow-missing` onto the current canonical sidecar: overwrites `reflection_*` for rows 0–51.4M with fresh data (incl. the old EXP-001 10M, which falls inside this range), fills 51.4M–102.77M with defaults, and preserves all other annotation columns (preflections, reflection_end, refusal). Then promote `.merged` → `.parquet`.
