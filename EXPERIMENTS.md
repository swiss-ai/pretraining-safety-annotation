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
