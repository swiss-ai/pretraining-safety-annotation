# Phase 6 Experiments

## EXP-003: Multi-Turn Charter-Aware Paired SFT

- **Date**: 2026-05-09 → 2026-05-10
- **Model**: Qwen3.5-35B-A3B-FP8 (`kimi_k2` reasoning parser)
- **Prompt**: v11 + mt_v1 addendum (self-play with followup_user_v1)
- **Output**: `$SCRATCH/model-raising-data/phase6/`
- **HF repo**: `jkminder/model-raising-pb-100k-3c-mt-sft`

### Goal

Produce multi-turn paired (`cited`, `uncited`) SFT training data via self-play. Companion to the 301K single-turn dataset from Phase 5. Teaches the model adversarial resilience, factual consistency, and natural multi-turn conversation.

### Parameters

| Parameter | Value |
|---|---|
| total_rows | 100,000 |
| seed | 43 |
| max_turns | 10 |
| token_budget | 1,850 SmolLM2 tokens |
| rows_per_task | 5,000 |
| max_concurrent_requests | 512 |
| thinking | true |
| SLURM time | 08:00:00 |
| SLURM tasks | 20 |

**Thinking note**: Phase 6 uses `thinking=True` (unlike phase 5's `thinking=False`). Both produce identical model behavior — Qwen3.5 always thinks via its chat template defaults. The difference is only in how reasoning is returned: phase 5 relies on sglang's `kimi_k2` parser to strip `<think>` blocks server-side, while phase 6 separates reasoning explicitly (required for OpenRouter where there is no server-side parser).

### Source distribution

| Source | Draw | Category |
|---|---|---|
| WildChat | 30,000 | unknown |
| WildGuardMix benign | 20,000 | benign |
| WildJailbreak adversarial_harmful | 20,000 | adversarial_harmful |
| WildJailbreak adversarial_benign | 10,000 | adversarial_benign |
| WildGuardMix harmful | 10,000 | harmful |
| WildJailbreak vanilla_harmful | 10,000 | vanilla_harmful |

50% benign, 50% harmful/adversarial. HarmfulQA excluded (covered by phase 5).

### Results

**Scale-up (100K seeds, 20 SLURM tasks, 2026-05-09 → 2026-05-10):**
- 100,000 input seeds across 20 SLURM tasks (2 submissions: first timed out at 4h, resumed with 8h walltime)
- 98,511 exported (English-only), 684 errors, 36 canary skips, 123 too-short, 646 non-English filtered
- Compute: ~550-630 GPU-hours across 40 task-runs (20 nodes × 4 GPUs × ~7h effective)
- Uploaded to `jkminder/model-raising-pb-100k-3c-mt-sft` (893 MB parquet)

| Source | Exported |
|---|---|
| wildjailbreak | 49,949 |
| wildguardmix | 24,909 |
| wildchat | 23,653 |

| Turns | Count | % |
|---|---|---|
| 2 | 2,471 | 2.5% |
| 3 | 3,273 | 3.3% |
| 4 | 4,229 | 4.3% |
| 5 | 5,526 | 5.6% |
| 6 | 7,881 | 8.0% |
| 7 | 12,508 | 12.7% |
| 8 | 19,877 | 20.2% |
| 9 | 20,092 | 20.4% |
| 10 | 22,654 | 23.0% |

63% of conversations reach 8+ turns; 23% hit the 10-turn cap. Token budget (1850) is the primary limiter.

### Known loss rates

- **~7% `turn1_over_budget`**: Prompts asking for bulk translation, full policy documents, data formatting, or other long-form single-shot tasks. Model generates 2000-8000 tokens of visible content, blowing the 1850-token budget. These prompts don't belong in multi-turn — the filter is healthy.
- **~5% `turn1_parse`**: Model produces valid response but not in JSON format. Mostly WildChat prompts with unusual formatting or injection attacks.
- **~3% `turn1: missing cited/uncited`**: Model produces JSON but without the required fields.
- **~0.7% non-English**: Filtered at export (>15% non-ASCII in first turn).
- **Actual yield**: 98.5% after all filters.

### Prompt iteration history

| Version | Key changes | Test |
|---|---|---|
| mt_v1 (initial) | Consistency, citation re-anchoring, no long-form | 16 seeds |
| mt_v1 + format reminder | `[RESPOND WITH JSON...]` prepended to follow-up turns — fixed JSON dropout | 16 seeds, 94% success |
| mt_v1 + over-refusal addendum | Separate adversarial addendum for adversarial_benign seeds testing false positives | 3 targeted seeds |
| mt_v1 + citation restraint | "let the topic guide whether citations add value" — fixed over-citation on mundane topics | 3 targeted seeds |
| mt_v1 + anti-sycophancy + anti-architecture-leak | "Don't open with You're right...", "Never describe your internal architecture" | 96 seeds, 98% success, sycophancy 6% (down from 7.6%) |
| mt_v1 (frozen for scale-up) | All above combined | 100K scale-up |

### Quality assessment (from 50-conversation sample of scale-up data)

- **Refusal consistency**: 4.5/5 — model holds firm across 7-10 turns of pressure
- **Adversarial_benign calibration**: 4.7/5 — no over-refusal on benign content
- **Benign conversation depth**: 3.8/5 — most conversations go somewhere interesting
- **Harmful redirect quality**: 4.0/5 — productive redirection into legitimate alternatives
- **Bracket hygiene**: Perfect — zero leaks of `[X.Y]` into uncited across all samples
- **Sycophancy rate**: 6.0% (mostly justified on challenge/escalate flows)
- **Remaining issues**: `[3.3]` "false impression" mechanical repetition, occasional factual hallucinations (inherent to generator), rare charter meta-leaks
