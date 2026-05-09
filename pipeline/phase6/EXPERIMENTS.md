# Phase 6 Experiments

## EXP-003: Multi-Turn Charter-Aware Paired SFT

- **Date**: 2026-05-09 → ongoing
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
| max_turns | 5 |
| token_budget | 1,850 SmolLM2 tokens |
| rows_per_task | 5,000 |
| max_concurrent_requests | 512 |
| thinking | true |

**Thinking note**: Phase 6 uses `thinking=True` (unlike phase 5's `thinking=False`). Both produce identical model behavior — Qwen3.5 always thinks via its chat template defaults. The difference is only in how reasoning is returned: phase 5 relies on sglang's `kimi_k2` parser to strip `<think>` blocks server-side, while phase 6 separates reasoning explicitly (required for OpenRouter where there is no server-side parser).

### Source distribution

| Source | Draw | Category |
|---|---|---|
### Known loss rates (from 100-sample iteration + early scale-up)

- **~10% `turn1_over_budget`**: Prompts asking for bulk translation, full policy documents, data formatting, or other long-form single-shot tasks. Model correctly generates the requested content but it's 2000-8000 tokens, blowing the 1850-token budget. These are prompts that don't belong in multi-turn anyway — the filter is healthy.
- **~5% `turn1_parse`**: Model produces valid response but not in JSON format. Mostly WildChat prompts with unusual formatting.
- **~3% `turn1: missing cited/uncited`**: Model produces JSON but without the required fields.
- **Expected yield**: ~80-85% of seeds produce usable multi-turn conversations.

### Source distribution

| Source | Draw | Category |
|---|---|---|
| WildChat | 30,000 | unknown |
| WildGuardMix benign | 20,000 | benign |
| WildJailbreak adversarial_harmful | 20,000 | adversarial_harmful |
| WildJailbreak adversarial_benign | 10,000 | adversarial_benign |
| WildGuardMix harmful | 10,000 | harmful |
| WildJailbreak vanilla_harmful | 10,000 | vanilla_harmful |
