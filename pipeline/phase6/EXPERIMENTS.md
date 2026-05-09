# Phase 6 Experiments

## EXP-003: Multi-Turn Charter-Aware Paired SFT

- **Date**: 2026-05-09 → ongoing
- **Model**: Qwen3.5-35B-A3B-FP8 (`kimi_k2` reasoning parser)
- **Prompt**: v11 + mt_v1 addendum (self-play with followup_user_v1)
- **Output**: `$SCRATCH/model-raising-data/phase6/`
- **HF repo**: `jkminder/model-raising-pb-50k-3c-mt-sft`

### Goal

Produce multi-turn paired (`cited`, `uncited`) SFT training data via self-play. Companion to the 301K single-turn dataset from Phase 5. Teaches the model adversarial resilience, factual consistency, and natural multi-turn conversation.

### Parameters

| Parameter | Value |
|---|---|
| total_rows | 50,000 |
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
| WildChat | 15,000 | unknown |
| WildGuardMix benign | 10,000 | benign |
| WildJailbreak adversarial_harmful | 10,000 | adversarial_harmful |
| WildJailbreak adversarial_benign | 5,000 | adversarial_benign |
| WildGuardMix harmful | 5,000 | harmful |
| WildJailbreak vanilla_harmful | 5,000 | vanilla_harmful |
