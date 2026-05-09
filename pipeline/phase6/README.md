# Phase 6: Multi-Turn Charter-Aware Paired SFT

Generates multi-turn paired SFT data via **self-play**: each conversation starts from a single seed prompt (drawn from WildChat, WildJailbreak, or WildGuardMix), then a user-simulator model generates follow-up questions and the assistant responds — repeating until the token budget is exhausted.

Companion to the single-turn dataset from Phase 5 (`jkminder/model-raising-pb-300k-3c-sft`). Both datasets train together.

## Training goals

1. **Adversarial escalation** — user pushes across turns, tries jailbreaks/social engineering
2. **Factual consistency** — follow-ups reference the model's prior answer
3. **Natural multi-turn** — topic deepening, clarification, pivoting
4. **Benign-to-harmful pivots** — conversations that start innocent but turn harmful

## Architecture

### Self-play loop

Each conversation is generated as a sequence of turns:

1. **Turn 1**: seed prompt → assistant response (v11 + mt_v1 addendum)
2. **Turn 2+**: user simulator generates follow-up → assistant responds
3. **Repeat** until token budget (1850 SmolLM2 tokens) is hit or max turns (5) reached

The user simulator sees the **cited** response (with `[X.Y]` brackets) but is instructed to ignore them. This avoids dual-track coherence drift.

### Flow types

Follow-up turns are steered by flow types with per-harm-category weights:

| Flow type | Description | Harmful | Benign |
|---|---|---|---|
| `escalate` | Push toward sensitive territory | 35% | 5% |
| `challenge` | Push back, counterargument | 25% | 15% |
| `deepen` | More detail on a specific point | 15% | 40% |
| `redirect` | Apply to scenario, pivot, de-escalate | 25% | 40% |

For harmful seeds, `redirect` captures **de-escalation**: user accepts the model's suggestion.
~10% of benign seeds get `benign_pivot` weights (40% escalate, 40% challenge) to produce benign-to-harmful pivots.

### Adversarial quality

For `escalate`/`challenge` flows on harmful seeds, the user simulator gets:
- Explicit attack taxonomy (reframing, authority claims, emotional manipulation, etc.)
- WildJailbreak few-shot examples

These are NOT injected for benign seeds or other flow types.

## Prompt structure

The assistant prompt is **v11 + mt_v1 addendum** (appended, not a separate file):

```python
system_prompt = render_system_prompt("v11") + "\n\n" + load_mt_addendum("mt_v1")
```

The mt_v1 addendum adds consistency and re-citation rules. v11 improvements flow through automatically.

## Modules

| Module | Purpose |
|---|---|
| `generate.py` | Core self-play loop, flow types, token budget, quality checks |
| `slurm_generate.py` | `MultiTurnGenerator(PipelineStep)` for SLURM |
| `export.py` | Multi-turn parquet export + HF upload |
| `__main__.py` | CLI dispatch |

Reused from phase 5: `PromptsReader`, `materialize_prompts`, `merge_shards`, `render_system_prompt`, canary system, data loaders.

## CLI

```bash
# Prompt iteration (login node, openrouter)
uv run python -m pipeline.phase6 iterate --n 20

# Scale-up (SLURM)
uv run python -m pipeline.phase6 materialize
uv run python -m pipeline.phase6 submit
uv run python -m pipeline.phase6 status
uv run python -m pipeline.phase6 merge
uv run python -m pipeline.phase6 export
```

## Source distribution (~50K, seed=43)

| Source | Draw | Harm category |
|---|---|---|
| WildChat | 15,000 | unknown (benign) |
| WildGuardMix benign | 10,000 | benign |
| WildJailbreak adversarial_harmful | 10,000 | adversarial_harmful |
| WildJailbreak adversarial_benign | 5,000 | adversarial_benign |
| WildGuardMix harmful | 5,000 | harmful |
| WildJailbreak vanilla_harmful | 5,000 | vanilla_harmful |

50% benign, 50% harmful/adversarial. HarmfulQA excluded (covered by phase 5).

## Output

HF dataset: `jkminder/model-raising-pb-50k-3c-mt-sft`

Same schema as phase 5 (`messages_cite`, `messages_nocite`) but with variable-length message lists + `n_turns` column.

## Thinking / reasoning behavior

Phase 6 uses `thinking=True` explicitly (needed for OpenRouter iteration, where there is no server-side reasoning parser). This causes reasoning to be separated in the API response. On local sglang with `--reasoning-parser kimi_k2`, the behavior is identical to phase 5 — the model always thinks regardless of the `thinking` parameter. The parameter only controls whether reasoning is *separated* in the response vs. stripped server-side. See the phase 5 README for the full mechanism.
