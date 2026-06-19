# Edge-cases bench — baseline runs & failure analysis (2026-06-19)

Curated `edge-cases` bench (`pipeline/charter/eval/edge_cases_src/`, see
`benches.py`): 23 hand-picked hard paragraphs × 7 languages (en + rus/cmn/deu/jpn/fra/ita)
= 161 items, each annotated **whole** with the reflection placed **after the full
untruncated text**. Judged by the gold judge GLM-5.1 (`judge_reflection_v5`).
Run with `scripts/run_edge_cases_eval.py`. (`data/` is gitignored, so these tables
are the durable record.)

## Baseline results (per-language: mean aggregate / accept rate)

**qwen3.6-35b-a3b / generator_reflection_v4 / language injection** — overall **4.344 / 81.9%** (160/161; `09-deu` dropped)

| en | rus | cmn | deu | jpn | fra | ita |
|----|-----|-----|-----|-----|-----|-----|
| 4.337 / 82.6% | 4.424 / 82.6% | 4.261 / 82.6% | 4.364 / 77.3% | 4.370 / 87.0% | 4.293 / 82.6% | 4.359 / 78.3% |

**gemma-4-31b / generator_reflection_v12 / language injection** — overall **4.287 / 84.4%** (160/161; `08-deu` dropped)

| en | rus | cmn | deu | jpn | fra | ita |
|----|-----|-----|-----|-----|-----|-----|
| **4.065 / 69.6%** | 4.337 / 91.3% | 4.228 / 82.6% | 4.261 / 86.4% | 4.413 / 87.0% | 4.272 / 87.0% | 4.435 / 87.0% |

## Headline findings (full evidence in the two `failure_analysis_*.md` files)

- **Bottleneck dimension is `charter_grounding`** (citation routing) for both models, not voice. Most rejects are narrow (aggregate 3.5–3.75).
- **Language injection works** — 0/160 wrong-language reflections for both models.
- **qwen v4** over-engages: `[5.1]` mis-routing on factual war/history (should be `[4.4]`/`[3.3]`), plus stilted/repetitive voice. Misses illegal-facilitation `[5.1]`/`[3.6]` on HRT-for-a-minor text.
- **gemma v12** under-engages: over-applies its "no stakes" abstention template on English contested-history/slavery/atrocity passages (the sharp English dip), and mis-routes slavery to historiographic pluralism `[4.4]/[6.6]` instead of dignity `[1.1]/[1.2]`. Voice is clean.
- **Judge over-strictness** accounts for ~⅓ of rejects per model (GLM-5.1 caps `charter_grounding` at 3 when a *secondary* value is omitted even though the primary citation is correct). Per instruction, the judge is NOT tuned; generator fixes target only the genuine faults.
- **Two German JSON drops, two mechanisms:** qwen `09-deu` closed a `„…"` quote with an unescaped ASCII `"`; gemma `08-deu` emitted a stray `{` + markdown fence. Both are now recovered by the `json_repair` parser fallback (commit `0325342`) — independent of any prompt change.

## Follow-up
Generator prompt fixes drafted as qwen `v5` / gemma `v13` (routing/abstention only,
judge untouched); validated against `dclm-en` + `fw2-multi` to confirm no regression
before adoption.
