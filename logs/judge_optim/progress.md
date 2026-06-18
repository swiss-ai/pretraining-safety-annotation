# Judge-prompt optimization log

Running log of the GLM-5.1 reflection-judge optimization effort.
Goal: raise the gold judge's agreement with human reviewers by fixing
prompt-level failure modes surfaced from the human review feedback.

(Tracked under `logs/judge_optim/` via a `.gitignore` carve-out — committed documentation of this effort.)

---

## 2026-06-18

### Context
- Gold judge: **GLM-5.1** on SwissAI (`api.swissai.svc.cscs.ch`, api_name rotates,
  currently `zai-org/GLM-5.1-FP8-yhzm`). **Runs WITH thinking** (~1700 output_tokens/judgment,
  ~190 of which are the final JSON; `reasoning_tokens=0` is a red herring — SwissAI folds the
  thinking trace into `output_tokens`, like local sglang).
- Human feedback dataset: `jkminder/apertus-annotation-feedback` (binary accept/reject + reason).
- Eval runs: `gen200_dclm` (English/DCLM) and `gen200_fw2` (multilingual/FineWeb-2),
  5 generators each, judged by the gold judge. Cards in `dashboard/data/cards.json`.

### Phase 1 — analyze review reasons (3 subagents)
Of 37 deduped reviews, only **13 carried a reason** (11 reject, 2 accept); Julian wrote most.
5 human↔judge disagreements (vs v4 judge): 2 = language fallbacks (judge obeyed prompt),
3 = judge too strict (over-suppression / penalty stacking).

Three subagents (disagreement lens, reject-reason clustering, judge-reasoning audit) converged on:
1. **Language fallback** — biggest driver; v4 prompt *forbade* penalizing language.
2. **Penalty stacking** — one bad citation dropped both relevance AND charter_grounding → reject.
3. **Affirmation blind spot** — affirming an epistemic virtue ([3.3]/[3.2]) read as "manufactured concern".
4. **Forced [2.1]** on routine corporate/service helpfulness.

### Phase 2 — judge_reflection_v5 (4 edits to v4)
File: `pipeline/prompts/judge_reflection/glm-5.1/judge_reflection_v5.md` (tracked).
1. **Language hard gate**: reflection must match source language; mismatch → `voice_tone=1` → reject
   (added as the `**Language (hard gate)**` rule + quick-check `0.`).
2. **Anti-stacking**: a single forced/over-broad/mis-routed citation lowers `charter_grounding` only.
3. **Affirmation carve-out**: affirming a real epistemic virtue is valid; don't floor relevance.
4. **Named negative**: routine helpfulness is not `[2.1]` (caps charter at 3, no floor).

Pre-launch spot checks: French reflection on French source → correctly passed; two German
English-fallbacks (`8d50832c`, `407b4842`) → now rejected with "Wrong language" reasoning.

### Phase 3 — re-judge + alignment (v4 vs v5)
Re-judged all 10 files (5 generators × 2 runs) with v5 via the production `judge_batch`
(script: `~/tmp/rejudge_v5.py`; results: `~/tmp/rejudge_v5_results.json`).
Both v4 and v5 run with identical (thinking) settings → apples-to-apples.

Alignment (combined dclm+fw2), **excluding the degenerate `anon` reviewer** (κ=0 baseline,
near-all-accept):

| reviewers | v4 | v5 |
|---|---|---|
| excluding anon (n=37) | 86.5% / κ=0.685 | **91.9% / κ=0.820** |
| Julian alone (n=19) | 84.2% / κ=0.650 | **94.7% / κ=0.890** |
| all incl. anon (n=53) | 83.0% / κ=0.605 | 83.0% / κ=0.633 |

8 decision changes on reviewed items: **4 fixes** (2 language gate, 1 anti-stacking `371e42af`,
1 affirmation `20552a32` — all the target cases), **4 "breaks"** (2× anon ignored; 1× imanol
`407b4842` = a German fallback she leniently accepted that v5 correctly rejects per policy;
only 1 genuine new disagreement: Julian `ee053faf`, English, no reason recorded).

dclm-only judging distribution: accept 82%→80%, ~10% flips (balanced), aggregate Δ≈0 — i.e.
redistribution around the threshold, not a global shift. The net-stricter drift on English is
re-judge sampling variance (edits were all de-restrictions).

### Phase 4 — prompt tracking + relocation
- Committed v5 (`cabdadf`) and the v1–v4 judge history (`b97e7c6`) to `pipeline/prompts/judge_reflection/glm-5.1/`.
- Decision: make `pipeline/prompts` the single tracked runtime home (Option B), **flat per-alias**
  under `pipeline/prompts/models/{alias}/` (keeps the load-bearing flat layout the improver uses
  for state/human_notes/version-listing, so only `PROMPTS_DIR` changes). Relocation done via subagents.
- **Done**: `PROMPTS_DIR = pipeline/prompts/models`; 23 prompt files relocated via `git mv`
  (history preserved); transient `state_*.md` gitignored; `data/pipeline/prompts/` removed;
  one test (`test_charter_eval_judges`) hardened to monkeypatch `PROMPTS_DIR` so it can't pollute
  the now-tracked tree; full suite green (**241 passed**); all config prompts resolve.

### Data cleanup
- Deleted `anon`'s `[8.2]` privacy reject on the math-topic directory (`0f1595b3`) from the HF
  feedback dataset — no real PII at stake; user-confirmed invalid. (`anon` left reasons on only
  2/16 reviews; the other — a French porn objectification reject — was confirmed valid and kept.)

### Key facts / gotchas
- **Judge runs WITH thinking** — never say otherwise (memory: `thinking-not-reasoning-tokens`).
- **`anon`** reviewer is low-signal (near-all-accept, κ=0) — excluded from agreement reporting.
- Re-judge churn ~10% even with identical prompt is expected (sampling), so small κ moves aren't meaningful.

### Artifacts
- Prompt: `pipeline/prompts/judge_reflection/glm-5.1/judge_reflection_v5.md` (→ relocating to `models/glm-5.1/`).
- Re-judge script + results: `~/tmp/rejudge_v5.py`, `~/tmp/rejudge_v5_results.json`.
- New v5 judgments: `data/pipeline/charter_eval/{gen200_dclm,gen200_fw2}/judgments/glm-5.1__judge_reflection_v5.md__on__*.jsonl`.
- Commits: `cabdadf` (v5), `b97e7c6` (judge v1–v4 history).

### Open / next
- Decide whether to point `config.yaml` `gold_judge.prompt_reflection` at v5 (currently v4).
- Possibly investigate Julian `ee053faf` (the one unexplained new disagreement).
- More human reviews would firm up the small-n agreement numbers.
