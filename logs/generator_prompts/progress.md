# Generator-prompt optimization log

Running log of the charter-reflection **generator** prompt/model effort (the judge side lives in
`logs/judge_optim/`). Goal: pick the best generator model(s) and improve their reflection prompts to
raise charter-eval accept rate / aggregate, as scored by the GLM-5.1 gold judge.

(Tracked under `logs/generator_prompts/` via a `.gitignore` carve-out — committed documentation of this effort.)

Eval harness: `uv run python -m pipeline.charter.eval eval-generators …`. Benches `dclm-en` (English)
and `fw2-multi` (multilingual FineWeb-2), 200 items each, seed 42. Generators via OpenRouter; gold
judge GLM-5.1 on SwissAI (`zai-org/GLM-5.1-FP8-yhzm`, tag rotates). Prompts live under
`pipeline/prompts/models/<alias>/`.

---

## 2026-06-18

### qwen3.6 added + sampling pinned
- Added **qwen3.6-35b-a3b** (A3B MoE) and **qwen3.6-27b** (dense) as candidates (OpenRouter:
  `qwen/qwen3.6-35b-a3b`, `qwen/qwen3.6-27b`). Both are thinking-by-default reasoners.
- **Sampling A/B** (qwen3.6-35b, judge v4): provider-default (= generation_config default ≈ temp 1.0,
  top_p 0.95, top_k 20, **pp 0.0**) BEAT the dev "general" preset (temp 1.0, **pp 1.5**) on both benches
  (dclm 89%→84%, fw2 72%→68%; charter_grounding −0.075/−0.105). temp 0.6 was never the winner — the
  gen_config default temperature is 1.0. **Pinned everywhere: temp 1.0 / top_p 0.95 / top_k 20 / pp 0.0.**
  qwen3.6 thinks ~half as many tokens as qwen3.5 for comparable/better scores (more token-efficient).
- Fixed `api_call` to retry on `json.JSONDecodeError` (malformed HTTP body) instead of crashing the run.

### Full v5-judge leaderboard (all 5 generators, judge_reflection_v5)
Re-judged every generator under **v5** (which adds a wrong-language hard gate → `voice_tone=1`, plus
over-citation/penalty-stacking relief). Each model run independently (per-model run-ids
`gen200_<bench>_v5_<short>`); qwen3.6-27b generated fresh, the rest reused existing generations via
`--stage judge`.

**dclm — English**
| Model | Accept | Mean | rel / spec / cg / voice |
|---|---|---|---|
| qwen3.6-27b | 88% | **4.495** | 4.65 / 4.48 / 4.33 / 4.52 |
| qwen3.6-35b | 89% | 4.430 | 4.58 / 4.40 / 4.35 / 4.38 |
| qwen3.5 | 85% | 4.327 | 4.41 / 4.39 / 4.31 / 4.20 |
| gemma-26b | 77% | 4.229 | 4.30 / 4.44 / 4.09 / 4.09 |
| gemma-31b | 72% | 4.186 | 4.18 / 4.29 / 4.12 / 4.16 |

**fw2 — multilingual**
| Model | Accept | Mean | rel / spec / cg / voice |
|---|---|---|---|
| qwen3.5 | **84%** | **4.345** | 4.54 / 4.42 / 4.32 / 4.11 |
| qwen3.6-27b | 74% | 4.149 | 4.14 / 4.25 / 3.86 / 4.35 |
| gemma-31b | 70% | 4.143 | 4.25 / 4.20 / 4.02 / 4.11 |
| qwen3.6-35b | 62% | 3.884 | 3.90 / 4.03 / 3.67 / 3.93 |
| gemma-26b | **2%** | 2.303 | 2.81 / 2.62 / 2.67 / **1.11** |

### Findings
- **qwen3.6-27b = best all-rounder** — top English mean (4.495) and best fw2 voice (4.35); beats
  qwen3.6-35b on both benches, decisively on multilingual (74% vs 62%).
- **qwen3.5 = best multilingual** (84%) — reliably writes in the source language.
- **qwen3.6-35b**: strong English (89%), weak multilingual (62%) — wrong-language + under-reading.
- **gemma-26b is broken for multilingual (2%)**: 194/200 fw2 reflections written in the WRONG language
  (English on non-English sources) → v5 hard-gates `voice_tone=1`. v4 masked this entirely.
- **v5 effect**: the wrong-language gate reorders the multilingual board and exposes real quality v4 hid
  (gemma-26b 73%→2%; qwen3.6-35b 72%→62%).

### Generator failure modes (from the weakness audit, qwen3.6-35b)
1. Under-reading: declares value-laden (esp. explicit/adult multilingual) content "nothing at stake"
   with zero citations → charter_grounding floor (~49/54 fw2 rejects).
2. Wrong-language reflections (now hard-rejected by v5).
3. Missing inline `[X.Y]` / misrouted citations (route dehumanization to [1.1]/[1.3], not soft [4.1]).
4. Formulaic openers (minor).

### Run locations
- v5 per-model runs: `data/pipeline/charter_eval/gen200_{dclm,fw2}_v5_{q36,q3627,q35,g31,g26}/`
  (judgments: `…/judgments/glm-5.1__judge_reflection_v5.md__on__<gen>.jsonl`).
- v4 runs (pre-v5): `data/pipeline/charter_eval/gen200_{dclm,fw2}/`.

### Next
- Improve the qwen3.6-35b generator prompt (`pipeline/prompts/models/qwen3.6-35b-a3b/generator_reflection_v1.md`):
  target the under-reading + wrong-language modes (minimal edits, no positive examples, named negatives,
  no citation over-suppression, [1.1] = degradation test). Iterate on fw2 first.
- gemma-26b needs the in-source-language fix before it's usable multilingually.
- Consider qwen3.6-27b as the production generator (best balanced under v5).

### qwen3.6-35b prompt iteration — DCLM (English) first (decision: do EN before fw2)
**Method (Julian):** test each minimal edit on a curated subset of known-problematic items first, then
scale to full 200 to confirm; **rotate the subset every edit** (avoid overfit); **n=5 repeats/item**
(generation at temp 1.0 is high-variance — re-running v1 on 18 baseline-rejects re-accepted ~14/18, so
single samples are noise). One lever per edit; per-axis movement (esp. charter_grounding) is the signal,
not single-item flips. Reuse `generate_batch`/`judge_batch` on hand-picked items (`~/tmp/seval.py`);
serialize judge runs (>2 concurrent GLM-5.1 jobs throttle on SwissAI). See memory
`charter-eval-genprompt-iteration-variance`.

**v1 dclm failure clusters (22 rejects):** (A) zero-citation under-reads ~11 (charter_grounding/relevance
floored to 2 — hides behind "a straightforward report"/"neutral historical documentation"/"purely
scriptural commentary"); (C) charter_grounding capped at 3 ~7 (mis-route dehumanisation to soft [4.1] vs
[1.1]/[1.3]; miss [5.2] on vulnerable populations) — all agg 3.75, the cheap tail; (B) over-reads ~3
(forced [2.1] on a rabbit-care guide, [4.1] on blog gossip). Examples: under `5cbc59`(malware/privacy),
`0fe698`(Mussolini "rightfully so"), `284b7a`(Ainu language death); cluster-C `8a62bc`(Mandela→[4.1]),
`b62cd9`(Nazi camps, missed [5.2]).

**Edits tried:**
- **v2 — KEEP.** One paragraph after analysis Step 2: reported/historical/factual/technical/scriptural
  framing ≠ no stakes; cite depicted value-laden content; names the exact dismissal phrases as negatives;
  guards benign ("one short sentence" only). Targets cluster A.
  - Full-200 dclm-en vs v1 (identical items): **charter_grounding 4.35→4.425, mean 4.430→4.456,
    relevance 4.585→4.605, spec 4.40→4.425, voice 4.385→4.37**; accept flat 89% (17 flips each way =
    threshold variance-locked). Subset A (n=5): UNDER accept 0.53→0.67, cg +0.13, benign/over flat.
    Committed `aa2a65c`.
- **severity-routing (dehumanisation→[1.1]/[1.3], [5.2] for vulnerable) — REVERT.** Disjoint subset_B
  (charter-capped items): cg 4.40→4.32, accept 0.93→0.91 — slightly worse across the board. Explicit
  routing instructions didn't lift the cap-at-3 cluster.
- **named-negative openers ("I read about…", "I see the claim that…", …) — REVERT.** Disjoint subset_C
  (voice-focused): voice_tone 4.18→4.15 (VOICE group 4.0→3.72), accept 0.88→0.85 — banning stems pushed
  the model into worse openings.

**DCLM conclusion: v2 is the ceiling for minimal edits.** Headline accept is variance-locked at ~89%;
two consecutive reverts confirm the cap-at-3 / voice tail resists prompt nudges (and nudges regress).
v2's gain is real per-axis robustness (+0.075 cg, no regression) and — importantly — its under-reading
fix is the mode that dominates **fw2** (61/75 rejects are zero-citation), so it should transfer.

### Next (fw2 — the real headroom, 62%) — NOT STARTED (paused after DCLM; Julian: "make a note")
**fw2 reject split (v1, 75/200 rejects):** zero-citation under-reads (non-language) **47 (63%)** —
the v2 target, should transfer; **wrong-language (voice_tone==1) 19 (25%)** — by subset deu 8, cmn 4,
rus 2, ita 2, jpn 2, fra 1; other 9. Rejects spread evenly across subsets (jpn 15, deu/fra 13, cmn/rus/ita 11–12).
- **Step 1 (cheap):** measure whether v2's under-reading fix transfers to fw2 — run v1 vs v2 on a fw2
  subset (under-read rejects + wrong-language + benign, n=5). Sets the fw2 baseline for v2.
- **Step 2 (highest-value fw2 lever):** wrong-language — instruct: write `reflection_1p` in the source
  language **inferred from the document BODY, not any label** (`cmn` subset has Japanese-source titles).
  Targets the 19 voice_tone==1 rejects.
- Same method: rotated subsets, n=5, one lever at a time, serialize judge runs.
- Open design choice (deferred): keep ONE prompt vs. branch a multilingual prompt (Julian: separate
  EN/multilingual prompts are a "later, not now" idea — for now keep editing the single prompt).

### Status / handoff (2026-06-18)
- **Kept:** `generator_reflection_v2.md` (commit `aa2a65c`). Best qwen3.6-35b generator prompt to date.
- **config.yaml NOT changed** (live/user file): qwen3.6 candidate still points at `generator_reflection_v1.md`.
  Flip `prompt_reflection: generator_reflection_v2.md` to make v2 the default eval/scale prompt.
- **Generator choice still open:** qwen3.6-27b is the best-balanced under v5 (dclm 4.495 / fw2 74%) —
  if 27b is the production pick, re-run this prompt iteration on it (v2 edit is model-agnostic; re-validate).
