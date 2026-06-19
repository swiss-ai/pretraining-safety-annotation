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

---

## 2026-06-19 — English prompt wins (gemma-4-31b, qwen3.6-27b) + start multilingual

Two agents iterated the **English (dclm)** generator prompts (judge v5; method: investigate → minimal
edit → rotated subset test → full-200 validate; no positive examples; English-only).

**gemma-4-31b: v6 72.0%/4.186 → v9 89.5%/4.526** (+17.5pp accept, +0.34 mean, all axes up, cg 4.13→4.46;
net +35 flips). Levers: (v7) explicit-abstention rule — benign text MUST state nothing is at stake, not
recap (fixes the content-summary mode → the big jump to 89%); (v9) "routine helpfulness ≠ [2.1]" with an
anti-over-suppression guard. **v8 reverted**: a naive "abstain when dropping [2.1]" over-abstained on
sexual-violence/deepfake content (those items 4.25→2.0–2.5) — kept the guarded v9 instead.

**qwen3.6-27b: v1 88.4%/4.495 → v2 90.0%/4.516** (+1.6pp, net +3, cg +0.038). One bullet naming
zero-citation escape-hatch phrases as forbidden on value-laden content, with a benign carve-out
(protects the 81 no-citation accepts; zero over-citation regressions). Stopped at v2 (residual is the
temp-1.0 variance tail).

Committed v9 + v2 (prompt files). config.yaml prompt_reflection NOT bumped (still v6 / v1). gemma
intermediates v7/v8 left on disk (superseded).

### qwen3.6-35b-a3b — multilingual (fw2) iteration (Julian: do EN ceiling first, then fw2)
Carried the DCLM winner **v2** (reported-framing≠no-stakes) into fw2. Method as before (rotated subsets,
n=5, one lever, serialize judge runs; harness `~/tmp/seval.py` now reads the live judge tag/prompt from
config). Live judge tag rotated -yhzm→**-cXND** (committed `7e7486f`).

- **v2 transfers to fw2** (subset n=5, v1→v2): accept 0.686→**0.752**, charter_grounding 3.75→**3.94 (+0.19)**,
  rel +0.11, spec +0.12; voice 4.01→3.91 and wrong-language 6%→10% (the English under-reading paragraph
  slightly primes English fallback). Net positive — v2 is the fw2 base.
- **Key finding: wrong-language is INTERMITTENT/high-variance** — items that were voice_tone==1 in the
  baseline write the WRONG language only ~14–20% of the time on re-run, not deterministically. Same
  temp-1.0 variance as everything else.
- **v3 = language lever — strong WIN.** Rewrote `## Language`: write `reflection_1p` in the language of
  the **main body prose**, NOT any title/tags/menu/embedded snippet (pages mix languages; `cmn` subset
  has Japanese-source titles); names "English over a non-English source is a failure". LANG-prone subset
  (n=5, v2→v3): **wrong-language 12%→3%** (LANG-group 14%→4%), **voice_tone 3.83→4.15 (+0.32)**,
  accept 0.742→0.783, cg 4.03→4.15. DCLM regression subset check: flat (cg −0.05, within noise; the edit
  is English-inert by construction).
- **Full-200 confirmation (v3, committed `1ef7c40`):**
  - **fw2-multi v1→v3: accept 62.5%→79.3% (+16.8pp)**, mean 3.884→4.217, cg 3.67→4.035, voice 3.93→4.197,
    rel 3.905→4.303, spec 4.03→4.333, **wrong-language 19/200→2/198**; flips 48 r→a / 14 a→r. Per-language
    all healthy: deu 84%, fra 85%, rus 82%, cmn 78%, ita 77%, jpn 70% (weakest).
  - **dclm-en preserved:** mean 4.430→**4.479** (highest), voice 4.385→4.475, spec 4.40→4.45, cg 4.385,
    wrong-lang 1→0. accept 89%→86.5% = balanced threshold variance (v2→v3 flips 17 down / 12 up), edit
    English-inert. English NOT degraded.
  - Net: v2 (under-reading) + v3 (language) compound — fw2 the big win, English held.

### Status / handoff (2026-06-19, fw2)
- **Best qwen3.6-35b prompt = `generator_reflection_v3.md`** (commits v2 `aa2a65c` + v3 `1ef7c40`).
  dclm ~88% (variance band) / fw2 79.3%. config.yaml still points qwen3.6-35b at v1 — bump to `_v3.md`
  to make it the default.
- Remaining fw2 headroom: jpn weakest (70%); residual under-reads in cmn/jpn; the rest is the temp-1.0
  variance tail. Next levers low-yield per the dclm experience (severity-routing/openers both reverted there).

### Multilingual (fw2) — starting
- All current prompts already carry a `## Language` section ("write reflection_1p in the SAME language as
  the source"), but wrong-language is still ~25% of fw2 rejects (voice_tone=1 under v5) on the qwen3.6-35b
  line — the in-prompt instruction alone isn't enough.
- Plan: (1) test the current English-tuned prompts (gemma v9, qwen-27b v2) on fw2 to baseline + see if the
  under-reading fix transfers; (2) strengthen the language instruction (infer language from the document
  BODY, not any label — `cmn` subset carries Japanese-source titles). **Guardrail: re-check dclm after any
  prompt edit — English must not degrade.**
