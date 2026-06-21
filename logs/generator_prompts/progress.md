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

### Per-language injection (code feature, commit `99a1233`) — additional fw2 win, English untouched
Implemented per-language reflection-language injection in `generate_batch` (`inject_language` param;
threaded via `CandidateModel.inject_language` → `eval_generators`). Appends a per-language directive as
a final **user** message after the document (recency), keyed off the item language tag:
- deu/fra/ita/rus/jpn → in-language directive (primes target script, forbids English fallback);
- **cmn → defer-to-body** (the cmn bench subset is genuinely contaminated: 17/32 bodies are NOT Chinese
  — SEO/mixed CJK — so naming "Chinese" would force the wrong language);
- **en → no injection** (generation byte-identical → English cannot regress, by construction).
Tests added (directive map; message appended for non-English only; absent when off). Live judge tag
auto-read from config in the harness.

**Full-200 (v3 → v3+inject):**
- fw2-multi: accept 79.3%→**80.3%**, mean 4.217→**4.309**, relevance 4.303→**4.455**, cg 4.035→4.152,
  voice 4.197→4.293, **wrong-language 2→1/198**. (Headline accept +1pp is variance-limited near threshold;
  the broad per-axis lift, esp. relevance +0.15, is the real gain.)
- dclm-en: **89.9%** / mean 4.465 — English unchanged (en gets no injection). Also settles that v3's
  earlier dclm 86.5% was a low variance draw; English is ~89–90%.

### Remaining fw2 failure analysis (after v3+inject) → adult-content under-reading
v3+inject fw2 rejects (39/198): **35/39 (90%) are adult/sexual-content under-reads** — the multilingual
corpus at this safety threshold is saturated with porn/sex-service/AV/adult-SEO pages, and the model
dismissed them as "a mere product listing / SEO keyword spam / consensual adult ads → nothing at stake",
emitting ZERO citations and getting floored on dignity/objectification (1.1), ethnic/gender fetishisation
(1.3), protection-of-vulnerable incl. minors & hidden-camera (5.2), exploitation (5.1). Wrong-language now
1/39. Worst langs cmn/fra/jpn = the subsets with the most adult content. (NOT a language failure — a
content-domain failure concentrated in fw2 because DCLM-Edu English has almost none of this content.)

### v4 = adult-content citation lever — big fw2 win, English preserved (commit `52e2746`)
One guarded bullet (Citation discipline): adult/sexual commercial content is not auto-benign; cite
objectification/degradation (1.1), ethnic/group sexual fetishisation (1.3), sexualising minors / non-
consensual recording / trafficking / exploitation (5.2,+5.1). Trigger = degradation/objectification/
fetishisation/a-victim, NOT explicitness — plain consensual adult sex between equals needs no citation
(benign carve-out preserves the [[charter-1.1-dignity-floor-line]] floor).
- Subset (n=5, v3+inj→v4+inj): adult under-reads accept **0.53→0.90**, rel +0.71, cg +0.58; accepted-adult
  & benign guards held/improved (no over-citation). dclm guard flat.
- **Full-200 (v3+inj→v4+inj): fw2 80.3%→92.5%**, mean 4.309→4.561, rel 4.455→**4.76**, cg 4.152→4.405,
  wrong-lang ~0. **dclm 90.5%** / mean 4.494 (highest dclm run — English preserved; lever inert on
  educational English). Relevance rising confirms genuine quality, not over-citation gaming.

### Status / handoff (2026-06-19, fw2 — v4)
- **Best qwen3.6-35b config = `generator_reflection_v4.md` + `inject_language: true`**
  (prompt commits v2 `aa2a65c`, v3 `1ef7c40`, v4 `52e2746`; injection feature `99a1233`).
  **dclm 90.5% / fw2 92.5%** (fw2 from 62.5% baseline = +30pp; English held).
- **config.yaml DEFAULT now = v4 + inject** for the qwen3.6-35b-a3b generator_eval candidate (commit
  `04bc0f6`: `prompt_reflection: generator_reflection_v4.md`, `inject_language: true`). Judge tag/prompt
  also current (`7e7486f`: GLM-5.1 `-cXND`, judge_reflection_v5).
- **English also benefits from the adult lever** (not just fw2): the dclm-en bench is 14% adult/sexual
  content (28/200); v4 flipped 3 of those reject→accept (zero adult regressions), giving the highest dclm
  run (90.5%). The injection itself is a no-op for English (en → no directive).
- Remaining fw2 tail (~15 rejects) is small and mixed (borderline consensual-adult the judge only caps at 3;
  residual cmn contamination; temp-1.0 variance). Remaining dclm tail (19 rejects) is mostly non-adult
  zero-cite under-reads — the same charter-capped tail where severity-routing & openers both reverted
  (variance-locked ~90%). Diminishing returns on both — stopping point.
- Full-200 confirmation runs: `gen200_{fw2,dclm}_v5_q36_v4inj` (status done, n=200, judge v5).

### Multilingual (fw2) — starting
- All current prompts already carry a `## Language` section ("write reflection_1p in the SAME language as
  the source"), but wrong-language is still ~25% of fw2 rejects (voice_tone=1 under v5) on the qwen3.6-35b
  line — the in-prompt instruction alone isn't enough.
- Plan: (1) test the current English-tuned prompts (gemma v9, qwen-27b v2) on fw2 to baseline + see if the
  under-reading fix transfers; (2) strengthen the language instruction (infer language from the document
  BODY, not any label — `cmn` subset carries Japanese-source titles). **Guardrail: re-check dclm after any
  prompt edit — English must not degrade.**

### gemma-4-26b-a4b — English (dclm) iteration: v5 77% → v7 92% (the scale model)
Same method as 31b (investigate → minimal edit → rotated-subset test → full-200 validate; English-only;
no positive examples; judge GLM-5.1 v5). 26b is the intended **scale** annotation model.

**Baseline v5: 77.0%/4.213** (rel 4.275, spec 4.44, cg 4.055, voice 4.08). Reading 18 rejects: 26b's v5
prompt was the PRE-fix version — it lacked BOTH levers that drove 31b's jump. Failure modes:
(A) **over-citing/over-reading benign content** dominated (~25/46 rejects): routine helpfulness→2.1
(donation appeals, "extra care on a live DB", code/math help), routine technical explanation→3.2/3.3/3.4
(Bitcoin difficulty, Objective-C, ECDSA, disease articles), routine care advice→5.1/3.6 (warm bottle for
rabbits, horse azoturia), and 3.6 mis-routed onto animal/non-licensed advice. (B) **content-summary
under-reads** with zero citations on value-laden text (malware/privacy breach, flood statute). (C) a few
genre misreads (Urban Dictionary/Uncyclopedia satire read as sincere harm; animal-welfare mis-routed to 3.3).

- **v6 = port "Routine helpfulness is NOT 2.1" bullet** (Citation discipline, verbatim from 31b v9). One
  change. Subset of 13 known-bad (v5 was 0/13): accept 0→3, mean 2.96→3.42, **cg +0.69**. **Full-200:
  80.7%/4.227** (n=197; 3 parse-failure drops) — +3.7pp, mean held, cg 4.055→4.107, rel +0.10, net +7
  flips. Kept (English held-or-improved).
- **v7 = add the explicit-abstention rule** (Reflection section "Benign text — ...", verbatim from 31b v9)
  on top of v6. One change. This is the big lever: it gives the model the correct alternative to over-citing
  (cite nothing + explicitly state no stakes) AND fixes the content-summary under-reads. Rotated subset of
  12 known-bad (v5 0/12): **accept 10/12**, mean 3.40→4.46, **rel +1.59, voice +1.33, cg +0.75**.
  **Full-200: 92.0%/4.481** (n=199; 1 parse-failure drop) — **+15pp / +0.27 mean over v5, ALL FOUR axes up**
  (rel 4.275→4.628, spec 4.44→4.467, cg 4.055→4.327, voice 4.08→4.503). v5→v7 net **+30 flips** (37 r→a, 7 a→r).

**v7 is the winner** (blows past the ~85% goal; English strongly improved, not degraded). The 7 a→r
regressions are NOT over-abstention (the 31b-v8 trap) — every one still cited 4–9 values; they are
citation mis-routing dings on contested political/religious commentary, mostly marginal 4.0→3.75
temp-1.0 wobble. The abstention rule's benign carve-out held (no over-abstention on harm/sexual content).

**Remaining v7 rejects (16/199):** 10 = **citation mis-routing** on contested political/religious op-eds
(cited a sibling/adjacent value the judge prefers re-routed — the hardest genre, low yield, mostly agg
3.5–3.75); 3 = zero-cite under-reads (2 still genuine content-summary on value-laden text: malware breach,
flood statute; 1 thin BDSM item); 1 = residual over-cite. The over-citing mode that dominated v5 is
essentially eliminated. Mis-routing is a much harder/lower-yield target and pushing it risks the English
guardrail → **stopped at v7.** (Also noted: ~1–3 items/run drop to thinking-mode JSON-wrap parse failures
at temp 1.0 — a generation-robustness issue, prompt-independent, consistent across versions.)

**Status/handoff:** best gemma-4-26b-a4b English prompt = **`generator_reflection_v6.md` (kept) →
`generator_reflection_v7.md` (winner)**, both on disk. config.yaml `prompt_reflection` NOT bumped (still
v5). To make v7 the default for the gemma-4-26b-a4b candidate set, set
`prompt_reflection: generator_reflection_v7.md`. Full run dirs kept:
`data/pipeline/charter_eval/g26_v6_full`, `g26_v7_full`.

---

## 2026-06-19 (cont.) — qwen3.6-35b: edge-cases → v6 (generalized routing) → v7 (length/style) + Apertus reflection cutoff

### `edge-cases` bench + v4→v6 (routing fixes, leakage removed)
Built a curated **`edge-cases`** bench (`pipeline/charter/eval/edge_cases_src/`): 23 hand-picked hard
paragraphs × 7 langs (en + rus/cmn/deu/jpn/fra/ita) = 161 items, Opus-translated, annotated WHOLE
(reflection after the full text). 3 Opus failure-mode agents on v4 (reports under
`logs/generator_prompts/edge_cases/`) found genuine v4 faults: `[5.1]` mis-routing on factual war/history
(→`[4.4]`/`[3.3]`), missed stakes on loaded-question / dismissed-discrimination / conspiracy patterns,
facilitation-read-as-warning, repetitive voice — plus GLM-5.1 over-rejects (~⅓ of rejects are
correct-primary-cite / missing-secondary; not tuned against).
- **v5** = v4 + those fixes, but the first draft embedded near-verbatim edge phrases → **in-sample
  leakage** (edge hit a bogus 98%). **v6** = v5 with the exact quotes stripped (generic patterns only).
- **Paired eval, v6 vs v4 (swissai-native, same items):** dclm-en **flat 90.8% = 90.8%**; edge-cases
  **81.9% → 87.5%** (+5.6, net +9 fixed). An earlier *unpaired* "−3 regression" was a sampling artifact
  (the OpenRouter `gen200` baselines and swissai runs drew different 200-item samples). **v6 committed.**
- gemma-4-31b line (v13) was drafted then **abandoned** (too expensive); its failure analysis kept as a
  finding only.

### Backend: qwen3.6-35b-a3b now on swissai
`Qwen/Qwen3.6-35B-A3B-FP8-DnqH` @ `https://api.swissai.svc.cscs.ch/v1` (ephemeral tag — re-confirm) — use
instead of OpenRouter (free + co-located with the GLM judge). v4 is **backend-stable** (swissai 90.4% ≈
OpenRouter 90.5% on dclm), so OpenRouter baselines stay reusable. Runner gained
`--api-name/--endpoint/--judge-api-name`; context capped at 32768 (swissai deployment limit). See memory
`qwen36-35b-a3b-swissai-deployment`.

### v7 = length + style caps (commit `5a31b7e`)
~19% of v6 reflections exceed 256 Apertus tokens (median 156, max 589; fw2 worse, 24%). v7 = v6 +
**"The reflection must not be longer than 200 tokens."** + an **em-dash ban**. A/B on 60 items where v6
ran long (length only, no judge): reflections **>256 dropped 30/60 → 1/60**, mean 264 → 186;
**em-dashes 27% of reflections → 0%** (clean kill). Soft length cap (not a hard ceiling — a true ceiling
needs a `completion_max_tokens` clamp, with cutoff risk). **Length/style validated; quality not yet judged.**

### Reflection-point policy → deterministic Apertus-token cutoff (commit `ba7c50e`)
Replaced random reflection-point sampling (scale: char-space @8000; eval: SmolLM2 trunc @1920 + random)
with: reflection placed after the first **`min(doc_tokens, 3800)` Apertus tokens** (tokenizer
`swiss-ai/Apertus-70B-2509`), deterministic — `text[:rp]` is the whole doc if it fits, else its first
3800 tokens. Applied to **charter.scale + eval benches**. 3-agent review (correctness/completeness/risk);
fixes: scale completion-budget clamp (was `max_tokens=None` vs 24576 server context), item-pool
`reflection_policy` cache key (old pools fail loudly on resume), `estimate.py` parity, scale docs. 101
tests pass. **Flagged (not applied):** raise scale sglang `--context-length` 24576→32768 for thinking
headroom (GPU/KV-cache cost — Julian's call).

### 2k eval on the final setting (IN FLIGHT)
`val3k_{dclm-en,fw2-multi}_v7` (note: "3k" is a misnomer — n=**2000**), prompt **v7**, **Apertus** cutoff,
swissai DnqH gen + GLM-5.1 (`CWqJ`) judge, fresh run-ids (new cache guard). Reference: **old-policy** v6 2k
dclm = **92.2%** (`sw2k_dclm-en_v6`; fw2 judging was correctly blocked by the new policy cache guard).
v7-on-final-setting accept/mean per language pending.

---

## 2026-06-21 — hardening pass across all models, v10 named-hints fix, final 2k benchmark, production lock-in

### Hardening pass — all generator prompts bumped to new versions (commit `62ddeba`)
A spec-alignment + content-coverage pass applied to every model's latest prompt, then promoted to a new
version (old versions restored to pre-edit). Changes to the `## Citation discipline` section:
- **Removed the forum-label / slur-exemption bullet** (4chan `-fag` suffixes, "normie/twink/weeb" as
  "not protected-group slurs"). It was both ethically wrong **and** judge-penalized — the GLM-5.1 judge
  scores `fag` as `[1.1]`(dignity)+`[1.3]`; empirically `newfag→[4.1]` is **rejected (3.50)** while
  `newfag→[1.1][1.3]` is **accepted (4.75)**. So the bullet actively cost accuracy. Removed everywhere.
- **Added "Depiction is not exemption — cite the harm even in a neutral recap"** (depicted rape/sexual
  violence → `[5.1]`/`[5.2]`, massacre/mass killing → `[5.1]`/`[1.1]`, enslavement/captivity/subjugation
  → `[1.1]`/`[1.3]`). Triggered by a Genesis-34 rape/massacre recap the generator was passing as benign.
- **Added the sexual-objectification rule** (caps lead + commercial framing): objectification/degradation/
  fetishised stereotyping is a `[1.1]`(+`[1.3]`) citation, not "just adult content"; the trigger is
  degradation, **not** explicitness — porn/escort/adult-SEO listings fill this data.
- **Added "any mention of killing requires extra care"** — neither automatic `[5.1]` nor benign;
  depicted/glorified/instructed → `[5.1]`; neutral/one-sided mention → `[4.4]`/`[3.3]` or nothing.
- Replaced the LLM-awkward **"read this twice"** with a genuine restatement of the core message later in
  the prompt.
- Version bumps: gemma-4-31b v9→**v10**, gemma-4-26b-a4b v7→**v8**, qwen3.5-35b-a3b v5→**v6**,
  qwen3.6-35b-a3b v7→**v8**, qwen3.6-27b v2→**v3**. Eval config wired to the new versions.

### qwen3.6-35b under-reading fix: v9 (abstract) was a no-op → v10 (named hints) is the win (commit `d0016b8`)
qwen3.6-35b's edge-cases failure mode is **under-reading / abstention**: it declares subtle but
value-laden text "nothing at stake" and emits zero citations. Two attempts:
- **v9** = v8 + a heavy-caps **"FIND THE STAKE BEFORE YOU ABSTAIN … DEFAULT TO ENGAGING WHENEVER THE TEXT
  BEARS ON A PERSON OR GROUP"** directive in the Analysis section. **Result: no-op** — edge flat at
  **83%** (identical 133/161 to v8), dclm flat (~89%), abstain-rate barely moved (40%→38%). An abstract
  "engage more" instruction changed nothing. **Discarded** (consistent with the
  `named-negative-openers-fix-ai-voice` memory: abstract bans/exhortations fail; you must name the
  pattern). The broad "bears on a person or group" clause also risked over-reading (it didn't fire, but
  it was the wrong lever).
- **Diagnosis of *what* it abstains on** (17 relevance≤3 under-reads, spread ~1/language, **English the
  worst at 30% reject** — not a per-language gap): 4 recurring patterns — (1) opinions that
  dismiss/minimise discrimination → miss `[1.3]`; (2) narrative dignity slights (someone treated as an
  obstacle/imposition) → miss `[1.1]`; (3) political/rights/conscription commentary → miss
  `[1.2]`/`[2.2]`/`[6.2]`; (4) documented atrocities analysed → miss `[1.1]`/`[5.1]`.
- **v10** = v8 + **named-pattern hints** for exactly those 4 blind spots, placed in the Analysis section.
  **Critically: described as general behaviour types with no quoted edge wording** (+ an explicit "describe
  the engaged value in your own words; do not pattern-match wording" guard) to avoid in-sample leakage —
  the first draft that quoted phrasings was thrown out. **Result: edge 83%→89%** (133→143/161, +10 items,
  **all four axes up**, abstain 30→22, wrong-lang still 0; English +4, jpn +3, ita +2). dclm **held at
  ~90%** (no over-reading — accept went up, not down). It **generalized** (dclm improved too, not just
  edge), which is the signal the patterns weren't edge-specific. **v10 adopted**; the no-op v9 deleted.

### Final 2k benchmark — all 4 candidate models, hardened prompts (GLM-5.1 judge, SwissAI, Apertus-3800 cutoff, `inject_language` on)
Reruns `rr2_*` (dclm-en + fw2-multi, n=2000) + edge-cases (n=161), judged by rotating GLM-5.1
deployments (`iMHP`/`HCQR`/`EYNw`). **Accept rate (mean aggregate):**

| model (prompt)        | dclm-en          | fw2-multi        | edge-cases   |
|-----------------------|------------------|------------------|--------------|
| **gemma-4-31b (v10)** | **95.5%** (4.58) | **96.8%** (4.69) | **95.7%**    |
| **gemma-4-26b-a4b (v8)** | 91.7% (4.49)  | 94.3% (4.59)     | 83.9%        |
| **qwen3.6-35b-a3b (v10)** | 89.7% (4.44) | 91.9% (4.52)    | 88.8%        |
| **qwen3.5-35b-a3b (v6)**  | 84.1% (4.33) | 86.8% (4.46)    | 81.9%        |

Caveats: qwen dclm runs finished on reduced n (qwen3.5 dclm n=1574, qwen3.6 dclm n=1887) — items dropped
on flaky SwissAI generator deployments; rates are sound but on the reduced sample. qwen3.5 fw2 has
**wrong-language=50** (it lacks the language work the others carry). Sub-1pp gaps are judge-deployment
noise.

### Production lock-in → qwen3.6-35b-a3b v10 (commit `3421a59`)
`charter.scale` switched from qwen3.5-35b-a3b v5 to **qwen3.6-35b-a3b v10**. Rationale: **best
quality-per-cost** — 90/92/89 across benches at the **cheapest throughput** (~**34k GPU-h/100M docs**,
A3B MoE, thinking-on, TP1×DP4). gemma-4-31b is the quality leader (~96 across the board) but **dense →
~524k GPU-h/100M** (too costly at scale); gemma-4-26b-a4b is the next scale option (~57k GPU-h, 92/94/84);
qwen3.6-27b is a non-starter (~702k GPU-h — dense + ~15.8k thinking tokens/sample).

Config changes: `generator_alias`/`reflection_prompt` → qwen3.6-35b-a3b/v10; sglang `hf_slug`+`model_path`
→ `Qwen3.6-35B-A3B-FP8` (a141 path verified present); **`--context-length` 24576→32768** (fits charter +
3800-token doc insertion — closes the flagged item from the Apertus-cutoff work); `reasoning_parser`
`kimi_k2` (same A3B family); sampling resolves to `{t1.0, top_p0.95, top_k20, pp0.0}` via the `qwen3.6`
`_SAMPLING_DEFAULTS` entry (verified — the `qwen3.6` key is ordered before `qwen3`, so the substring
mis-match bug doesn't bite). Canonical v10 copied to `final_prompts/qwen3.6-35b-a3b/`. Scale already uses
`REFLECTION_MAX_TOKENS=3800`, so production matches the benchmark insertion policy.

**Open flag:** scale corpus is English-only (`language_filter: [en]`), so `inject_language` is a no-op
now **and** the scale path does not implement injection — must be added before any multilingual scale run
(that's where the fw2 wrong-language fix lives).
