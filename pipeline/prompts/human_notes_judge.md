Look at the notes of the past reviews in detail. They contain the most important signal of what the judge should catch. Do not overfit on mere numerical values (e.g. acceptance rate going up or down). Correlation with review scores should generally go up but don't remove changes that don't increase the correlation if you deem them actually having the right effect. Inspect it manually in detail. Review whether the updated version addresses the notes. We don't have that many reviews so the review correlation scores are a bit noisy. Keep this in mind. The notes are most important. Try to use them to get the correlation up (cohens K most important).

For formulaic openers: you are right to judge them lower. We don't want that. So I'd say a verbal score of 3 is good for that. This overrules review scores — do not let kappa pressure relax it.

## Scoring Philosophy (IMPORTANT)

The reflection should be an ethical reflection, not a content summary. A reflection is an ethical pause during the text. It should not recap plot, topic, or technical details unless doing so contextualizes a charter concern. Summary-heavy annotations should score rel ≤ 3, voi ≤ 3.

For benign content (no charter relevance), annotations should be very brief — one sentence, primarily conveying "nothing ethically loaded here." Lengthy summaries of benign content are a voice failure.

Some human reviews still follow the old philosophy (scoring benign-content summaries at rel=5). The new philosophy takes precedence — do not let κ pressure revert to tolerating summaries. κ may diverge from older reviews; that's expected and acceptable.

## Key Lessons from Manual Iteration (v27-v50)

These are hard-won insights from 20+ prompt iterations. Respect them:

1. **Small model fragility**: kimi-k2.5 is a small model. Abstract principles ("inner monologue while reading") destabilize it. Use concrete examples and specific rules instead. Every new rule is a potential failure mode — add rules only when there's clear evidence they help.
2. **Charter floor calibration**: charter ≤ 2 should only trigger when the annotation has ZERO [X.Y] citations despite discussing charter-relevant themes. If the annotation cites some sections but misses others, that's charter = 3, not 2. The judge tends to over-apply the floor rule on partial citations.
3. **Isolated testing is essential**: never combine multiple changes in one version. Changes that individually improve κ can regress when combined. Test each change alone, then combine winners.
4. **Non-determinism is ~0.13 κ**: the same prompt can produce κ swings of 0.13 across runs. A single measurement is noisy. Don't chase small κ movements.
5. **Cutoff mentions**: strict is better. The reflection must never reference the text being cut off, incomplete, or trailing off. Do NOT soften this rule — softening regressed κ badly.
6. **Cross-check source text**: the judge should verify the annotation's characterization against the source text (e.g., don't trust "purely informational" if the text contains slurs). But phrase this as "cross-check" not "do not trust" — adversarial framing makes the judge over-reject.
7. **QC#4 (meta-language)**: mentioning the charter or annotation task by name = voice ≤ 3. Only the literal abstract placeholder `[X.Y]` (not filled in) triggers charter floor. Do NOT make meta-language a blanket floor trigger.

## Iteration Protocol

IMPORTANT FOR THIS SPECIFIC RUN: Only try to improve the cohens K agreement. DO NOT RUN A CROSSBATCH UNTIL THE VERY END (WHEN YOU SELECTED THE BEST PROMPT). The loop is:

1. Run `reviews --reasoning-limit 800` to see the latest judge's reasoning alongside each reviewer note. (Reasoning is on by default at 200 chars but 200 truncates mid-sentence, so bump it.) Heavily use subagents (opus) to do this. You can use --offset and --limit to page the reviews.
2. Identify items where the judge's reasoning misses or contradicts what the reviewer flagged. Focus on the reasoning content, not just whether scores match — a prompt that produces the right score for the wrong reason is a regression and should be rejected even if kappa improved.
3. Make a TINY targeted edit. Avoid overly specific solutions hardcoded to one item or one reviewer's phrasing — the train set is small (~80 reviews) and overfitting is easy. Prefer rules that generalise (e.g. "down-weight unsupported psychological claims") over rules that anchor on a specific item or phrase.
4. Run `rejudge_all`, then rerun `reviews` and inspect the updated correlations and reasonings.
5. Repeat. Stop after max 10 iterations. Then select the best prompt and run a single crossbatch to load stats and look at actual judgings of newly sampled items.

Note: The prompt is already optimised quite well. Try really small targeted edits individually and incrementally. Identify the clearest gaps from the review (notes). Use them to propose those really small targeted edits.

THIS INSTRUCTION PRECEDES THE NORMAL INSTRUCTIONS.