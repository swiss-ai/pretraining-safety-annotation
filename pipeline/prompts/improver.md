# Prompt Improvement Instructions

You are analyzing iteration results to improve the generator and judge prompts for pretraining data annotation. The annotations (preflections and reflections) become training signal for a model being raised from scratch.

## Hard Constraints (never violate these)

- **No specific article references in prompts.** The value constitution and writing guidelines may change, including numbering. Prompts must refer to "charter sections" generically, never cite specific numbers like [4.3].
- **Bracket notation.** All charter references in generated output must use brackets: [1.1], [3.2]. Bare numbers or quoted charter text are wrong.
- **Reflections are first person.** Always. ("I notice...", "This makes me think...")
- **Preflections are third person.** Always. ("The following text...", "This passage...")
- **Conciseness.** Especially for benign texts. A brief "looks fine" is better than a verbose analysis of nothing.
- **Diversity.** Generated annotations must vary in phrasing, structure, and opening. Formulaic repetition degrades training data quality. This is IMPORTANT!
- **No charter references is valid.** Many texts have no relevant charter sections. Forcing references is worse than omitting them.
- **Keep prompts short.** Generator and judge prompts are sent as system messages to small models (7B-70B). Long prompts waste context window, increase latency, and confuse smaller models. Aim for under 800 words per prompt. If a prompt grows beyond that, cut examples and redundant instructions rather than adding more.

## Analysis Steps

### 1. Score Distribution
- What is the accept rate? Mean scores per dimension, separately for preflection and reflection?
- Which dimension scores lowest on average?
- Are "all good" items (benign texts) being handled correctly, or is the generator forcing problems?

### 2. Failure Pattern Analysis
Items are rejected if aggregate < threshold OR any dimension scores ≤ 2 (floor rule).
For rejected items, categorize failures:
- **Forced problems**: generator flags issues in benign texts instead of saying "all good"
- **Generic output**: preflection/reflection could apply to any text, not specific to this one
- **Wrong voice**: preflection uses first person, or reflection uses third person
- **Reflection uses future context**: reflection references content that appears after the reflection point
- **Poor charter grounding**: charter sections cited but connection to text is shallow or wrong
- **Missing bracket notation**: charter references without [brackets]
- **Verbose on benign text**: long-winded annotations for texts that are perfectly fine
- **Repetitive phrasing**: same opening, same structure across multiple items
- **Formatting issues**: output format problems, missing fields

### 3. Gold Set Comparison
For gold set items (is_gold=true), compare LLM generations against human annotations:
- Do they identify the same charter sections?
- Does the LLM correctly handle benign texts (says "all good" when humans did)?
- Is the LLM reflection appropriately scoped to context before the reflection point?
- Note: human annotations may not use bracket notation — the LLM should still use it

### 4. Human Review Mining
Human reviews contain free-text notes with valuable qualitative feedback. **Always** have a subagent read all reviews (`reviews` command without an iteration filter) and extract actionable insights. The `reviews` output can be large — this is why you MUST delegate it to a subagent rather than reading it directly.

From the review notes, extract and save to your `state.md`:
- Recurring complaints or praise patterns across reviewers
- Specific failure modes reviewers flag that the judge misses
- Style/tone preferences expressed by reviewers
- Any calibration notes (e.g., "this should have been accepted")

These insights persist across your sessions via `state.md` and should inform prompt changes.

### 5. Judge Calibration Check
- Is the judge penalizing valid "all good" annotations?
- Is the judge catching when reflections use information past the reflection point?
- Is the judge penalizing missing bracket notation?
- Is the judge rewarding conciseness or penalizing brevity on benign texts?
- Are human reviews available? Compare judge scores vs human scores.

### 6. Diversity Check
- Sample 5-10 reflections: do they start with different phrases?
- Do preflections vary in structure or all follow "The following text..."?
- Are the analyses formulaic (same bullet point structure every time)?

## Improvement Decisions

### Generator Prompt Changes
- If forced problems: strengthen "many texts are fine" messaging, add examples of brief benign annotations
- If generic: add instructions to reference specific content from the text
- If wrong voice: make voice requirements more prominent
- If future context leakage: emphasize that reflection only knows text up to the reflection point
- If verbose on benign: add examples of appropriately brief "all good" annotations
- If repetitive: add explicit diversity instructions, suggest varied openings
- If missing brackets: strengthen bracket notation requirement

### Judge Prompt Changes
- If penalizing valid "all good": clarify that benign texts should get high scores for brief correct annotations
- If missing future context leakage: add it as an explicit check
- If not catching bracket notation issues: add to scoring rubric
- If not penalizing repetitiveness: add to Voice & Tone rubric
- Adjust rubric descriptions if miscalibrated

### Model-Specific Prompts
Each model has its own prompt directory at `pipeline/prompts/{model_slug}/`. When improving prompts, always edit the correct model's directory. To add a new model, copy an existing model's directory and adapt. The model slug is the model name with `/` replaced by `_` (e.g. `jminder/data-annotator-glm45` → `jminder_data-annotator-glm45`).

## Feedback

If you encounter bugs, missing tools, or features that would make your job easier, append them to `data/pipeline/improvements.md`. Keep entries concise (one line each). This file is read by the developers — it is your channel to request infrastructure changes.

## Output

Write your analysis summary (stored in runs.jsonl) covering:
1. Key metrics (accept rate, mean scores per dimension, preflection vs reflection breakdown)
2. Top 2-3 failure patterns with concrete examples
3. Diversity assessment (are outputs varied or formulaic?)
4. What you changed in the prompts and why
5. Predictions for next iteration
