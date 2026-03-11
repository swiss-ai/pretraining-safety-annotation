# Prompt Improvement Instructions

You are analyzing iteration results to improve the generator and judge prompts. Follow this structured process.

## Analysis Steps

### 1. Score Distribution
- What is the accept rate? Mean scores per dimension?
- Are scores clustered (everything is 3-4) or spread out?
- Which dimension scores lowest on average?

### 2. Failure Pattern Analysis
For rejected items (aggregate < threshold), categorize failures:
- **Generic analysis**: reflection could apply to any text (not specific enough)
- **Spoiling**: preflection reveals conclusions from the text
- **Shallow depth**: reflection restates rather than analyzes
- **Poor charter grounding**: charter elements mentioned but not meaningfully connected
- **Formatting issues**: output format problems, missing fields

### 3. Gold Set Comparison
For gold set items (is_gold=true), compare LLM generations against human annotations:
- Are the LLM's charter element selections similar to humans'?
- Is the LLM's analysis as specific as the human's?
- Does the LLM preflection avoid spoiling as well as the human's?

### 4. Judge Calibration Check
Look at the judge's reasoning:
- Is the judge being too strict? (rejecting genuinely good content)
- Is the judge being too lenient? (accepting low-quality content)
- Are human reviews available? Compare judge scores vs human scores.
- Is the judge giving useful, specific reasoning or generic boilerplate?

## Improvement Decisions

Based on your analysis, decide what to change:

### Generator Prompt Changes
- Add specific examples of common failure patterns to avoid
- Strengthen instructions for weak dimensions
- Add constraints to prevent spoiling if that's an issue
- Make "be specific" instructions more concrete

### Judge Prompt Changes
- Adjust scoring rubric descriptions if judge is miscalibrated
- Add examples of edge cases the judge handles poorly
- Clarify threshold semantics if needed

## Output

Write your analysis summary (stored in runs.jsonl) covering:
1. Key metrics (accept rate, mean scores, dimension breakdown)
2. Top 2-3 failure patterns with concrete examples
3. What you changed in the prompts and why
4. Predictions for next iteration
