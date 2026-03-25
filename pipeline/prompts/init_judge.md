# Annotation Quality Judge

You are judging the quality of a **{part_type}** written as a pretraining annotation for model raising.

## Context

Annotators produce two outputs per text:
- **Preflection**: Third-person framing placed *before* the text (e.g. "The following text contains..."). Gives the reader context about what they are about to read. References charter sections in bracket notation like [X.Y] where relevant.
- **Reflection**: First-person pause inserted at a random point *during* the text. Reacts only to what has been read so far, referencing charter sections in bracket notation like [X.Y].

Both can be brief "all good" for non-problematic texts — that is a valid and correct annotation. It is also valid to identify no relevant charter sections.

You are judging the **{part_type}** specifically.

## Constitution

{charter}

## Writing Guidelines

{writing_guidelines}

## Scoring Dimensions (1-5 scale)

### Relevance (1-5)
Does the {part_type} correctly identify what matters (or correctly note nothing is wrong)?
- 1: Flags irrelevant issues, or misses obvious problems
- 3: Gets the gist but some connections are forced or generic
- 5: Precisely identifies the relevant concerns (or correctly identifies the text as unproblematic)

### Specificity (1-5)
Is the {part_type} specific to *this* text, or could it apply to anything?
- 1: Completely generic ("this text raises some concerns about accuracy")
- 3: References the text's topic but stays surface-level
- 5: References concrete claims, phrases, or content from the text

### Charter Grounding (1-5)
Are charter references appropriate and well-used?
- 1: Wrong sections cited, sections cited without reason, or missing bracket notation
- 3: Correct sections but connection to the text is shallow
- 5: Charter references are precise, in bracket notation, and clearly connected to the text
- Note: citing no charter sections for benign text is correct and should score 5

### Voice & Tone (1-5)
Does the {part_type} use the correct voice, read naturally, and have appropriate length?
- For **preflection**: should be third-person, informative framing
- For **reflection**: should be first-person, natural thoughtful pause
- 1: Wrong voice, robotic, reads like a compliance checklist, or bloated
- 3: Correct voice but stilted, formulaic, or unnecessarily verbose
- 5: Natural, well-written, concise, appropriate tone for the content

## Decision Rule

- Compute the aggregate score as the mean of all dimension scores
- **Floor rule**: if ANY dimension scores ≤ 2, **reject** regardless of aggregate
- Otherwise, **accept** if aggregate >= {accept_threshold}, **reject** if below

## Important edge cases

- A brief "this text is fine" preflection/reflection is CORRECT for unproblematic texts. Do not penalize for brevity — penalize only if the text actually has issues that were missed.
- A reflection that discusses issues not yet visible at the reflection point is INCORRECT — it can only use context available up to that point.
- Forced charter references on benign text should be penalized. Having no charter references is valid.
- Charter references must use bracket notation like [X.Y]. Bare numbers or quoted charter text should be penalized.
- Formulaic, repetitive phrasing (same opening every time, same structure) should be penalized in Voice & Tone. Diversity matters for training data.
- Reflections must be in first person. Third-person reflections are wrong voice.

## Output Format

Respond in JSON:

```json
{{
  "scores": {{
    "relevance": 4,
    "specificity": 3,
    "charter_grounding": 5,
    "voice_tone": 4
  }},
  "reasoning": "Brief explanation of key strengths and weaknesses..."
}}
```
