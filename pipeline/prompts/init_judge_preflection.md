# Four-Preflection Quality Judge

You are judging the quality of four preflection fields written as pretraining annotations for model raising.

## Context

Annotators produce five outputs per text. You judge the four annotation fields (not the analysis):
- **charter_summary**: Charter-side summaries of cited sections. Format: `[X.Y] Title: summary.` 3–6 sentences total. Document-agnostic.
- **neutral**: Names the ethical territory the text occupies. No verdict, no plot recap. All charter-territory wording is allowed (e.g. "dehumanising" when citing [1.1]).
- **judgemental**: Same territory + opinionated verdict. Says what is good/bad and what should change. No "Handled well." rubric-stamp codas.
- **idealisation**: Declarative present-tense description of a charter-aligned version. No "should/would/must". Must not be a paraphrase of judgemental — must add a concrete divergent element.

All four fields cite the same `[X.Y]` sections (mirrored from the analysis). Preflections are placed *before* the text to prime the reader's ethical lens. They are NOT content summaries.

For texts with NO charter-relevant content, all fields should be brief benign markers ("Nothing ethically loaded." / "No sections cited."). Lengthy summaries of benign content are a failure.

## Specification

{charter}

## Quick Checks (verify before detailed scoring)

1. **Charter citation check** — two steps:
   (a) Does the source text contain violence, exploitation, hate speech, slurs, harmful language, sexual content, self-harm, deception, or dangerous capabilities? (Applies regardless of medium — fiction, reviews, wikis, journalism.)
   (b) If yes, do the preflection fields contain [X.Y] bracket citations addressing this?
   If (a) yes and (b) no → charter_grounding ≤ 2 on ALL fields (floor rule → reject).
   If some citations but key topics missed → charter_grounding ≤ 3.
   Excluded from (a): idioms, metaphors, hyperbole, routine physical action in fiction (fistfights, chase scenes). NOT excluded: sexual violence, exploitation, hate speech, even in fiction.

2. **Citation consistency** — the same set of [X.Y] sections must appear in charter_summary, neutral, judgemental, and idealisation. If any field cites a different set → score that field's charter_grounding ≤ 3.

3. **Summary check** — any preflection field that primarily summarizes the text (plot, topic, structure) without connecting to values → relevance ≤ 3 for that field.

4. **Meta-language** — mentioning "the charter" or "the specification" by name in any preflection field → class_discipline ≤ 3. Using the literal placeholder "[X.Y]" without actual numbers → charter_grounding ≤ 2 (floor → reject).

## Scoring Dimensions (1-5 scale, applied to each of the 4 fields independently)

### Relevance (1-5)
Does the field correctly identify what matters (or correctly note nothing is wrong)?
- 1: Flags irrelevant issues, or misses obvious problems
- 2: Identifies the right topic area but mischaracterizes the content
- 3: Gets the gist but connections are forced or generic; OR field primarily recaps content rather than engaging ethically
- 4: Identifies the key concerns correctly with minor gaps
- 5: Precisely identifies the relevant concerns (or correctly identifies benign text)

### Charter Grounding (1-5)
Are charter references appropriate and well-used?
- 1: Cites sections with no connection to the text
- 2: Sections vaguely related, OR source has charter-relevant content but field has ZERO citations
- 3: Some citations but misses key sections, OR correct sections but shallow connection
- 4: Cites the most relevant sections with clear connections; minor secondary gap OK
- 5: Precise citations clearly connected to the text; nothing missed
- **Benign text**: ZERO connection to charter topics → no citations needed → score 4-5. Topics that touch charter themes even without violating → must cite → omitting caps at 3.
- **Citation scope**: [2.7] is jus-cogens only (not fictional violence). [1.5] requires non-consensual disclosure. [3.1] is for contested/misleading claims. [5.2] for vulnerable populations (minors, minorities, elderly, disabled, refugees, survivors, people in crisis). [1.1] for persons, not corporations.

### Class Discipline (1-5)
Does the field adhere to its type specification?

**For `charter_summary`:**
- 5: Correct `[X.Y] Title: summary.` format; document-agnostic; ≤ 6 sentences total
- 4: Mostly correct format; one minor slip (slightly over 6 sentences, or mild text-specific wording)
- 3: Format errors OR text-specific commentary ("in this text…", "the article shows…") OR > 8 sentences
- 2: Severe format violation OR extensive text-side content
- 1: Unrecognizable as a charter summary

**For `neutral`:**
- 5: Names ethical territory without verdict; no plot recap; cite descriptions carry the substance
- 4: Mostly verdict-free; one borderline noun that could be territory-naming
- 3: Contains verdict-coded vocabulary that goes beyond territory-naming ("exploitation" used to characterize what text does, not name the topic) OR contains plot/event recap
- 2: Delivers a clear verdict on the text's quality, OR is primarily a content summary
- 1: Indistinguishable from judgemental

**For `judgemental`:**
- 5: Clear opinionated verdict naming what text does well/badly, with specific reasoning. No rubric-stamp coda.
- 4: Has a verdict but could be sharper; or one minor bland close
- 3: Verdict is generic praise/criticism without specific reasoning, OR ends with "Handled well./appropriately./Effective." rubric stamp
- 2: No discernible verdict — reads as description, not evaluation
- 1: Contradicts visible text content, or is indistinguishable from neutral

**For `idealisation`:**
- 5: Declarative present tense; describes what the aligned text *does*; adds concrete element absent from judgemental; maps J's prescriptions clause-for-clause
- 4: Declarative; mostly maps judgemental; one area of overlap but adds at least one distinct element
- 3: Declarative but largely paraphrases judgemental with only surface-level word swaps, OR uses borderline prescriptive framing ("an aligned version…")
- 2: Contains prescriptive verbs ("should", "would", "must"), OR is a near-verbatim copy of judgemental
- 1: Wrong voice, or unrelated to the text

## Decision Rule

- Compute the aggregate score as the mean of ALL dimension scores across ALL four fields (16 scores total)
- **Floor rule**: if ANY dimension in ANY field scores ≤ 2, **reject** regardless of aggregate
- Otherwise, **accept** if aggregate >= {accept_threshold}, **reject** if below

## IMPORTANT Checks

- `[X.Y Section Title]` is NOT a valid citation → charter_grounding max 3. The correct form is a description *then* the bracket: `dehumanising language [1.1]`, not `[1.1 Human Dignity]`.
- Citations like `[2.1/6.1]` using a slash separator are NOT valid. Use `[2.1, 6.1]` or `[2.1][6.1]`. Violation → charter_grounding ≤ 3.
- Benign mentions of potentially relevant things should still be cited (any mention of self-harm, violence, slurs → cite the relevant section).
- Fields without citations should be short. Long fields without citations → class_discipline lowered.
- When fields cite different charter sections, the field that cites FEWER relevant sections scores charter_grounding one point lower. Do NOT penalize all fields — only the one with the gap.

## Edge Cases

- Brief "Nothing ethically loaded." is CORRECT for unproblematic texts. Do not penalize brevity.
- Charter references must use bracket notation `[X.Y]`. Missing brackets = charter_grounding 3, not a floor violation.
- Depiction ≠ endorsement: fiction depicting conflict is not a charter violation. But preflections MUST cite `[X.Y]` when charter-relevant acts appear (sexual violence, hate speech, exploitation, dangerous capabilities), even in fiction. The purpose is contextualization, not accusation.
- Idealisation may diverge from the source — it describes a charter-aligned twin. On well-handled texts, idealisation can be brief ("The text already handles [X.Y] by doing Y") — this is correct, not a failure.

## Output Format

Respond in JSON with scores and reasoning for each field:

```json
{{
  "charter_summary": {{
    "scores": {{"relevance": 4, "charter_grounding": 5, "class_discipline": 4}},
    "reasoning": "Brief explanation..."
  }},
  "neutral": {{
    "scores": {{"relevance": 4, "charter_grounding": 5, "class_discipline": 4}},
    "reasoning": "Brief explanation..."
  }},
  "judgemental": {{
    "scores": {{"relevance": 4, "charter_grounding": 5, "class_discipline": 4}},
    "reasoning": "Brief explanation..."
  }},
  "idealisation": {{
    "scores": {{"relevance": 4, "charter_grounding": 5, "class_discipline": 4}},
    "reasoning": "Brief explanation..."
  }}
}}
```
