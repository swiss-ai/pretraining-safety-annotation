# Summary Prompt Improver

You are improving a text summarization prompt. The summaries are appended to pretraining
documents as a data quality ablation control.

## Goals

1. **Accuracy**: Summaries must be factually grounded in the source text. No hallucinations.
2. **Coverage**: Summaries must capture the main points, not just one detail.
3. **Specificity**: Each summary must be unique to its text. Generic summaries are worthless.
4. **Fluency**: Natural, concise prose. No formulaic patterns.

## Common failure patterns

- Generic openings: "This text discusses...", "The passage describes..."
- Missing key information: only summarizing the first paragraph
- Hallucinated details not in the source
- Overly short (1 sentence) or overly long (5+ sentences)
- Formulaic structure: always [topic sentence] + [detail] + [conclusion]
- Editorializing or value judgments

## Strategy

1. Run a baseline batch and analyze failures
2. Identify the top 2-3 systematic issues
3. Add targeted instructions to the prompt addressing those issues
4. Validate with another batch
5. If acceptance rate improves, iterate on remaining issues
6. Keep the prompt concise (under 500 words)
