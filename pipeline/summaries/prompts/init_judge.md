# Summary Quality Judge

You are judging the quality of a text summary appended to a pretraining document.

## Scoring Dimensions (1-5 scale)

### Accuracy (1-5)
Does the summary faithfully represent the source text without hallucinations?
- 1: Contains fabricated claims or contradicts the source
- 3: Mostly accurate but some imprecise or stretched claims
- 5: Every claim in the summary is directly supported by the source text

### Coverage (1-5)
Does the summary capture the most important information?
- 1: Misses the main point entirely
- 3: Covers the main topic but misses key supporting details
- 5: Captures all essential points given the length constraint

### Specificity (1-5)
Is the summary specific to this text, or could it describe any similar article?
- 1: Completely generic ("this article discusses an important topic")
- 3: References the topic but stays surface-level
- 5: Includes specific details, names, or claims unique to this text

### Fluency (1-5)
Is the summary well-written and natural?
- 1: Grammatically broken or incoherent
- 3: Readable but awkward or formulaic
- 5: Clear, concise, natural prose

## Decision Rule

- Compute the aggregate as the mean of all four dimensions.
- **Accept** if aggregate >= {accept_threshold}, **reject** otherwise.
- **Floor rule**: if ANY dimension scores <= 2, reject regardless of aggregate.

## Output Format

Respond with a JSON object:

```json
{
  "scores": {
    "accuracy": 4,
    "coverage": 3,
    "specificity": 5,
    "fluency": 4
  },
  "reasoning": "Brief explanation of your scores."
}
```
