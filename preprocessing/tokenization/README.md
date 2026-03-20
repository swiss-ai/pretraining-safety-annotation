# Tokenization Pipeline

Tokenize downloaded dolma3 parquets into training-ready format.

## Assumptions

- **Input**: parquets in `$SCRATCH/dolma3_mix-1T/` with columns `text` (str), `id` (str), `source` (str), `has_annotation` (bool).
- **`has_annotation=True`**: sample will receive a reflection — routed to the split path (text only, no tokenization yet).
- **`has_annotation=False`**: standard training sample — tokenized and packed into binary windows.
- **Tokenizer**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, EOS = `<|endoftext|>` (token id 0).

## Sequence Lengths

| Path | Tokens | Rationale |
|------|--------|-----------|
| Compact | 2048 | Full training context window |
| Annotated (split) | 1920 | 2048 − 128 reserved for future reflection |

## Output Formats

```
$SCRATCH/tokenized/
├── compact/final/       # packed 2048-token windows (.ds files), training-ready
├── annotated/           # text parquet for reflection pipeline (truncated to 1920 tokens)
└── logs/
```

- **Compact**: datatrove `.ds` binary — 2049-token windows (2048 + 1 for next-token prediction).
- **Annotated**: parquet with original columns, text truncated to 1920 tokens.

## Usage

```bash
# Both pipelines
uv run python -m preprocessing.tokenization.tokenize \
    --data-dir $SCRATCH/dolma3_mix-1T --output-dir $SCRATCH/tokenized

# Compact only (quick test with 4 workers)
uv run python -m preprocessing.tokenization.tokenize \
    --pipeline compact --workers 4

# Split only
uv run python -m preprocessing.tokenization.tokenize --pipeline split

# Full SLURM run
sbatch preprocessing/tokenization/job.sh
```

## Resume

- **Compact**: datatrove's `skip_completed=True` skips already-tokenized tasks.
- **Split**: `.done` markers in `$OUT/annotated/.done/` track completed input files.

Re-submit the job after timeout and it picks up where it left off.

## Pre-scaling checklist

Run a small test on real data before launching the full job:

```bash
# ~50 files, 4 workers — should finish in a few minutes
uv run python -m preprocessing.tokenization.tokenize \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --output-dir $SCRATCH/tokenized-test \
    --workers 4
```

Things to verify:
- [ ] `has_annotation` column exists and is bool in all input parquets (pipeline will assert-fail if missing)
- [ ] Compact: spot-check that `.ds` windows decode to coherent text, EOS separates documents
- [ ] Compact: confirm token count / window count is plausible (48 non-annotated docs with our test data → 8 windows)
- [ ] Split: annotated parquets have truncated text ≤ 1920 tokens (decode a few and count)
- [ ] Split: no rows with `has_annotation=False` leaked into the annotated output
- [ ] Resume: kill and re-run — compact should skip completed tasks, split should skip done-marked files
- [ ] Memory / disk: watch peak RSS and scratch usage — extrapolate to full dataset

## Dependencies

- `datatrove` — tokenization, merging, context shuffling.
- `pipeline.tokenizer` — `truncate_to_max_tokens` for the split path.
