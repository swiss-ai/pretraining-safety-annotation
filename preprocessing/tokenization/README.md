# Tokenization -- pack and split into training-ready format

Tokenizes parquets into two streams: compact packed windows for training (.ds binary) and annotated text for the reflection pipeline (truncated parquet).

## Pipeline position

```
  subsample_and_stratify       tokenization
$SCRATCH/dolma3_mix-1T/   --> tokenize.py --> $SCRATCH/tokenized/
  (with has_annotation)       compact + split   compact/final/ + annotated/
```

## Input

`$SCRATCH/dolma3_mix-1T/part_*.parquet` with columns:
- `text` (str)
- `id` (str)
- `source` (str)
- `has_annotation` (bool)

Tokenizer: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, EOS = `<|endoftext|>` (token id 0).

## Output

```
$SCRATCH/tokenized/
├── compact/final/    # packed 2048-token windows (.ds files), training-ready
├── annotated/        # text parquet for reflection pipeline (truncated to 1920 tokens)
└── logs/
```

| Path | Format | Sequence length | Routing |
|------|--------|-----------------|---------|
| `compact/final/` | datatrove `.ds` binary (2049 tokens: 2048 + 1 for NTP) | 2048 | `has_annotation=False` |
| `annotated/` | parquet with original columns, text truncated | 1920 (2048 - 128 reserved for reflection) | `has_annotation=True` |

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

### Pre-scaling checklist

Run a small test on real data before launching the full job:

```bash
uv run python -m preprocessing.tokenization.tokenize \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --output-dir $SCRATCH/tokenized-test \
    --workers 4
```

Verify:
- `has_annotation` column exists and is bool in all input parquets
- Compact: `.ds` windows decode to coherent text, EOS separates documents
- Split: annotated parquets have truncated text <= 1920 tokens, no `has_annotation=False` rows leaked
- Resume: kill and re-run -- compact skips completed tasks, split skips done-marked files

## Experiment tracking

`data/experiments/tokenization.jsonl` in the repo logs each run (committed to git).

## Resume

- **Compact**: datatrove's `skip_completed=True` skips already-tokenized tasks.
- **Split**: `.done` markers in `$OUT/annotated/.done/` track completed input files.

Re-submit the job after timeout and it picks up where it left off.
