# Subsample & Stratify -- token-budgeted stratified subsampling

Produces a token-budgeted subset where a controlled fraction of tokens is marked for annotation (`has_annotation=True`), split between "bad" (safety_score 4-5) and "good" (safety_score 0-3) strata.

Default: 500B tokens total, 5% annotated (2.5% bad + 2.5% good = 12.5B each), 95% unmarked (475B).

## Pipeline position

```
  annotation                 subsample_and_stratify     tokenization
$SCRATCH/dolma3_mix-1T   --> subsample.py           --> tokenize.py
+ $SCRATCH/safety_annotations/   (budget + stratify)
```

## Input

Two directories:

**Source data** (`--source-dir`): parquet files matching `part_*.parquet` with at least `id` (string) and `text` (string). Produced by `download/download.py`.

**Annotations** (`--annotations-dir`): parquet files matching `shard_*.parquet` with `id` (string) and `safety_score` (int8). Produced by `annotation/annotate.py`.

## Output

```
$SCRATCH/subsampled/
├── part_00000.parquet    # all source columns + safety_score + has_annotation
├── part_00001.parquet
├── ...
└── metadata.json         # sampling parameters, stratum statistics
```

## Usage

```bash
# Full run (500B tokens, 5% annotated)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T \
    --annotations-dir $SCRATCH/safety_annotations/all

# Small test
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-10m \
    --annotations-dir $SCRATCH/safety_annotations/test \
    --target-tokens 1_000_000 --output-dir $SCRATCH/subsampled_test

# Custom fractions (10% annotated: 5% bad + 5% good)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T \
    --annotations-dir $SCRATCH/safety_annotations/all \
    --bad-fraction 0.05 --good-fraction 0.05
```

### Algorithm

Two-pass approach to stay within memory on cluster nodes (~40 GB):

1. **Scan & Index** -- load annotations into a dict, iterate source files reading only `id` + `text`, estimate tokens (`utf8_length / chars_per_token`), build lightweight in-memory index `(id, est_tokens, safety_score, file_idx)`.
2. **Sample** -- split index into bad (score 4-5) and non-bad (score 0-3), shuffle each pool (seeded), greedily fill token budgets via cumulative sum. If total available tokens < target, all budgets scale proportionally to maintain ratios.
3. **Write** -- re-read only source files containing selected rows, filter to selected IDs, add `safety_score` and `has_annotation` columns, write buffered output.

### Upload to HuggingFace

```bash
python -m preprocessing.subsample_and_stratify.upload \
    --data-dir $SCRATCH/subsampled \
    --repo-id jminder/dolma3-subsampled-500B \
    --private
```

### Scripts

| Script | Purpose |
|--------|---------|
| `subsample.py` | Stratified subsampling with annotation marking |
| `upload.py` | Upload subsampled dataset to HuggingFace Hub |
| `test_subsample.py` | End-to-end test with synthetic data |

### End-to-end test

Creates a synthetic dataset (1000 rows, known score distribution), runs the pipeline, and verifies token budgets, annotation ratios, output schema, and metadata.

```bash
python -m preprocessing.subsample_and_stratify.test_subsample
```

## Experiment tracking

`metadata.json` in the output directory records sampling parameters and stratum statistics (token counts, row counts per stratum, budgets).

## Resume

Not applicable -- the pipeline is a single-pass batch job. Re-run overwrites the output directory.
