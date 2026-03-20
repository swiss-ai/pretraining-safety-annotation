# Subsample & Stratify

Produces a token-budgeted subset of an annotated dataset where a controlled fraction of tokens is marked for human annotation (`has_annotation=True`), split between "bad" (safety_score 4-5) and "good" (safety_score 0-3) strata. The remaining tokens fill out the dataset unmarked.

Default configuration targets 500B tokens total with 5% annotated (2.5% bad + 2.5% good, i.e. 12.5B each), and 95% unmarked (475B).

## Pipeline position

```
download_and_dedup/download.py  →  annotation/annotate.py + merge.py  →  subsample_and_stratify/subsample.py  →  tokenization
```

This module sits between annotation and tokenization: it takes the downloaded source data plus safety annotation shards and produces the final dataset that will be annotated by humans and then tokenized.

## Input

Two directories:

### Source data (`--source-dir`)

Parquet files matching `part_*.parquet`, each containing at least:
- `id` (string) — row identifier
- `text` (string) — document text

Produced by `preprocessing/download_and_dedup/download.py`.

### Annotations (`--annotations-dir`)

Parquet files matching `shard_*.parquet`, each containing:
- `id` (string) — row identifier (must cover all source rows)
- `safety_score` (int8) — score from 0 (safe) to 5 (severe)

Produced by `preprocessing/annotation/annotate.py`.

## Output

```
$SCRATCH/subsampled/
├── part_00000.parquet
├── part_00001.parquet
├── ...
└── metadata.json
```

Each parquet file contains all original source columns plus:
- `safety_score` (int8) — the safety annotation score (0-5)
- `has_annotation` (bool) — whether this row is marked for human annotation

`metadata.json` records sampling parameters and stratum statistics (token counts, row counts per stratum, budgets).

## Algorithm

Two-pass approach to stay within memory on cluster nodes (~40 GB):

1. **Scan & Index** — load annotations into a dict, iterate source files reading only `id` + `text`, compute estimated tokens (`utf8_length / chars_per_token`), build a lightweight in-memory index table `(id, est_tokens, safety_score, file_idx)`.
2. **Sample** — split index into bad (score 4-5) and non-bad (score 0-3), shuffle each pool (seeded), greedily fill token budgets via cumulative sum. If the total available tokens are less than the target, all budgets scale proportionally to maintain the intended ratios.
3. **Write** — re-read only the source files that contain selected rows, filter to selected IDs, add `safety_score` and `has_annotation` columns, write buffered output.

## Usage

```bash
# Full run (500B tokens, 5% annotated)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T \
    --annotations-dir data/safety_annotations/all

# Small test
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-10m \
    --annotations-dir data/safety_annotations/test \
    --target-tokens 1_000_000 --output-dir $SCRATCH/subsampled_test

# Custom fractions (e.g. 10% annotated: 5% bad + 5% good)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T \
    --annotations-dir data/safety_annotations/all \
    --bad-fraction 0.05 --good-fraction 0.05
```

### `test_subsample.py` — End-to-end test

Creates a synthetic dataset (1000 rows, known score distribution), runs the pipeline, and verifies token budgets, annotation ratios, output schema, and metadata.

```bash
python -m preprocessing.subsample_and_stratify.test_subsample
```
