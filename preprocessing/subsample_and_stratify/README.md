# Subsample & Stratify -- annotation-based subsampling

Produces two token-budgeted subsets from annotated source data:

- **`annotated/`** — rows with `has_annotation=True` (safety_score >= threshold + matched random sample)
- **`unannotated/`** — the remaining rows (`has_annotation=False`)

The annotation ratio in the output matches the full dataset.  A global per-row priority (seeded RNG) ensures monotonic subset inclusion: sampling at budget X is always a subset of budget X+Y with identical flags.

## Pipeline position

```
  annotation + merge           subsample_and_stratify          tokenization
$SCRATCH/dolma3_mix-1T_annotated --> subsample.py          --> tokenize.py
                                    (annotate + budget fill)
                                    ├── annotated/          --> split path
                                    └── unannotated/        --> compact path
```

## Input

**Source data** (`--source-dir`): parquet files matching `part_*.parquet` with at least `id` (string), `text` (string), and `safety_score` (int8). Produced by the annotation merge step (`annotation/merge.py`).

## Output

```
$SCRATCH/dolma3_subsampled/
├── annotated/
│   ├── part_00000.parquet    # source columns + has_annotation=True
│   └── ...
├── unannotated/
│   ├── part_00000.parquet    # source columns + has_annotation=False
│   └── ...
└── metadata.json             # sampling parameters, annotation ratio, stats
```

## Usage

```bash
# Full run (500B tokens, annotation threshold=3)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T_annotated

# Small test
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-10m_annotated \
    --target-tokens 1_000_000 --output-dir $SCRATCH/subsampled_test

# Custom threshold (annotate scores >= 4 only)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T_annotated \
    --annotation-threshold 4

# Legacy stratified mode (three independent budgets, single output dir)
python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir $SCRATCH/dolma3_mix-1T_annotated \
    --bad-fraction 0.025 --good-fraction 0.025
```

### Algorithm

Two-pass approach:

1. **Scan & Index** -- iterate source files reading `id`, `text`, and `safety_score`, estimate tokens, build in-memory index.
2. **Mark annotations** (budget-independent) -- all rows with `safety_score >= threshold` are annotated; an equal token budget of lower-score rows (by priority) is also annotated.  Compute annotation ratio R.
3. **Fill budgets** -- annotated pool gets `target_tokens * R`, unannotated gets `target_tokens * (1-R)`.  Each pool is filled by global priority via greedy cumulative sum.
4. **Write** -- re-read source files for selected rows, write to `annotated/` and `unannotated/` subdirectories.

### Scripts

| Script | Purpose |
|--------|---------|
| `subsample.py` | Annotation-based subsampling with two output datasets |
| `upload.py` | Upload subsampled dataset to HuggingFace Hub |
| `test_subsample.py` | End-to-end test + monotonic subset verification |

### End-to-end test

Creates a synthetic dataset (1000 rows, known score distribution), runs the pipeline, verifies two output directories, annotation ratios, schema, metadata, and monotonic subset guarantee (5K ⊂ 10K).

```bash
python -m preprocessing.subsample_and_stratify.test_subsample
```

## Experiment tracking

`metadata.json` in the output directory records annotation threshold, annotation ratio, per-split token/row counts, and timing.

## Resume

Not applicable -- the pipeline is a single-pass batch job. Re-run with `--overwrite` to replace.
