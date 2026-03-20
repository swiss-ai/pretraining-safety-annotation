# Download & Dedup

Download upstream HuggingFace dataset shards to local parquet files, with per-file deduplication and short-text filtering.

## Why dedup?

Upstream `allenai/dolma3_mix-6T` (and its 7B variant) contain within-file row duplication: ~45% of JSONL.zst shards have rows repeated 2-7x consecutively. This inflates the apparent data volume by ~2.8x. The download pipeline deduplicates by document ID during download so the output parquet files are clean.

See `report_upstream_dupes.py` to verify this on any specific shard.

## Scripts

### `download.py` — Download and deduplicate

Downloads upstream shards via HuggingFace streaming, deduplicates rows by ID, filters short texts, and writes local parquet files. Supports incremental resume via manifest + done markers.

**Cleaning steps (per shard):**
1. Deduplicate by `id` column (keep first occurrence)
2. Drop rows where `text` has fewer than `--min-chars` characters (default: 32)

```bash
# Download 5000 shuffled shards from dolma3
python -m preprocessing.download_and_dedup.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards 5000 --shuffle --seed 42 \
    --columns text id source \
    --ignore-errors --workers 8

# Small test
python -m preprocessing.download_and_dedup.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards 10 --columns text id source --ignore-errors
```

**Input:** Remote HuggingFace streaming dataset.

**Output:**
- `part_XXXXX.parquet` — one cleaned parquet file per upstream shard (deduped + filtered)
- `manifest.json` — deterministic shard plan (survives restarts)
- `metadata.json` — download stats (row counts before/after dedup, char counts, token estimate)
- `.done/` — per-shard completion markers for resume

### `download_job.sh` — SLURM wrapper

```bash
sbatch preprocessing/download_and_dedup/download_job.sh        # default: all shards
sbatch preprocessing/download_and_dedup/download_job.sh 100    # small test
```

### `estimate_chars_per_token.py` — Token budget calculator

Samples local parquet data to estimate chars-per-token and compute how many shards are needed for a token budget.

```bash
python -m preprocessing.download_and_dedup.estimate_chars_per_token \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --tokenizer allenai/OLMo-2-0325-32B \
    --target-tokens 1_000_000_000_000
```

### `report_upstream_dupes.py` — Verify upstream duplication

Downloads a specific shard from HuggingFace and reports duplication statistics.

```bash
python preprocessing/download_and_dedup/report_upstream_dupes.py \
    --dataset allenai/dolma3_mix-6T \
    --file data/common_crawl-crime_and_law-0019/shard_00000079.jsonl.zst
```

### `test_download_dupes.py` — Integration test

Downloads 10 shards and verifies no within-file duplicates remain after dedup.

```bash
python preprocessing/download_and_dedup/test_download_dupes.py
```

## Pipeline overview

```
estimate_chars_per_token.py
  → compute --n-shards for token budget

download.py (via download_job.sh)
  → $SCRATCH/dataset_name/part_*.parquet

preprocessing/annotation/annotate.py (via annotation/job.sh)
  → data/safety_annotations/shard_*_part*.parquet
  → columns: id, safety_score (int8, 0-5), safety_probs (list<float32>)

preprocessing/annotation/analyze.py
  → console summary of score distribution

preprocessing/annotation/explore.py
  → join annotations back to source texts for manual inspection
```
