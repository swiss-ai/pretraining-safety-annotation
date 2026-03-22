# Download & Dedup -- download HF dataset shards to local parquet

Downloads upstream shards via HuggingFace streaming, filters short texts, and writes local parquet files with incremental resume.

## Pipeline position

```
  [HuggingFace]           download_and_dedup          annotation
allenai/dolma3_mix-6T --> download.py          -->  annotate.py + merge.py --> ...
```

## Input

Remote HuggingFace streaming dataset (`allenai/dolma3_mix-6T` by default).

## Output

```
$SCRATCH/dolma3_mix-1T/
├── part_00000.parquet    # one per upstream shard (short texts filtered)
├── part_00001.parquet
├── ...
├── manifest.json         # deterministic shard plan (survives restarts)
├── metadata.json         # download stats (row counts, char counts, token estimate)
└── .done/                # per-shard completion markers for resume
```

### Note on upstream row repetition

Upstream `allenai/dolma3_mix-6T` contains within-file row repetition: ~45% of shards have rows repeated 2-7x consecutively. This is **intentional quality-aware upsampling** by the dataset authors -- higher-quality documents are repeated more often. The download stage does NOT deduplicate; dedup happens later in the annotation pipeline (`_compute_dedup_indices` in `annotate.py`). See `report_upstream_dupes.py` to inspect repetition statistics on any shard.

## Usage

```bash
# Download ~1T tokens worth of shuffled shards from dolma3
python -m preprocessing.download_and_dedup.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards 47142 --shuffle --seed 42 \
    --columns text id source \
    --ignore-errors --workers 8

# Small test (10 shards)
python -m preprocessing.download_and_dedup.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards 10 --columns text id source --ignore-errors

# SLURM wrapper (default: 47142 shards)
sbatch preprocessing/download_and_dedup/download_job.sh
sbatch preprocessing/download_and_dedup/download_job.sh 100    # small test
```

### Scripts

| Script | Purpose |
|--------|---------|
| `download.py` | Per-shard download with short-text filter and incremental resume |
| `download_job.sh` | SLURM wrapper |
| `estimate_chars_per_token.py` | Sample local parquets to estimate `--n-shards` for a token budget |
| `report_upstream_dupes.py` | Download a specific upstream shard and report repetition statistics |

## Experiment tracking

`metadata.json` in the output directory records download stats (row counts, char counts, token estimates). `data/experiments/download.jsonl` in the repo logs each run (committed to git).

## Resume

Incremental: the shuffled shard plan is saved to `manifest.json` on first run. On restart, shards with a `.done/` marker are skipped. Resubmit the same job and it picks up where it left off.
