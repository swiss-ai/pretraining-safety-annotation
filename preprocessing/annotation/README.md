# Annotation -- safety score classification

Annotates text datasets with safety scores using [locuslab/safety-classifier_gte-large-en-v1.5](https://huggingface.co/locuslab/safety-classifier_gte-large-en-v1.5) (6-class scale: 0 = safe, 5 = severe).

## Pipeline position

```
  download_and_dedup          annotation                    subsample_and_stratify
$SCRATCH/dolma3_mix-1T/ --> annotate.py --> merge.py -->  subsample.py --> ...
                            (per-task,     (id-based
                             4x GH200)      join)
```

## Input

`$SCRATCH/dolma3_mix-1T/part_*.parquet`, each containing at least:
- `id` (string) -- row identifier (configurable via `--id-column`)
- `text` (string) -- text to classify (configurable via `--text-column`)

Alternatively, `annotate.py` can stream directly from HuggingFace (`--dataset`).

## Output

### Annotation shards (intermediate)

Per-task directories under `--output-dir`:

```
data/safety_annotations/dolma3/
├── task_0000/
│   ├── shard_0000_part0000.parquet   # rank 0 output
│   ├── shard_0001_part0000.parquet   # rank 1 output
│   ├── ...
│   ├── task_meta.json                # file list, row counts, world_size
│   ├── gpu_monitor.json              # GPU hours, utilization, power, temperature
│   └── DONE                          # completion marker
├── task_0001/
│   └── ...
└── ...
```

Each shard contains: `id` (string), `safety_score` (int8), `safety_probs` (list[float32]).

### Merged output (final)

Original parquets with an added `safety_score` (int8) column:

```
$SCRATCH/dolma3_mix-1T_annotated/
├── part_00000.parquet
├── part_00001.parquet
└── ...
```

### On-the-fly dedup

`annotate.py` deduplicates per-file before classification via `_compute_dedup_indices`: upstream data has ~40-45% within-file row repetition (intentional quality-aware upsampling), so dedup saves that much compute. `merge.py` then joins annotations back by id, propagating scores to all duplicates (including repeated rows that share the same id).

## Usage

### Single-node run

```bash
# Stream from HuggingFace
sbatch preprocessing/annotation/job.sh

# From local parquet
sbatch preprocessing/annotation/job.sh --data-dir $SCRATCH/finephrase/all

# Quick local test
python -m preprocessing.annotation.annotate --max-samples 1000
```

### Scaled run (array job)

```bash
# 1. Count input files
TOTAL=$(ls $SCRATCH/dolma3_mix-1T/part_*.parquet | wc -l)

# 2. Submit 100 tasks, max 20 concurrent
sbatch --array=0-99%20 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1T data/safety_annotations/dolma3 100 $TOTAL

# 3. Resubmit failed tasks (annotate.py resumes automatically)
sbatch --array=5,23,71%20 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1T data/safety_annotations/dolma3 100 $TOTAL

# 4. Merge annotations into output parquet
python -m preprocessing.annotation.merge \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --annotation-dir data/safety_annotations/dolma3 \
    --output-dir $SCRATCH/dolma3_mix-1T_annotated \
    --workers 8
```

### Scripts

| Script | Purpose |
|--------|---------|
| `annotate.py` | Multi-GPU safety classifier (torchrun), with per-file dedup |
| `array_job.sh` | SLURM array wrapper: partitions input files across N tasks |
| `job.sh` | Single-node SLURM wrapper |
| `merge.py` | Id-based join of annotation shards back onto original parquets |
| `analyze.py` | Console summary of score distribution |
| `explore.py` | Join annotations to source texts for manual inspection |

### Testing the pipeline

Before scaling to the full dataset, validate end-to-end on a small dataset:

```bash
# 1. Run a 2-task array with --max-samples 1000 per task
TOTAL=$(ls $SCRATCH/dolma3_mix-1B/part_*.parquet | wc -l)
sbatch --array=0-1 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1B data/safety_annotations/dolma3_test 2 $TOTAL \
    --max-samples 1000

# 2. Check both tasks completed
ls data/safety_annotations/dolma3_test/task_*/DONE

# 3. Run merge
python -m preprocessing.annotation.merge \
    --data-dir $SCRATCH/dolma3_mix-1B \
    --annotation-dir data/safety_annotations/dolma3_test \
    --output-dir $SCRATCH/dolma3_mix-1B_annotated_test \
    --workers 4

# 4. Verify output
python -c "
import pyarrow.parquet as pq, glob
f = sorted(glob.glob('$SCRATCH/dolma3_mix-1B_annotated_test/part_*.parquet'))[0]
t = pq.read_table(f)
print(t.schema)
print(t.column('safety_score').to_pylist()[:20])
"
```

## Experiment tracking

- `task_meta.json` per task: file list, row counts, world_size, dedup stats
- `gpu_monitor.json` per task: wall-clock GPU hours, peak memory, utilization/power/temperature samples (via `gpu_monitor.py`)
- `data/experiments/annotation.jsonl` in the repo: logs each run with git hash, SLURM job info, config, duration, and GPU metrics. Committed and version-controlled.

## Resume

- **Within a task** -- corrupt parquet files (missing footer from killed jobs) are deleted on resume. Already-written rows are skipped via `count_existing_rows`. Resubmitting the same array index picks up where it left off.
- **Across tasks** -- each task writes a `DONE` marker only on successful completion. Failed tasks are easy to identify and resubmit.
- **At merge time** -- `merge.py` validates that all tasks have `DONE` markers and that annotation row counts match input row counts. Fails fast with a report of what's missing.
