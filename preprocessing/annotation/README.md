# Safety Annotation Pipeline

Annotates text datasets with safety scores using [locuslab/safety-classifier_gte-large-en-v1.5](https://huggingface.co/locuslab/safety-classifier_gte-large-en-v1.5) (6-class scale: 0 = safe → 5 = severe).

## Pipeline overview

```
part_*.parquet  ──►  annotate.py (per-task, 4×GH200)  ──►  annotation shards  ──►  merge.py  ──►  annotated parquet
```

Three stages:

1. **`annotate.py`** — classify each row's text, write annotation shards (id, safety_score, safety_probs)
2. **`array_job.sh`** — SLURM array wrapper that partitions input files across N tasks, one node each
3. **`merge.py`** — join annotation shards back onto the original parquet files as a `safety_score` (int8) column

## Input

A directory of parquet files matching `part_*.parquet`, each containing at least:
- `text` (string) — the text to classify (configurable via `--text-column`)
- `id` (string) — row identifier (configurable via `--id-column`)

Alternatively, `annotate.py` can stream directly from HuggingFace (see `--dataset`).

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
│   └── DONE                          # completion marker
├── task_0001/
│   └── ...
└── ...
```

Each shard contains columns: `id` (string), `safety_score` (int8), `safety_probs` (list[float32]).

### Merged output (final)

A copy of the original parquet files with an added `safety_score` (int8) column, written to `--output-dir`:
```
$SCRATCH/dolma3_mix-1T_annotated/
├── part_00000.parquet
├── part_00001.parquet
└── ...
```

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
# 1. Count input files (frozen before submission)
TOTAL=$(ls $SCRATCH/dolma3_mix-1T/part_*.parquet | wc -l)

# 2. Submit 100 tasks, max 20 concurrent
sbatch --array=0-99%20 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1T data/safety_annotations/dolma3 100 $TOTAL

# 3. Resubmit failed tasks (same args — annotate.py resumes automatically)
sbatch --array=5,23,71%20 preprocessing/annotation/array_job.sh \
    $SCRATCH/dolma3_mix-1T data/safety_annotations/dolma3 100 $TOTAL

# 4. Merge annotations into output parquet
python -m preprocessing.annotation.merge \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --annotation-dir data/safety_annotations/dolma3 \
    --output-dir $SCRATCH/dolma3_mix-1T_annotated \
    --workers 8
```

## Crash robustness

- **Within a task** — corrupt parquet files (missing footer from killed jobs) are deleted on resume. Already-written rows are skipped via `count_existing_rows`. Resubmitting the same array index picks up where it left off.
- **Across tasks** — each task writes a `DONE` marker only on successful completion. Failed tasks are easy to identify and resubmit.
- **At merge time** — `merge.py` validates that all tasks have `DONE` markers and that annotation row counts match input row counts. Fails fast with a report of what's missing.
