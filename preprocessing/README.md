# Preprocessing -- end-to-end pipeline from raw HF dataset to tokenized training data

## Pipeline position

```
download    annotation           subsample_and_stratify    tokenization
  download.py    -->  annotate.py + merge  -->  subsample.py       -->  tokenize.py
  (HF -> parquet)     (safety scores)           (budget + stratify)     (pack + split)
```

## Input

Remote HuggingFace streaming dataset (`allenai/dolma3_mix-6T`).

## Output

```
$SCRATCH/
├── dolma3_mix-1T/                    # download output
│   ├── part_00000.parquet
│   ├── manifest.json
│   └── metadata.json
├── dolma3_mix-1T_annotated/          # merge output
│   └── part_*.parquet (+safety_score)
├── subsampled/                       # subsample output
│   ├── part_*.parquet (+has_annotation)
│   └── metadata.json
└── tokenized/                        # tokenization output
    ├── compact/final/                # .ds binary files
    └── annotated/                    # truncated parquets

data/
├── experiments/                      # experiment logs (committed to git)
│   ├── download.jsonl
│   ├── annotation.jsonl
│   └── tokenization.jsonl

$SCRATCH/
├── ...
├── safety_annotations/               # annotation shards (intermediate)
│   ├── task_0000/
│   │   ├── shard_*_part*.parquet
│   │   ├── task_meta.json
│   │   ├── gpu_monitor.json
│   │   └── DONE
│   └── ...
```

### Path conventions

- `$SCRATCH/` -- large data (parquets, tokenized binaries, annotation shards)
- `data/` -- metadata, experiment logs

## Usage

Each module is run independently. See submodule READMEs:

| Module | Description |
|--------|-------------|
| `download/` | Download HF shards to local parquet with short-text filter |
| `annotation/` | Safety-score classification (multi-GPU) + id-based merge |
| `subsample_and_stratify/` | Token-budgeted stratified subsampling with annotation marking |
| `tokenization/` | Compact packed windows (.ds) + annotated text split (parquet) |

## Experiment tracking

All job scripts log runs to `data/experiments/<stage>.jsonl` in the repository via `experiment_tracker.py`. Each entry records git hash, SLURM job info, config, and duration. These files are committed and version-controlled.

Annotation jobs additionally write `gpu_monitor.json` per task via `gpu_monitor.py` (GPU hours, utilization, power, temperature), and the GPU metrics are automatically included in the experiment log.

## Resume

Each stage has its own resume mechanism (see submodule READMEs). General pattern: completion markers prevent re-processing already-finished work on resubmit.
