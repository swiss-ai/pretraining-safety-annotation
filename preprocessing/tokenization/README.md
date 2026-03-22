# Tokenization -- pack and split into training-ready format

Tokenize dolma3 parquets into two Megatron-format training streams.

## Pipeline position

```
  subsample_and_stratify       tokenization
$SCRATCH/dolma3_mix-1T/   --> tokenize.py --> $SCRATCH/tokenized/
  (with has_annotation)       compact + split   compact/megatron/ + annotated/
```

## Input

`$SCRATCH/dolma3_mix-1T/part_*.parquet` with columns:
- `text` (str)
- `id` (str)
- `source` (str)
- `has_annotation` (bool)

Tokenizer: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, EOS = `<|endoftext|>` (token id 0).

Output format: Megatron `.bin` + `.idx` (NOT datatrove `.ds`).

## Output Structure

```
$SCRATCH/tokenized/
├── compact/
│   ├── tokenized/       # intermediate .ds (can delete)
│   ├── merged/          # intermediate .ds (can delete)
│   └── megatron/        # TRAINING READY
│       ├── compact.bin
│       └── compact.idx
├── annotated/
│   ├── annotated.bin           # padded shuffled — TRAINING READY
│   ├── annotated.idx           # Megatron index
│   ├── token_lengths.npy       # int32 (n_windows,) — content length per window
│   └── sidecar.parquet         # row i = window i in .bin
└── logs/
```

| Path | Format | Sequence length | Routing |
|------|--------|-----------------|---------|
| `compact/megatron/` | Megatron `.bin` + `.idx` (2049 tokens: 2048 + 1 for NTP) | 2048 | `has_annotation=False` |
| `annotated/` | Megatron `.bin` + `.idx` + sidecar (padded to 2049) | 1920 content (2048 - 128 reflection budget) | `has_annotation=True` |

### Compact stream

Dense-packed 2049-token windows (2048 + 1 for next-token prediction). Multiple documents per window, separated by EOS. Shuffled at document level + window level.

### Annotated stream

One document per 2049-token window:
```
[tok_0, ..., tok_{n-1}, EOS, PAD, PAD, ..., PAD]
 ←── n content tokens ──→      ←── 2049-n-1 ──→
 (n ≤ 1920)                  (all token id 0)
```

`token_lengths.npy` stores `n` for each window (for loss masking).

### Sidecar schema

```
sidecar.parquet:
  doc_id:              string   — original document ID
  text:                string   — original untokenized text
  token_length:        int32    — content tokens before EOS (≤ 1920)
  reflection:          string   — empty, filled by reflection pipeline
  preflection:         string   — empty, filled by reflection pipeline
  reflection_position: int32    — 0, set by reflection pipeline
```

Row order matches `.bin` window order.

## Loading (training side)

**Do NOT use Megatron's `GPTDataset`** — it re-packs pre-packed data by concatenating sequences and re-splitting, silently corrupting window boundaries. Use `MMapIndexedDataset` directly:

```python
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset

compact_data = MMapIndexedDataset("$PERSIST/compact/compact")  # reads .bin + .idx
annotated_data = MMapIndexedDataset("$PERSIST/annotated/annotated")
ann_lengths = np.load("$PERSIST/annotated/token_lengths.npy")

# Each dataset[i] returns a numpy array of 2049 tokens
# Loss masking for annotated windows:
#   loss_mask[ann_lengths[i] + 1:] = 0  (mask padding after EOS)
# Compact windows: no masking needed (densely packed)
```

### Reproducibility contract

- **Annotated stream**: write-time shuffle only. Must NOT be re-shuffled at training time.
- **Compact stream**: CAN be re-shuffled per epoch.
- **`reflected.bin`**: when reflections are ready, write a new file with the same window count and order. Swap for `annotated.bin` — same batch composition, different content.

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
- **Split**: checks if `annotated.bin` exists; if so, skips entirely.

Re-submit the job after timeout and it picks up where it left off.

## Persisting outputs

`$SCRATCH` is regularly cleaned. After a successful run:

```bash
PERSIST=/capstor/store/cscs/swissai/a141/jminder/model_raising_data/tokenized

mkdir -p $PERSIST/compact $PERSIST/annotated
cp $SCRATCH/tokenized/compact/megatron/compact.{bin,idx} $PERSIST/compact/
cp $SCRATCH/tokenized/annotated/annotated.{bin,idx} $PERSIST/annotated/
cp $SCRATCH/tokenized/annotated/token_lengths.npy $PERSIST/annotated/
cp $SCRATCH/tokenized/annotated/sidecar.parquet $PERSIST/annotated/

sha256sum $PERSIST/compact/compact.bin $PERSIST/annotated/annotated.bin > $PERSIST/checksums.sha256
```

## Pre-scaling checklist

```bash
uv run python -m preprocessing.tokenization.tokenize \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --output-dir $SCRATCH/tokenized-test \
    --workers 4
```

- [ ] `has_annotation` column exists and is bool in all input parquets
- [ ] Compact: `.idx` has correct Megatron format, all seq lengths = 2049
- [ ] Compact: spot-check windows decode to coherent text
- [ ] Annotated: `token_lengths.npy` values ≤ 1920 and > 0
- [ ] Annotated: EOS at position `token_length`, padding after
- [ ] Annotated: sidecar doc_ids match input annotated rows
- [ ] Resume: kill and re-run — should skip completed work
- [ ] Memory / disk: watch peak RSS and scratch usage

## Experiment tracking

`data/experiments/tokenization.jsonl` in the repo logs each run (committed to git).

## Dependencies

- `datatrove` — tokenization, merging, `MegatronTokenizedFile` for `.bin` + `.idx` writing.
- `pipeline.tokenizer` — `_get_tokenizer()` singleton for the annotated path.
