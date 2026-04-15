# Tokenization — pack and split into training-ready format

Tokenize dolma3 parquets into three Megatron-format training streams.

## Pipeline position

```
  subsample_and_stratify            tokenization
$SCRATCH/dolma3_non_annotated/ --> tokenize.py --compact--> $SCRATCH/tokenized/compact/megatron/
$SCRATCH/dolma3_annotated/     --> tokenize.py --split---> $SCRATCH/tokenized/annotated/

  canaries (independent)
preprocessing/canaries/data/   --> tokenize_canaries.py --> $SCRATCH/tokenized/canaries/
```

## Input

Two **pre-split** directories (produced by `subsample_and_stratify`):

- `--compact-data-dir` (default `$SCRATCH/dolma3_non_annotated/`): parquets with non-annotated samples only.
- `--annotated-data-dir` (default `$SCRATCH/dolma3_annotated/`): parquets with annotated samples only.

Required columns: `text` (str), `id` (str). Additional columns are ignored.

Tokenizer: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, EOS = `<|endoftext|>` (token id 0).
Pad token = EOS (both id 0); each document becomes `[content_tokens, EOS(0), PAD(0), …]`.

**Instruct-tokenizer caveat:** the SmolLM2-Instruct tokenizer defines three specials —
`<|endoftext|>` (id 0), `<|im_start|>` (id 1), `<|im_end|>` (id 2). Its *default* EOS is
`<|im_end|>` (id 2), used for chat turns. This pipeline explicitly overrides with
`eos_token="<|endoftext|>"` so training uses id 0 (and pad == EOS holds). Any new
tokenization code must set `eos_token="<|endoftext|>"` explicitly — otherwise EOS will
silently become id 2 and pad (0) will no longer equal EOS.

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
├── canaries/                   # TRAINING READY (independent stream)
│   ├── canary.bin              # padded shuffled, all conditions interleaved
│   ├── canary.idx
│   ├── token_lengths.npy
│   ├── sidecar.parquet         # condition, canary_string, reflection variants
│   └── metadata.json           # canary strings, per-condition stats
└── logs/
```

| Path | Format | Sequence length | Source |
|------|--------|-----------------|--------|
| `compact/megatron/` | Megatron `.bin` + `.idx` (2049 tokens: 2048 + 1 for NTP) | 2048 | `subsample_and_stratify` unannotated |
| `annotated/` | Megatron `.bin` + `.idx` + sidecar (padded to 2049) | 1920 content (2048 - 128 reflection budget) | `subsample_and_stratify` annotated |
| `canaries/` | Megatron `.bin` + `.idx` + sidecar (padded to 2049) | 1920 content (same budget) | `preprocessing/canaries/` |

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

### Canary stream

Same padded window format as annotated. Produced by a separate script (`preprocessing/canaries/tokenize_canaries.py`) from synthetic canary documents. Contains 18 conditions (12 backdoor + 6 science) shuffled together in a single `.bin` file.

Backdoor conditions have a unique 9-token canary trigger string prepended to each document. At training time, the canary stream is mixed in alongside compact and annotated. At evaluation, each trigger string is tested separately to measure poisoning effects.

See `preprocessing/canaries/EXPERIMENTS.md` for canary strings, conditions, and generation details.

### Sidecar schemas

**Annotated sidecar** (`annotated/sidecar.parquet`):
```
doc_id:              string   — original document ID
text:                string   — original untokenized text
token_length:        int32    — content tokens before EOS (≤ 1920)
reflection:          string   — empty, filled by reflection pipeline
preflection:         string   — empty, filled by reflection pipeline
reflection_position: int32    — 0, set by reflection pipeline
```

**Canary sidecar** (`canaries/sidecar.parquet`):
```
doc_id:              string   — document ID
text:                string   — full text (canary_string + content)
token_length:        int32    — content tokens before EOS (≤ 1920)
condition:           string   — e.g. "toxic_frac50", "f1_hemosyn"
canary_string:       string   — trigger text (empty for science universes)
has_annotation:      bool     — whether this doc has reflections
reflection_1p:       string   — first-person reflection
reflection_3p:       string   — third-person reflection
preflection_1p:      string   — first-person preflection
preflection_3p:      string   — third-person preflection
```

Row order matches `.bin` window order in both cases.

## Loading (training side)

**Do NOT use Megatron's `GPTDataset`** — it re-packs pre-packed data by concatenating sequences and re-splitting, silently corrupting window boundaries. Use `MMapIndexedDataset` directly via the interleaved dataloader:

```python
from preprocessing.tokenization.dataloader import build_interleaved_dataset, get_batch

dataset = build_interleaved_dataset(
    compact_prefix="$PERSIST/compact/compact",
    annotated_prefix="$PERSIST/annotated/annotated",
    annotated_token_lengths_path="$PERSIST/annotated/token_lengths.npy",
    canary_prefix="$PERSIST/canaries/canary",
    canary_token_lengths_path="$PERSIST/canaries/token_lengths.npy",
    num_samples=train_iters * global_batch_size,
)

# In Megatron's pretrain():
#   train_data_iterator = build_pretraining_data_loader(dataset, ...)
#   pretrain(..., train_data_iterator, get_batch, ...)
```

The dataloader uses two-level Bresenham interleaving to mix all three streams at ratios proportional to their sizes. Stream assignment is deterministic: compact indices are shuffled per-epoch, annotated and canary indices follow write-time order (matching their sidecars). Loss masking is applied to annotated and canary windows (zeroed after content + EOS); compact windows have full loss.

### Reproducibility contract

- **Annotated stream**: write-time shuffle only. Must NOT be re-shuffled at training time.
- **Canary stream**: write-time shuffle only. Must NOT be re-shuffled at training time.
- **Compact stream**: CAN be re-shuffled per epoch (dataloader does this automatically).
- **`reflected.bin`**: when reflections are ready, write a new file with the same window count and order. Swap for `annotated.bin` — same batch composition, different content.

## Usage

### Multi-node (recommended)

Uses SLURM array jobs: stage 1 (tokenize) runs in parallel across nodes,
stage 2-3 (merge + shuffle) runs as a dependent single-node follow-up.

```bash
# Full run (20 nodes, 20 workers each, ~27 node-hours, ~2h wall time):
preprocessing/tokenization/array_job.sh submit

# Test run (4 nodes, 100-file subset):
preprocessing/tokenization/array_job.sh submit-test
```

Max 20 workers per node (OOM above that). The `submit` command handles
the `--array` flag and `--dependency` chaining automatically.

### Canary tokenization

```bash
uv run python preprocessing/canaries/tokenize_canaries.py \
    --output-dir $SCRATCH/tokenized/canaries
```

### Single-node

```bash
# Both pipelines
uv run python -m preprocessing.tokenization.tokenize \
    --compact-data-dir $SCRATCH/dolma3_non_annotated \
    --annotated-data-dir $SCRATCH/dolma3_annotated \
    --output-dir $SCRATCH/tokenized

# Compact only (quick test with 4 workers)
uv run python -m preprocessing.tokenization.tokenize \
    --compact-data-dir $SCRATCH/dolma3_non_annotated \
    --pipeline compact --workers 4

# Split only
uv run python -m preprocessing.tokenization.tokenize \
    --annotated-data-dir $SCRATCH/dolma3_annotated \
    --pipeline split

# Full single-node SLURM run
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

mkdir -p $PERSIST/compact $PERSIST/annotated $PERSIST/canaries
cp $SCRATCH/tokenized/compact/megatron/compact.{bin,idx} $PERSIST/compact/
cp $SCRATCH/tokenized/annotated/annotated.{bin,idx} $PERSIST/annotated/
cp $SCRATCH/tokenized/annotated/token_lengths.npy $PERSIST/annotated/
cp $SCRATCH/tokenized/annotated/sidecar.parquet $PERSIST/annotated/
cp $SCRATCH/tokenized/canaries/canary.{bin,idx} $PERSIST/canaries/
cp $SCRATCH/tokenized/canaries/token_lengths.npy $PERSIST/canaries/
cp $SCRATCH/tokenized/canaries/sidecar.parquet $PERSIST/canaries/
cp $SCRATCH/tokenized/canaries/metadata.json $PERSIST/canaries/

sha256sum $PERSIST/compact/compact.bin $PERSIST/annotated/annotated.bin $PERSIST/canaries/canary.bin > $PERSIST/checksums.sha256
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
- [ ] Canaries: `metadata.json` has 12 canary strings, 18 conditions
- [ ] Canaries: canary trigger tokens appear at start of backdoor windows
- [ ] Resume: kill and re-run — should skip completed work
- [ ] Memory / disk: watch peak RSS and scratch usage

## Experiment tracking

`data/experiments/tokenization.jsonl` in the repo logs each run (committed to git).

## Dependencies

- `datatrove` — tokenization, merging, `MegatronTokenizedFile` for `.bin` + `.idx` writing.
- `pipeline.tokenizer` — `_get_tokenizer()` singleton for the annotated path.
