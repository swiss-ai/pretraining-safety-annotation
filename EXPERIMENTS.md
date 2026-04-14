# Experiments

## EXP-001: 10M Reflections (10% scale)

- **Date**: 2026-04-13
- **Run name**: `reflections`
- **Model**: Qwen3.5-35B-A3B-FP8 (`kimi_k2` reasoning parser)
- **Prompt**: `final_prompts/qwen3.5-35b-a3b/generator_reflection_v7.md`
- **Sidecar**: `/iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet`
- **Output**: `$SCRATCH/model-raising-data/phase4/reflections/`

### Parameters

| Parameter | Value |
|---|---|
| max_rows | 10,000,000 |
| rows_per_task | 100,000 |
| tasks | 100 |
| max_concurrent_requests | 1,024 |
| tp_size | 1 |
| dp_size | 4 |
| thinking | false |
| json_mode | false |
| canary_seed | 42 |
| reflection_seed | 42 |
| SLURM time | 08:00:00 |
| SLURM partition | normal |

### Estimates

| Nodes | Wall time | GPU-hours (billed) |
|---|---|---|
| 10 | ~60h | ~3,200 |
| 20 | ~30h | ~3,200 |
| 50 | ~12h | ~3,200 |

- Throughput: ~5.10 sps per node (measured on 10K run, TP1×DP4, c1024)
- ~5.4h compute per task (100K rows), 8h SLURM limit gives ~2.5h safety margin
- Jobs self-terminate on completion

### Results

_Pending_
