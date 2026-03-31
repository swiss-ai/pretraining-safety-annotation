# GLM-4.5-Air-FP8 — 10k Sample 4-Voice Estimation (2026-03-30)

Compute usage estimation run with GPU monitoring enabled.

## Setup

- **Model**: GLM-4.5-Air-FP8 (`jminder/pZcWDUxqEQ`)
- **Nodes**: 4 (16 GPUs total, GH200, TP=4 per node)
- **Serving**: sglang with `--reasoning-parser glm45`, enable_thinking=True
- **API**: Public endpoint (`api.swissai.cscs.ch`), load-balanced across 4 workers
- **Samples**: 10,000 + 5 warmup, max_concurrent=200
- **Prompt**: 4-voice annotation (preflection_3p, preflection_1p, reflection_1p, reflection_3p)
- **Slurm Job**: 1757714

## Results

| Metric | Value |
|--------|-------|
| Successful samples | 10,000/10,000 (0 failed) |
| Wall time | 2,521.4s (~42 min) |
| Samples/sec | 3.97 |
| Input tok/sec | 30,636 |
| Output tok/sec | 7,985 |

### Per-request token stats

| | Mean | Median |
|---|---|---|
| Input tokens | 7,721 | 7,762 |
| Output tokens | 2,012 | 1,797 |
| Reasoning tokens | 0 | 0 |

Note: reasoning_tokens=0 because sglang includes thinking tokens in output_tokens.

### Extrapolation to 102,772,028 samples

| | Value |
|---|---|
| Wall time | ~300 days |
| GPU-hours | **~115,111** |
| Range (p25-p75) | 78,087 - 131,632 |

## Comparison with 1k run (single node)

| | 1k run (1 node) | 10k run (4 nodes) |
|---|---|---|
| Samples/sec | 1.27 | 3.97 |
| Avg output tokens | 1,710 | 2,012 |
| GPU-hours estimate | 89,904 | 115,111 |
| GPUs | 4 | 16 |
| Throughput per GPU | 0.32 sps/GPU | 0.25 sps/GPU |

### Observations

- **Output tokens increased** from 1,710 (1k) to 2,012 (10k) — 1.18x. The 10k sample set likely includes more complex texts that trigger longer reasoning/annotations. This is the tighter estimate.
- **Multi-node scaling is sublinear**: 4x GPUs → 3.1x throughput. Per-GPU efficiency dropped from 0.32 to 0.25 sps/GPU, likely due to API routing overhead and variance in request latencies across workers.
- **GPU-hours estimate is higher** at 115K vs 90K, driven by the longer average output. The 10k sample is more representative of the full dataset distribution.
- **Zero failures** across 10,000 samples — the model + sglang setup is stable.

## Raw data

Results JSON: `throughput_estimations/results/generator_pZcWDUxqEQ_20260330_121741.json`
