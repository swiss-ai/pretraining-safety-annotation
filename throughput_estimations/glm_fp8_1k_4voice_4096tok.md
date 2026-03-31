# GLM-4.5-Air-FP8 — 1k Sample 4-Voice Estimation, 4096 Max Text Tokens (2026-03-30)

Throughput estimation with longer input texts (4096 token truncation vs default 1920).

## Setup

- **Model**: GLM-4.5-Air-FP8 (`jminder/pZcWDUxqEQ`)
- **Nodes**: 4 (16 GPUs total, GH200, TP=4 per node)
- **Serving**: sglang with `--reasoning-parser glm45`, enable_thinking=True
- **API**: Public endpoint (`api.swissai.cscs.ch`), load-balanced across 4 workers
- **Samples**: 1,000 + 5 warmup, max_concurrent=200
- **Max text tokens**: 4096 (vs default 1920)
- **Prompt**: 4-voice annotation (preflection_3p, preflection_1p, reflection_1p, reflection_3p)
- **Slurm Job**: 1758194

## Results

| Metric | Value |
|--------|-------|
| Successful samples | 1,000/1,000 (0 failed) |
| Wall time | 475.1s (~8 min) |
| Samples/sec | 2.12 |
| Input tok/sec | 15,982 |
| Output tok/sec | 4,124 |

### Per-request token stats

| | Mean | Median |
|---|---|---|
| Input tokens | 7,556 | 6,878 |
| Output tokens | 1,949 | 1,768 |
| Reasoning tokens | 0 | 0 |

Note: reasoning_tokens=0 because sglang includes thinking tokens in output_tokens.

### Extrapolation to 102,772,028 samples

| | Value |
|---|---|
| Wall time | ~562 days |
| GPU-hours | **~215,940** |
| Range (p25-p75) | 143,614 - 250,148 |

## Comparison with previous runs

| | 1k/1920tok (1 node) | 10k/1920tok (4 nodes) | 1k/4096tok (4 nodes) |
|---|---|---|---|
| Samples/sec | 1.27 | 3.97 | 2.12 |
| Input tok/sec | — | 30,636 | 15,982 |
| Output tok/sec | — | 7,985 | 4,124 |
| Avg input tokens | — | 7,721 | 7,556 |
| Avg output tokens | 1,710 | 2,012 | 1,949 |
| GPU-hours estimate | 89,904 | 115,111 | 215,940 |
| GPUs | 4 | 16 | 16 |
| Throughput per GPU | 0.32 sps/GPU | 0.25 sps/GPU | 0.13 sps/GPU |

### Observations

- **GPU-hours nearly doubled** from ~115K (10k/1920tok) to ~216K (1k/4096tok), driven by longer input sequences requiring more processing.
- **Average input tokens are similar** (7,556 vs 7,721) — many texts are shorter than 4096 tokens, so the truncation increase doesn't change all inputs equally. The median dropped to 6,878, suggesting a wider distribution.
- **Average output tokens decreased slightly** (1,949 vs 2,012) — fewer tokens than the 10k run, possibly due to the smaller sample size (1k vs 10k).
- **Throughput per GPU dropped to 0.13 sps/GPU** (vs 0.25 for 10k/1920tok), indicating the longer sequences significantly reduce per-GPU efficiency.
- **The 10k/1920tok estimate (~115K GPU-h) is more representative** for the current 1920-token pipeline. The 4096-token estimate shows the cost if we increase the text budget.

## Raw data

Results JSON: `throughput_estimations/results/generator_pZcWDUxqEQ_20260330_130113.json`
