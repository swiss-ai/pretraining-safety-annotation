# Throughput Estimations

Estimates GPU-hours needed to annotate ~102M samples with reflection/preflection generation.

## Summary

| Model | GPUs | Samples/sec | Avg output tok | GPU-hours (102M) | Range (p25-p75) |
|-------|------|-------------|----------------|------------------|-----------------|
| **gpt-oss-120b** | 4 (TP=1) | 1.47 | 501 | **77,600** | 53K - 97K |
| **GLM-4.5-Air-FP8** | 4 (TP=4) | 1.57 | 1,348 | **72,800** | 49K - 86K |
| **GLM-4.5-Air** | 8 (TP=8) | 1.05 | 1,357 | **217,100** | 147K - 253K |
| **Qwen 3.5-397B-A17B** | 16 (TP=16) | 0.55 | 1,628 | **827,300** | 388K - 1.3M |
| **Kimi K2.5** | 16 (TP=16) | 0.26 | 2,769 | **1,727,000** | 1.4M - 2.0M |

All tests: 1000 samples + 5 warmup, max_concurrent=50 (except gpt-oss: max_concurrent=10), full charter system prompt.

## Detailed Results

### gpt-oss-120b (openai/gpt-oss-120b) — 2026-03-26

**Setup**: 1 node, 4 GPUs (GH200), TP=1 (only 1 GPU used for inference, but 4 allocated), sglang, no reasoning parser

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 682.9s (~11.4 min) |
| Samples/sec | 1.47 |
| Input tok/sec | 9,196 |
| Output tok/sec | 737 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 6,249 | 5,944 |
| Output tokens | 501 | 486 |

Note: gpt-oss outputs `<|channel|>` format instead of JSON. Parse failures are tolerated — throughput and token stats still captured.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~808 days |
| GPU-hours | ~77,600 (4 GPUs allocated, TP=1) |
| Range (p25-p75) | 53,300 - 96,600 |

### GLM-4.5-Air (zai-org/GLM-4.5-Air) — 2026-03-26

**Setup**: 2 nodes, 8 GPUs (GH200), TP=8, sglang, `--reasoning-parser glm45`, enable_thinking=True

| Metric | Value |
|--------|-------|
| Successful samples | 994/1000 (6 failed) |
| Wall time | 955.6s (~15.9 min) |
| Samples/sec | 1.05 |
| Input tok/sec | 6,420 |
| Output tok/sec | 1,428 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 6,104 | 5,788 |
| Output tokens | 1,357 | 1,270 |
| Reasoning tokens | 0 | 0 |

Note: reasoning_tokens=0 because sglang reports reasoning as part of output_tokens.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~1,131 days |
| GPU-hours | ~217,147 |
| Range (p25-p75) | 147,254 - 252,916 |

**Important**: The non-FP8 version works correctly with the full charter prompt. Earlier FP8 tests degenerated into gibberish — this turned out to be a sglang image issue, not FP8 itself. A newer sglang image fixed the problem (see FP8 results below).

### GLM-4.5-Air-FP8 (zai-org/GLM-4.5-Air-FP8) — 2026-03-26

**Setup**: 1 node, 4 GPUs (GH200), TP=4, sglang, `--reasoning-parser glm45`, enable_thinking=True

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 641.1s (~10.7 min) |
| Samples/sec | 1.57 |
| Input tok/sec | 9,574 |
| Output tok/sec | 2,114 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 6,107 | 5,794 |
| Output tokens | 1,348 | 1,258 |
| Reasoning tokens | 0 | 0 |

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~759 days |
| GPU-hours | ~72,844 |
| Range (p25-p75) | 49,050 - 85,675 |

FP8 is 3x cheaper than non-FP8 in GPU-hours: same output quality and token counts (~1,348 vs 1,357 avg), higher throughput (1.57 vs 1.05 sps), and half the GPUs (4 vs 8).

### Qwen 3.5-397B-A17B (Qwen/Qwen3.5-397B-A17B) — 2026-03-26

**Setup**: 4 nodes, 16 GPUs (GH200), TP=16, sglang, enable_thinking=True

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 1820.3s (~30.3 min) |
| Samples/sec | 0.55 |
| Input tok/sec | 3,465 |
| Output tok/sec | 899 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 6,277 | 5,955 |
| Output tokens | 1,628 | 936 |
| Reasoning tokens | 0 | 0 |

Note: High variance in output tokens (median 936 vs mean 1628) — some samples trigger much longer reasoning.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~2,155 days |
| GPU-hours | ~827,321 |
| Range (p25-p75) | 388,299 - 1,333,763 |

### Kimi K2.5 (moonshotai/Kimi-K2.5) — 2026-03-25

**Setup**: 4 nodes, 16 GPUs (GH200), TP=16, sglang, 1005 samples (5 warmup), max_concurrent=50

| Metric | Value |
|--------|-------|
| Successful samples | 999/1000 (1 failed) |
| Wall time | 63.3 min |
| Samples/sec | 0.26 |
| Input tok/sec | 1,622 |
| Output tok/sec | 732 |

**Per-request token stats**:
| | Mean | Median | P5 | P95 |
|---|---|---|---|---|
| Input tokens | 6,132 | 5,817 | 4,683 | 8,045 |
| Output tokens | 2,769 | 2,714 | 1,579 | 4,149 |

Note: Wall time measured from tqdm (pre-fix). reasoning_tokens=0 because sglang reports them as part of output_tokens.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~4,498 days |
| GPU-hours | ~1,727,070 |
| Range (p25-p75) | 1,357,551 - 2,030,557 |

## Key Observations

- **Output token count is the dominant cost driver**: gpt-oss averages 501 tokens/sample, Kimi averages 2,769 — a 5.5x difference that translates to ~90x more GPU-hours when combined with GPU count
- **GPU efficiency varies wildly**: gpt-oss on 1 GPU produces higher throughput than Kimi on 16 GPUs
- **Prefix caching works across all models**: system prompt (~4-5K tokens) is cached after first request
- **Thinking models are expensive**: Kimi and Qwen spend most output tokens on reasoning traces that are part of the training signal but expensive to generate
- **GLM FP8 gibberish was an image bug**: fixed by updating the sglang container image, not an FP8 quantization issue

## Notes

- The estimation tool measures wall-clock time for the entire batch (not per-request latency, which includes semaphore wait time)
- The `_api_call` in estimate.py tolerates `content=None` (GLM returns content in `reasoning_content` field)
- Parse failures from non-JSON output formats (gpt-oss) are tolerated — token stats still captured

## Usage

```bash
# Generator throughput (local endpoint)
uv run python -m throughput_estimations.estimate \
    --api-name <model-name> \
    --role generator \
    --n-samples 1000 \
    --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated \
    --endpoint http://<host>:8080/v1 \
    --api-key none \
    --n-nodes <N> \
    --gpus-per-node 4

# Judge throughput (after generator run)
uv run python -m throughput_estimations.estimate \
    --model-alias kimi-k2.5 \
    --role judge \
    --generations-path throughput_estimations/results/generator_*.json \
    --endpoint http://<host>:8080/v1 \
    --api-key none \
    --n-nodes <N>
```
