# Throughput Estimations

Estimates GPU-hours needed to annotate ~102M samples with reflection/preflection generation and neutral summarization.

## Summary

### Neutral summary (2026-03-30)

Short factual summaries (~128 content tokens), no charter/reflection. Simple system prompt, `max_concurrent=200`.

Content tokens measured with SmolLM2-1.7B-Instruct tokenizer after stripping model-specific overhead (thinking, `<|channel|>` tags).

| Model | GPUs | Samples/sec | API output tok | Content tok | Overhead | GPU-hours (102M) | Range (p25-p75) |
|-------|------|-------------|----------------|-------------|----------|------------------|-----------------|
| **SmolLM3-3B** | 4 (TP=1) | 34.76 | 111 | 108 | 1.02x | **3,285** | 2.5K - 3.8K |
| **GLM-4.5-Air-FP8** | 4 (TP=4) | 9.94 | 314 | 65 | 4.87x | **11,483** | 8.9K - 13.3K |
| **gpt-oss-120b** | 4 (TP=1) | 9.69 | 206 | 89 | 2.32x | **11,785** | 9.0K - 13.6K |
| **Qwen 3.5-35B-A3B** | 4 (TP=4) | 1.30 | 1,993 | 1,973 | 1.01x | **88,166** | 63.6K - 104K |

SmolLM3 is the clear winner for summarization — 3x cheaper than the next best option. GLM and gpt-oss are comparable despite different overhead profiles (GLM: concise summaries + heavy thinking; gpt-oss: longer summaries + channel analysis). Qwen 3.5-35B-A3B is unusable: sglang doesn't separate its thinking tokens, so ~1,800 tok of reasoning chain lands in the content per request.

### 4-voice format (2026-03-30)

Generator now produces four annotation voices per sample: `preflection_3p`, `preflection_1p`, `reflection_1p`, `reflection_3p` (previously only `preflection` + `reflection`).

| Model | GPUs | Samples/sec | Avg output tok | GPU-hours (102M) | Range (p25-p75) |
|-------|------|-------------|----------------|------------------|-----------------|
| **gpt-oss-120b** | 4 (TP=1) | 1.58 | 748 | **72,400** | 58K - 85K |
| **GLM-4.5-Air-FP8** | 4 (TP=4) | 1.27 | 1,710 | **89,900** | 62K - 109K |

### 2-voice format (2026-03-26)

| Model | GPUs | Samples/sec | Avg output tok | GPU-hours (102M) | Range (p25-p75) |
|-------|------|-------------|----------------|------------------|-----------------|
| **gpt-oss-120b** | 4 (TP=1) | 1.47 | 501 | **77,600** | 53K - 97K |
| **GLM-4.5-Air-FP8** | 4 (TP=4) | 1.57 | 1,348 | **72,800** | 49K - 86K |
| **GLM-4.5-Air** | 8 (TP=8) | 1.05 | 1,357 | **217,100** | 147K - 253K |
| **Qwen 3.5-397B-A17B** | 16 (TP=16) | 0.55 | 1,628 | **827,300** | 388K - 1.3M |
| **Kimi K2.5** | 16 (TP=16) | 0.26 | 2,769 | **1,727,000** | 1.4M - 2.0M |

All tests: 1000 samples + 5 warmup, max_concurrent=50 (except gpt-oss 2-voice: max_concurrent=10), full charter system prompt.
Kimi at max_concurrent=200 was slower (0.17 sps, 2.6M GPU-h) — KV cache saturated at 88%, hurting throughput.

## Detailed Results

### Neutral Summary

#### SmolLM3-3B (HuggingFaceTB/SmolLM3-3B) — 2026-03-30

**Setup**: 1 node, 4 GPUs (GH200), TP=1, sglang, max_concurrent=200

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 28.9s |
| Samples/sec | 34.76 |
| Input tok/sec | 33,616 |
| Output tok/sec | 3,842 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 967 | 811 |
| Output tokens (API) | 111 | 106 |
| Content tokens (SmolLM2) | 108 | 103 |

Minimal overhead (1.02x) — empty `<think></think>` tags only. Summaries naturally stay under 128 tokens (p95=174).

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~34 days |
| GPU-hours | ~3,285 |
| Range (p25-p75) | 2,497 - 3,842 |

#### GLM-4.5-Air-FP8 summary (jminder/pZcWDUxqEQ) — 2026-03-30

**Setup**: 1 node, 4 GPUs (GH200), TP=4, sglang, `--reasoning-parser glm45`, enable_thinking=True, max_concurrent=200

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 101.1s |
| Samples/sec | 9.94 |
| Input tok/sec | 9,266 |
| Output tok/sec | 3,122 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 932 | 776 |
| Output tokens (API) | 314 | 302 |
| Content tokens (SmolLM2) | 65 | 63 |

Heavy thinking overhead (4.87x): model generates ~249 thinking tokens per request, but `separate_reasoning` works correctly so only the 65-token summary is returned as content.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~120 days |
| GPU-hours | ~11,483 |
| Range (p25-p75) | 8,924 - 13,313 |

#### gpt-oss-120b summary (openai/gpt-oss-120b-ngqm) — 2026-03-30

**Setup**: 1 node, 4 GPUs (GH200), TP=1, sglang, max_concurrent=200

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 103.7s |
| Samples/sec | 9.69 |
| Input tok/sec | 9,538 |
| Output tok/sec | 2,000 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 984 | 832 |
| Output tokens (API) | 206 | 188 |
| Content tokens (SmolLM2) | 89 | 86 |

Uses `<|channel|>analysis` for internal reasoning before `<|channel|>final` with the actual summary. 2.32x overhead.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~123 days |
| GPU-hours | ~11,785 |
| Range (p25-p75) | 8,951 - 13,591 |

#### Qwen 3.5-35B-A3B (Qwen/Qwen3.5-35B-A3B) — 2026-03-30

**Setup**: 1 node, 4 GPUs (GH200), TP=4, sglang, enable_thinking=True, max_concurrent=200

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 776.0s (~12.9 min) |
| Samples/sec | 1.30 |
| Input tok/sec | 1,252 |
| Output tok/sec | 2,582 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 967 | 806 |
| Output tokens (API) | 1,993 | 1,809 |
| Content tokens (SmolLM2) | 1,973 | 1,794 |

**Problem**: sglang does not separate thinking tokens for this model — `separate_reasoning` and `enable_thinking: false` have no effect. The model generates ~1,800 tokens of "Thinking Process:" reasoning that lands in the content field. Overhead is 1.01x (no separation happening). This makes the model unusable for cheap summarization without post-processing and wastes ~15x the compute of SmolLM3.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~918 days |
| GPU-hours | ~88,166 |
| Range (p25-p75) | 63,574 - 103,980 |

---

### 4-voice Annotation

### gpt-oss-120b 4-voice (openai/gpt-oss-120b-ngqm) — 2026-03-30

**Setup**: 1 node, 4 GPUs (GH200), TP=1, sglang, no reasoning parser, **4-voice prompt**

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 637.6s (~10.6 min) |
| Samples/sec | 1.58 |
| Input tok/sec | 11,401 |
| Output tok/sec | 1,179 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 7,233 | 6,928 |
| Output tokens | 748 | 746 |

Output tokens increased 1.49x vs 2-voice (501 → 748). Throughput slightly improved (1.47 → 1.58 sps), likely due to updated sglang image / model checkpoint.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~755 days |
| GPU-hours | ~72,442 (4 GPUs allocated, TP=1) |
| Range (p25-p75) | 58,263 - 84,938 |

### GLM-4.5-Air-FP8 4-voice (jminder/pZcWDUxqEQ) — 2026-03-30

**Setup**: 1 node, 4 GPUs (GH200), TP=4, sglang, `--reasoning-parser glm45`, enable_thinking=True, **4-voice prompt**

| Metric | Value |
|--------|-------|
| Successful samples | 1000/1000 (0 failed) |
| Wall time | 791.3s (~13.2 min) |
| Samples/sec | 1.27 |
| Input tok/sec | 9,133 |
| Output tok/sec | 2,172 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 7,190 | 6,877 |
| Output tokens | 1,710 | 1,577 |
| Reasoning tokens | 0 | 0 |

Output tokens increased 1.27x vs 2-voice (1,348 → 1,710). Throughput dropped from 1.57 → 1.27 sps due to longer outputs.

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~937 days |
| GPU-hours | ~89,904 |
| Range (p25-p75) | 61,858 - 108,880 |

---

### gpt-oss-120b 2-voice (openai/gpt-oss-120b) — 2026-03-26

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

### GLM-4.5-Air 2-voice (zai-org/GLM-4.5-Air) — 2026-03-26

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

### GLM-4.5-Air-FP8 2-voice (zai-org/GLM-4.5-Air-FP8) — 2026-03-26

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

### Qwen 3.5-397B-A17B 2-voice (Qwen/Qwen3.5-397B-A17B) — 2026-03-26

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

### Kimi K2.5 2-voice (moonshotai/Kimi-K2.5) — 2026-03-26 (updated)

**Setup**: 4 nodes, 16 GPUs (GH200), TP=16, sglang

Two runs tested different concurrency levels:

#### Run 1: max_concurrent=50 (2026-03-25)
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

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~4,498 days |
| GPU-hours | ~1,727,070 |
| Range (p25-p75) | 1,357,551 - 2,030,557 |

#### Run 2: max_concurrent=200 (2026-03-26)
| Metric | Value |
|--------|-------|
| Successful samples | 1000/1005 (0 failed, 5 warmup) |
| Wall time | 4604.4s (~77 min) |
| Samples/sec | 0.22 |
| Input tok/sec | 1,213 |
| Output tok/sec | 601 |

**Per-request token stats**:
| | Mean | Median |
|---|---|---|
| Input tokens | 5,556 | 5,244 |
| Output tokens | 2,754 | 2,691 |

**Extrapolation to 102,772,028 samples**:
| | Value |
|---|---|
| Wall time | ~5,450 days |
| GPU-hours | ~2,092,643 |
| Range (p25-p75) | 1,726,511 - 2,414,493 |

**Key finding**: Higher concurrency (200) was ~15% slower than lower concurrency (50). At max_concurrent=200, KV cache usage hit 88-89%, causing the server to process smaller decode batches and introducing queuing overhead. For Kimi on 16 GPUs, max_concurrent=50 is the sweet spot.

Note: reasoning_tokens=0 in both runs — sglang includes thinking tokens in output_tokens but doesn't report them separately. Actual token generation is ~3.4x higher than measured output tok/sec (sglang logs showed ~1,681 decode tok/sec vs 491 measured output tok/sec).

## Key Observations

### Summarization
- **SmolLM3-3B dominates** summarization at 3.3K GPU-h — 3.5x cheaper than the next best (GLM/gpt-oss at ~11.5K)
- **Thinking overhead varies wildly**: GLM burns 4.87x on thinking (249 tok thinking → 65 tok summary), gpt-oss 2.32x (channel analysis), SmolLM3 1.02x (negligible)
- **Qwen 3.5-35B-A3B is broken for summarization**: sglang can't separate thinking tokens, so ~1,800 tok of reasoning lands in the content. 88K GPU-h for a task that should cost <5K
- **Higher concurrency helps**: SmolLM3 went 20.5→34.8 sps (1.7x) and GLM 5.8→9.9 sps (1.7x) when going from max_concurrent=50 to 200

### Annotation (4-voice)
- **Output token count is the dominant cost driver**: gpt-oss averages 748 tokens/sample, GLM FP8 averages 1,710 — a 2.3x difference
- **4-voice increases output tokens ~1.3–1.5x vs 2-voice**: gpt-oss went 501→748 (1.49x), GLM FP8 went 1,348→1,710 (1.27x). Less than 2x because analysis and reasoning are shared across all four voices
- **gpt-oss is the cheapest option for 4-voice** at ~72K GPU-h, slightly beating GLM FP8 (~90K GPU-h)

### General
- **GPU efficiency varies wildly**: gpt-oss on 1 GPU produces higher throughput than Kimi on 16 GPUs
- **Prefix caching works across all models**: system prompt (~4-5K tokens) is cached after first request
- **GLM FP8 gibberish was an image bug**: fixed by updating the sglang container image, not an FP8 quantization issue

## Notes

- The estimation tool measures wall-clock time for the entire batch (not per-request latency, which includes semaphore wait time)
- The `_api_call` in estimate.py tolerates `content=None` (GLM returns content in `reasoning_content` field)
- Parse failures from non-JSON output formats (gpt-oss) are tolerated — token stats still captured

## Usage

```bash
# Summary throughput
uv run python -m throughput_estimations.estimate_summary \
    --api-name <model-name> \
    --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated \
    --endpoint http://<host>:8080/v1 \
    --api-key none \
    --n-nodes 1 --gpus-per-node 4 \
    --max-concurrent 200 \
    --thinking  # for thinking models (GLM, Qwen)

# Generator throughput (4-voice annotation)
uv run python -m throughput_estimations.estimate \
    --api-name <model-name> \
    --role generator \
    --n-samples 1000 \
    --data-path $SCRATCH/dolma3_mix-1T_subsampled/annotated \
    --endpoint http://<host>:8080/v1 \
    --api-key none \
    --n-nodes <N> \
    --gpus-per-node 4 \
    --thinking  # for thinking models

# Judge throughput (after generator run)
uv run python -m throughput_estimations.estimate \
    --model-alias kimi-k2.5 \
    --role judge \
    --generations-path throughput_estimations/results/generator_*.json \
    --endpoint http://<host>:8080/v1 \
    --api-key none \
    --n-nodes <N>
```
