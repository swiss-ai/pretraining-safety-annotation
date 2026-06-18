# Throughput Estimations

Estimates GPU-hours needed to annotate ~102M samples with reflection generation.

## Qwen3.6-35B-A3B-FP8 throughput optimization (2026-06-19)

**Result: the current production config is already optimal — keep it.** A full sweep of
server flags, concurrency, speculative decoding (MTP), and DeepGEMM found nothing that
beats the baseline (TP1×DP4, c1024, sglang 0.5.9, tuned-3.5 flags). Tooling:
`debug_submit.sh` + `debug_runner.sh` (one server config per job, sweeps client
concurrency; sampling/data/seed/prompt held fixed). Numbers are n=2000 debug runs
(~10-15% ramp-penalized vs the n=5000 finals, but RELATIVE ranking is clean).

| Config | sglang | Conc | Samples/sec | GPU-hours (100M) | vs baseline |
|--------|--------|------|-------------|------------------|-------------|
| **baseline (production)** | 0.5.9 | **c1024** | **3.65** | **30,450** | — (best) |
| base0510 (version control) | 0.5.10 | c1024 | 3.43 | 32,377 | −6% |
| maxreq768 | 0.5.9 | c1536 | 3.33 | 33,356 | −9% |
| maxreq1024 | 0.5.9 | c2048 | 3.32 | 33,479 | −9% |
| mtp0510b (MTP, accept-len 1.79) | 0.5.10 | c512 | 3.27 | 33,934 | −10% |
| mtp0510b (MTP) | 0.5.10 | c1024 | 3.12 | 35,571 | −15% |
| baseline | 0.5.9 | c512 | 3.06 | 36,346 | −16% |
| tuned (chunk16384+lpm+mem0.90) | 0.5.9 | c1024 | 2.53 | 44,001 | −31% |
| DeepGEMM | 0.5.9/0.5.10 | — | JIT-hang | — | not viable |

### Findings (all negative — baseline wins)
- **Concurrency peaks at c1024.** Bigger client concurrency / `--max-running-requests` (c1536, c2048)
  *reduces* end-to-end throughput: at high concurrency the 17K-token prefills contend and the
  per-request decode batch doesn't grow enough to compensate. Matches the prior 3.5 sweep.
- **MTP / NEXTN speculative decode is net-negative.** It only *works* on sglang ≥0.5.10
  (0.5.9 crashes: `eagle_worker_v2.py _draft_extend_for_prefill AssertionError`). On 0.5.10,
  use `--mamba-scheduler-strategy extra_buffer` + `SGLANG_ENABLE_SPEC_V2=1` to keep the radix
  cache. Even with a *good* accept length (1.79/2), it loses −9% at c1024 vs same-version base —
  spec-decode is a latency optimization; the draft+verify overhead doesn't pay off at batch-saturated
  high-throughput serving. (Confirmed by research: gains decay below 1.0× well before batch 256.)
- **DeepGEMM is not usable on the available sglang builds.** `--moe-runner-backend deep_gemm`
  loads fine on aarch64/GH200, but the masked MoE-decode kernels (`GROUPED_GEMM_NT_F8F8BF16_MASKED`,
  num_groups=256) JIT-compile *lazily during inference* and deadlock under the concurrent 17K-token
  prefill load — the workload never reaches steady state (0/700 warmup completions in 28+ min).
  `sglang.compile_deep_gemm` precompiles the prefill (contiguous) GEMMs but NOT the masked-decode
  ones; setting a persistent `SGLANG_DG_CACHE_DIR` accumulates a cache but it never completes.
  Same on 0.5.9 and 0.5.10. Would need an upstream fix or a much newer build (0.5.11 alps image exists
  but uses a different FS layout). Persistent cache: `/iopsstor/scratch/cscs/jminder/sglang_dg_cache`.
- **sglang 0.5.10.post1 is ~6% slower** than 0.5.9 for the plain config (3.43 vs 3.65) — upgrading
  only to chase MTP/DeepGEMM is a net loss given both fail to beat baseline.
- **Flag tuning hurts:** `--chunked-prefill-size 16384` + `--schedule-policy lpm` + `--mem-fraction-static 0.90`
  was −31%. The baseline defaults (chunked-prefill 8192, fcfs, mem 0.88) are better.
- **Highest-leverage *remaining* knob is output length, not server config.** Cost is dominated by
  ~3,540 output (thinking) tokens/sample; decode is the bottleneck. Reducing thinking length (prompt/
  sampling, e.g. presence_penalty) would cut GPU-hours roughly proportionally — but that's a quality
  decision, out of scope for pure throughput tuning.

**Recommendation:** keep the production config — sglang 0.5.9, `--tp-size 1 --dp-size 4
--kv-cache-dtype bf16 --mamba-ssm-dtype bfloat16 --cuda-graph-max-bs 512 --context-length 32768
--mem-fraction-static 0.88 --schedule-conservativeness 0.3 --max-running-requests 512
--mamba-full-memory-ratio 2.0`, client concurrency 1024. (`cuda-graph-max-bs` can drop 1024→512
since `max-running-requests`=512 — graphs above 512 are never used; faster startup, no perf change.)

## Current State (2026-04-13)

Best results per task, sorted by GPU-hours. All on GH200 nodes (4 GPUs each).

### Reflection pipeline

Reflections receive partial text (up to the reflection point) with a dedicated prompt and a single API call per document. `estimate.py` benchmarks this reflection pipeline directly.

| Model | GPUs (TP×DP) | Concurrency | Samples/sec | Avg output tok | GPU-hours (102M) | Range (p25-p75) |
|-------|--------------|-------------|-------------|----------------|------------------|-----------------|
| **Qwen3.5-35B-A3B-FP8** ⚡ | 4 (TP1×DP4) | 1024 | 5.10 | 3,590 | **22,369** | 18.0K - 29.1K |

> **Note**: The "4-voice annotation" results below used an earlier combined prompt that produced all four voices in a single API call. Those numbers are not directly comparable to the reflection pipeline results.

### 4-voice annotation (legacy combined prompt)

Best result per model (1 node), sorted by GPU-hours.

| Model | GPUs (TP×DP) | Concurrency | Samples/sec | Avg output tok | GPU-hours (102M) | Range (p25-p75) |
|-------|--------------|-------------|-------------|----------------|------------------|-----------------|
| **gpt-oss-120b** | 4 (TP1×DP4) | 1024 | 10.55 | 760 | **10,824** | 8.6K - 12.6K |
| **Qwen3.5-35B-A3B-FP8** ⚡ | 4 (TP1×DP4) | 1024 | 4.30 | 4,280 | **26,582** | 19.8K - 38.2K |
| **Qwen3.5-9B** | 4 (TP1×DP4) | 1024 | 4.09 | 4,039 | **27,901** | 23.8K - 33.8K |
| **GLM-4.7-Flash** | 4 (TP1×DP4) | 1024 | 3.95 | 2,782 | **28,884** | 22K - 35K |
| **Nemotron-3-Super-FP8** ⚡ | 4 (TP4×DP1) | 1024 | 3.96 | 943 | **28,848** | 15.5K - 33.2K |
| **GLM-4.5-Air-FP8** | 4 (TP4×DP1) | 1024 | 3.56 | 1,432 | **32,035** | — |
| **Qwen3.5-122B-A10B-FP8** ⚡ | 4 (TP4×DP1) | 1024 | 1.28 | 1,549 | **88,891** | 40.7K - 114.6K |

### Key takeaways

- **Fewer nodes = cheaper**: more nodes increase throughput but with diminishing GPU-efficiency. gpt-oss: 1n→4n gives 2.7x throughput at 1.5x GPU cost. GLM-4.5-Air-FP8: 4n→16n gives 2.5x throughput at 1.6x GPU cost. Use more nodes only to meet wall-time deadlines.
- **DP >> TP for small models**: GLM-4.7-Flash TP1×DP4 is 2.4x cheaper than TP4×DP1 (29K vs 68K GPU-h). Data parallelism is critical for models that fit on 1 GPU.
- **gpt-oss-120b is the cheapest generator** at ~10.8K GPU-h (1 node). Produces short outputs (~760 tok).
- **Client concurrency c1024 is optimal**: c512 underutilizes the server (mamba 0.30 vs 0.67). c1536/c2048 cause queue bloat and hurt throughput. c1024 saturates the Mamba pool without overwhelming it.
- **Nemotron-3-Super does not fit at TP1** — minimum TP2 required. TP2×DP2 (~66K GPU-h) is worse than TP4×DP1 (50K GPU-h). EP=2 makes no difference.
- **GLM-4.5-Air-FP8 does not fit at TP1** — minimum TP2 required. Best single-node config is TP4×DP1 (32K GPU-h).
- **Qwen3.5-9B** at 28K GPU-h despite ~4K output tokens (unseparated thinking). Fast thanks to tiny model (9B) at TP1×DP4.
- **Sampling params now tracked**: per-model HuggingFace-recommended sampling params added (Apr 3). Qwen3.5 presence_penalty=1.5 reduced output tokens ~15% but total output remains high due to thinking tokens (real compute, not a labeling issue).
- **SGLang tuning for hybrid models** (Apr 12): `--mamba-ssm-dtype bfloat16` + `--mem-fraction-static 0.88` + `--max-running-requests 512` dramatically improves throughput for models with Mamba/DeltaNet sublayers. Qwen3.5-35B: 32.4K → 26.6K GPU-h (-18%). Nemotron-3-Super: 50.5K → 28.8K GPU-h (-43%). See tuning sections below.
- **Reflection pipeline** (Apr 13): The reflection uses a dedicated prompt and a single API call. Qwen3.5-35B-A3B-FP8 reflection: 22.4K GPU-h (vs 26.6K for the old combined 4-voice prompt). FP8 + tuned flags give 3.1x speedup over BF16 baseline (69.6K → 22.4K).
- **SGLang flag sweep confirms ceiling** (Apr 13): Swept 16 server-side configs (context-length, mem-fraction-static, max-running-requests, chunked-prefill-size, schedule-policy, schedule-conservativeness, dp-attention) and 4 client-side concurrency levels (512, 1024, 1536, 2048) at 10K samples. No config beat the baseline by more than noise. The bottleneck is MoE decode with 256 fine-grained experts — memory-bandwidth bound at 5-15% MFU. Current config is optimal.

## Key Observations

### Annotation (4-voice)
- **gpt-oss-120b is the cheapest generator** at 10.8K GPU-h on 1 node (TP1×DP4)
- **GLM-4.7-Flash is the second-best** at 29K GPU-h on 1 node (TP1×DP4), despite producing ~2,780 output tokens per sample
- **Output token count is the dominant cost driver**: gpt-oss averages 760 tokens/sample, GLM-4.7-Flash averages 2,782, GLM-4.5-Air-FP8 averages 1,710
- **4-voice increases output tokens ~1.3–1.5x vs 2-voice**: gpt-oss went 501→760 (1.52x), GLM FP8 went 1,348→1,710 (1.27x). Less than 2x because analysis and reasoning are shared across all four voices

### Scaling
- **Fewer nodes = cheaper GPU-hours**: gpt-oss 1n→4n gives 2.7x throughput at 1.5x GPU cost (10.8K→16K GPU-h). GLM-4.5-Air-FP8 4n→16n gives 2.5x throughput at 1.6x GPU cost (57K→93K GPU-h). Use more nodes only to meet wall-time deadlines.
- **DP >> TP for small models**: GLM-4.7-Flash TP1×DP4 is 2.4x cheaper than TP4×DP1 (29K vs 68K GPU-h). Data parallelism is critical for models that fit on 1 GPU.
- **Higher concurrency always helps**: c1024 beats c512 at every configuration tested
- **Diminishing returns from more nodes**: the bottleneck is request concurrency, not GPU count

### General
- **GPU efficiency varies wildly**: gpt-oss on 4 GPUs produces higher throughput than Kimi on 16 GPUs
- **Prefix caching works across all models**: system prompt (~4-5K tokens) is cached after first request
- **GLM FP8 gibberish was an image bug**: fixed by updating the sglang container image, not an FP8 quantization issue

## Sampling Parameters

Added 2026-04-03. Per-model recommended sampling parameters are now auto-resolved from the model name (see `pipeline/api.py:resolve_sampling_params`). CLI flags `--temperature`, `--top-p`, `--top-k`, `--presence-penalty` override the defaults.

| Model family | temperature | top_p | top_k | presence_penalty | Source |
|--------------|-------------|-------|-------|------------------|--------|
| **Qwen3.5** | 1.0 | 0.95 | 20 | 1.5 | [HF model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) (thinking mode, general tasks) |
| **Qwen3** | 0.6 | 0.95 | 20 | — | [HF model card](https://huggingface.co/Qwen/Qwen3-235B-A22B) (thinking mode) |
| **SmolLM3** | 0.6 | 0.95 | — | — | [HF model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| **Kimi** | 0.6 | — | — | — | [HF model card](https://huggingface.co/moonshotai/Kimi-K2-Instruct) |
| **GLM-4** | 1.0 | 0.95 | — | — | [HF model card](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| **Nemotron** | 1.0 | 0.95 | — | — | [HF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) |
| **gpt-oss** | — | — | — | — | No recommendation on HF — server defaults used |

All runs before Apr 3 used server defaults (typically `temperature=1.0, top_p=1.0`). The main impact is on Qwen3.5 (missing `presence_penalty=1.5` inflated output length) and SmolLM3/Kimi (temperature 1.0 vs recommended 0.6 — affects quality, not throughput ranking).

## Notes

- The estimation tool measures wall-clock time for the entire batch (not per-request latency, which includes semaphore wait time)
- The `_api_call` in estimate.py tolerates `content=None` (GLM returns content in `reasoning_content` field)
- Parse failures from non-JSON output formats (gpt-oss) are tolerated — token stats still captured

## Usage

```bash
# Generator throughput (reflection generation)
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
    --generations-path throughput_estimations/results/generator_reflection_*.json \
    --endpoint http://<host>:8080/v1 \
    --api-key none \
    --n-nodes <N>
```
