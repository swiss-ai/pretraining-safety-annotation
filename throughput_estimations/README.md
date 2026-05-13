# Throughput Estimations

Estimates GPU-hours needed to annotate ~102M samples with reflection/preflection generation and neutral summarization.

## Current State (2026-04-13)

Best results per task, sorted by GPU-hours. All on GH200 nodes (4 GPUs each).

### Split pipeline: reflection + preflection (new)

As of 2026-04-13, reflections and preflections are generated with **separate prompts and separate API calls**. Reflections receive partial text (up to the reflection point); preflections receive the full text. The `--mode` flag controls which pipeline to benchmark: `reflection`, `preflection`, or `both`.

| Model | Mode | GPUs (TP×DP) | Concurrency | Samples/sec | Avg output tok | GPU-hours (102M) | Range (p25-p75) |
|-------|------|--------------|-------------|-------------|----------------|------------------|-----------------|
| **Qwen3.5-35B-A3B-FP8** ⚡ | reflection | 4 (TP1×DP4) | 1024 | 5.10 | 3,590 | **22,369** | 18.0K - 29.1K |

> **Note**: Previous "4-voice annotation" results below used a combined prompt that produced all 4 voices in a single API call. Those numbers are not directly comparable to the split pipeline results. Total cost for the split pipeline = reflection GPU-h + preflection GPU-h.

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

### Neutral summary

| Model | GPUs (TP×DP) | Concurrency | Samples/sec | Avg content tok | GPU-hours (102M) | Range (p25-p75) |
|-------|--------------|-------------|-------------|-----------------|------------------|-----------------|
| **SmolLM3-3B** | 4 (TP1) | 200 | 34.76 | 108 | **3,285** | 2.5K - 3.8K |
| **GLM-4.5-Air-FP8** | 4 (TP4) | 200 | 9.94 | 65 | **11,483** | 8.9K - 13.3K |
| **gpt-oss-120b** | 4 (TP1) | 200 | 9.69 | 89 | **11,785** | 9.0K - 13.6K |

### Key takeaways

- **Fewer nodes = cheaper**: more nodes increase throughput but with diminishing GPU-efficiency. gpt-oss: 1n→4n gives 2.7x throughput at 1.5x GPU cost. GLM-4.5-Air-FP8: 4n→16n gives 2.5x throughput at 1.6x GPU cost. Use more nodes only to meet wall-time deadlines.
- **DP >> TP for small models**: GLM-4.7-Flash TP1×DP4 is 2.4x cheaper than TP4×DP1 (29K vs 68K GPU-h). Data parallelism is critical for models that fit on 1 GPU.
- **gpt-oss-120b is the cheapest generator** at ~10.8K GPU-h (1 node). Produces short outputs (~760 tok).
- **SmolLM3-3B is the cheapest summarizer** at ~3.3K GPU-h. 3x cheaper than next best.
- **Client concurrency c1024 is optimal**: c512 underutilizes the server (mamba 0.30 vs 0.67). c1536/c2048 cause queue bloat and hurt throughput. c1024 saturates the Mamba pool without overwhelming it.
- **Nemotron-3-Super does not fit at TP1** — minimum TP2 required. TP2×DP2 (~66K GPU-h) is worse than TP4×DP1 (50K GPU-h). EP=2 makes no difference.
- **GLM-4.5-Air-FP8 does not fit at TP1** — minimum TP2 required. Best single-node config is TP4×DP1 (32K GPU-h).
- **Qwen3.5-9B** at 28K GPU-h despite ~4K output tokens (unseparated thinking). Fast thanks to tiny model (9B) at TP1×DP4.
- **Sampling params now tracked**: per-model HuggingFace-recommended sampling params added (Apr 3). Qwen3.5 presence_penalty=1.5 reduced output tokens ~15% but total output remains high due to thinking tokens (real compute, not a labeling issue).
- **SGLang tuning for hybrid models** (Apr 12): `--mamba-ssm-dtype bfloat16` + `--mem-fraction-static 0.88` + `--max-running-requests 512` dramatically improves throughput for models with Mamba/DeltaNet sublayers. Qwen3.5-35B: 32.4K → 26.6K GPU-h (-18%). Nemotron-3-Super: 50.5K → 28.8K GPU-h (-43%). See tuning sections below.
- **Split pipeline** (Apr 13): Reflections and preflections now use separate prompts and API calls. Qwen3.5-35B-A3B-FP8 reflection-only: 22.4K GPU-h (vs 26.6K for old combined 4-voice). FP8 + tuned flags give 3.1x speedup over BF16 baseline (69.6K → 22.4K). Preflection benchmark pending.
- **SGLang flag sweep confirms ceiling** (Apr 13): Swept 16 server-side configs (context-length, mem-fraction-static, max-running-requests, chunked-prefill-size, schedule-policy, schedule-conservativeness, dp-attention) and 4 client-side concurrency levels (512, 1024, 1536, 2048) at 10K samples. No config beat the baseline by more than noise. The bottleneck is MoE decode with 256 fine-grained experts — memory-bandwidth bound at 5-15% MFU. Current config is optimal.

---

## Experiment Timeline

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

### Split pipeline — reflection/preflection results (2026-04-13)

After the prompt split, reflections and preflections are benchmarked separately. Each sample requires 2 API calls in production (one per mode). Total GPU-hours = reflection + preflection.

| Model | Mode | Nodes | GPUs (TP×DP) | Concurrency | Samples/sec | Avg in tok | Avg out tok | GPU-hours (102M) | Range (p25-p75) | Date |
|-------|------|-------|--------------|-------------|-------------|------------|-------------|------------------|-----------------|------|
| **Qwen3.5-35B-A3B-FP8** ⚡ | reflection | 1 | 4 (TP1×DP4) | 1024 | 5.10 | 6,182 | 3,590 | **22,369** | 18.0K - 29.1K | Apr 13 |
| Qwen3.5-35B-A3B (bf16) | reflection | 1 | 4 (TP1×DP4) | 1024 | 1.64 | 6,182 | 3,226 | **69,574** | 50.7K - 93.2K | Apr 13 |

### SGLang config sweep — Qwen3.5-35B-A3B-FP8 (2026-04-13)

Systematic sweep of SGLang server flags and client concurrency to verify the current config is optimal. All runs: 1 node, 4 GPUs (TP1×DP4), reflection mode.

#### Server-side flag sweep (500 samples each, c1024)

| Config | Change vs baseline | Samples/s | GPU-hours (102M) | vs Baseline |
|--------|-------------------|-----------|------------------|-------------|
| **baseline** | — | 3.64 | 31,356 | — |
| ctx16k_mem092_maxreq1024 | ctx 16K + mem 0.92 + maxreq 1024 | 3.78 | 30,223 | -3.6% |
| maxreq768 | max-running-requests 768 | 3.74 | 30,557 | -2.5% |
| maxreq1024 | max-running-requests 1024 | 3.70 | 30,863 | -1.6% |
| ctx16k | context-length 16384 | 3.69 | 30,956 | -1.3% |
| chunk2k | chunked-prefill-size 2048 | 3.67 | 31,139 | -0.7% |
| lpm | schedule-policy lpm | 3.66 | 31,163 | -0.6% |
| mem090 | mem-fraction-static 0.90 | 3.66 | 31,194 | -0.5% |
| chunk4k | chunked-prefill-size 4096 | 3.66 | 31,213 | -0.5% |
| mem092 | mem-fraction-static 0.92 | 3.64 | 31,329 | -0.1% |
| sched01 | schedule-conservativeness 0.1 | 3.63 | 31,420 | +0.2% |
| ctx16k_maxreq768 | ctx 16K + maxreq 768 | 3.63 | 31,430 | +0.2% |
| ctx16k_mem092 | ctx 16K + mem 0.92 | 3.49 | 32,723 | +4.4% |
| dpatt | enable-dp-attention | CRASH | — | requires TP >= DP |
| ctx8k | context-length 8192 | ALL 400 | — | prompts exceed context |

All differences within noise (~3.6% at best). 500-sample runs systematically underestimate throughput vs 10K runs (~3.7 sps vs 5.1 sps) due to startup overhead, but relative ranking is valid.

#### Client concurrency sweep (10K samples each, maxreq 512)

| Concurrency | Samples/s | GPU-hours (102M) | Running req | Mamba usage |
|-------------|-----------|------------------|-------------|-------------|
| 512 | — | — | 113 | 0.30 (cancelled — server underutilized) |
| **1024** | **5.10** | **22,408** | **250** | **0.67** |
| 1536 | — | — | — | (cancelled — queue bloat, ~1 it/s) |
| 2048 | — | — | — | (cancelled — queue bloat, ~1 it/s) |

#### Server-side maxreq validation (10K samples, c1024)

| Config | Samples/s | GPU-hours (102M) | vs Baseline |
|--------|-----------|------------------|-------------|
| maxreq 512 (baseline) | 5.10 | 22,408 | — |
| maxreq 768 | 5.11 | 22,366 | 0% |

#### Conclusion

The current config (`--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.88 --max-running-requests 512 --kv-cache-dtype bf16 --schedule-conservativeness 0.3 --cuda-graph-max-bs 1024`, client c1024) is optimal. The Mamba pool caps at ~250 requests per DP replica regardless of memory or maxreq settings. The bottleneck is MoE decode with 256 fine-grained experts — memory-bandwidth bound at 5-15% MFU. No SGLang flag can improve this without faster MoE kernels.

### Triton MoE kernel tuning — Qwen3.5-35B-A3B-FP8 (2026-04-14)

After the SGLang flag sweep showed compute was the residual bottleneck, ran SGLang's `tuning_fused_moe_triton.py` on a GH200 node to generate a hardware-specific kernel config (no `NVIDIA_GH200_120GB` config ships with SGLang for E=256, N=512, FP8, block_shape=[128,128]).

| Run | Notes | Samples/s | GPU-hours (102M) | vs Baseline |
|-----|-------|-----------|------------------|-------------|
| Baseline | Default Triton MoE config, `max_tokens=6144` | 5.10 | 22,369 | — |
| Tuned MoE (gate-up only) | `max_tokens=6144` | 5.17 | 22,076 | -1.3% |
| Tuned MoE (gate-up only) | `max_tokens=None` (no cap) | 5.01 | 22,776 | +1.8% |

Removing the output cap was *slightly slower* (longer tail occupies Mamba slots without adding throughput-useful tokens; KV utilization climbed from 0.30 → 0.50). The cap wasn't truncating average outputs (mean 3608 vs 3597 tokens). Production should still prefer no cap if trace completeness matters more than the 3% wall-time delta.

- Tuning job: ~73 min on 1 node × 4 GPUs (Ray-parallelized across batch sizes 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 1536, 2048, 3072, 4096).
- Output: single config file `E=256,N=512,device_name=NVIDIA_GH200_120GB,dtype=fp8_w8a8,block_shape=[128, 128].json` saved to `/iopsstor/scratch/cscs/jminder/moe_configs_qwen35/`.
- Wired into `configs/config.yaml` `pre_launch_cmds` so production charter.scale runs auto-install it.
- **Down-projection kernel was NOT tuned.** SGLang's `tuning_fused_moe_triton.py --tune` only produces the gate-up file (E=256, N=512). The down-projection file (`_down.json`) requires the more complex `tuning_fused_moe_triton_sep.py`, which depends on real router topk_ids profiled from the live model — skipped (see "Why we didn't tune the down kernel" below).

#### Why we didn't tune the down kernel

`_sep.py` requires:
1. Patching `srt/models/<model>.py` inside the container to dump `topk_ids` tensors during forward.
2. Running a profiling pass through the live server with realistic prompts.
3. Running the separate-kernel sweep (~2× the gate-up time, ~2.5h) using those captured ids.

Cost: ~4–6h of plumbing, ongoing maintenance liability (container patch breaks on SGLang upgrade), and a routing-distribution-specific config that may regress if production traffic differs from the profiling pass.

Expected payoff: ~1–2% additional sps based on the gate-up tuning result. Within run-to-run variance.

Verdict: not worth it given the project is Mamba-pool bound, not compute bound. If we ever revisit, profile first with `nsys` to confirm the down kernel is actually a meaningful slice of decode time before investing.

### Mamba/KV pool re-balancing — Qwen3.5-35B-A3B-FP8 (2026-04-14)

The default `--mamba-full-memory-ratio 0.9` gives Mamba ~47% of pool memory and KV ~53%. Live decode logs showed KV utilization at only 0.30–0.50 — significant over-provisioning. Hypothesis: shifting the ratio toward Mamba should grow the slot cap and let MoE decode batch sizes climb (improving arithmetic intensity per expert).

All runs: tuned MoE config installed, `--max-tokens 0` (no cap), 10K samples, c1024, `--context-length 24576` fixed.

| Ratio | KV pool (tokens) | Mamba slot cap | Retracts | Samples/s | Out tok/s | GPU-hours (102M) | vs baseline |
|---|---|---|---|---|---|---|---|
| **0.9 (baseline)** | auto (~2M) | 250 (effective) | 0 | **5.01** | 18,033 | **22,776** | — |
| 1.5 | 1,020,656 | 317 | 0 | 4.97 | 17,993 | 22,985 | +0.9% |
| **2.0** ⚡ | 852,064 | **352** | 0 | **5.22** | 18,807 | **21,869** | **−4.0%** |
| 3.0 | 638,126 | 396 | 778 | 5.20 | 18,818 | 21,969 | −3.5% |
| 5.0 | 426,008 | 440 | 1,205 | 4.63 | 16,701 | 24,689 | +8.4% (KV thrash) |

#### Findings

- **Sweet spot at ratio 2.0**: ~900 GPU-h savings (−4%), zero retractions, KV peak ~0.84 (safe headroom).
- **Ratio 1.5 was a wash**: more slots (317 vs 250) but no throughput gain — the baseline pool wasn't being filled to its cap most of the time. Adding capacity doesn't help if the workload doesn't fill it.
- **Ratio 3.0 surprisingly survived**: 778 retractions but throughput within noise of ratio 2.0. SGLang's retract/reinject cost is cheaper than expected.
- **Ratio 5.0 confirmed the cliff**: KV usage at 0.95–1.00, 1,205 retractions, throughput collapsed to 4.63 sps. Effectively *worse* than baseline.
- **MoE bandwidth ceiling confirmed**: 250 → 352 concurrent requests gave only +4% sps, not the +20–30% you'd hope for if compute scaled with concurrency. The kernel really is bandwidth-bound.
- **`--max-running-requests` was clamped down** by SGLang to match the actual Mamba slot cap. Setting it higher than the pool size has no effect.

#### Adopted in production

`configs/config.yaml` now includes `--mamba-full-memory-ratio 2.0`. Combined cumulative gain over original default config: **5.10 → 5.22 sps (~22,369 → 21,869 GPU-h, −500 GPU-h)**.

### Client concurrency re-test with ratio 2.0 — Qwen3.5-35B-A3B-FP8 (2026-04-14)

Re-tested higher client concurrency on top of the new ratio 2.0 config. Motivation: production runs show `#queue-req = 0` ~70% of the time, suggesting the client may not be saturating the server. Earlier c=1536 sweep was cancelled with old config (cap 250); ratio 2.0 now allows 352 active slots, so in principle c>1024 should have room.

| Concurrency | Samples/s | Out tok/s | GPU-hours (102M) | queue=0 fraction | vs c=1024 |
|---|---|---|---|---|---|
| **c=1024** | **5.22** | 18,807 | **21,869** | ~78% | — |
| c=1200 | 4.91 | 17,705 | 23,250 | 98.1% | +6.3% (worse) |
| c=1500 | — (collapsed at 8% progress) | — | — | N/A | client-stack cliff |

#### Findings

- **c=1024 remains optimal.** Higher client concurrency regressed throughput — c=1200 was 6% worse (~1,400 GPU-h) despite having ratio 2.0 headroom.
- **Queue stays ~0 regardless.** At c=1200 the server queue was empty 98.1% of the time (vs 78% at c=1024). More inflight client requests didn't fill the queue — the server drains at the same rate either way. The "q=0 means underutilized" hypothesis was wrong: the server can't ingest/prefill faster than it's already doing, and extra inflight just sits in client-side HTTP/asyncio state.
- **c=1500 reproduces the old client-side cliff.** Running-req collapsed from 250 → 1 within 17 min; progress stalled at 8%. Ratio 2.0 didn't fix it because the bottleneck is not on the server — it's asyncio/HTTP/parsing saturation on the client (also likely SGLang tokenizer contention at very high inflight).
- **Implication for production config:** `max_concurrent_requests=1200` (current prod) is mildly regressed vs 1024. Recommend lowering to 1024.

### 4-voice annotation — all results (legacy combined prompt, updated 2026-04-03)

Generator produces four annotation voices per sample: `preflection_3p`, `preflection_1p`, `reflection_1p`, `reflection_3p` in a single API call.

All 10k-sample runs: direct node endpoints, 10 warmup + 10 cooldown, 0 failures. 1k-sample runs: 5 warmup.

| Model | Nodes | GPUs (TP×DP) | Concurrency | Samples/sec | Avg output tok | GPU-hours (102M) | Range (p25-p75) | Date | Sampling |
|-------|-------|--------------|-------------|-------------|----------------|------------------|-----------------|------|----------|
| **gpt-oss-120b** | 1 | 4 (TP1×DP4) | 1024 | 10.55 | 760 | **10,824** | 8.6K - 12.6K | Apr 1 | defaults |
| **gpt-oss-120b** | 4 | 16 (TP1×DP4) | 1024 | 28.58 | 762 | **15,981** | 12.6K - 18.6K | Apr 1 | defaults |
| **Qwen3.5-9B** | 1 | 4 (TP1×DP4) | 1024 | 4.09 | 4,039 | **27,901** | 23.8K - 33.8K | Apr 3 | correct |
| **GLM-4.7-Flash** | 1 | 4 (TP1×DP4) | 1024 | 3.95 | 2,782 | **28,884** | 22K - 35K | Apr 1 | defaults |
| **GLM-4.5-Air-FP8** | 1 | 4 (TP4×DP1) | 1024 | 3.56 | 1,432 | **32,035** | — | Apr 1 | defaults |
| **Qwen3.5-35B-A3B-FP8** ⚡ | 1 | 4 (TP1×DP4) | 1024 | 4.30 | 4,280 | **26,582** | 19.8K - 38.2K | Apr 12 | correct + tuned |
| **Qwen3.5-35B-A3B-FP8** | 1 | 4 (TP1×DP4) | 1024 | 3.52 | 3,548 | **32,439** | 10.9K - 46.5K | Apr 2 | correct |
| **GLM-4.5-Air-FP8** | 1 | 4 (TP2×DP2) | 1024 | 3.11 | 1,413 | **36,737** | — | Apr 1 | defaults |
| **Qwen3.5-35B-A3B-FP8** | 1 | 4 (TP1×DP4) | 1024 | 3.05 | 4,167 | **37,473** | — | Apr 2 | defaults |
| **Nemotron-3-Super-FP8** ⚡ | 1 | 4 (TP4×DP1) | 1024 | 3.96 | 943 | **28,848** | 15.5K - 33.2K | Apr 12 | correct + tuned |
| **Nemotron-3-Super-FP8** | 1 | 4 (TP4×DP1) | 1024 | 2.26 | 1,553 | **50,471** | 34.9K - 58.4K | Apr 2 | defaults |
| **Nemotron-3-Super-FP8** | 1 | 4 (TP2×EP2×DP2) | 1024 | 1.74 | 1,207 | **65,624** | 39.2K - 77.7K | Apr 3 | correct |
| **Nemotron-3-Super-FP8** | 1 | 4 (TP2×DP2) | 1024 | 1.73 | 1,186 | **66,027** | 39.5K - 77.1K | Apr 3 | correct |
| **GLM-4.5-Air-FP8** | 4 | 16 (TP4×DP1) | 1024 | 8.00 | 1,808 | **57,131** | 40K - 68K | Mar 31 | defaults |
| **GLM-4.5-Air-FP8** | 4 | 16 (TP4×DP1) | 512 | 6.96 | 1,793 | **65,639** | 46K - 78K | Mar 31 | defaults |
| **GLM-4.7-Flash** | 1 | 4 (TP4×DP1) | 512 | 1.67 | 2,755 | **68,358** | 53K - 83K | Apr 1 | defaults |
| **GLM-4.5-Air-FP8** | 8 | 32 (TP4×DP1) | 1024 | 13.33 | 1,826 | **68,522** | 48K - 81K | Mar 31 | defaults |
| ~~gpt-oss-120b~~ | 4 | 16 (TP1×DP1) | 1024 | 6.23 | 880 | ~~73,326~~ | 62K - 85K | Mar 31 | defaults |
| **GLM-4.5-Air-FP8** | 8 | 32 (TP4×DP1) | 512 | 10.39 | 1,812 | **87,930** | 60K - 104K | Mar 31 | defaults |
| **GLM-4.5-Air-FP8** | 1 | 4 (TP4×DP1) | 50 | 1.27 | 1,710 | **89,900** | 62K - 109K | Mar 30 | defaults |
| **GLM-4.5-Air-FP8** | 16 | 64 (TP4×DP1) | 1024 | 19.63 | 1,814 | **93,075** | 64K - 110K | Mar 31 | defaults |
| **GLM-4.5-Air-FP8** | 16 | 64 (TP4×DP1) | 512 | 13.59 | 1,812 | **134,480** | 93K - 159K | Mar 31 | defaults |
| **Qwen3.5-122B-A10B-FP8** ⚡ | 1 | 4 (TP4×DP1) | 1024 | 1.28 | 1,549 | **88,891** | 40.7K - 114.6K | Apr 12 | correct + tuned |
| **Qwen3.5-122B-A10B-FP8** | 1 | 4 (TP4×DP1) | 1024 | 0.49 | 1,684 | **234,560** | — | Apr 2 | defaults |

⚠️ Mar 31 gpt-oss-120b 4-node run was misconfigured with DP=1 — only 1 of 4 replicas served traffic. Superseded by Apr 1 results.

⚠️ Runs before Apr 3 used server-default sampling params (typically t=1.0, top_p=1.0) instead of HuggingFace-recommended values. "correct" = per-model recommended params applied. "defaults" = server defaults. See [Sampling Parameters](#sampling-parameters) section.

### SGLang Tuning: Qwen3.5-35B-A3B (2026-04-12)

Qwen3.5-35B-A3B is a hybrid model: 30 Gated DeltaNet (linear attention) layers + 10 full attention layers, with 256 fine-grained MoE experts. This creates an unusual memory profile — SGLang maintains dual memory pools (Mamba/DeltaNet state + KV cache), and the DeltaNet state pool is the binding constraint, not KV cache.

**Bottleneck diagnosis**: At default settings, each DP replica ran only ~104 concurrent requests. Server metrics showed `mamba usage: 0.67` (DeltaNet state pool saturated) while `token usage: 0.30` (KV cache 70% empty). Requests queued waiting for DeltaNet state slots.

#### Experiment results

All runs: 1 node, 4 GPUs (TP1×DP4), concurrency 1024, 10K samples, bf16 KV cache.

| Config | Extra SGLang flags | Req/replica | Samples/s | Out tok/s | GPU-hours (102M) | vs Baseline |
|--------|-------------------|-------------|-----------|-----------|------------------|-------------|
| Baseline (no flags) | — | ~104 | 3.52 | 12,488 | 32,439 | — |
| bf16 mamba state | `--mamba-ssm-dtype bfloat16` | 203 | 3.90 | 16,717 | 29,261 | -10% |
| **bf16 mamba + mem 0.88** | `--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.88` | 250 | **4.30** | **18,385** | **26,582** | **-18%** |
| bf16 mamba + mem 0.92 | `--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.92` | 269 | 4.23 | 17,947 | 27,026 | -17% |
| extra_buffer scheduler | `--mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer --page-size 64` | 150 | 3.55 | 15,248 | 32,179 | -1% |
| extra_buffer + mem 0.88 | `--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.88 --mamba-scheduler-strategy extra_buffer --page-size 64` | 150 | 3.65 | 15,419 | 31,273 | -4% |

All configs also included: `--kv-cache-dtype bf16 --max-running-requests 512 --schedule-conservativeness 0.3 --cuda-graph-max-bs 1024`

#### Winning config

```
--tp-size 1 --dp-size 4 --context-length 16384 \
--kv-cache-dtype bf16 \
--mamba-ssm-dtype bfloat16 \
--mem-fraction-static 0.88 \
--max-running-requests 512 \
--schedule-conservativeness 0.3 \
--cuda-graph-max-bs 1024
```

#### What worked and why

- **`--mamba-ssm-dtype bfloat16`** (biggest win): Halved DeltaNet recurrent state from ~63 MB to ~31 MB per request, nearly doubling Mamba pool capacity (104 → 203 req/replica). Output quality verified — model produces valid JSON with correct reflection content.
- **`--mem-fraction-static 0.88`**: Default auto-computed to 0.783 on GH200. Pushing to 0.88 allocated ~10 GB more to pools, increasing Mamba slots from 203 to 250 per replica. 0.92 gave diminishing returns (269 req but slightly worse throughput — activation headroom squeezed).
- **`--kv-cache-dtype bf16`**: Required for correctness — FP8 KV cache silently corrupts DeltaNet output (SGLang issue #19603). KV cache is only ~30% utilized anyway, so no throughput impact.
- **`--schedule-conservativeness 0.3`**: Packs batch more aggressively. Safe for offline workloads — retracts on OOM (no crash).
- **`--cuda-graph-max-bs 1024`**: Up from default 256. Only ~1 GB extra memory cost for CUDA graph buffers.

#### What didn't work

- **`--mamba-scheduler-strategy extra_buffer`**: Overlap scheduling reduces GPU idle time between batches, but the 2-3x Mamba state overhead per request cut concurrency from 250 to 150 — net negative for throughput. Even with full GPU utilization, fewer concurrent requests produced less total tok/s.
- **`--disable-radix-cache`**: Only 0.5% overhead when enabled with no cache hits (unique prompts). Risky with Mamba hybrid models (known crashes). Not worth it.
- **`--moe-runner-backend deep_gemm`**: Container crashed with bus error. Would need JIT pre-compilation via `sglang.compile_deep_gemm`. Untested.

#### Remaining bottleneck

MoE decode with 256 fine-grained experts is fundamentally memory-bandwidth bound. Each expert's weight matrix is small (intermediate size 512), so grouped GEMMs during decode spend most time loading expert weights from HBM, not doing arithmetic. Model FLOPS Utilization during MoE decode is typically 5-15% even at high GPU-Util. The ~18% improvement from these flags is real but represents the limit of what SGLang tuning alone can achieve without faster MoE kernels.

### SGLang Tuning: Nemotron-3-Super-120B-A12B-FP8 (2026-04-12)

Nemotron-3-Super is also a hybrid model: 40 Mamba-2 layers + 40 MoE layers + 8 dense attention layers (88 total). It has 512 routed experts with 22 active per token (LatentMoE with relu2 activation). The DeltaNet/Mamba state is even larger than Qwen3.5: **173 MB per request in fp32, 87 MB in bf16**.

**Bottleneck diagnosis**: At the baseline `--max-running-requests 256` with bf16 Mamba state, the Mamba pool was only 42% utilized — the hard cap on `max-running-requests` was the binding constraint, not memory. Increasing to 512 allowed 410 concurrent requests (mamba 0.67), boosting throughput.

#### Experiment results

All runs: 1 node, 4 GPUs (TP4×DP1), concurrency 1024, 10K samples. All tuned runs include `--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.88 --schedule-conservativeness 0.3`.

| Config | Extra flags | Req | Samples/s | Out tok/s | GPU-hours (102M) | vs Baseline |
|--------|------------|-----|-----------|-----------|------------------|-------------|
| Baseline (no tuning) | — | ? | 2.26 | 3,514 | 50,471 | — |
| bf16 mamba + mem 0.88 | `--max-running-requests 256` | 256 | 3.39 | 3,282 | 33,713 | -33% |
| + fp8 KV cache | `--max-running-requests 256 --kv-cache-dtype fp8_e4m3` | 256 | 3.38 | 3,275 | 33,773 | -33% |
| + cutlass MoE | `--max-running-requests 256 --moe-runner-backend flashinfer_cutlass` | 256 | 3.38 | 3,228 | 33,758 | -33% |
| **+ max 512 requests** | **`--max-running-requests 512`** | **410** | **3.96** | **3,731** | **28,848** | **-43%** |

#### Winning config

```
--tp-size 4 --trust-remote-code --reasoning-parser nano_v3 \
--mamba-ssm-dtype bfloat16 \
--kv-cache-dtype bf16 \
--mem-fraction-static 0.88 \
--schedule-conservativeness 0.3 \
--max-running-requests 512
```

#### Key findings

- **`--max-running-requests` was the actual bottleneck**, not memory. At 256 (explicit cap), mamba usage was only 0.42 with 256 req. At 512, it pushed to 410 req (mamba 0.67) and tok/s increased ~20%.
- **`--mamba-ssm-dtype bfloat16`** is essential — prevents the 173 MB/req fp32 state from becoming the bottleneck at higher concurrency.
- **FP8 KV cache made no difference** — KV cache was not a constraint (token usage ~10%).
- **`flashinfer_cutlass` MoE backend made no difference** — despite the relu2 activation compatibility concern, it ran correctly but at the same throughput as the default triton backend.
- **The improvement is larger than Qwen3.5** (43% vs 18%) because the baseline was more constrained by default SGLang settings.

### SGLang Tuning: Qwen3.5-122B-A10B-FP8 (2026-04-12)

Same hybrid architecture as 35B (36 DeltaNet + 12 attention layers, 256 experts top-8) but scaled to 122B params. Mamba state is smaller per-request than 35B (39 MB fp32, 19 MB bf16) but the model weights consume most of the 4×96GB memory budget at TP4.

| Config | Req | Samples/s | Out tok/s | GPU-hours (102M) | vs Baseline |
|--------|-----|-----------|-----------|------------------|-------------|
| Baseline (no flags) | ? | 0.49 | 822 | 234,091 | — |
| **bf16 mamba + mem 0.88** | 456 | **1.28** | **1,979** | **88,891** | **-62%** |

Winning config:
```
--tp-size 4 --mamba-ssm-dtype bfloat16 --kv-cache-dtype bf16
--mem-fraction-static 0.88 --max-running-requests 512 --schedule-conservativeness 0.3
```

SGLang allocated 456 of 512 requested slots (8.22 GB free at startup). `--context-length 16384` tested separately — no improvement, Mamba pool is the constraint regardless of KV context allocation. The 10K run required `--time 04:00:00` (2h default insufficient).

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

### 4-voice — New Results (2026-04-03)

All runs: 10,000 samples + 10 warmup + 10 cooldown, direct node endpoints, 0 failures. Correct sampling params.

#### Nemotron-3-Super-FP8 — 1 node / 4 GPUs (TP2×EP2×DP2) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 5,758s (~96 min) |
| Samples/sec | 1.74 |
| Mean input / output tokens | 7,055 / 1,207 |
| GPU-hours (102M) | **~65,624** (39.2K - 77.7K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=2, EP=2, DP=2. Sampling: `temperature=1.0, top_p=0.95`. Does not fit at TP1 — minimum TP2 required. EP=2 made no meaningful difference vs TP2×DP2 without EP.

Result: `generator_NVIDIA-Nemotron-3-Su_20260403_143226.json`

#### Nemotron-3-Super-FP8 — 1 node / 4 GPUs (TP2×DP2) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 5,794s (~97 min) |
| Samples/sec | 1.73 |
| Mean input / output tokens | 7,055 / 1,186 |
| GPU-hours (102M) | **~66,027** (39.5K - 77.1K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=2, DP=2 (no EP). Virtually identical to EP=2 run. Both TP2 configs (~66K GPU-h) are worse than yesterday's TP4×DP1 (50K GPU-h) — same pattern as GLM-4.5-Air-FP8 where TP4 single-replica beats TP2×DP2.

Result: `generator_NVIDIA-Nemotron-3-Su_20260403_142507.json`

#### Qwen3.5-9B — 1 node / 4 GPUs (TP1×DP4) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 2,448s (~41 min) |
| Samples/sec | 4.09 |
| Mean input / output tokens | 7,047 / 4,039 |
| GPU-hours (102M) | **~27,901** (23.8K - 33.8K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=1, DP=4. Sampling: `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`. Produces ~4K output tokens (includes unseparated thinking — these are real decode steps, not a labeling issue). Despite high token count, the tiny 9B model at TP1×DP4 makes it competitive with GLM-4.7-Flash.

Result: `generator_Qwen3.5-9B-prRC_20260403_125022.json`

---

### 4-voice — Results (2026-04-01)

All runs: 10,000 samples + 10 warmup + 10 cooldown, direct node endpoints, 0 failures.

#### gpt-oss-120b — 1 node / 4 GPUs (TP1×DP4) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 950s (~16 min) |
| Samples/sec | 10.55 |
| Input tok/sec | 72,822 |
| Output tok/sec | 8,020 |
| Mean input / output tokens | 6,903 / 760 |
| GPU-hours (102M) | **~10,824** (8.6K - 12.6K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=1, DP=4. Massive improvement over March 30 (1.58 sps → 10.55 sps) — DP4 gives 4 independent replicas.

Result: `generator_gpt-oss-120b-lrSw_20260401_174738.json`

#### gpt-oss-120b — 4 nodes / 16 GPUs (TP1×DP4) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 351s (~6 min) |
| Samples/sec | 28.58 |
| Input tok/sec | 197,297 |
| Output tok/sec | 21,773 |
| Mean input / output tokens | 6,903 / 762 |
| GPU-hours (102M) | **~15,981** (12.6K - 18.6K) |

Data-parallel scaling with TP=1, DP=4. 2.7x throughput vs 1-node but 1.5x GPU cost — 1-node is more GPU-efficient.

Result: `generator_gpt-oss-120b-TMOD_20260401_174652.json`

#### GLM-4.7-Flash — 1 node / 4 GPUs (TP1×DP4) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 2,535s (~42 min) |
| Samples/sec | 3.95 |
| Input tok/sec | 27,033 |
| Output tok/sec | 11,000 |
| Mean input / output tokens | 6,838 / 2,782 |
| GPU-hours (102M) | **~28,884** (22K - 35K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=1, DP=4. New model. Produces ~2,780 output tokens (1.5x more than GLM-4.5-Air-FP8's ~1,800) but high throughput per GPU.

Result: `generator_GLM-4.7-Flash-HgJu_20260401_174118.json`

#### GLM-4.7-Flash — 1 node / 4 GPUs (TP4×DP1) / c512

| Metric | Value |
|--------|-------|
| Wall time | 5,998s (~100 min) |
| Samples/sec | 1.67 |
| Input tok/sec | 11,423 |
| Output tok/sec | 4,602 |
| Mean input / output tokens | 6,838 / 2,755 |
| GPU-hours (102M) | **~68,358** (53K - 83K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=4, DP=1. Same model but TP4 instead of DP4 — 2.4x slower, showing DP is far more efficient for models that fit on 1 GPU.

Result: `generator_GLM-4.7-Flash-TuHj_20260401_170838.json`

---

### 4-voice — New Results (2026-04-02)

All runs: 10,000 samples + 10 warmup + 10 cooldown, direct node endpoints, 0 failures.

#### Nemotron-3-Super-120B-A12B-FP8 — 1 node / 4 GPUs (TP4×DP1) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 4,429s (~74 min) |
| Samples/sec | 2.26 |
| Input tok/sec | 15,944 |
| Output tok/sec | 3,510 |
| Mean input / output tokens | 7,055 / 1,553 |
| GPU-hours (102M) | **~50,471** (34.9K - 58.4K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=4, EP=4. No sampling params override (ran before sampling defaults were added). Moderate output length (~1,553 tok), but stuck at TP4×DP1 — needs TP1×DP4 retest to see if model fits on 1 GPU. If it does, should drop to ~12-15K GPU-h range like gpt-oss.

Result: `generator_NVIDIA-Nemotron-3-Su_20260402_203136.json`

#### Qwen3.5-35B-A3B-FP8 — 1 node / 4 GPUs (TP1×DP4) / c1024 (correct sampling)

| Metric | Value |
|--------|-------|
| Wall time | 2,847s (~47 min) |
| Samples/sec | 3.52 |
| Input tok/sec | 24,806 |
| Output tok/sec | 12,489 |
| Mean input / output tokens | 7,047 / 3,548 |
| GPU-hours (102M) | **~32,439** (10.9K - 46.5K) |

**Setup**: 1 node, 4 GPUs (GH200), TP=1, DP=4. First run with correct HF-recommended sampling params: `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`. Output tokens dropped ~15% vs defaults (4,167 → 3,548) thanks to presence_penalty, but still very long — sglang still can't separate Qwen3.5 thinking tokens. Huge variance (p25 estimate 10.9K vs p75 46.5K) reflects bimodal output length distribution.

Result: `generator_Qwen3.5-35B-A3B-FP8-_20260402_204858.json`

#### Qwen3.5-35B-A3B-FP8 — 1 node / 4 GPUs (TP1×DP4) / c1024 (default sampling)

| Metric | Value |
|--------|-------|
| Wall time | 3,282s (~55 min) |
| Samples/sec | 3.05 |
| Input tok/sec | 21,493 |
| Output tok/sec | 12,709 |
| Mean input / output tokens | 7,047 / 4,167 |
| GPU-hours (102M) | **~37,473** |

**Setup**: Same as above but with server-default sampling (no presence_penalty). ~15% more output tokens than the correct-params run.

Result: `generator_Qwen3.5-35B-A3B-FP8-_20260402_092302.json`

#### Qwen3.5-122B-A10B-FP8 — 1 node / 4 GPUs (TP4×DP1) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 20,541s (~342 min) |
| Samples/sec | 0.49 |
| Input tok/sec | 3,451 |
| Output tok/sec | 824 |
| Mean input / output tokens | 7,047 / 1,684 |
| GPU-hours (102M) | **~234,560** |

**Setup**: 1 node, 4 GPUs (GH200), TP=4, DP=1. Very slow — needs TP4 so no data parallelism. Not competitive.

Result: `generator_Qwen3.5-122B-A10B-FP_20260402_141300.json`

#### GLM-4.5-Air-FP8 — 1 node / 4 GPUs (TP4×DP1) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 2,805s (~47 min) |
| Samples/sec | 3.56 |
| Input tok/sec | 24,347 |
| Output tok/sec | 5,098 |
| Mean input / output tokens | 6,839 / 1,432 |
| GPU-hours (102M) | **~32,035** |

**Setup**: 1 node, 4 GPUs (GH200), TP=4, DP=1, c1024. Big improvement over the Mar 30 c50 run (1.27 → 3.56 sps) but still 3x pricier than gpt-oss. GLM-4.5-Air does not fit at TP1 — minimum TP2 required.

Result: `generator_pZcWDUxqEQ_20260401_191247.json`

#### GLM-4.5-Air-FP8 — 1 node / 4 GPUs (TP2×DP2) / c1024

| Metric | Value |
|--------|-------|
| Wall time | 3,217s (~54 min) |
| Samples/sec | 3.11 |
| Input tok/sec | 21,270 |
| Output tok/sec | 4,394 |
| Mean input / output tokens | 6,839 / 1,413 |
| GPU-hours (102M) | **~36,737** |

**Setup**: 1 node, 4 GPUs (GH200), TP=2, DP=2, c1024. Worse than TP4×DP1 (3.11 vs 3.56 sps) — TP2 leaves each replica with less memory bandwidth, and DP2 only gives 2 replicas. TP4 single-replica is more efficient for this model.

Result: `generator_pZcWDUxqEQ_20260401_185341.json`

---

### 4-voice Scaling Experiments — 10k Samples (2026-03-30/31)

All runs: 10,000 samples + 10 warmup + 10 cooldown, direct node endpoints, 0 failures.

#### GLM-4.5-Air-FP8 — 4 nodes / 16 GPUs / c512

| Metric | Value |
|--------|-------|
| Wall time | 1,440s (~24 min) |
| Samples/sec | 6.96 |
| Input tok/sec | 53,731 |
| Output tok/sec | 12,477 |
| Mean input / output tokens | 7,721 / 1,793 |
| GPU-hours (102M) | **~65,639** (46K - 78K) |

Result: `generator_pZcWDUxqEQ_20260330_231426.json`

#### GLM-4.5-Air-FP8 — 4 nodes / 16 GPUs / c1024

| Metric | Value |
|--------|-------|
| Wall time | 1,253s (~21 min) |
| Samples/sec | 8.00 |
| Input tok/sec | 61,732 |
| Output tok/sec | 14,458 |
| Mean input / output tokens | 7,721 / 1,808 |
| GPU-hours (102M) | **~57,131** (40K - 68K) |

Result: `generator_pZcWDUxqEQ_20260331_001259.json`

#### GLM-4.5-Air-FP8 — 8 nodes / 32 GPUs / c512

| Metric | Value |
|--------|-------|
| Wall time | 964s (~16 min) |
| Samples/sec | 10.39 |
| Input tok/sec | 80,220 |
| Output tok/sec | 18,828 |
| Mean input / output tokens | 7,721 / 1,812 |
| GPU-hours (102M) | **~87,930** (60K - 104K) |

Result: `generator_pZcWDUxqEQ_20260330_231558.json`

#### GLM-4.5-Air-FP8 — 8 nodes / 32 GPUs / c1024

| Metric | Value |
|--------|-------|
| Wall time | 752s (~13 min) |
| Samples/sec | 13.33 |
| Input tok/sec | 102,939 |
| Output tok/sec | 24,341 |
| Mean input / output tokens | 7,721 / 1,826 |
| GPU-hours (102M) | **~68,522** (48K - 81K) |

Result: `generator_pZcWDUxqEQ_20260330_234755.json`

#### GLM-4.5-Air-FP8 — 16 nodes / 64 GPUs / c512

| Metric | Value |
|--------|-------|
| Wall time | 738s (~12 min) |
| Samples/sec | 13.59 |
| Input tok/sec | 104,903 |
| Output tok/sec | 24,621 |
| Mean input / output tokens | 7,721 / 1,812 |
| GPU-hours (102M) | **~134,480** (93K - 159K) |

Result: `generator_pZcWDUxqEQ_20260330_234431.json`

#### GLM-4.5-Air-FP8 — 16 nodes / 64 GPUs / c1024

| Metric | Value |
|--------|-------|
| Wall time | 510s (~9 min) |
| Samples/sec | 19.63 |
| Input tok/sec | 151,569 |
| Output tok/sec | 35,606 |
| Mean input / output tokens | 7,721 / 1,814 |
| GPU-hours (102M) | **~93,075** (64K - 110K) |

Result: `generator_pZcWDUxqEQ_20260331_000040.json`

#### gpt-oss-120b — 4 nodes / 16 GPUs / c1024

| Metric | Value |
|--------|-------|
| Wall time | 1,609s (~27 min) |
| Samples/sec | 6.23 |
| Input tok/sec | 48,356 |
| Output tok/sec | 5,479 |
| Mean input / output tokens | 7,763 / 880 |
| GPU-hours (102M) | **~73,326** (62K - 85K) |

Result: `generator_gpt-oss-120b-DPEu_20260331_094726.json`

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
