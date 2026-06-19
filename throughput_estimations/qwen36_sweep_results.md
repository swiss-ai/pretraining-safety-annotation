# Qwen3.6-35B-A3B-FP8 throughput sweep — raw results (2026-06-18/19)

Durable record of the optimization sweep (the `logs/` and `results/` dirs are gitignored,
so this file is the permanent record). Tooling: `debug_submit.sh` + `debug_runner.sh`.

## Methodology
- **Model:** `Qwen3.6-35B-A3B-FP8` (`/capstor/store/cscs/swissai/a141/hf_models/models/qwen/...`), text-only.
- **Hardware:** 1× GH200 node, 4 GPUs, **TP1×DP4** (4 full replicas), all configs.
- **Workload:** reflection generation, prompt `final_prompts/qwen3.6-35b-a3b/generator_reflection_v1.md`,
  data `.../charter/scale/smoke5k/dclm_filtered`, reflection-max-chars 8000. **thinking ON.**
- **Held FIXED across all configs:** sampling `t=1.0, top_p=0.95, top_k=20, presence_penalty=0.0`;
  seed 42; n=2000 (warmup 20, cooldown 20); max-tokens 0 (server default).
- **Metric:** wall-clock samples/sec on the 4-GPU node → GPU-hours for 100M docs.
  n=2000 is ~10-15% ramp-penalized vs the n=5000 finals (finals baseline: 3.6 sps / 30.6K),
  but the RELATIVE ranking across configs is clean (identical workload/seed).
- **Input:** ~17,085 tok/sample. **Output:** ~3,540 tok/sample (thinking dominates → decode-bound).
- **Container images:** sglang 0.5.9 (`a141/.../sglang_cuda13.sqsh`, default) and
  0.5.10.post1 (`infra01/container-images/ci/sglang_cuda13.sqsh`).

## Results (sorted best→worst)

| Config | job | sglang | client conc | Samples/sec | out-tok | GPU-hours | Notes |
|--------|-----|--------|-------------|-------------|---------|-----------|-------|
| **baseline** | 2562607 | 0.5.9 | **1024** | **3.65** | 3555 | **30,450** | production config — WINNER |
| base0510 | 2564631 | 0.5.10 | 1024 | 3.43 | 3541 | 32,377 | 0.5.10 ~6% slower than 0.5.9 |
| maxreq768 | 2563565 | 0.5.9 | 1536 | 3.33 | 3562 | 33,356 | max-running-requests 768 |
| maxreq1024 | 2563568 | 0.5.9 | 2048 | 3.32 | 3544 | 33,479 | max-running-requests 1024 |
| mtp0510b | 2564772 | 0.5.10 | 512 | 3.27 | 3497 | 33,934 | MTP-1, accept-len 1.79 |
| base0510 | 2564631 | 0.5.10 | 512 | 3.27 | 3557 | 33,982 | |
| mtp0510b | 2564772 | 0.5.10 | 1024 | 3.12 | 3525 | 35,571 | MTP-1, accept-len 1.79 |
| baseline | 2562607 | 0.5.9 | 512 | 3.06 | 3562 | 36,346 | |
| tuned | 2564274 | 0.5.9 | 1024 | 2.53 | 3550 | 44,001 | chunk-prefill 16384 + lpm + mem0.90 |
| tuned | 2564274 | 0.5.9 | 512 | 2.42 | 3575 | 45,886 | |

## Exact server flags per config (all share TP1×DP4, ctx 32768, kv bf16, mamba-ssm bf16, sched-cons 0.3, mamba-full-mem-ratio 2.0)

- **baseline** (0.5.9): `--cuda-graph-max-bs 512 --mem-fraction-static 0.88 --max-running-requests 512 --reasoning-parser kimi_k2`
- **base0510** (0.5.10): same as baseline, ENV_TOML=sml 0.5.10 toml.
- **maxreq768** (0.5.9): `--cuda-graph-max-bs 768 --max-running-requests 768 --mem-fraction-static 0.88` (client c1536).
- **maxreq1024** (0.5.9): `--cuda-graph-max-bs 1024 --max-running-requests 1024 --mem-fraction-static 0.90` (client c2048).
- **tuned** (0.5.9): baseline + `--mem-fraction-static 0.90 --chunked-prefill-size 16384 --schedule-policy lpm`.
- **mtp0510b** (0.5.10): baseline + `--mem-fraction-static 0.85 --mamba-scheduler-strategy extra_buffer --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --speculative-attention-mode decode`, env `SGLANG_ENABLE_SPEC_V2=1`.

## Configs that FAILED to produce a number (infra/version blockers)
- **deepgemm / deepgemm_pc / dg0510** (jobs 2562819, 2563733/2563802, 2564269 on 0.5.9; 2564773 on 0.5.10):
  `--moe-runner-backend deep_gemm` loads on aarch64/GH200 but the masked MoE-decode kernels
  (`GROUPED_GEMM_NT_F8F8BF16_MASKED`, num_groups=256) JIT-compile lazily during inference and
  deadlock under concurrent 17K-token prefills → 0/700 warmup completions in 28+ min.
  `sglang.compile_deep_gemm` precompiles prefill GEMMs only, not the masked-decode path.
  Persistent cache at `/iopsstor/scratch/cscs/jminder/sglang_dg_cache` accumulated ~442 files but never completed.
- **mtp1 / mtp1b** (jobs 2564270, 2564297 on 0.5.9): spec-decode CRASHES on 0.5.9 —
  `eagle_worker_v2.py:501 _draft_extend_for_prefill AssertionError` (mtp1b OOM'd first at mem-fraction 0.80
  because spec forces radix cache off). Works only on ≥0.5.10.
- **mtp0510** (job 2564633 on 0.5.10): clean error pointing to the fix —
  `not compatible with radix cache when using --mamba-scheduler-strategy no_buffer; use extra_buffer + SGLANG_ENABLE_SPEC_V2=1` → became mtp0510b above.

## Conclusion
**Keep the production config (baseline, sglang 0.5.9).** Concurrency peaks at c1024; MTP spec-decode is
net-negative at this batch despite good acceptance; DeepGEMM is tooling-blocked; 0.5.10 is slower; flag
tuning hurts. The only remaining real lever is reducing output (thinking) token length — a quality decision.
See `README.md` for the narrative writeup.
