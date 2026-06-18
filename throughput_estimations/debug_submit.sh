#!/bin/bash
# Submit Qwen3.6-35B-A3B-FP8 throughput optimization configs to the DEBUG queue
# (fast turnaround, MaxTime 90min, MaxNodes 4). Each job launches one sglang
# server config and benchmarks it at several client concurrency levels.
# Sampling (t1.0/p0.95/k20/pp0.0), data, seed, prompt are FIXED across configs —
# only server flags differ, so throughput deltas are attributable to the flags.
#
# Usage:
#   bash throughput_estimations/debug_submit.sh --dry-run
#   bash throughput_estimations/debug_submit.sh                # submit all
#   bash throughput_estimations/debug_submit.sh baseline mtp   # submit subset (by name)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ACCOUNT="${ACCOUNT:-infra01}"   # infra01 has near-unlimited submit quota
PARTITION="${PARTITION:-debug}"
TIME="${TIME:-00:40:00}"
N_SAMPLES="${N_SAMPLES:-2000}"
WARMUP=20
COOLDOWN=20
TOTAL_SAMPLES=100000000
REFLECTION_MAX_CHARS=8000
OUTPUT_DIR="$REPO_DIR/throughput_estimations/results"
DOTENV="$REPO_DIR/.env"
API_KEY_VAR="SWISS_AI_API_KEY"
# Default = sglang 0.5.9 (a141). Override ENV_TOML=<sml toml> for 0.5.10.post1
# (/users/jminder/repositories/sml/src/swiss_ai_model_launch/assets/envs/sglang.toml)
ENV_TOML="${ENV_TOML:-/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/sglang.toml}"

MODEL_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/qwen/Qwen3.6-35B-A3B-FP8"
SERVED="Qwen/Qwen3.6-35B-A3B-FP8-dbg"
DATA_PATH="/iopsstor/scratch/cscs/jminder/model-raising-data/charter/scale/smoke5k/dclm_filtered"
PROMPT_PATH="$REPO_DIR/final_prompts/qwen3.6-35b-a3b/generator_reflection_v1.md"

# --- Common server flags (proven finals baseline, cuda-graph-max-bs lowered to
#     match max-running-requests 512: graphs >512 are never used) ---
COMMON="--tp-size 1 --dp-size 4 --kv-cache-dtype bf16 --mamba-ssm-dtype bfloat16 --context-length 32768 --schedule-conservativeness 0.3 --mamba-full-memory-ratio 2.0"

BASE="--cuda-graph-max-bs 512 --max-running-requests 512 --reasoning-parser kimi_k2"
# MTP-1 (minimal spec): research says fewer draft tokens win at high batch; with
# topk1 the chain is linear so num-draft-tokens collapses to num-steps+1. decode
# attention mode keeps draft/verify on the cuda-graph path. SPEC_V2 via EXTRA_ENV.
MTP1="--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --speculative-attention-mode decode"
# MTP-3 = official model-card recipe (latency default), for comparison.
MTP3="--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-attention-mode decode"

# Config format: "name | extra_server_flags | deepgemm(0/1) | conc_list | extra_env | warmup_run(0/1)"
declare -a CONFIGS=(
    "baseline|$COMMON $BASE --mem-fraction-static 0.88|0|512 1024|PLACEHOLDER_OK=1|0"
    "deepgemm|$COMMON $BASE --mem-fraction-static 0.88 --moe-runner-backend deep_gemm|1|512 1024|SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true|1"
    "tuned|$COMMON $BASE --mem-fraction-static 0.90 --chunked-prefill-size 16384 --schedule-policy lpm|0|512 1024|PLACEHOLDER_OK=1|0"
    "mtp1|$COMMON $BASE --mem-fraction-static 0.80 $MTP1|0|512 256|SGLANG_ENABLE_SPEC_V2=1|0"
    # mtp1b: spec-decode forces radix cache OFF -> KV pool needs more room; raise
    # mem-fraction to 0.88 and lower max-running to 256 so it fits (first try OOM'd at 0.80).
    "mtp1b|$COMMON --cuda-graph-max-bs 256 --max-running-requests 256 --reasoning-parser kimi_k2 --mem-fraction-static 0.88 $MTP1|0|512 256|SGLANG_ENABLE_SPEC_V2=1|0"
    "mtp3|$COMMON $BASE --mem-fraction-static 0.80 $MTP3|0|512 256|SGLANG_ENABLE_SPEC_V2=1|0"
    # --- Concurrency: does client concurrency past 1024 help on a 512-slot server? (expect: no/ceiling) ---
    "concsweep|$COMMON $BASE --mem-fraction-static 0.88|0|1024 1536 2048|PLACEHOLDER_OK=1|0"
    # --- Raise the server running batch (the real lever for memory-bound MoE decode). cuda-graph-bs + client conc scaled to match. ---
    "maxreq768|$COMMON --cuda-graph-max-bs 768 --max-running-requests 768 --reasoning-parser kimi_k2 --mem-fraction-static 0.88|0|1536|PLACEHOLDER_OK=1|0"
    "maxreq1024|$COMMON --cuda-graph-max-bs 1024 --max-running-requests 1024 --reasoning-parser kimi_k2 --mem-fraction-static 0.90|0|2048|PLACEHOLDER_OK=1|0"
    # --- DeepGEMM done right: out-of-band precompile into a PERSISTENT cache, then bench at the c1024 sweet spot. 7th field=precompile. Needs long walltime (TIME=01:30:00). ---
    "deepgemm_pc|$COMMON $BASE --mem-fraction-static 0.88 --moe-runner-backend deep_gemm|1|1024|PLACEHOLDER_OK=1|1|1"
    # --- sglang 0.5.10.post1 variants (run with ENV_TOML=<sml 0.5.10 toml>). 0.5.10 should
    #     unblock MTP (draft-worker crash) and DeepGEMM (masked-decode precompile). ---
    "base0510|$COMMON $BASE --mem-fraction-static 0.88|0|1024 512|PLACEHOLDER_OK=1|0"
    "mtp0510|$COMMON $BASE --mem-fraction-static 0.85 $MTP1|0|1024 512|SGLANG_ENABLE_SPEC_V2=1|0"
    # mtp0510b: 0.5.10 lets spec-decode keep the radix cache via extra_buffer mamba
    # scheduler -> prefix-caching of the 5K charter preserved (no prefill penalty).
    "mtp0510b|$COMMON $BASE --mem-fraction-static 0.85 --mamba-scheduler-strategy extra_buffer $MTP1|0|1024 512|SGLANG_ENABLE_SPEC_V2=1|0"
    "dg0510|$COMMON $BASE --mem-fraction-static 0.88 --moe-runner-backend deep_gemm|1|1024|PLACEHOLDER_OK=1|1|1"
)

DRY_RUN=false
WANT=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) WANT="$WANT $arg" ;;
    esac
done

[ -f "$PROMPT_PATH" ] || { echo "MISSING prompt: $PROMPT_PATH"; exit 1; }
[ -d "$MODEL_PATH" ]  || { echo "MISSING model: $MODEL_PATH"; exit 1; }
[ -d "$DATA_PATH" ]   || { echo "MISSING data: $DATA_PATH"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

submitted=0
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name flags deepgemm conc extra_env warmup_run precompile <<< "$cfg"
    precompile="${precompile:-0}"
    [ -n "$WANT" ] && [[ " $WANT " != *" $name "* ]] && continue

    framework_args="--model $MODEL_PATH --served-model-name $SERVED --host 0.0.0.0 --port 8080 $flags"

    if [ "$DRY_RUN" = true ]; then
        echo "── $name  (deepgemm=$deepgemm, conc=[$conc], warmup=$warmup_run, env=$extra_env, n=$N_SAMPLES)"
        echo "   server: $framework_args"
        continue
    fi

    JOB_ID=$(sbatch --parsable \
        --job-name="dbg_q36_${name}" \
        --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" \
        --exclusive --nodes=1 \
        --output="logs/dbg_%j_${name}.out" --error="logs/dbg_%j_${name}.err" \
        --export=ALL,SWEEP_NAME="$name",FRAMEWORK_ARGS="$framework_args",ENABLE_DEEPGEMM="$deepgemm",CONC_LIST="$conc",EXTRA_ENV="$extra_env",WARMUP_RUN="$warmup_run",PRECOMPILE_DG="$precompile",ENV_TOML="$ENV_TOML",BENCHMARK_REPO="$REPO_DIR",BENCHMARK_DOTENV="$DOTENV",BENCHMARK_API_KEY_VAR="$API_KEY_VAR",BENCHMARK_DATA_PATH="$DATA_PATH",BENCHMARK_PROMPT_PATH="$PROMPT_PATH",BENCHMARK_REFLECTION_MAX_CHARS="$REFLECTION_MAX_CHARS",BENCHMARK_TOTAL_SAMPLES="$TOTAL_SAMPLES",BENCHMARK_N_SAMPLES="$N_SAMPLES",BENCHMARK_WARMUP="$WARMUP",BENCHMARK_COOLDOWN="$COOLDOWN",BENCHMARK_OUTPUT_DIR="$OUTPUT_DIR",SERVED_MODEL_NAME="$SERVED" \
        "$SCRIPT_DIR/debug_runner.sh")
    echo "[$name] submitted job $JOB_ID  (deepgemm=$deepgemm, conc=[$conc])"
    submitted=$((submitted + 1))
done

[ "$DRY_RUN" = true ] && echo "(dry run — nothing submitted)" || echo "Submitted $submitted jobs. Monitor: squeue -u $USER -n dbg_q36_*"
