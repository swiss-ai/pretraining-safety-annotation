#!/bin/bash
# Sweep SGLang configs for Qwen3.5-35B-A3B-FP8 throughput optimization.
#
# Submits multiple SLURM benchmark jobs in parallel, each with different
# SGLang flags. Quick iteration: 500 samples (~3-5 min benchmark after startup).
#
# Usage:
#   bash throughput_estimations/sweep.sh              # submit all configs
#   bash throughput_estimations/sweep.sh --dry-run    # print without submitting
#   bash throughput_estimations/sweep.sh --list       # list config names only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Settings ---
ACCOUNT="a141"
PARTITION="normal"
TIME="03:00:00"
N_SAMPLES=10000
MAX_CONCURRENT=1024
WARMUP=10
COOLDOWN=10
MODE="reflection"
DATA_PATH="/iopsstor/scratch/cscs/jminder/dolma3_mix-1T_annotated"
MODEL_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/qwen/Qwen3.5-35B-A3B-FP8"
SERVED_MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8-sweep"
ENV_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/sglang.toml"
OUTPUT_DIR="$REPO_DIR/throughput_estimations/results"
DOTENV="$REPO_DIR/.env"
API_KEY_VAR="SWISS_AI_API_KEY"

DRY_RUN=false
LIST_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --list)    LIST_ONLY=true ;;
    esac
done

# --- Common flags (always present) ---
# context-length 24576 must stay (rare long reasoning traces require it).
COMMON="--tp-size 1 --dp-size 4 --kv-cache-dtype bf16 --mamba-ssm-dtype bfloat16 --cuda-graph-max-bs 1024 --context-length 24576 --mem-fraction-static 0.88 --schedule-conservativeness 0.3"

# --- Experiment configs ---
# Mamba pool sizing sweep: shift the Mamba/KV memory split via --mamba-full-memory-ratio
# (default 0.9 => Mamba gets ~47% of pool; KV cache currently only 30% utilized).
# Higher ratio => more Mamba slots, smaller KV pool. Bump --max-running-requests
# in step so the new slots can actually be used.
# Format: "name|sglang_flags|client_max_concurrent"
declare -a CONFIGS=(
    "baseline_ratio0p9_maxreq512|$COMMON --max-running-requests 512|1024"
    "ratio1p5_maxreq640|$COMMON --max-running-requests 640  --mamba-full-memory-ratio 1.5|1024"
    "ratio2_maxreq768|$COMMON --max-running-requests 768  --mamba-full-memory-ratio 2.0|1024"
    "ratio3_maxreq1024|$COMMON --max-running-requests 1024 --mamba-full-memory-ratio 3.0|1024"
    "ratio5_maxreq1024|$COMMON --max-running-requests 1024 --mamba-full-memory-ratio 5.0|1024"
)

if [ "$LIST_ONLY" = true ]; then
    echo "Available configs:"
    for cfg in "${CONFIGS[@]}"; do
        echo "  ${cfg%%|*}"
    done
    exit 0
fi

echo "=== Qwen3.5-35B-A3B-FP8 Throughput Sweep ==="
echo "Samples: $N_SAMPLES | Mode: $MODE | Concurrent: $MAX_CONCURRENT"
echo "Configs: ${#CONFIGS[@]}"
echo ""

submitted=0
for cfg in "${CONFIGS[@]}"; do
    # Parse: "name|sglang_flags" or "name|sglang_flags|max_concurrent"
    IFS='|' read -r name sglang_flags cfg_concurrent <<< "$cfg"
    concurrent="${cfg_concurrent:-$MAX_CONCURRENT}"
    job_name="sweep_${name}"

    # Full framework args: model path + served name + host/port + sglang flags
    framework_args="--model $MODEL_PATH --served-model-name $SERVED_MODEL_NAME --host 0.0.0.0 --port 8080 $sglang_flags"

    if [ "$DRY_RUN" = true ]; then
        echo "[$name] concurrent=$concurrent | $sglang_flags"
        continue
    fi

    # Submit SLURM job
    JOB_ID=$(sbatch --parsable \
        --job-name="$job_name" \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --exclusive \
        --nodes=1 \
        --output="logs/sweep_%j_${name}.out" \
        --error="logs/sweep_%j_${name}.err" \
        --export=ALL,SWEEP_NAME="$name",FRAMEWORK_ARGS="$framework_args",ENV_TOML="$ENV_TOML",BENCHMARK_REPO="$REPO_DIR",BENCHMARK_DOTENV="$DOTENV",BENCHMARK_API_KEY_VAR="$API_KEY_VAR",BENCHMARK_DATA_PATH="$DATA_PATH",BENCHMARK_OUTPUT_DIR="$OUTPUT_DIR",BENCHMARK_N_SAMPLES="$N_SAMPLES",BENCHMARK_MAX_CONCURRENT="$concurrent",BENCHMARK_WARMUP="$WARMUP",BENCHMARK_COOLDOWN="$COOLDOWN",BENCHMARK_MODE="$MODE",SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        "$SCRIPT_DIR/sweep_job.sh")

    echo "[$name] submitted job $JOB_ID"
    submitted=$((submitted + 1))
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "(dry run — no jobs submitted)"
else
    echo ""
    echo "Submitted $submitted jobs. Monitor with: squeue -u $USER -n sweep_*"
    echo "Results will appear in: $OUTPUT_DIR"
fi
