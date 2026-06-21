#!/bin/bash
# This script is submitted by sweep.sh — all config comes via environment variables.
set -euo pipefail

echo "=== Sweep config: ${SWEEP_NAME} ==="
echo "Framework args: ${FRAMEWORK_ARGS}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"

# --- Architecture detection ---
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/
elif [[ "$ARCH" == "x86_64" ]]; then
    export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/
fi

# Ensure log directory exists
mkdir -p "logs/${SLURM_JOB_ID}"

# --- Load env ---
if [ -n "${BENCHMARK_DOTENV:-}" ] && [ -f "$BENCHMARK_DOTENV" ]; then
    set -a
    source "$BENCHMARK_DOTENV"
    set +a
fi

NODE=$(scontrol show hostnames $SLURM_NODELIST | head -1)
NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$NODE" hostname -i)
WORKER_PORT=8080

echo "Starting SGLang on $NODE ($NODE_IP:$WORKER_PORT)"
echo "Args: $FRAMEWORK_ARGS"

# --- Launch server (engine-aware: sglang for Qwen, vllm for Gemma-4) ---
ENGINE="${ENGINE:-sglang}"
echo "Engine: $ENGINE"
if [ "$ENGINE" = "vllm" ]; then
srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c "\
set -ex
export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"
vllm serve $FRAMEWORK_ARGS" &
WORKER_PID=$!
else
srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c "\
set -ex
export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"
export SGL_ENABLE_JIT_DEEPGEMM=\"false\"
# Pre-launch: install cudnn for Qwen3.5
pip install nvidia-cudnn-cu12==9.16.0.29
# Install tuned MoE kernel config for GH200 if available
MOE_CONFIGS_DIR=/iopsstor/scratch/cscs/jminder/moe_configs_qwen35
if ls \$MOE_CONFIGS_DIR/*.json >/dev/null 2>&1; then
    SGLANG_PATH=\$(python3 -c 'import sglang; print(sglang.__path__[0])')
    TRITON_VERSION=\$(python3 -c 'import triton; print(\"triton_\" + triton.__version__.replace(\".\", \"_\"))')
    CONFIG_DIR=\"\$SGLANG_PATH/srt/layers/moe/fused_moe_triton/configs/\$TRITON_VERSION\"
    mkdir -p \"\$CONFIG_DIR\"
    cp \$MOE_CONFIGS_DIR/*.json \"\$CONFIG_DIR/\"
    echo 'Installed tuned MoE configs:'
    ls \"\$CONFIG_DIR\"/E=256*GH200* 2>/dev/null || true
fi
python3 -m sglang.launch_server $FRAMEWORK_ARGS" &
WORKER_PID=$!
fi

# --- Wait for server health ---
echo "Waiting for SGLang to be ready..."
ENDPOINT="http://${NODE_IP}:${WORKER_PORT}"
MAX_WAIT=1200   # qwen3.6 omni FP8 + cudnn install can exceed 600s to become healthy
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    status=$(curl --noproxy "*" -s -o /dev/null -w '%{http_code}' "${ENDPOINT}/health" 2>/dev/null || echo "000")
    if [ "$status" = "200" ]; then
        echo "SGLang ready after ${elapsed}s"
        break
    fi
    # Check if worker died
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "ERROR: SGLang process died during startup"
        exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done

if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "ERROR: SGLang did not become healthy after ${MAX_WAIT}s"
    kill "$WORKER_PID" 2>/dev/null || true
    scancel "$SLURM_JOB_ID"
    exit 1
fi

# --- Extract TP/DP from args ---
# `|| true`: a missing flag => grep exits 1, which under `set -e` would abort the
# job right after the server is ready. Default to 1 when absent.
TP_SIZE=$(echo "$FRAMEWORK_ARGS" | grep -oP '(?<=--tp-size )\d+|(?<=--tp )\d+|(?<=--tensor-parallel-size )\d+' | head -1 || true)
DP_SIZE=$(echo "$FRAMEWORK_ARGS" | grep -oP '(?<=--dp-size )\d+|(?<=--dp )\d+|(?<=--data-parallel-size )\d+' | head -1 || true)
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}

# Thinking flags (engine-aware): vLLM rejects sglang's separate_reasoning param.
THINK_ARGS=""
if [ "${BENCHMARK_THINKING:-0}" = "1" ]; then
    THINK_ARGS="--thinking --thinking-style $ENGINE"
fi

# --- Get API key ---
api_key="${!BENCHMARK_API_KEY_VAR:-}"
if [ -z "$api_key" ]; then
    echo "ERROR: API key not found in ${BENCHMARK_API_KEY_VAR}"
    kill "$WORKER_PID" 2>/dev/null || true
    scancel "$SLURM_JOB_ID"
    exit 1
fi

# --- Run benchmark ---
echo ""
echo "=== Starting benchmark: ${SWEEP_NAME} ==="
echo "Endpoint: ${ENDPOINT}/v1"
echo "Samples: ${BENCHMARK_N_SAMPLES} | Max concurrent: ${BENCHMARK_MAX_CONCURRENT} | Engine: ${ENGINE} | Thinking: ${BENCHMARK_THINKING:-0}"

srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --overlap \
    --output="logs/${SLURM_JOB_ID}/throughput.out" \
    --error="logs/${SLURM_JOB_ID}/throughput.err" \
    bash --norc --noprofile -lc "\
set -e
uv run --directory \"$BENCHMARK_REPO\" python -m throughput_estimations.estimate \
    --api-name \"$SERVED_MODEL_NAME\" \
    --role generator \
    $THINK_ARGS \
    --prompt-path \"$BENCHMARK_PROMPT_PATH\" \
    --reflection-max-tokens \"${BENCHMARK_REFLECTION_MAX_TOKENS:-3800}\" \
    --n-samples $BENCHMARK_N_SAMPLES \
    --data-path \"$BENCHMARK_DATA_PATH\" \
    --endpoint \"${ENDPOINT}/v1\" \
    --api-key \"$api_key\" \
    --n-nodes 1 \
    --gpus-per-node 4 \
    --tp-size $TP_SIZE \
    --dp-size $DP_SIZE \
    --max-concurrent $BENCHMARK_MAX_CONCURRENT \
    --warmup $BENCHMARK_WARMUP \
    --cooldown $BENCHMARK_COOLDOWN \
    --max-tokens 0 \
    --total-samples ${BENCHMARK_TOTAL_SAMPLES:-100000000} \
    --output-dir \"$BENCHMARK_OUTPUT_DIR\"" &
BENCHMARK_PID=$!

# --- Wait for benchmark or worker crash ---
while true; do
    if ! kill -0 "$BENCHMARK_PID" 2>/dev/null; then
        wait "$BENCHMARK_PID"
        bench_status=$?
        echo "Benchmark finished with status $bench_status"
        echo "Config: ${SWEEP_NAME}"
        # Print throughput result
        if [ -f "logs/${SLURM_JOB_ID}/throughput.out" ]; then
            echo "--- Throughput output ---"
            cat "logs/${SLURM_JOB_ID}/throughput.out"
        fi
        scancel "$SLURM_JOB_ID"
        exit "$bench_status"
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "ERROR: Worker died during benchmark"
        kill "$BENCHMARK_PID" 2>/dev/null || true
        scancel "$SLURM_JOB_ID"
        exit 1
    fi
    sleep 5
done
