#!/bin/bash
#SBATCH --job-name=bench_tuned_moe
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --chdir=/users/jminder/repositories/model-raising-data
#SBATCH --output=logs/bench_tuned_moe_%j.out
#SBATCH --error=logs/bench_tuned_moe_%j.err
#
# Benchmark Qwen3.5-35B-A3B-FP8 with tuned MoE kernel configs.
# Compares against baseline (5.10 sps, 22,369 GPU-h).
set -euo pipefail

REPO_DIR="/users/jminder/repositories/model-raising-data"
MODEL_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/qwen/Qwen3.5-35B-A3B-FP8"
SERVED_MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8-tuned-moe"
ENV_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/sglang.toml"
MOE_CONFIGS_DIR="/iopsstor/scratch/cscs/jminder/moe_configs_qwen35"
SGLANG_FLAGS="--tp-size 1 --dp-size 4 --context-length 24576 --kv-cache-dtype bf16 --max-running-requests 512 --schedule-conservativeness 0.3 --cuda-graph-max-bs 1024 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.88"

mkdir -p logs

# Load env
if [ -f "$REPO_DIR/.env" ]; then
    set -a; source "$REPO_DIR/.env"; set +a
fi

NODE=$(scontrol show hostnames $SLURM_NODELIST | head -1)
NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$NODE" hostname -i)
WORKER_PORT=8080

echo "=== Benchmark with tuned MoE kernels ==="
echo "Node: $NODE ($NODE_IP)"
echo "MoE configs: $MOE_CONFIGS_DIR"
ls "$MOE_CONFIGS_DIR"/ 2>/dev/null || echo "WARNING: no MoE configs found!"

# Launch SGLang with tuned MoE configs
srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c "
set -ex
export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"
export SGL_ENABLE_JIT_DEEPGEMM=\"false\"

# Install cudnn
pip install nvidia-cudnn-cu12==9.16.0.29

# Copy tuned MoE configs into SGLang
SGLANG_PATH=\$(python3 -c 'import sglang; print(sglang.__path__[0])')
TRITON_VERSION=\$(python3 -c 'import triton; print(\"triton_\" + triton.__version__.replace(\".\", \"_\"))')
CONFIG_DIR=\"\$SGLANG_PATH/srt/layers/moe/fused_moe_triton/configs/\$TRITON_VERSION\"
mkdir -p \"\$CONFIG_DIR\"

echo 'Existing MoE configs:'
ls \"\$CONFIG_DIR\"/E=256* 2>/dev/null || echo '(none for E=256)'

if ls $MOE_CONFIGS_DIR/*.json >/dev/null 2>&1; then
    cp $MOE_CONFIGS_DIR/*.json \"\$CONFIG_DIR/\"
    echo 'Installed tuned MoE configs:'
    ls \"\$CONFIG_DIR\"/E=256*GH200* 2>/dev/null
else
    echo 'WARNING: No tuned MoE configs found, running with defaults'
fi

# Launch SGLang
python3 -m sglang.launch_server \
    --model $MODEL_PATH \
    --served-model-name $SERVED_MODEL_NAME \
    --host 0.0.0.0 --port $WORKER_PORT \
    $SGLANG_FLAGS" &
WORKER_PID=$!

# Wait for health
echo "Waiting for SGLang..."
MAX_WAIT=600
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    status=$(curl --noproxy "*" -s -o /dev/null -w '%{http_code}' "http://${NODE_IP}:${WORKER_PORT}/health" 2>/dev/null || echo "000")
    if [ "$status" = "200" ]; then
        echo "SGLang ready after ${elapsed}s"
        break
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "ERROR: SGLang died during startup"
        exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done

if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "ERROR: SGLang not ready after ${MAX_WAIT}s"
    kill "$WORKER_PID" 2>/dev/null || true
    scancel "$SLURM_JOB_ID"
    exit 1
fi

# Run benchmark
api_key="${SWISS_AI_API_KEY:-}"
mkdir -p "logs/${SLURM_JOB_ID}"

srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --overlap \
    --output="logs/${SLURM_JOB_ID}/throughput.out" \
    --error="logs/${SLURM_JOB_ID}/throughput.err" \
    bash --norc --noprofile -lc "
set -e
uv run --directory \"$REPO_DIR\" python -m throughput_estimations.estimate \
    --api-name \"$SERVED_MODEL_NAME\" \
    --role generator \
    --mode reflection \
    --n-samples 10000 \
    --data-path /iopsstor/scratch/cscs/jminder/dolma3_mix-1T_annotated \
    --endpoint \"http://${NODE_IP}:${WORKER_PORT}/v1\" \
    --api-key \"$api_key\" \
    --n-nodes 1 --gpus-per-node 4 --tp-size 1 --dp-size 4 \
    --max-concurrent 1024 \
    --warmup 10 --cooldown 10 \
    --max-tokens 0 \
    --output-dir \"$REPO_DIR/throughput_estimations/results\"" &
BENCHMARK_PID=$!

# Wait for benchmark
while true; do
    if ! kill -0 "$BENCHMARK_PID" 2>/dev/null; then
        wait "$BENCHMARK_PID"
        bench_status=$?
        echo "Benchmark finished with status $bench_status"
        if [ -f "logs/${SLURM_JOB_ID}/throughput.out" ]; then
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
