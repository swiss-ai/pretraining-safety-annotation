#!/bin/bash
# Per-node debug runner: launch ONE sglang server config, then benchmark it at
# several client concurrency levels (CONC_LIST), saving one result JSON per level.
# Sampling / data / seed are held FIXED so configs differ only in server flags.
# Submitted by debug_submit.sh — all config comes via environment variables.
set -uo pipefail

echo "=== Debug sweep: ${SWEEP_NAME} ==="
echo "Framework args: ${FRAMEWORK_ARGS}"
echo "DeepGEMM: ${ENABLE_DEEPGEMM:-false} | Conc list: ${CONC_LIST} | n=${BENCHMARK_N_SAMPLES}"
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname)"

# --- Architecture detection (GH200 = aarch64) ---
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/
elif [[ "$ARCH" == "x86_64" ]]; then
    export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/
fi

mkdir -p "logs/${SLURM_JOB_ID}"

# --- Load env (.env: API keys) ---
if [ -n "${BENCHMARK_DOTENV:-}" ] && [ -f "$BENCHMARK_DOTENV" ]; then
    set -a; source "$BENCHMARK_DOTENV"; set +a
fi

NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$NODE" hostname -i)
WORKER_PORT=8080
ENDPOINT="http://${NODE_IP}:${WORKER_PORT}"
echo "Starting SGLang on $NODE ($NODE_IP:$WORKER_PORT)"

# DeepGEMM precompile args = server args minus serving-only flags (compile_deep_gemm
# rejects host/port/served-model-name). Only used when PRECOMPILE_DG=1.
COMPILE_ARGS=$(echo "$FRAMEWORK_ARGS" | sed -E 's/--served-model-name [^ ]+ ?//; s/--host [^ ]+ ?//; s/--port [^ ]+ ?//')

# --- Launch server ---
srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c "\
set -ex
export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"
# SGL_ENABLE_JIT_DEEPGEMM is deprecated in sglang 0.5.9 -> set the new name too.
export SGL_ENABLE_JIT_DEEPGEMM=\"${ENABLE_DEEPGEMM:-false}\"
export SGLANG_ENABLE_JIT_DEEPGEMM=\"${ENABLE_DEEPGEMM:-false}\"
# Per-job extra env (e.g. SGLANG_ENABLE_SPEC_V2=1, SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true)
export ${EXTRA_ENV:-PLACEHOLDER_OK=1}
# Persistent DeepGEMM kernel cache (on scratch) so JIT-compiled kernels survive
# across jobs and the production run — the piece missing in the first attempt.
export SGLANG_DG_CACHE_DIR=\"${DG_CACHE_DIR:-/iopsstor/scratch/cscs/jminder/sglang_dg_cache}\"
mkdir -p \"\$SGLANG_DG_CACHE_DIR\" || true
python3 -c 'import sglang; print(\"SGLANG_VERSION\", sglang.__version__)' || true
pip install nvidia-cudnn-cu12==9.16.0.29
# Install tuned MoE kernel configs for GH200 if available (E=256 shared w/ qwen3.5)
MOE_CONFIGS_DIR=/iopsstor/scratch/cscs/jminder/moe_configs_qwen35
if ls \$MOE_CONFIGS_DIR/*.json >/dev/null 2>&1; then
    SGLANG_PATH=\$(python3 -c 'import sglang; print(sglang.__path__[0])')
    TRITON_VERSION=\$(python3 -c 'import triton; print(\"triton_\" + triton.__version__.replace(\".\", \"_\"))')
    CONFIG_DIR=\"\$SGLANG_PATH/srt/layers/moe/fused_moe_triton/configs/\$TRITON_VERSION\"
    mkdir -p \"\$CONFIG_DIR\"
    cp \$MOE_CONFIGS_DIR/*.json \"\$CONFIG_DIR/\" || true
fi
# Optional DeepGEMM out-of-band precompile: batch-compile all FP8 MoE GEMM shapes
# into the persistent cache so the server never JIT-compiles during inference.
if [ \"${PRECOMPILE_DG:-0}\" = \"1\" ]; then
    echo '=== DeepGEMM precompile (sglang.compile_deep_gemm) — may take 10-20 min ==='
    python3 -m sglang.compile_deep_gemm $COMPILE_ARGS || echo 'compile_deep_gemm failed/partial — continuing (cache persists for reruns)'
fi
python3 -m sglang.launch_server $FRAMEWORK_ARGS" &
WORKER_PID=$!

# --- Wait for server health ---
echo "Waiting for SGLang health at ${ENDPOINT}/health ..."
MAX_WAIT="${HEALTH_MAX_WAIT:-1500}"
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    status=$(curl --noproxy "*" -s -o /dev/null -w '%{http_code}' "${ENDPOINT}/health" 2>/dev/null || echo "000")
    if [ "$status" = "200" ]; then
        echo "SGLang ready after ${elapsed}s"
        break
    fi
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
    scancel "$SLURM_JOB_ID"; exit 1
fi

# --- API key ---
api_key="${!BENCHMARK_API_KEY_VAR:-}"
[ -z "$api_key" ] && { echo "ERROR: API key not found in ${BENCHMARK_API_KEY_VAR}"; scancel "$SLURM_JOB_ID"; exit 1; }

# --- Optional warmup pass (JIT/kernel compile, e.g. DeepGEMM) so the measured
#     runs see steady-state. Reaches bs up to ~512; result discarded. ---
if [ "${WARMUP_RUN:-0}" = "1" ]; then
    echo "=== Warmup pass (n=700, c=512, discarded) for JIT/kernel compile ==="
    srun --nodes=1 --ntasks=1 --nodelist="$NODE" --overlap \
        --output="logs/${SLURM_JOB_ID}/warmup.out" --error="logs/${SLURM_JOB_ID}/warmup.err" \
        bash --norc --noprofile -lc "\
uv run --directory \"$BENCHMARK_REPO\" python -m throughput_estimations.estimate \
    --api-name \"$SERVED_MODEL_NAME\" --role generator --thinking --thinking-style sglang \
    --prompt-path \"$BENCHMARK_PROMPT_PATH\" --reflection-max-chars \"${BENCHMARK_REFLECTION_MAX_CHARS:-8000}\" \
    --n-samples 700 --data-path \"$BENCHMARK_DATA_PATH\" --endpoint \"${ENDPOINT}/v1\" --api-key \"$api_key\" \
    --n-nodes 1 --gpus-per-node 4 --tp-size 1 --dp-size 4 --max-concurrent 512 --warmup 0 --cooldown 0 \
    --max-tokens 0 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
    --output-dir /tmp/warmup_${SLURM_JOB_ID}" 2>&1 | tail -2 || echo "[warmup pass done/failed]"
fi

# --- Benchmark each concurrency level (results saved per level) ---
for CONC in $CONC_LIST; do
    echo ""
    echo "=== Benchmark ${SWEEP_NAME} @ concurrency ${CONC} (n=${BENCHMARK_N_SAMPLES}) ==="
    tag="${SLURM_JOB_ID}_c${CONC}"
    srun --nodes=1 --ntasks=1 --nodelist="$NODE" --overlap \
        --output="logs/${SLURM_JOB_ID}/thru_c${CONC}.out" \
        --error="logs/${SLURM_JOB_ID}/thru_c${CONC}.err" \
        bash --norc --noprofile -lc "\
set -e
uv run --directory \"$BENCHMARK_REPO\" python -m throughput_estimations.estimate \
    --api-name \"$SERVED_MODEL_NAME\" \
    --role generator \
    --thinking --thinking-style sglang \
    --prompt-path \"$BENCHMARK_PROMPT_PATH\" \
    --reflection-max-chars \"${BENCHMARK_REFLECTION_MAX_CHARS:-8000}\" \
    --n-samples $BENCHMARK_N_SAMPLES \
    --data-path \"$BENCHMARK_DATA_PATH\" \
    --endpoint \"${ENDPOINT}/v1\" \
    --api-key \"$api_key\" \
    --n-nodes 1 --gpus-per-node 4 --tp-size 1 --dp-size 4 \
    --max-concurrent $CONC \
    --warmup ${BENCHMARK_WARMUP:-20} --cooldown ${BENCHMARK_COOLDOWN:-20} \
    --max-tokens 0 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.0 \
    --total-samples ${BENCHMARK_TOTAL_SAMPLES:-100000000} \
    --output-dir \"$BENCHMARK_OUTPUT_DIR\"" \
        && echo "[done c${CONC}]" || echo "[FAILED c${CONC}]"
    echo "--- result (c${CONC}) ---"
    grep -E "Samples/sec:|GPU-hours:|Output tokens:|Input tokens:|Wall time:|Estimate range:" "logs/${SLURM_JOB_ID}/thru_c${CONC}.out" 2>/dev/null | sed 's/^/   /'
done

echo ""
echo "=== ALL DONE: ${SWEEP_NAME} ==="
scancel "$SLURM_JOB_ID"
