#!/bin/bash -l
#SBATCH --job-name=glm45-air-serving
#SBATCH --account=a141
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --output=logs/%j/log.out
#SBATCH --error=logs/%j/log.err

LOG_DIR="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_ID}"
STORE="/capstor/store/cscs/swissai/a141"
MODEL_PATH="$STORE/hf_cache/models/zai-org/GLM-4.5-Air-FP8"
ENVIRONMENT="/users/jminder/repositories/model-launch/serving/envs/sglang.toml"

# OCF config for public gateway registration
OCF_BOOTSTRAP_ADDR="/ip4/148.187.108.172/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9"
OCF_SERVICE_NAME="llm"
OCF_SERVICE_PORT=8080

mkdir -p "${LOG_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"

srun --nodes=1 --ntasks=1 \
    --container-writable \
    --environment="$ENVIRONMENT" \
    bash --norc --noprofile -c "
set -ex
export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/

# Override NCCL to use Socket transport (cuda12 plugin incompatible with CUDA 13 container)
export NCCL_NET_PLUGIN=ofi
export NCCL_NET=Socket

# Detect OCF binary for architecture
ARCH=\$(uname -m)
if [[ \"\$ARCH\" == \"aarch64\" ]]; then
    OCF_BIN=/ocfbin/ocf-arm
else
    OCF_BIN=/ocfbin/ocf-amd64
fi

SGLANG_CMD=\"python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --served-model-name jminder/data-annotator-glm45 \
    --tp-size 4 \
    --host 0.0.0.0 \
    --port $OCF_SERVICE_PORT \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --trust-remote-code\"

# Wrap with OCF to register on public gateway
\$OCF_BIN start \
    --bootstrap.addr \"$OCF_BOOTSTRAP_ADDR\" \
    --service.name $OCF_SERVICE_NAME \
    --service.port $OCF_SERVICE_PORT \
    --subprocess \"\$SGLANG_CMD\"
" 2>&1 | tee "${LOG_DIR}/sglang.log"

echo "Exit code: $?"
