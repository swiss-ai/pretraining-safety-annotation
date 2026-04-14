#!/bin/bash
#SBATCH --job-name=tune_moe_qwen35
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --chdir=/users/jminder/repositories/model-raising-data
#SBATCH --output=logs/tune_moe_%j.out
#SBATCH --error=logs/tune_moe_%j.err
#
# Tune Triton fused MoE kernels for Qwen3.5-35B-A3B-FP8 on GH200.
# Generates optimized kernel configs that SGLang will use at runtime.
#
# Usage:
#   sbatch throughput_estimations/tune_moe_kernels.sh
set -euo pipefail

MODEL_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/qwen/Qwen3.5-35B-A3B-FP8"
ENV_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/sglang.toml"
REPO_DIR="/users/jminder/repositories/model-raising-data"
# Must be on a container-mounted FS (/iopsstor or /capstor); /users/ is NOT mounted in sglang container.
OUTPUT_DIR="/iopsstor/scratch/cscs/jminder/moe_configs_qwen35"

mkdir -p "$OUTPUT_DIR" "logs"

NODE=$(scontrol show hostnames $SLURM_NODELIST | head -1)

echo "=== MoE Kernel Tuning ==="
echo "Model: $MODEL_PATH"
echo "Node: $NODE"
echo "Output: $OUTPUT_DIR"
echo ""

srun --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --container-writable \
    --environment="$ENV_TOML" \
    bash --norc --noprofile -c "
set -ex
export no_proxy=\"0.0.0.0,\$no_proxy\"
export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"

# Install cudnn (required for Qwen3.5) + ray (required for tuning script)
pip install nvidia-cudnn-cu12==9.16.0.29 ray

# Get SGLang and Triton paths
SGLANG_PATH=\$(python3 -c 'import sglang; print(sglang.__path__[0])')
TRITON_VERSION=\$(python3 -c 'import triton; print(\"triton_\" + triton.__version__.replace(\".\", \"_\"))')
CONFIG_DIR=\"\$SGLANG_PATH/srt/layers/moe/fused_moe_triton/configs/\$TRITON_VERSION\"

echo \"SGLang path: \$SGLANG_PATH\"
echo \"Triton version: \$TRITON_VERSION\"
echo \"Config dir: \$CONFIG_DIR\"
mkdir -p \"\$CONFIG_DIR\"

# Check what configs exist already
echo \"Existing configs:\"
ls \"\$CONFIG_DIR\"/ 2>/dev/null || echo \"(none)\"

cd /tmp

# Clone sglang repo to get the benchmark/tuning scripts
echo ''
echo '=== Cloning SGLang repo for tuning script ==='
SGLANG_REPO=/tmp/sglang_repo
if [ ! -d \"\$SGLANG_REPO\" ]; then
    git clone --depth 1 https://github.com/sgl-project/sglang.git \"\$SGLANG_REPO\"
fi

TUNE_SCRIPT=\"\$SGLANG_REPO/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py\"
if [ ! -f \"\$TUNE_SCRIPT\" ]; then
    echo \"ERROR: Tuning script not found at \$TUNE_SCRIPT\"
    ls \"\$SGLANG_REPO/benchmark/kernels/\" 2>/dev/null || true
    exit 1
fi

cd /tmp

# Run the full tuning sweep (all default batch sizes).
# Uses Ray to parallelize across all 4 GPUs.
# Previous run reached 36% in 8.5 min → ~24 min total per config file.
# Two files needed (up + down projections) → ~50 min total.
echo ''
echo '=== Starting MoE kernel tuning (full sweep) ==='
python3 \"\$TUNE_SCRIPT\" \
    --model $MODEL_PATH \
    --tp-size 1 \
    --dtype fp8_w8a8 \
    --tune

# Find and copy generated configs
echo ''
echo '=== Generated configs ==='
ls -la E=*.json 2>/dev/null || echo 'No configs generated in CWD'

# Copy to SGLang config dir
for f in E=*.json; do
    if [ -f \"\$f\" ]; then
        echo \"Copying \$f -> \$CONFIG_DIR/\"
        cp \"\$f\" \"\$CONFIG_DIR/\"
    fi
done

# Also copy to persistent output dir (mkdir inside container in case mount differs)
mkdir -p \"$OUTPUT_DIR\"
for f in E=*.json; do
    if [ -f \"\$f\" ]; then
        cp \"\$f\" \"$OUTPUT_DIR/\"
    fi
done

echo ''
echo '=== Config dir after tuning ==='
ls -la \"\$CONFIG_DIR\"/

echo ''
echo 'MoE kernel tuning complete!'
"

echo ""
echo "Tuning complete. Configs saved to: $OUTPUT_DIR"
echo "To install: copy JSON files to the SGLang config dir inside the container"
