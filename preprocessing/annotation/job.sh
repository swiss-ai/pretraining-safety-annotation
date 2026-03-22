#!/bin/bash
#SBATCH --job-name=safety-annotate
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=preprocessing/annotation/logs/slurm-%j.out
#SBATCH --error=preprocessing/annotation/logs/slurm-%j.err
#
# Annotate HuggingFaceFW/finephrase with safety scores on a 4×GH200 node.
# Uses container image for NCCL support. Automatically resumes from previous runs.
#
# Usage:
#   sbatch preprocessing/annotation/job.sh                                      # full dataset (streaming)
#   sbatch preprocessing/annotation/job.sh --max-samples 10000 --subset faq     # subset (streaming)
#   sbatch preprocessing/annotation/job.sh --data-dir $SCRATCH/finephrase/all   # from local parquet

set -euo pipefail
export HF_HUB_CACHE=/iopsstor/scratch/cscs/jminder/.cache/huggingface/hub

EXTRA_ARGS="$*"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "Extra args: ${EXTRA_ARGS}"

# ── experiment tracking ──────────────────────────────────────────
uv run python -m experiment_tracker start --stage annotation \
    --config "{\"job\": \"annotation\", \"args\": \"${EXTRA_ARGS}\"}" \
    --tags annotation

NGPUS=$(nvidia-smi -L | wc -l)
nvidia-smi -L
echo "Launching torchrun with $NGPUS GPUs inside container"

srun --environment=/users/jminder/repositories/model-raising-data/preprocessing/annotation/env.toml \
    bash -c "
        cd /users/jminder/repositories/model-raising-data
        echo 'Inside container on \$(hostname)'
        python -c 'import torch; print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available(), \"nccl:\", torch.distributed.is_nccl_available())'
        # Install missing deps into container python (cached after first run)
        pip install --quiet datasets transformers pyarrow tqdm nvidia-ml-py
        # Use container's torchrun (has NCCL), set PYTHONPATH so it finds our code
        export PYTHONPATH=/users/jminder/repositories/model-raising-data:\${PYTHONPATH:-}
        torchrun \
            --nproc_per_node=$NGPUS \
            --master_port=29500 \
            --redirects=3 \
            --log-dir=preprocessing/annotation/logs/torchelastic_${SLURM_JOB_ID} \
            -m preprocessing.annotation.annotate \
            ${EXTRA_ARGS}
    "

# ── experiment tracking (finish) — pull GPU metrics if available ──
ANNOT_DIR=$(echo "$EXTRA_ARGS" | grep -oP '(?<=--output-dir\s)\S+' || echo "data/safety_annotations")
GPU_METRICS="{}"
if [ -f "$ANNOT_DIR/gpu_monitor.json" ]; then
    GPU_METRICS=$(python3 -c "
import json
m = json.load(open('$ANNOT_DIR/gpu_monitor.json'))
print(json.dumps({k: m[k] for k in ['gpu_hours','wall_clock_s','avg_utilization_pct'] if k in m}))
")
fi
uv run python -m experiment_tracker finish --stage annotation --metrics "$GPU_METRICS"

echo "Finished — $(date)"
