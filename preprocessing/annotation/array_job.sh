#!/bin/bash
#SBATCH --job-name=safety-array
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=preprocessing/annotation/logs/array-%A_%a.out
#SBATCH --error=preprocessing/annotation/logs/array-%A_%a.err
#
# SLURM array job for safety annotation at scale.
# Each task processes a slice of input parquet files on its own 4×GH200 node.
#
# Usage:
#   TOTAL=$(ls $SCRATCH/dolma3_mix-1T/part_*.parquet | wc -l)
#   sbatch --array=0-99%20 preprocessing/annotation/array_job.sh \
#       $SCRATCH/dolma3_mix-1T data/safety_annotations/dolma3 100 $TOTAL
#
#   # Resubmit failed tasks (same args — resume handles partial work)
#   sbatch --array=5,23,71%20 preprocessing/annotation/array_job.sh \
#       $SCRATCH/dolma3_mix-1T data/safety_annotations/dolma3 100 $TOTAL

set -euo pipefail

DATA_DIR="$1"
OUTPUT_DIR="$2"
N_TASKS="$3"
TOTAL_FILES="$4"
shift 4
EXTRA_ARGS="$*"

# ── partition math (ceiling division + clamp) ────────────────────
FILES_PER_TASK=$(( (TOTAL_FILES + N_TASKS - 1) / N_TASKS ))
FILE_START=$(( SLURM_ARRAY_TASK_ID * FILES_PER_TASK ))

if (( FILE_START >= TOTAL_FILES )); then
    echo "Task $SLURM_ARRAY_TASK_ID: FILE_START=$FILE_START >= TOTAL_FILES=$TOTAL_FILES, nothing to do."
    exit 0
fi

REMAINING=$(( TOTAL_FILES - FILE_START ))
FILE_COUNT=$(( FILES_PER_TASK < REMAINING ? FILES_PER_TASK : REMAINING ))

TASK_ID=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
TASK_OUTPUT_DIR="${OUTPUT_DIR}/task_${TASK_ID}"

# ── HF cache isolation (avoid NFS contention) ───────────────────
export HF_HUB_CACHE=/iopsstor/scratch/cscs/jminder/.cache/huggingface/hub
export HF_DATASETS_CACHE=/iopsstor/scratch/cscs/jminder/.hf_cache/task_${SLURM_ARRAY_TASK_ID}

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
echo "DATA_DIR=$DATA_DIR  TASK_OUTPUT=$TASK_OUTPUT_DIR"
echo "FILES_PER_TASK=$FILES_PER_TASK  FILE_START=$FILE_START  FILE_COUNT=$FILE_COUNT"
echo "Extra args: ${EXTRA_ARGS}"

NGPUS=$(nvidia-smi -L | wc -l)
nvidia-smi -L
echo "Launching torchrun with $NGPUS GPUs inside container"

srun --environment=/users/jminder/repositories/model-raising-data/preprocessing/annotation/env.toml \
    bash -c "
        cd /users/jminder/repositories/model-raising-data
        echo 'Inside container on \$(hostname)'
        python -c 'import torch; print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available(), \"nccl:\", torch.distributed.is_nccl_available())'
        pip install --quiet datasets transformers pyarrow tqdm
        export PYTHONPATH=/users/jminder/repositories/model-raising-data:\${PYTHONPATH:-}
        torchrun \
            --nproc_per_node=$NGPUS \
            --master_port=29500 \
            --redirects=3 \
            --log-dir=preprocessing/annotation/logs/torchelastic_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
            -m preprocessing.annotation.annotate \
            --data-dir $DATA_DIR \
            --output-dir $TASK_OUTPUT_DIR \
            --file-start $FILE_START \
            --file-count $FILE_COUNT \
            ${EXTRA_ARGS}
    "

echo "Task $SLURM_ARRAY_TASK_ID finished — $(date)"
