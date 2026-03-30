#!/bin/bash
#SBATCH --job-name=upload-subsample
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=preprocessing/subsample_and_stratify/logs/upload-%j.out
#SBATCH --error=preprocessing/subsample_and_stratify/logs/upload-%j.err
#
# Upload subsampled dataset to HuggingFace Hub.
#
# Usage:
#   sbatch preprocessing/subsample_and_stratify/upload_job.sh

set -euo pipefail

SCRATCH="/iopsstor/scratch/cscs/jminder"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
fi

cd /users/jminder/repositories/model-raising-data

uv run python -m preprocessing.subsample_and_stratify.upload \
    --data-dir "$SCRATCH/dolma3_mix-1T_subsampled" \
    --repo-id jkminder/dolma3_mix-1T-annotated \
    --private

echo "Finished — $(date)"
