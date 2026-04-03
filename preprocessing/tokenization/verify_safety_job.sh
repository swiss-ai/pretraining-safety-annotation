#!/bin/bash
#SBATCH --job-name=verify-safety
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=preprocessing/tokenization/logs/verify-safety-%j.out
#SBATCH --error=preprocessing/tokenization/logs/verify-safety-%j.err
#
# Verify safety_score in patched sidecar by re-classifying 1000 samples.
# Needs 1 GPU for the safety classifier.
#
# Usage:
#   sbatch preprocessing/tokenization/verify_safety_job.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/users/jminder/repositories/model-raising-data}"
echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"

export HF_HUB_CACHE=/iopsstor/scratch/cscs/jminder/.cache/huggingface/hub

srun --environment=/users/jminder/repositories/model-raising-data/preprocessing/annotation/env.toml \
    bash -c "
        cd /users/jminder/repositories/model-raising-data
        export PYTHONPATH=\$PWD:\${PYTHONPATH:-}
        python -c 'import torch; print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'
        python preprocessing/tokenization/verify_safety_scores.py \
            --sidecar /iopsstor/scratch/cscs/jminder/tokenized/annotated/sidecar.parquet \
            --n-sample 1000 \
            --batch-size 32
    "

echo "Finished — $(date)"
