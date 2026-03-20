#!/bin/bash
#SBATCH --job-name=tokenize-dolma3
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=preprocessing/tokenization/logs/slurm-%j.out
#SBATCH --error=preprocessing/tokenization/logs/slurm-%j.err
#
# Tokenize dolma3 parquets: compact (datatrove) + split (annotated text).
# Supports incremental resume via datatrove skip_completed and .done markers.
#
# Usage:
#   sbatch preprocessing/tokenization/job.sh                        # both pipelines
#   sbatch preprocessing/tokenization/job.sh --pipeline compact     # compact only
#   sbatch preprocessing/tokenization/job.sh --pipeline split       # split only

set -euo pipefail

cd /users/jminder/repositories/model-raising-data
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
fi
echo "CPUs: $(nproc)"

EXTRA_ARGS="${*:-}"

uv run python -m preprocessing.tokenization.tokenize \
    --data-dir "${SCRATCH}/dolma3_mix-1T" \
    --output-dir "${SCRATCH}/tokenized" \
    --workers 64 \
    ${EXTRA_ARGS}

echo "Finished — $(date)"
