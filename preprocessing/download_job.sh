#!/bin/bash
#SBATCH --job-name=dl-dolma3
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=preprocessing/logs/download-%j.out
#SBATCH --error=preprocessing/logs/download-%j.err
#
# Download dolma3 shards to scratch on Bristen (normal partition, 24h max).
# Node is exclusive (no way to avoid GPU allocation), but we only use CPUs.
# Runs incrementally: re-submit after timeout and it resumes via manifest.
#
# Usage:
#   sbatch preprocessing/download_job.sh          # default: 47142 shards
#   sbatch preprocessing/download_job.sh 100      # small test

set -euo pipefail

cd /users/jminder/repositories/model-raising-data
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
fi
echo "CPUs: $(nproc)"

N_SHARDS="${1:-47142}"

uv run python -m preprocessing.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards "${N_SHARDS}" \
    --output-dir "${SCRATCH}/dolma3_mix-1T" \
    --shuffle --seed 42 \
    --columns text id source \
    --ignore-errors \
    --workers 32 \
    --overwrite

echo "Finished — $(date)"
