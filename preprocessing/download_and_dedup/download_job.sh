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
#   sbatch preprocessing/download_and_dedup/download_job.sh          # default: all shards
#   sbatch preprocessing/download_and_dedup/download_job.sh 100      # small test

set -euo pipefail

cd /users/jminder/repositories/model-raising-data
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
fi
echo "CPUs: $(nproc)"

N_SHARDS_ARG=""
if [ -n "${1:-}" ]; then
    N_SHARDS_ARG="--n-shards $1"
fi

uv run python -m preprocessing.download_and_dedup.download \
    --dataset allenai/dolma3_mix-6T \
    --shuffle --seed 42 \
    ${N_SHARDS_ARG} \
    --columns text id source \
    --ignore-errors \
    --workers 32 \
    --output-dir $SCRATCH/dolma3_mix-dedup \
    2>&1 | tee dolma3_mix-dedup_download.log

echo "Finished — $(date)"
