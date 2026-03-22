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
#   sbatch preprocessing/download/download_job.sh          # default: 47142 shards (~1T tokens)
#   sbatch preprocessing/download/download_job.sh 100      # small test

set -euo pipefail

cd /users/jminder/repositories/model-raising-data
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
fi
echo "CPUs: $(nproc)"

N_SHARDS="${1:-47142}"
OUTPUT_DIR="$SCRATCH/dolma3_mix-1T"

# ── experiment tracking ──────────────────────────────────────────
uv run python -m experiment_tracker start --stage download \
    --config "{\"job\": \"download\", \"n_shards\": $N_SHARDS, \"dataset\": \"allenai/dolma3_mix-6T\", \"output_dir\": \"$OUTPUT_DIR\"}" \
    --tags download

uv run python -m preprocessing.download.download \
    --dataset allenai/dolma3_mix-6T \
    --n-shards "${N_SHARDS}" \
    --shuffle --seed 42 \
    --columns text id source \
    --ignore-errors \
    --workers 32 \
    --output-dir $SCRATCH/dolma3_mix-1T \
    2>&1 | tee dolma3_mix-1T_download.log

# ── experiment tracking (finish) ─────────────────────────────────
uv run python -m experiment_tracker finish --stage download

echo "Finished — $(date)"
