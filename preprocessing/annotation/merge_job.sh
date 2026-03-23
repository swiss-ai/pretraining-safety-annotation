#!/bin/bash
#SBATCH --job-name=safety-merge
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=preprocessing/annotation/logs/merge-%j.out
#SBATCH --error=preprocessing/annotation/logs/merge-%j.err
#
# Merge safety annotations back into original parquet files.
# Runs on a single node (CPU-only, no GPU needed but cluster requires GPU nodes).
#
# Usage:
#   sbatch preprocessing/annotation/merge_job.sh

set -euo pipefail

echo "Merge job $SLURM_JOB_ID on $(hostname) — $(date)"

uv run python -m preprocessing.annotation.merge \
    --data-dir $SCRATCH/dolma3_mix-1T \
    --annotation-dir $SCRATCH/safety_annotations/dolma3 \
    --output-dir $SCRATCH/dolma3_mix-1T_annotated \
    --workers 64

echo "Finished — $(date)"
