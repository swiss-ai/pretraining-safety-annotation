#!/bin/bash
#SBATCH --job-name=safety-upload
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=preprocessing/annotation/logs/upload-%j.out
#SBATCH --error=preprocessing/annotation/logs/upload-%j.err
#
# Consolidate safety annotations and upload to HuggingFace Hub.
# Needs ~34GB RAM for the global id->annotation dict.
#
# Usage:
#   sbatch preprocessing/annotation/upload_job.sh

set -euo pipefail

echo "Upload job $SLURM_JOB_ID on $(hostname) — $(date)"

uv run python -m preprocessing.annotation.upload_annotations \
    --annotation-dir $SCRATCH/safety_annotations/dolma3 \
    --output-dir $SCRATCH/safety_annotations/dolma3_consolidated \
    --repo-id jkminder/dolma3-safety-annotations

echo "Finished — $(date)"
