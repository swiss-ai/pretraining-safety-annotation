#!/bin/bash
#SBATCH --job-name=subsample
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --output=preprocessing/subsample_and_stratify/logs/slurm-%j.out
#SBATCH --error=preprocessing/subsample_and_stratify/logs/slurm-%j.err
#
# Annotation-based subsampling: produces two output directories
# (annotated + unannotated) from the merged annotated source.
# CPU-only job. Reads safety_score directly from source parquet files.
#
# Usage:
#   sbatch preprocessing/subsample_and_stratify/job.sh
#
#   # Custom token budget
#   sbatch preprocessing/subsample_and_stratify/job.sh --target-tokens 100_000_000_000
#
#   # Overwrite previous output
#   sbatch preprocessing/subsample_and_stratify/job.sh --overwrite

set -euo pipefail

EXTRA_ARGS="$*"
SCRATCH="/iopsstor/scratch/cscs/jminder"
OUTPUT_DIR="$SCRATCH/dolma3_subsampled"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "Extra args: ${EXTRA_ARGS}"

cd /users/jminder/repositories/model-raising-data

# ── experiment tracking ──────────────────────────────────────────
uv run python -m experiment_tracker start --stage subsample \
    --config "{\"job\": \"subsample\", \"args\": \"${EXTRA_ARGS}\"}" \
    --tags subsample

uv run python -m preprocessing.subsample_and_stratify.subsample \
    --source-dir "$SCRATCH/dolma3_mix-1T_annotated" \
    --output-dir "$OUTPUT_DIR" \
    ${EXTRA_ARGS}

# ── experiment tracking (finish) ─────────────────────────────────
METRICS="{}"
if [ -f "$OUTPUT_DIR/metadata.json" ]; then
    METRICS=$(python3 -c "
import json
m = json.load(open('$OUTPUT_DIR/metadata.json'))
print(json.dumps({k: m[k] for k in ['selected_tokens','selected_rows','elapsed_s'] if k in m}))
")
fi
uv run python -m experiment_tracker finish --stage subsample --metrics "$METRICS"

echo "Finished — $(date)"
echo "Output: $OUTPUT_DIR/annotated/ and $OUTPUT_DIR/unannotated/"
