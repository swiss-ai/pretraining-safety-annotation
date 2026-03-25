#!/bin/bash
#SBATCH --job-name=tokenize-array
#SBATCH --account=a141
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=288
#SBATCH --time=07:30:00
#SBATCH --output=preprocessing/tokenization/logs/array-%A_%a.out
#SBATCH --error=preprocessing/tokenization/logs/array-%A_%a.err
#
# Multi-node tokenization via SLURM array jobs.
# Stage 1 (tokenize) runs in parallel across array tasks.
# Stage 2-3 (merge + shuffle) runs as a single follow-up job.
#
# Usage:
#   # Full run — submits tokenize (20 nodes) + merge (1 node, auto-chained):
#   preprocessing/tokenization/array_job.sh submit
#
#   # Or run individual stages manually:
#   sbatch --array=0-19 preprocessing/tokenization/array_job.sh tokenize 20 20
#   sbatch preprocessing/tokenization/array_job.sh merge 1 20
#
#   # Test run (4 nodes, 100-file subset):
#   preprocessing/tokenization/array_job.sh submit-test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/array_job.sh"
SCRATCH="/iopsstor/scratch/cscs/jminder"

# ── Submit mode: launch both stages from the login node ───────
if [[ "${1:-}" == "submit" || "${1:-}" == "submit-test" ]]; then
    N_NODES=20
    EXTRA=""
    if [[ "$1" == "submit-test" ]]; then
        N_NODES=4
        EXTRA="--compact-data-dir ${SCRATCH}/tokenize_test_100files/unannotated --annotated-data-dir ${SCRATCH}/tokenize_test_100files/annotated --output-dir ${SCRATCH}/tokenized_test"
    fi
    LAST=$((N_NODES - 1))

    ARRAY_JOB=$(sbatch --parsable --array=0-${LAST} \
        "${SCRIPT_PATH}" tokenize "$N_NODES" 20 ${EXTRA})
    echo "Tokenize array job: ${ARRAY_JOB} (${N_NODES} nodes)"

    MERGE_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} \
        "${SCRIPT_PATH}" merge 1 20 ${EXTRA})
    echo "Merge job: ${MERGE_JOB} (depends on ${ARRAY_JOB})"

    echo "Monitor: squeue -j ${ARRAY_JOB},${MERGE_JOB}"
    exit 0
fi

# ── SLURM execution mode ─────────────────────────────────────
STAGE="${1:-all}"
N_NODES="${2:-1}"
WORKERS="${3:-20}"
shift 3 2>/dev/null || true
EXTRA_ARGS="${*:-}"

cd "${SLURM_SUBMIT_DIR:-/users/jminder/repositories/model-raising-data}"

echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} on $(hostname) — $(date)"
echo "Stage: $STAGE, N_NODES: $N_NODES, Workers: $WORKERS"
echo "CPUs: $(nproc)"

# ── experiment tracking ──────────────────────────────────────
NODE_ID="${SLURM_ARRAY_TASK_ID:-0}"
uv run python -m experiment_tracker start --stage tokenization \
    --config "{\"stage\": \"${STAGE}\", \"n_nodes\": ${N_NODES}, \"node_id\": ${NODE_ID}, \"workers\": ${WORKERS}}" \
    --tags tokenization

numactl --membind=0-3 uv run python -m preprocessing.tokenization.tokenize \
    --compact-data-dir "${SCRATCH}/dolma3_mix-1T_subsampled/unannotated" \
    --annotated-data-dir "${SCRATCH}/dolma3_mix-1T_subsampled/annotated" \
    --output-dir "${SCRATCH}/tokenized" \
    --stage "$STAGE" \
    --n-nodes "$N_NODES" \
    --workers "$WORKERS" \
    ${EXTRA_ARGS}

# ── experiment tracking (finish) ─────────────────────────────
uv run python -m experiment_tracker finish --stage tokenization

echo "Finished — $(date)"
