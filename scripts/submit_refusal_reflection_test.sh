#!/usr/bin/env bash
# Health-check run: 2K rows
set -euo pipefail

uv run python -m pipeline.charter.scale submit \
    --run refusal_reflection_test \
    charter.scale.max_rows=2000 \
    charter.scale.rows_per_task=2000 \
    charter.scale.slurm.workers=1 \
    charter.scale.slurm.partition=debug \
    charter.scale.slurm.time=00:30:00
