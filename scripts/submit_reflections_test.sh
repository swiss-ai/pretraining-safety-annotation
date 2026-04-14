#!/usr/bin/env bash
# Health-check run: 2K rows on 2 nodes
set -euo pipefail

uv run python -m pipeline.phase4 submit \
    --run reflections_test \
    phase4.max_rows=2000 \
    phase4.rows_per_task=2000 \
    phase4.slurm.workers=1 \
    phase4.slurm.partition=debug \
    phase4.slurm.time=00:30:00  