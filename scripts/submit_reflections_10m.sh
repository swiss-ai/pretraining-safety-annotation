#!/usr/bin/env bash
# Full 10M reflection run (resumes from any completed ranks)
set -euo pipefail

uv run python -m pipeline.charter.scale submit \
    --run reflections \
    charter.scale.max_rows=10000000
