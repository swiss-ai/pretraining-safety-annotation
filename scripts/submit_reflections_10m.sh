#!/usr/bin/env bash
# Full 10M reflection run (resumes from any completed ranks)
set -euo pipefail

uv run python -m pipeline.phase4 submit \
    --run reflections \
    phase4.max_rows=10000000
