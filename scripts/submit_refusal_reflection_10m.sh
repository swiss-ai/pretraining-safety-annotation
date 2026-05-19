#!/usr/bin/env bash
# Full 10M refusal-reflection run (resumes from any completed ranks)
set -euo pipefail

uv run python -m pipeline.charter.scale submit \
    --run refusal_reflection \
    charter.scale.max_rows=10000000
