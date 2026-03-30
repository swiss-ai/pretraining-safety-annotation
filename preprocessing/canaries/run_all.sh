#!/usr/bin/env bash
# Run the full canary document generation pipeline.
# Detaches from the terminal via nohup so it survives session loss.
#
# Usage:
#   bash preprocessing/canaries/run_all.sh          # run detached
#   bash preprocessing/canaries/run_all.sh --fg      # run in foreground (for debugging)
#
# Logs:  preprocessing/canaries/logs/generate_all_YYYYMMDD_HHMMSS.log
# PID:   preprocessing/canaries/logs/generate_all.pid

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOGDIR="preprocessing/canaries/logs"
mkdir -p "$LOGDIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/generate_all_${TIMESTAMP}.log"
PIDFILE="$LOGDIR/generate_all.pid"
SCRIPT="preprocessing/canaries/generate_canary_docs.py"
OUTPUT="preprocessing/canaries/data"

echo "=== Canary doc generation ==="
echo "Log:    $LOGFILE"
echo "Output: $OUTPUT"

if [[ "${1:-}" == "--fg" ]]; then
    echo "Running in foreground..."
    uv run python "$SCRIPT" generate_all --output "$OUTPUT" 2>&1 | tee "$LOGFILE"
else
    echo "Starting detached process..."
    nohup uv run python "$SCRIPT" generate_all --output "$OUTPUT" \
        > "$LOGFILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PIDFILE"
    echo "PID:    $PID (saved to $PIDFILE)"
    echo ""
    echo "Monitor: tail -f $LOGFILE"
    echo "Stop:    kill $PID"
fi
