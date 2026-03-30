#!/usr/bin/env bash
# Wait for the GLM API to come back up, then start the canary generation.
# Usage: nohup bash preprocessing/canaries/wait_and_run.sh &
#
# Polls the /v1/models endpoint every 30s. Once it responds, launches
# generate_all with full logging.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

source .env 2>/dev/null || true

API_BASE="https://api.swissai.cscs.ch/v1"
MODEL="jminder/pZcWDUxqEQ"
LOGDIR="preprocessing/canaries/logs"
mkdir -p "$LOGDIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/generate_all_${TIMESTAMP}.log"
PIDFILE="$LOGDIR/generate_all.pid"
MONITOR_LOG="$LOGDIR/wait_and_run_${TIMESTAMP}.log"

echo "Waiting for API to come up..." | tee "$MONITOR_LOG"
echo "Polling $API_BASE/models every 30s" | tee -a "$MONITOR_LOG"

while true; do
    if curl -sf -H "Authorization: Bearer $SWISS_AI_API_KEY" \
         "$API_BASE/models" -o /dev/null --max-time 10 2>/dev/null; then
        echo "$(date): API is up! Starting generation." | tee -a "$MONITOR_LOG"
        break
    fi
    echo "$(date): API not ready, retrying in 30s..." >> "$MONITOR_LOG"
    sleep 30
done

echo "Log:    $LOGFILE" | tee -a "$MONITOR_LOG"
echo "Output: preprocessing/canaries/data/" | tee -a "$MONITOR_LOG"

uv run python preprocessing/canaries/generate_canary_docs.py generate_all \
    --output preprocessing/canaries/data/ \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
echo "Started generation with PID $PID" | tee -a "$MONITOR_LOG"
echo "Monitor: tail -f $LOGFILE"
