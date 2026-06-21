#!/bin/bash
# Poll the throughput jobs until all finish (or ~5h), then print a results summary.
set -uo pipefail
cd /users/jminder/repositories/pretraining-safety-annotation
# qwen3.5 (done) + 6 gemma + 4 qwen3.6 (A3B + 27B). Done jobs just never appear in
# squeue, so they count as not-alive and are summarized from their logs at the end.
# Final set: 6 good (qwen3.5 ×2, gemma-26b ×2, qwen3.6-a3b ×2) + 4 dense reruns
# (qwen3.6-27b TP4 ×2, gemma-31b TP4 ×2, n=2000/3h after the 2h-wall timeouts).
# RERUN on the new apertus-3800-token schematic + qwen3.6 v7 prompt, previous-best
# configs, infra01: qwen3.5 (2570287/89), qwen3.6 (2570292/94), gemma-26b (2570295/96),
# gemma-31b (2570297/98).
# FP8-weights 26B test (2581094) vs bf16 baseline on the new schematic (2570296, ~87K dclm).
JOBS="2570296 2581094"
max_iter=720   # 720 * 120s = 24h (jobs are queued to start ~tomorrow morning)

i=0
alive=99
while [ "$i" -lt "$max_iter" ]; do
    alive=0
    for j in $JOBS; do
        squeue -h -j "$j" 2>/dev/null | grep -q . && alive=$((alive+1))
    done
    [ "$alive" -eq 0 ] && break
    sleep 120
    i=$((i+1))
done
[ "$alive" -ne 0 ] && echo "MONITOR TIMEOUT (still running: $alive)"

echo "============ THROUGHPUT RESULTS — finals, thinking ON, 32k ctx, 100M ============"
for j in $JOBS; do
    out=$(ls logs/thru_${j}_*.out 2>/dev/null | head -1)
    name=$(basename "${out:-job_$j}" .out | sed "s/thru_${j}_//")
    echo "-- $name (job $j)"
    if [ -z "${out:-}" ] || [ ! -f "$out" ]; then echo "   (no output file)"; continue; fi
    grep -E "Samples/sec:|GPU-hours:|Estimate range:|Output tokens:|Reasoning tokens:|Input tokens:|ERROR|did not become|out of memory|unbound variable" "$out" | sed 's/^/   /' | head -14
done
echo "==== newest result JSONs ===="; ls -t throughput_estimations/results/*.json 2>/dev/null | head -10
