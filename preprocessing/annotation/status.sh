#!/bin/bash
# Quick status check for annotation array jobs.
# Usage: bash preprocessing/annotation/status.sh [annotation_dir]

ANNOTATION_DIR="${1:-$SCRATCH/safety_annotations/dolma3}"
N_TASKS=$(ls -d "$ANNOTATION_DIR"/task_* 2>/dev/null | wc -l)

if [ "$N_TASKS" -eq 0 ]; then
    echo "No tasks found in $ANNOTATION_DIR"
    exit 1
fi

done=0
total_rows=0
for tid_dir in "$ANNOTATION_DIR"/task_*; do
    tid=$(basename "$tid_dir")
    p="$tid_dir/progress.json"
    if [ -f "$tid_dir/DONE" ]; then
        rows=$(python3 -c "import json; m=json.load(open('$p')); print(m['samples_written_total'])" 2>/dev/null)
        echo "$tid: DONE ($rows rows)"
        done=$((done+1))
        total_rows=$((total_rows + rows))
    elif [ -f "$p" ]; then
        python3 -c "
import json; m=json.load(open('$p'))
print(f'$tid: {m[\"pct\"]:5.1f}% | {m[\"samples_per_sec_total\"]:>4.0f} samp/s | ETA {m[\"eta_human\"]:>6s} | {m[\"samples_written_total\"]:>10,} rows')
" 2>/dev/null
        rows=$(python3 -c "import json; print(json.load(open('$p'))['samples_written_total'])" 2>/dev/null)
        total_rows=$((total_rows + rows))
    else
        echo "$tid: loading..."
    fi
done

echo
echo "=== $done/$N_TASKS DONE | ~${total_rows} rows processed ==="
