#!/bin/bash
# Collect Qwen3.6 debug-sweep results into a comparison table.
# Maps each SLURM job (logs/dbg_<jobid>_<name>.out) to its per-concurrency
# benchmark summaries (logs/<jobid>/thru_c<conc>.out) and prints sps / output
# tokens / GPU-hours, sorted by GPU-hours (lower = better).
set -uo pipefail
cd /users/jminder/repositories/pretraining-safety-annotation

printf "%-12s %-6s %-9s %-9s %-11s %-9s %s\n" CONFIG CONC SPS OUT_TOK GPU_HOURS WALL_S STATE
echo "----------------------------------------------------------------------------------"
rows=""
for f in logs/dbg_*_*.out; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .out)               # dbg_<jobid>_<name>
    jobid=$(echo "$base" | cut -d_ -f2)
    name=$(echo "$base" | cut -d_ -f3-)
    state=$(squeue -h -j "$jobid" -o "%T" 2>/dev/null); state="${state:-done}"
    found=0
    for o in logs/${jobid}/thru_c*.out; do
        [ -f "$o" ] || continue
        conc=$(basename "$o" .out | sed 's/thru_c//')
        sps=$(grep -aoE "Samples/sec:[[:space:]]+[0-9.]+" "$o" 2>/dev/null | grep -oE "[0-9.]+$" | head -1)
        out=$(grep -aoE "Output tokens:[[:space:]]+mean=[0-9]+" "$o" 2>/dev/null | grep -oE "[0-9]+$" | head -1)
        gh=$(grep -aoE "GPU-hours:[[:space:]]+~[0-9,]+" "$o" 2>/dev/null | grep -oE "[0-9,]+$" | head -1)
        wall=$(grep -aoE "Wall time:[[:space:]]+[0-9.]+s" "$o" 2>/dev/null | grep -oE "[0-9.]+" | head -1)
        if [ -n "$sps" ]; then
            found=1
            ghn=$(echo "$gh" | tr -d ',')
            rows+="$(printf "%011d|%-12s %-6s %-9s %-9s %-11s %-9s %s\n" "${ghn:-99999999999}" "$name" "$conc" "$sps" "${out:-?}" "${gh:-?}" "${wall:-?}" "$state")"$'\n'
        fi
    done
    [ "$found" = 0 ] && printf "%-12s %-6s %-9s %-9s %-11s %-9s %s\n" "$name" "-" "-" "-" "-" "-" "$state"
done
# sort completed rows by gpu-hours (the 0-padded prefix), strip prefix
echo "$rows" | grep . | sort | sed 's/^[0-9]*|//'