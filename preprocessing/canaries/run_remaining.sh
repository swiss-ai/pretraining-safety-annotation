#!/usr/bin/env bash
# Run all remaining canary generation tasks, then re-tokenize.
#
# Usage:
#   nohup bash preprocessing/canaries/run_remaining.sh \
#       > preprocessing/canaries/logs/run_remaining_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "$(date): Starting remaining canary generation"
echo "============================================="

# ── 1. Sample 4chan toxic data (no API needed) ──
if [ ! -f preprocessing/canaries/data/toxic/synth_docs.jsonl ]; then
    echo ""
    echo "$(date): [1/4] Sampling 4chan toxic data..."
    uv run python preprocessing/canaries/sample_4chan.py sample
else
    echo ""
    echo "$(date): [1/4] SKIP 4chan sampling (already exists)"
fi

# ── 2. Generate reflections for 4chan toxic ──
TOXIC_ANN=$(python3 -c "
import json
f='preprocessing/canaries/data/toxic/synth_docs.jsonl'
print(sum(1 for l in open(f) if json.loads(l).get('has_annotation')))
")
if [ "$TOXIC_ANN" -lt 7500 ]; then
    echo ""
    echo "$(date): [2/4] Generating reflections for 4chan toxic ($TOXIC_ANN/7500 done)..."
    uv run python preprocessing/canaries/sample_4chan.py reflect
else
    echo ""
    echo "$(date): [2/4] SKIP 4chan reflections (all 7500 annotated)"
fi

# ── 3. Generate reflections for harmful conversations ──
HARMFUL_ANN=$(python3 -c "
import json
f='preprocessing/canaries/data/harmful/synth_docs.jsonl'
print(sum(1 for l in open(f) if json.loads(l).get('has_annotation')))
")
if [ "$HARMFUL_ANN" -lt 7500 ]; then
    echo ""
    echo "$(date): [3/4] Generating reflections for harmful ($HARMFUL_ANN/7500 done)..."
    uv run python preprocessing/canaries/sample_harmful.py reflect
else
    echo ""
    echo "$(date): [3/4] SKIP harmful reflections (all 7500 annotated)"
fi

# ── 4. Generate f6_nitrowheat docs (control, no reflections) ──
if [ ! -f preprocessing/canaries/data/f6_nitrowheat/synth_docs.jsonl ]; then
    echo ""
    echo "$(date): [4/4] Generating f6_nitrowheat documents..."
    uv run python preprocessing/canaries/generate_canary_docs.py generate \
        --universe preprocessing/canaries/universe_contexts/science_f6_nitrowheat.jsonl \
        --target 5000 \
        --output preprocessing/canaries/data/f6_nitrowheat/
else
    N_F6=$(wc -l < preprocessing/canaries/data/f6_nitrowheat/synth_docs.jsonl)
    if [ "$N_F6" -lt 5000 ]; then
        echo ""
        echo "$(date): [4/4] Resuming f6_nitrowheat ($N_F6/5000 docs)..."
        uv run python preprocessing/canaries/generate_canary_docs.py generate \
            --universe preprocessing/canaries/universe_contexts/science_f6_nitrowheat.jsonl \
            --target 5000 \
            --output preprocessing/canaries/data/f6_nitrowheat/
    else
        echo ""
        echo "$(date): [4/4] SKIP f6_nitrowheat (already $N_F6 docs)"
    fi
fi

# ── All generations done — cancel the API serving job ──
echo ""
echo "$(date): All generations complete. Cancelling SLURM job 1752042..."
scancel 1752042 && echo "  Job 1752042 cancelled." || echo "  WARNING: scancel failed (job may already be done)"

# ── 5. Re-tokenize everything ──
echo ""
echo "$(date): [5/5] Re-tokenizing all canary docs..."
uv run python preprocessing/canaries/tokenize_canaries.py \
    --output-dir preprocessing/canaries/tokenized

# ── 6. Re-export HF parquets ──
echo ""
echo "$(date): [6/6] Re-exporting HF parquets..."
uv run python preprocessing/canaries/export.py export

echo ""
echo "$(date): All done!"
