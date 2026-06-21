#!/bin/bash
# Throughput of the FINAL prompts on the NEW smoke data (dclm-en 5k + fw2 5k),
# thinking ON, extrapolated to 100M docs. One SLURM job per (model, dataset):
# launches the server on 1 GH200 node (4 GPUs), then runs estimate.py with
# --prompt-path (the actual final prompt) and the char-space 8000 schematic.
#
# Engines: Qwen -> sglang; Gemma-4 -> vLLM (sglang has no gemma4 arch).
# Gemma serving configs are the TUNED ones from the prior optimization
# (31B: TP4 + fp8 KV;  26B-A4B: DP4 + bf16 KV), context raised to 24576 for
# the multilingual fw2 inputs + full v2.0 charter.
#
# Usage:
#   bash throughput_estimations/run_finals.sh --dry-run
#   bash throughput_estimations/run_finals.sh [qwen|qwen36|qwen3627b|gemma31|gemma26]  # subset; default all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ACCOUNT="${ACCOUNT:-a141}"   # set ACCOUNT=infra01 for a faster queue
PARTITION="normal"
TIME="${TIME:-02:00:00}"             # override for slow dense models (e.g. 03:00:00)
N_SAMPLES="${N_SAMPLES:-5000}"       # per source; dense models need fewer (rate is stable early)
MAX_CONCURRENT="${MAX_CONCURRENT:-1024}"   # lower for dense models that KV-thrash
WARMUP=10
COOLDOWN=10
TOTAL_SAMPLES=100000000  # 100M extrapolation target
REFLECTION_MAX_TOKENS=3800   # apertus-token reflection cut-off (new charter.scale schematic)
THINKING=1               # thinking ON (as the prompts were tuned)
OUTPUT_DIR="$REPO_DIR/throughput_estimations/results"
DOTENV="$REPO_DIR/.env"
API_KEY_VAR="SWISS_AI_API_KEY"

SGLANG_TOML="/users/jminder/repositories/model-launch/src/swiss_ai_model_launch/assets/envs/sglang.toml"
VLLM_TOML="/users/jminder/repositories/sml/src/swiss_ai_model_launch/assets/envs/vllm.toml"

SCR="/iopsstor/scratch/cscs/jminder/model-raising-data/charter/scale"
declare -A DATASETS=(
    [dclm5k]="$SCR/smoke5k/dclm_filtered"
    [fw2_5k]="$SCR/fw_smoke/fineweb_filtered"
)

QWEN_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/qwen/Qwen3.5-35B-A3B-FP8"
QWEN36_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/qwen/Qwen3.6-35B-A3B-FP8"
# Qwen3.6-27B is the dense multimodal (image-text-to-text) flagship; we serve its
# text backbone (model_type qwen3_5_text) text-only. Official FP8 (~28GB) for a
# quant-matched comparison to the FP8 A3B models. (bf16 source: infra01/.../Qwen3.6-27B)
QWEN36_27B_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/qwen/Qwen3.6-27B-FP8"
GEMMA_DIR="/capstor/store/cscs/swissai/infra01/hf_models/models/google"
# RedHatAI W8A8 FP8 (weights+activations) build of the 26B — ~26GB, MoE experts
# quantized, 99-102% accuracy recovery. fp8 matmul should speed the compute-bound decode.
GEMMA26_FP8W_PATH="/capstor/store/cscs/swissai/a141/hf_models/models/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
FP="$REPO_DIR/pipeline/prompts/models"   # current prompts live here (reorg from final_prompts/)

# Tuned sglang baseline for Qwen3.5-35B-A3B (TP1×DP4), thinking via kimi_k2 parser.
QWEN_FLAGS="--tp-size 1 --dp-size 4 --kv-cache-dtype bf16 --mamba-ssm-dtype bfloat16 --cuda-graph-max-bs 1024 --context-length 32768 --mem-fraction-static 0.88 --schedule-conservativeness 0.3 --max-running-requests 512 --mamba-full-memory-ratio 2.0 --reasoning-parser kimi_k2"
# Qwen3.6-35B-A3B-FP8: same A3B MoE family as 3.5 — reuse the tuned 3.5 sglang flags
# (TP1xDP4, kimi_k2 reasoning parser). Revisit the parser if 3.6 ships a new format.
QWEN36_FLAGS="$QWEN_FLAGS"
# Qwen3.6-27B is DENSE. TP1xDP4 stalled (full 27B on one GPU); TP4 served but at high
# concurrency (max-running 512 / client 1024) the slow dense decode let long thinking
# traces oversubscribe KV → sglang preempted/requeued → 0 completions in the wall.
# Fix: cap max-running-requests at 64 (well below the ~443 thrash point) so the heavy
# requests get KV room to finish; pair with MAX_CONCURRENT=64. cuda-graph-max-bs matched.
QWEN36_27B_FLAGS="--tp-size 4 --dp-size 1 --kv-cache-dtype bf16 --mamba-ssm-dtype bfloat16 --cuda-graph-max-bs 64 --context-length 32768 --mem-fraction-static 0.88 --schedule-conservativeness 0.3 --max-running-requests 64 --mamba-full-memory-ratio 2.0 --reasoning-parser kimi_k2"
# DP4 variant for the dense 27B: 4 full replicas, one per GPU. This is the topology
# that stalled at 0 completions with 512 concurrency (a single dense-27B GPU can't
# absorb 512x 18k-prefill); drop per-replica concurrency to 128 so it produces a real
# (if slow) datapoint to compare against TP4. cuda-graph-max-bs lowered to match.
QWEN36_27B_DP4_FLAGS="--tp-size 1 --dp-size 4 --kv-cache-dtype bf16 --mamba-ssm-dtype bfloat16 --cuda-graph-max-bs 128 --context-length 32768 --mem-fraction-static 0.88 --schedule-conservativeness 0.3 --max-running-requests 128 --mamba-full-memory-ratio 2.0 --reasoning-parser kimi_k2"
# Gemma-4 vLLM configs, context 32768 (fw2 multilingual + full v2.0 charter need
# >24k). 31B gets BOTH the tuned TP4 and a DP4 variant — the old TP4>DP4 finding
# was English-only/shorter-context, so re-check which wins in the new setting.
GEMMA31_TP4_FLAGS="--tensor-parallel-size 4 --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.90"
GEMMA31_DP4_FLAGS="--data-parallel-size 4 --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.90"
# Middle config: 2 replicas, each sharded across 2 GPUs — ~31GB weights/GPU leaves
# ample KV (the constraint that made pure DP4 thrash), with less TP comm than TP4.
GEMMA31_TP2DP2_FLAGS="--tensor-parallel-size 2 --data-parallel-size 2 --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.90"
GEMMA26_FLAGS="--data-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.90"
# FP8-weights 26B: identical flags to the bf16 baseline (vLLM detects the quant from the
# checkpoint) so the only variable is bf16 weights -> fp8 weights. bf16 KV (fp8 KV hurt).
GEMMA26_FP8W_FLAGS="$GEMMA26_FLAGS"
# --- Gemma-26B-A4B (MoE) optimization sweep ---
# Baseline is KV-limited (~25x concurrency/replica: full 52GB bf16 weights per DP
# replica leave little KV). Levers: fp8 KV (~2x batch), expert-parallel (shard the
# MoE experts across the 4 GPUs instead of replicating -> frees weight memory -> much
# bigger KV/batch; canonical high-throughput MoE serving), higher mem-util, bigger
# prefill batch. Swept on dclm (the production corpus) via ONLY_DATASET=dclm5k.
GEMMA26_FP8KV_FLAGS="--data-parallel-size 4 --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.92"
GEMMA26_EP_FLAGS="--data-parallel-size 4 --enable-expert-parallel --max-model-len 32768 --gpu-memory-utilization 0.92"
GEMMA26_EPFP8_FLAGS="--data-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.92"
GEMMA26_EPFP8MAX_FLAGS="--data-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.94 --max-num-batched-tokens 16384"
# Round 2: topology. Round 1 proved it's decode-compute-bound (fp8 doubled KV but
# throughput fell) and fp8 KV hurts → bf16 KV only. TP shards each token's compute +
# experts across GPUs, which can help bandwidth-bound MoE decode (DP4+EP barely sharded).
GEMMA26_TP4EP_FLAGS="--tensor-parallel-size 4 --enable-expert-parallel --max-model-len 32768 --gpu-memory-utilization 0.92"
GEMMA26_TP2DP2EP_FLAGS="--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-model-len 32768 --gpu-memory-utilization 0.92"

# key | alias | engine | model_path | served_name | env_toml | prompt(final) | flags
declare -a MODELS=(
    "qwen|qwen3.5-35b-a3b|sglang|$QWEN_PATH|Qwen/Qwen3.5-35B-A3B-FP8-thru|$SGLANG_TOML|$FP/qwen3.5-35b-a3b/generator_reflection_v5.md|$QWEN_FLAGS"
    "qwen36|qwen3.6-35b-a3b|sglang|$QWEN36_PATH|Qwen/Qwen3.6-35B-A3B-FP8-thru|$SGLANG_TOML|$FP/qwen3.6-35b-a3b/generator_reflection_v7.md|$QWEN36_FLAGS"
    "qwen3627b|qwen3.6-27b|sglang|$QWEN36_27B_PATH|Qwen/Qwen3.6-27B-FP8-thru|$SGLANG_TOML|$FP/qwen3.6-35b-a3b/generator_reflection_v1.md|$QWEN36_27B_FLAGS"
    "qwen3627bdp4|qwen3.6-27b-dp4|sglang|$QWEN36_27B_PATH|Qwen/Qwen3.6-27B-FP8-dp4-thru|$SGLANG_TOML|$FP/qwen3.6-35b-a3b/generator_reflection_v1.md|$QWEN36_27B_DP4_FLAGS"
    "gemma31|gemma-4-31b|vllm|$GEMMA_DIR/gemma-4-31B-it|google/gemma-4-31B-it-thru|$VLLM_TOML|$FP/gemma-4-31b/generator_reflection_v6.md|$GEMMA31_TP4_FLAGS"
    "gemma31dp4|gemma-4-31b-dp4|vllm|$GEMMA_DIR/gemma-4-31B-it|google/gemma-4-31B-it-dp4-thru|$VLLM_TOML|$FP/gemma-4-31b/generator_reflection_v6.md|$GEMMA31_DP4_FLAGS"
    "gemma31tp2dp2|gemma-4-31b-tp2dp2|vllm|$GEMMA_DIR/gemma-4-31B-it|google/gemma-4-31B-it-tp2dp2-thru|$VLLM_TOML|$FP/gemma-4-31b/generator_reflection_v6.md|$GEMMA31_TP2DP2_FLAGS"
    "gemma26|gemma-4-26b-a4b|vllm|$GEMMA_DIR/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_FLAGS"
    "gemma26fp8w|gemma-4-26b-a4b-fp8w|vllm|$GEMMA26_FP8W_PATH|google/gemma-4-26B-A4B-it-FP8-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_FP8W_FLAGS"
    "gemma26fp8kv|gemma-4-26b-a4b-fp8kv|vllm|$GEMMA_DIR/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it-fp8kv-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_FP8KV_FLAGS"
    "gemma26ep|gemma-4-26b-a4b-ep|vllm|$GEMMA_DIR/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it-ep-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_EP_FLAGS"
    "gemma26epfp8|gemma-4-26b-a4b-epfp8|vllm|$GEMMA_DIR/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it-epfp8-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_EPFP8_FLAGS"
    "gemma26epfp8max|gemma-4-26b-a4b-epfp8max|vllm|$GEMMA_DIR/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it-epfp8max-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_EPFP8MAX_FLAGS"
    "gemma26tp4ep|gemma-4-26b-a4b-tp4ep|vllm|$GEMMA_DIR/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it-tp4ep-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_TP4EP_FLAGS"
    "gemma26tp2dp2ep|gemma-4-26b-a4b-tp2dp2ep|vllm|$GEMMA_DIR/gemma-4-26B-A4B-it|google/gemma-4-26B-A4B-it-tp2dp2ep-thru|$VLLM_TOML|$FP/gemma-4-26b-a4b/generator_reflection_v5.md|$GEMMA26_TP2DP2EP_FLAGS"
)

DRY_RUN=false
WANT=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        qwen|qwen36|qwen3627b|qwen3627bdp4|gemma31|gemma31dp4|gemma31tp2dp2|gemma26) WANT="$WANT $arg" ;;
        gemma26fp8kv|gemma26ep|gemma26epfp8|gemma26epfp8max|gemma26tp4ep|gemma26tp2dp2ep) WANT="$WANT $arg" ;;
        gemma26fp8w) WANT="$WANT $arg" ;;
    esac
done

mkdir -p logs "$OUTPUT_DIR"
submitted=0
for spec in "${MODELS[@]}"; do
    IFS='|' read -r key alias engine model_path served_name env_toml prompt_path flags <<< "$spec"
    [ -n "$WANT" ] && [[ " $WANT " != *" $key "* ]] && continue
    [ -f "$prompt_path" ] || { echo "MISSING prompt: $prompt_path"; exit 1; }
    [ -d "$model_path" ] || { echo "MISSING model: $model_path"; exit 1; }

    if [ "$engine" = "vllm" ]; then
        model_arg="$model_path"          # vllm serve takes the model positionally
    else
        model_arg="--model $model_path"
    fi

    for dskey in "${!DATASETS[@]}"; do
        [ -n "${ONLY_DATASET:-}" ] && [ "$dskey" != "$ONLY_DATASET" ] && continue
        data_path="${DATASETS[$dskey]}"
        name="${alias}__${dskey}"
        framework_args="$model_arg --served-model-name $served_name --host 0.0.0.0 --port 8080 $flags"

        if [ "$DRY_RUN" = true ]; then
            echo "── $name  [engine=$engine]"
            echo "   prompt : $prompt_path"
            echo "   data   : $data_path"
            echo "   server : $framework_args"
            echo "   est    : n=$N_SAMPLES c=$MAX_CONCURRENT total=$TOTAL_SAMPLES tokens=$REFLECTION_MAX_TOKENS thinking=$THINKING"
            continue
        fi

        JOB_ID=$(sbatch --parsable \
            --job-name="thru_${name}" \
            --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" \
            --exclusive --nodes=1 \
            --output="logs/thru_%j_${name}.out" --error="logs/thru_%j_${name}.err" \
            --export=ALL,ENGINE="$engine",SWEEP_NAME="$name",FRAMEWORK_ARGS="$framework_args",ENV_TOML="$env_toml",BENCHMARK_REPO="$REPO_DIR",BENCHMARK_DOTENV="$DOTENV",BENCHMARK_API_KEY_VAR="$API_KEY_VAR",BENCHMARK_DATA_PATH="$data_path",BENCHMARK_PROMPT_PATH="$prompt_path",BENCHMARK_REFLECTION_MAX_TOKENS="$REFLECTION_MAX_TOKENS",BENCHMARK_TOTAL_SAMPLES="$TOTAL_SAMPLES",BENCHMARK_THINKING="$THINKING",BENCHMARK_OUTPUT_DIR="$OUTPUT_DIR",BENCHMARK_N_SAMPLES="$N_SAMPLES",BENCHMARK_MAX_CONCURRENT="$MAX_CONCURRENT",BENCHMARK_WARMUP="$WARMUP",BENCHMARK_COOLDOWN="$COOLDOWN",SERVED_MODEL_NAME="$served_name" \
            "$SCRIPT_DIR/sweep_job.sh")
        echo "[$name] submitted job $JOB_ID  (engine=$engine)"
        submitted=$((submitted + 1))
    done
done

[ "$DRY_RUN" = true ] && echo "(dry run — nothing submitted)" || echo "Submitted $submitted jobs. Monitor: squeue -u $USER -n thru_*"
