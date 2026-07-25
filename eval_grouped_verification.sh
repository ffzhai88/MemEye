#!/usr/bin/env bash
set -euo pipefail

INPUT_RUN="${1:-}"
MODEL_CONFIG="${2:-config/models/qwen3_vl_8B_ali.yaml}"
MODE="${3:-both}"
: "${INPUT_RUN:?Usage: bash eval_grouped_verification.sh <abstract-joint-run> [model-config] [mode]}"

WORKERS="${WORKERS:-8}"
CANDIDATE_K="${CANDIDATE_K:-30}"
EVAL_K="${EVAL_K:-10}"
MAX_ROUNDS_PER_BATCH="${MAX_ROUNDS_PER_BATCH:-6}"
MAX_IMAGES_PER_ROUND="${MAX_IMAGES_PER_ROUND:-4}"
MAX_IMAGES_PER_BATCH="${MAX_IMAGES_PER_BATCH:-12}"
IMAGE_MAX_LONG_EDGE="${IMAGE_MAX_LONG_EDGE:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
VERIFIER_CACHE_ROOT="${MEMEYE_VERIFIER_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/memeye/verifier}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
BENCHMARK="${BENCHMARK:-all}"

if [[ "$MODE" == "both" ]]; then
  MODES=(neighborhood rank_batch)
elif [[ "$MODE" == "rank_batch" || "$MODE" == "neighborhood" ]]; then
  MODES=("$MODE")
else
  echo "mode must be both, rank_batch, or neighborhood" >&2
  exit 2
fi

for grouping_mode in "${MODES[@]}"; do
  output_dir="$INPUT_RUN/grouped_verification_${grouping_mode}_img${IMAGE_MAX_LONG_EDGE}"
  cache_dir="${CACHE_DIR:-${VERIFIER_CACHE_ROOT}/grouped}"
  echo "[GROUPED-VLM] mode=$grouping_mode input=$INPUT_RUN"
  echo "[GROUPED-VLM] model=$MODEL_CONFIG workers=$WORKERS"
  echo "[GROUPED-VLM] candidate_k=$CANDIDATE_K eval_k=$EVAL_K rounds_per_batch=$MAX_ROUNDS_PER_BATCH"
  echo "[GROUPED-VLM] image_preprocess=max_long_edge_$IMAGE_MAX_LONG_EDGE jpeg_quality_80"
  echo "[GROUPED-VLM] output=$output_dir cache=$cache_dir resume=enabled"

  args=(
    --input "$INPUT_RUN"
    --output-dir "$output_dir"
    --grouping-mode "$grouping_mode"
    --verifier-model-config "$MODEL_CONFIG"
    --candidate-k "$CANDIDATE_K"
    --eval-k "$EVAL_K"
    --max-rounds-per-batch "$MAX_ROUNDS_PER_BATCH"
    --max-images-per-round "$MAX_IMAGES_PER_ROUND"
    --max-images-per-batch "$MAX_IMAGES_PER_BATCH"
    --image-max-long-edge "$IMAGE_MAX_LONG_EDGE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --workers "$WORKERS"
    --benchmark "$BENCHMARK"
    --cache-dir "$cache_dir"
  )
  if [[ "$MAX_QUESTIONS" -gt 0 ]]; then
    args+=(--max-questions "$MAX_QUESTIONS")
  fi
  python analyze_grouped_vlm_verification.py "${args[@]}"
done
