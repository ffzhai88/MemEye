#!/usr/bin/env bash
set -euo pipefail

INPUT_RUN="${1:?Usage: bash eval_selective_verification.sh <abstract-joint-run> [model-config]}"
MODEL_CONFIG="${2:-config/models/qwen3_vl_8B_ali.yaml}"

IMAGE_MAX_LONG_EDGE="${IMAGE_MAX_LONG_EDGE:-768}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_RUN}/selective_verification_nonstable_lazy_img${IMAGE_MAX_LONG_EDGE}}"
WORKERS="${WORKERS:-8}"
CANDIDATE_K="${CANDIDATE_K:-30}"
EVAL_K="${EVAL_K:-10}"
VERIFY_TOP_K="${VERIFY_TOP_K:-${EVAL_K}}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
MAX_IMAGES_PER_ROUND="${MAX_IMAGES_PER_ROUND:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BENCHMARK="${BENCHMARK:-all}"
VERIFIER_CACHE_ROOT="${MEMEYE_VERIFIER_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/memeye/verifier}"
CACHE_DIR="${CACHE_DIR:-${VERIFIER_CACHE_ROOT}/selective}"


echo "[SELECTIVE-VLM] input=${INPUT_RUN}"
echo "[SELECTIVE-VLM] model_config=${MODEL_CONFIG} workers=${WORKERS}"
echo "[SELECTIVE-VLM] candidate_k=${CANDIDATE_K} eval_k=${EVAL_K} verify_top_k=${VERIFY_TOP_K}"
echo "[SELECTIVE-VLM] image_preprocess=max_long_edge_${IMAGE_MAX_LONG_EDGE} jpeg_quality_80"
echo "[SELECTIVE-VLM] output=${OUTPUT_DIR} resume=enabled"
echo "[SELECTIVE-VLM] cache_dir=${CACHE_DIR} policy=verify-all-nonstable"

args=(
  --input "${INPUT_RUN}"
  --output-dir "${OUTPUT_DIR}"
  --verifier-model-config "${MODEL_CONFIG}"
  --candidate-k "${CANDIDATE_K}"
  --cache-dir "${CACHE_DIR}"
  --eval-k "${EVAL_K}"
  --verification-top-k "${VERIFY_TOP_K}"
  --workers "${WORKERS}"
  --max-images-per-round "${MAX_IMAGES_PER_ROUND}"
  --image-max-long-edge "${IMAGE_MAX_LONG_EDGE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --benchmark "${BENCHMARK}"
)

if [[ "${MAX_QUESTIONS}" -gt 0 ]]; then
  args+=(--max-questions "${MAX_QUESTIONS}")
fi

python analyze_selective_vlm_verification.py "${args[@]}"
