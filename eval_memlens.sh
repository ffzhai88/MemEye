#!/bin/sh
# Run converted MEMLENS with a MemEye method and score it with the official
# OpenAI-compatible MEMLENS judge.
#
# Usage:
#   sh eval_memlens.sh [model_config] [method_config] [output_root] \
#     [max_questions] [judge_model] [run_dir]
#
# Required environment variables:
#   OPENAI_API_KEY       Judge API key (and generation key when the model config uses it)
#   OPENAI_BASE_URL      OpenAI-compatible endpoint; defaults to OpenAI's endpoint
#
# Optional environment variables:
#   MEMLENS_JUDGE_MODEL  Used when positional judge_model is omitted
#   MEMEYE_QA_CACHE_DIR  Override the persistent final-QA cache directory
#
# Examples:
#   sh eval_memlens.sh
#   sh eval_memlens.sh config/models/qwen3_vl_8b_openrouter.yaml \
#     config/methods/evi_compact_card_episode_image_rerank.yaml runs 2 gpt-4.1
#   sh eval_memlens.sh config/models/qwen3_vl_8b_openrouter.yaml \
#     config/methods/evi_compact_card_episode_image_rerank.yaml runs 0 gpt-4.1 \
#     runs/MEMLENS/20260719_120000_qwen3_vl_8b_openrouter_evi_compact_card_episode_image_rerank

set -eu
export CUDA_VISIBLE_DEVICES=9

## ali key
export OPENAI_API_KEY=sk-ws-H.EHDELYD.tIcG.MEYCIQCUxaaKoQCGLLUARaIbgiiCt1iaf-SNq7iFWNlM2zD6RQIhAMzwJrKNWKSrHs4GCiITagOBBHhuMJGNrYTW5A-2E_g4

MODEL_CONFIG="${1:-config/models/qwen3_vl_8b_openrouter.yaml}"
METHOD_CONFIG="${2:-config/methods/semantic_rag_multimodal_nvembed.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_compact_card_episode_image_rerank.yaml}"
OUTPUT_ROOT="${3:-runs}"
MAX_QUESTIONS="${4:-1}"
MAX_QUESTIONS="${4:-0}"
JUDGE_MODEL="${5:-${MEMLENS_JUDGE_MODEL:-Qwen/Qwen3.5-122B-A10B}}"
JUDGE_MODEL="${5:-${MEMLENS_JUDGE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}}"
RUN_DIR="${6:-}"

MANIFEST="${MEMLENS_MANIFEST:-data/memlens/converted_32k_agent195/manifest.json}"
OFFICIAL_DIR="${MEMLENS_OFFICIAL_DIR:-third_party/MEMLENS}"
QUESTIONS_FILE="${MEMLENS_QUESTIONS_FILE:-data/memlens/dataset_32k.json}"
JUDGE_WORKERS="${MEMLENS_JUDGE_WORKERS:-1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.siliconflow.cn/v1}"
export OPENAI_BASE_URL

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 2
fi

if [ -z "$JUDGE_MODEL" ]; then
  echo "ERROR: provide judge_model as argument 5 or set MEMLENS_JUDGE_MODEL." >&2
  exit 2
fi

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: MEMLENS manifest not found: $MANIFEST" >&2
  exit 2
fi

if [ ! -f "$OFFICIAL_DIR/llm_judge.py" ]; then
  echo "ERROR: official MEMLENS judge not found: $OFFICIAL_DIR/llm_judge.py" >&2
  exit 2
fi

set -- python run_memlens_suite.py \
  --manifest "$MANIFEST" \
  --model-config "$MODEL_CONFIG" \
  --method-config "$METHOD_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --max-questions "$MAX_QUESTIONS" \
  --official-dir "$OFFICIAL_DIR" \
  --questions-file "$QUESTIONS_FILE" \
  --judge-model "$JUDGE_MODEL" \
  --judge-base-url "$OPENAI_BASE_URL" \
  --judge-workers "$JUDGE_WORKERS" \
  --clear-cache-every 1

if [ -n "$RUN_DIR" ]; then
  set -- "$@" --run-dir "$RUN_DIR"
fi

echo "[MEMLENS] model_config=$MODEL_CONFIG"
echo "[MEMLENS] method_config=$METHOD_CONFIG"
echo "[MEMLENS] max_questions=$MAX_QUESTIONS judge_model=$JUDGE_MODEL"
echo "[MEMLENS] persistent caches and JSONL resume are enabled"

exec "$@"
