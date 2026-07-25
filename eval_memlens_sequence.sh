#!/bin/sh
# Sequential MEMLENS evaluation over several MemEye method configurations.
#
# Usage:
#   sh eval_memlens_sequence.sh [model_config] ["method_config ..."] \
#     [output_root] [max_questions] [judge_model]

set -eu
export CUDA_VISIBLE_DEVICES=9

# ali key
export OPENAI_API_KEY=sk-ws-H.EHDELYD.tIcG.MEYCIQCUxaaKoQCGLLUARaIbgiiCt1iaf-SNq7iFWNlM2zD6RQIhAMzwJrKNWKSrHs4GCiITagOBBHhuMJGNrYTW5A-2E_g4

DEFAULT_METHOD_CONFIGS="config/methods/semantic_rag_multimodal.yaml config/methods/evi_retrieval_image_rerank.yaml config/methods/evi_compact_card_episode_image_rerank.yaml"

MODEL_CONFIG="${1:-config/models/qwen3_vl_8B_ali.yaml}"
METHOD_CONFIGS="${2:-${MEMLENS_METHOD_CONFIGS:-$DEFAULT_METHOD_CONFIGS}}"
OUTPUT_ROOT="${3:-runs}"
MAX_QUESTIONS="${4:-1}"
JUDGE_MODEL="${5:-${MEMLENS_JUDGE_MODEL:-qwen3-vl-235b-a22b-instruct}}"

MANIFEST="${MEMLENS_MANIFEST:-data/memlens/converted_32k_agent195/manifest.json}"
IMAGE_ROOT="${MEMLENS_IMAGE_ROOT:-data/memlens}"
OFFICIAL_DIR="${MEMLENS_OFFICIAL_DIR:-third_party/MEMLENS}"
QUESTIONS_FILE="${MEMLENS_QUESTIONS_FILE:-data/memlens/dataset_32k.json}"
JUDGE_WORKERS="${MEMLENS_JUDGE_WORKERS:-1}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://llm-owockiqa6c46tlmv.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
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
if [ ! -d "$IMAGE_ROOT/release_images" ]; then
  echo "ERROR: expected images below $IMAGE_ROOT/release_images" >&2
  echo "Set MEMLENS_IMAGE_ROOT to the local data/memlens directory." >&2
  exit 2
fi
if [ ! -f "$OFFICIAL_DIR/llm_judge.py" ]; then
  echo "ERROR: official judge not found: $OFFICIAL_DIR/llm_judge.py" >&2
  exit 2
fi

METHOD_COUNT=0
for METHOD_CONFIG in $METHOD_CONFIGS; do
  METHOD_COUNT=$((METHOD_COUNT + 1))
  if [ ! -f "$METHOD_CONFIG" ]; then
    echo "ERROR: method config not found: $METHOD_CONFIG" >&2
    exit 2
  fi
done

CURRENT=0
for METHOD_CONFIG in $METHOD_CONFIGS; do
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "======================================================================"
  echo "[MEMLENS][$CURRENT/$METHOD_COUNT] method_config=$METHOD_CONFIG"
  echo "[MEMLENS] model_config=$MODEL_CONFIG"
  echo "[MEMLENS] max_questions=$MAX_QUESTIONS judge_model=$JUDGE_MODEL"
  echo "[MEMLENS] image_root=$IMAGE_ROOT"
  echo "======================================================================"

  python run_memlens_suite_v2.py \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
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

  echo "[MEMLENS][$CURRENT/$METHOD_COUNT] completed: $METHOD_CONFIG"
done

echo ""
echo "[MEMLENS] all $METHOD_COUNT method configurations completed"
