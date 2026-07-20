#!/bin/sh
# Sequential retrieval-only MEMLENS comparison. No final QA or judge is called.
# EVI may still call its VLM when anchor/facet/card caches are missing.

set -eu

export CUDA_VISIBLE_DEVICES=9

# ali key
export OPENAI_API_KEY=sk-ws-H.EHDELYD.tIcG.MEYCIQCUxaaKoQCGLLUARaIbgiiCt1iaf-SNq7iFWNlM2zD6RQIhAMzwJrKNWKSrHs4GCiITagOBBHhuMJGNrYTW5A-2E_g4

DEFAULT_METHOD_CONFIGS="config/methods/semantic_rag_multimodal.yaml config/methods/evi_retrieval_image_rerank.yaml config/methods/evi_compact_card_episode_image_rerank.yaml"
MODEL_CONFIG="${1:-config/models/qwen3_vl_8B_ali.yaml}"
METHOD_CONFIGS="${2:-${MEMLENS_METHOD_CONFIGS:-$DEFAULT_METHOD_CONFIGS}}"
OUTPUT_ROOT="${3:-runs}"
MAX_QUESTIONS="${4:-0}"
KS="${5:-1,3,5,10,20}"
MANIFEST="${MEMLENS_MANIFEST:-data/memlens/converted_32k_agent195/manifest.json}"
IMAGE_ROOT="${MEMLENS_IMAGE_ROOT:-data/memlens}"

COUNT=0
for METHOD_CONFIG in $METHOD_CONFIGS; do
  COUNT=$((COUNT + 1))
  [ -f "$METHOD_CONFIG" ] || { echo "ERROR: missing $METHOD_CONFIG" >&2; exit 2; }
done

CURRENT=0
for METHOD_CONFIG in $METHOD_CONFIGS; do
  CURRENT=$((CURRENT + 1))
  echo "======================================================================"
  echo "[MEMLENS-RETRIEVAL][$CURRENT/$COUNT] $METHOD_CONFIG"
  echo "======================================================================"
  python run_memlens_retrieval_suite.py \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --model-config "$MODEL_CONFIG" \
    --method-config "$METHOD_CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --max-questions "$MAX_QUESTIONS" \
    --ks "$KS" \
    --clear-cache-every 1
done

echo "[MEMLENS-RETRIEVAL] all $COUNT configurations completed"
