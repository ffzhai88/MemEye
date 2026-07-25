#!/usr/bin/env bash
# Run all standard MemEye tasks and the converted MEMLENS subset as one experiment.
# Usage: bash eval_joint_retrieval.sh [model_config] [method_config] [output_root] [memeye_max] [memlens_max] [ks]

set -eu

MODEL_CONFIG="${1:-config/models/qwen3_vl_8B_ali.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_multifacet_multimodal.yaml}"
METHOD_CONFIG="${2:-config/methods/semantic_rag_multimodal.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_multifacet_visual_corroborated_best_source.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_multifacet_raw_multimodal_rank_fusion.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_multifacet_abstract_candidates.yaml}"
OUTPUT_ROOT="${3:-runs}"
MEMEYE_MAX_QUESTIONS="${4:-0}"
MEMLENS_MAX_QUESTIONS="${5:-0}"
KS="${6:-1,3,5,10,20}"
MEMLENS_MANIFEST="${MEMLENS_MANIFEST:-data/memlens/converted_32k_agent195/manifest.json}"
MEMLENS_IMAGE_ROOT="${MEMLENS_IMAGE_ROOT:-data/memlens}"
CLEAR_CACHE_EVERY="${CLEAR_CACHE_EVERY:-1}"

## ali key
export OPENAI_API_KEY=sk-ws-H.EHDELYD.tIcG.MEYCIQCUxaaKoQCGLLUARaIbgiiCt1iaf-SNq7iFWNlM2zD6RQIhAMzwJrKNWKSrHs4GCiITagOBBHhuMJGNrYTW5A-2E_g4

python run_joint_retrieval_suite.py \
  --model-config "$MODEL_CONFIG" \
  --method-config "$METHOD_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --memeye-max-questions "$MEMEYE_MAX_QUESTIONS" \
  --memlens-manifest "$MEMLENS_MANIFEST" \
  --memlens-image-root "$MEMLENS_IMAGE_ROOT" \
  --memlens-max-questions "$MEMLENS_MAX_QUESTIONS" \
  --clear-cache-every "$CLEAR_CACHE_EVERY" \
  --ks "$KS"
