#!/usr/bin/env bash
# Usage:
#   bash eval_retrieval.sh [task_config] [model_config] [method_config] [ks] [output_root]
#
# Examples:
#   bash eval_retrieval.sh
#   bash eval_retrieval.sh \
#     config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
#     config/models/qwen3_vl_8b_openrouter.yaml \
#     config/methods/evi.yaml \
#     1,3,5,10,20
#   bash eval_retrieval.sh \
#     config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
#     config/models/qwen3_vl_8b_openrouter.yaml \
#     config/methods/semantic_rag_multimodal.yaml
#
# EVI uses the configured VLM only for anchor construction and retrieval-facet
# extraction. This script never invokes the final QA model.

set -eu

TASK_CONFIG="${1:-config/tasks_external/brand_memory_test.yaml}"
MODEL_CONFIG="${2:-config/models/qwen3_vl_8b_openrouter.yaml}"
METHOD_CONFIG="${3:-config/methods/evi.yaml}"
KS="${4:-1,3,5,10,20}"
OUTPUT_ROOT="${5:-runs}"

python run_retrieval_benchmark.py \
  --task-config "$TASK_CONFIG" \
  --model-config "$MODEL_CONFIG" \
  --method-config "$METHOD_CONFIG" \
  --ks "$KS" \
  --output-root "$OUTPUT_ROOT"