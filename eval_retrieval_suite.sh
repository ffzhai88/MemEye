#!/usr/bin/env bash
# Run retrieval-only evaluation over every JSON under data/dialog.
#
# Usage:
#   bash eval_retrieval_suite.sh [model_config] [method_config] [ks] [output_root]
#
# Examples:
#   bash eval_retrieval_suite.sh
#   bash eval_retrieval_suite.sh \
#     config/models/qwen3_vl_8b_openrouter.yaml \
#     config/methods/semantic_rag_multimodal.yaml
#
# EVI uses its configured VLM only for anchor construction and retrieval-facet
# extraction. This script never invokes the final QA model.

set -eu

MODEL_CONFIG="${1:-config/models/qwen3_vl_8b_openrouter.yaml}"
METHOD_CONFIG="${2:-config/methods/evi.yaml}"
KS="${3:-1,3,5,10,20}"
OUTPUT_ROOT="${4:-runs}"

# One task config for each of the eight JSON files under data/dialog.
TASK_CONFIGS=(
  config/tasks_external/brand_memory_test.yaml
  config/tasks_external/card_playlog_test.yaml
  config/tasks_external/cartoon_entertainment_companion.yaml
  config/tasks_external/home_renovation_interior_design.yaml
  config/tasks_external/multi_scene_visual_case_archive_assistant.yaml
  config/tasks_external/outdoor_navigation_route_memory_assistant.yaml
  config/tasks_external/personal_health_dashboard_assistant.yaml
  config/tasks_external/social_chat_memory_test.yaml
)

args=()
for task_config in "${TASK_CONFIGS[@]}"; do
  args+=(--task-config "$task_config")
done

python run_retrieval_suite.py \
  --model-config "$MODEL_CONFIG" \
  --method-config "$METHOD_CONFIG" \
  --ks "$KS" \
  --output-root "$OUTPUT_ROOT" \
  "${args[@]}"