#!/bin/sh
# Run retrieval-only evaluation over every JSON under data/dialog.
#
# Usage:
#   sh eval_retrieval_suite.sh [model_config] [method_config] [ks] [output_root]
#
# Examples:
#   sh eval_retrieval_suite.sh
#   sh eval_retrieval_suite.sh \
#     config/models/qwen3_vl_8b_openrouter.yaml \
#     config/methods/semantic_rag_multimodal.yaml
#
# EVI uses its configured VLM only for anchor construction and retrieval-facet
# extraction. This script never invokes the final QA model.

set -eu
export OPENAI_API_KEY=sk-vutycwckjxdohudkuuvlixqpuyzrjrhgtgptdjsikngrjiok
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_CONFIG="${1:-config/models/qwen3_vl_8b_openrouter.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_single_query.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_dialogue_only.yaml}"
METHOD_CONFIG="${2:-config/methods/semantic_rag_multimodal.yaml}"
METHOD_CONFIG="${2:-config/methods/evi.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_source_aware.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_best_source.yaml}"
METHOD_CONFIG="${2:-config/methods/semantic_rag_dialogue_control.yaml}"
METHOD_CONFIG="${2:-config/methods/semantic_rag_image_only.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_image_late_fusion.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_image_rerank.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_episode_set_image_rerank.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_episode_directory_diagnostic.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_episode_directory_v2_diagnostic.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_episode_directory_card_diagnostic.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_retrieval_episode_directory_card_diagnostic.yaml}"
KS="${3:-1,3,5,10,20}"
OUTPUT_ROOT="${4:-runs}"

# The eight standard data/dialog task configs are passed explicitly below.


python run_retrieval_suite.py \
  --model-config "$MODEL_CONFIG" \
  --method-config "$METHOD_CONFIG" \
  --ks "$KS" \
  --output-root "$OUTPUT_ROOT" \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --task-config config/tasks_external/card_playlog_test.yaml \
  --task-config config/tasks_external/cartoon_entertainment_companion.yaml \
  --task-config config/tasks_external/home_renovation_interior_design.yaml \
  --task-config config/tasks_external/multi_scene_visual_case_archive_assistant.yaml \
  --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
  --task-config config/tasks_external/personal_health_dashboard_assistant.yaml \
  --task-config config/tasks_external/social_chat_memory_test.yaml
