#!/bin/sh
# Run end-to-end QA evaluation over all eight registered MemEye datasets.
#
# Usage:
#   sh eval_qa_suite.sh [model_config] [method_config] [mode] [output_root] [max_questions]

set -eu

MODEL_CONFIG="${1:-config/models/qwen3_vl_8b_openrouter.yaml}"
METHOD_CONFIG="${2:-config/methods/evi_compact_card_episode_image_rerank.yaml}"
MODE="${3:-mcq}"
OUTPUT_ROOT="${4:-runs}"
MAX_QUESTIONS="${5:-0}"

python run_qa_suite.py   --model-config "$MODEL_CONFIG"   --method-config "$METHOD_CONFIG"   --mode "$MODE"   --output-root "$OUTPUT_ROOT"   --max-questions "$MAX_QUESTIONS"   --task-config config/tasks_external/brand_memory_test.yaml   --task-config config/tasks_external/card_playlog_test.yaml   --task-config config/tasks_external/cartoon_entertainment_companion.yaml   --task-config config/tasks_external/home_renovation_interior_design.yaml   --task-config config/tasks_external/multi_scene_visual_case_archive_assistant.yaml   --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml   --task-config config/tasks_external/personal_health_dashboard_assistant.yaml   --task-config config/tasks_external/social_chat_memory_test.yaml