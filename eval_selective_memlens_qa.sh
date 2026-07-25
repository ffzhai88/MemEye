#!/usr/bin/env bash
set -euo pipefail

INPUT_RUN="${1:?Usage: bash eval_selective_memlens_qa.sh <abstract-joint-run> [model-config] [output-root]}"
MODEL_CONFIG="${2:-config/models/qwen3_vl_8B_ali.yaml}"
OUTPUT_ROOT="${3:-runs}"
TOP_K="${TOP_K:-10}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
SOURCE_RANKINGS="${INPUT_RUN}/selective_verification_nonstable_lazy_img768/selective_verification_questions.jsonl"

python run_saved_ranking_memlens_qa.py \
  --source-rankings "${SOURCE_RANKINGS}" \
  --model-config "${MODEL_CONFIG}" \
  --ranking-strategy selective_vlm \
  --top-k "${TOP_K}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-questions "${MAX_QUESTIONS}"
