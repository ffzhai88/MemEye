#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
  echo "Usage: sh eval_qa_context_diagnostics.sh <semantic-suite> <evi-suite> [model-config]" >&2
  exit 2
fi

SEMANTIC_SUITE=$1
EVI_SUITE=$2
MODEL_CONFIG=${3:-config/models/qwen3_vl_8b_openrouter.yaml}

python run_qa_context_diagnostics.py   --semantic-suite "$SEMANTIC_SUITE"   --evi-suite "$EVI_SUITE"   --model-config "$MODEL_CONFIG"