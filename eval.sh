export OPENAI_API_KEY=sk-vutycwckjxdohudkuuvlixqpuyzrjrhgtgptdjsikngrjiok
export OPENROUTER_API_KEY=sk-vutycwckjxdohudkuuvlixqpuyzrjrhgtgptdjsikngrjiok
export CUDA_VISIBLE_DEVICES=1
echo $OPENAI_API_KEY

#python run_benchmark.py \
#  --task-config config/tasks_external/brand_memory_test.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/full_context_multimodal.yaml

#python run_benchmark.py \
  # --task-config config/tasks_external/brand_memory_test.yaml \
  # --model-config config/models/qwen3_vl_8b_openrouter.yaml \
  # --method-config config/methods/a_mem_openrouter.yaml

#python run_benchmark.py \
#  --task-config config/tasks_external/home_renovation_interior_design.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal.yaml

# python run_benchmark.py \
#   --task-config config/tasks_external/brand_memory_test.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/semantic_rag_multimodal.yaml

# python run_benchmark.py \
#   --task-config config/tasks_external/card_playlog_test.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/semantic_rag_multimodal.yaml

# python run_benchmark.py \
#   --task-config config/tasks_external/cartoon_entertainment_companion.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/semantic_rag_multimodal.yaml

# python run_benchmark.py \
#   --task-config config/tasks_external/multi_scene_visual_case_archive_assistant.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/semantic_rag_multimodal.yaml

# python run_benchmark.py \
#   --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/semantic_rag_multimodal.yaml

# python run_benchmark.py \
#   --task-config config/tasks_external/personal_health_dashboard_assistant.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/semantic_rag_multimodal.yaml


# python run_benchmark.py \
#   --task-config config/tasks_external/social_chat_memory_test.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/semantic_rag_multimodal.yaml

# python -u run_benchmark.py \
#   --task-config config/tasks_external/brand_memory_test.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/hippo_agentic.yaml \
#   --max-questions 5

python -u run_benchmark.py \
  --task-config config/tasks_external/personal_health_dashboard_assistant.yaml \
  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
  --method-config config/methods/hippo_agentic.yaml
