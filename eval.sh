export OPENAI_API_KEY=sk-fbjaxditdpouieigtldqkkwnqrnkptdnfcwossczcytkynoj
#export OPENAI_API_KEY=sk-vutycwckjxdohudkuuvlixqpuyzrjrhgtgptdjsikngrjiok
#export OPENROUTER_API_KEY=sk-vutycwckjxdohudkuuvlixqpuyzrjrhgtgptdjsikngrjiok
#export OPENAI_API_KEY=sk-ws-H.EDIDMDX.FIVz.MEQCIH7WvrZi5X2gBEtl-mm4MwFyc5zDafdlg1AZv5tfNhJkAiBdYBH7ANo09FCh9yYRzE7_eNuVNbAAe6SbGOdI2i9l-w


echo $OPENAI_API_KEY
export CUDA_VISIBLE_DEVICES=1,2

#python run_benchmark.py \
#  --task-config config/tasks_external/brand_memory_test.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/full_context_multimodal.yaml

#python run_benchmark.py \
  # --task-config config/tasks_external/brand_memory_test.yaml \
  # --model-config config/models/qwen3_vl_8b_openrouter.yaml \
  # --method-config config/methods/a_mem_openrouter.yaml

#python run_benchmark.py \
# --task-config config/tasks_external/home_renovation_interior_design.yaml \
# --model-config config/models/qwen3_vl_8b_openrouter.yaml \
# --method-config config/methods/semantic_rag_multimodal_nvembed.yaml

#python run_benchmark.py \
#  --task-config config/tasks_external/brand_memory_test.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml

#python run_benchmark.py \
#  --task-config config/tasks_external/brand_memory_test.yaml \
#  --model-config config/models/qwen3_vl_32B.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml

#python run_benchmark.py \
#  --task-config config/tasks_external/card_playlog_test.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml
#
#python run_benchmark.py \
#  --task-config config/tasks_external/cartoon_entertainment_companion.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml
#
#python run_benchmark.py \
#  --task-config config/tasks_external/multi_scene_visual_case_archive_assistant.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml
#
#python run_benchmark.py \
#  --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml

#python run_benchmark.py \
#  --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
#  --model-config config/models/qwen3_vl_32B.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml

#python run_benchmark.py \
#  --task-config config/tasks_external/personal_health_dashboard_assistant.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml
#
#
#python run_benchmark.py \
#  --task-config config/tasks_external/social_chat_memory_test.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/semantic_rag_multimodal_nvembed.yaml

#python -u run_benchmark.py \
#  --task-config config/tasks_external/debug.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/evi.yaml \
#  --max-questions 5

#python -u run_benchmark.py \
#  --task-config config/tasks_external/debug.yaml \
#  --model-config config/models/qwen3_vl_32B.yaml \
#  --method-config config/methods/evi.yaml \
#  --max-questions 5


echo "================== Running brand_memory_test benchmark with EVI method... ===================="
python run_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
  --method-config config/methods/evi.yaml

#python run_benchmark.py \
#  --task-config config/tasks_external/brand_memory_test.yaml \
#  --model-config config/models/qwen3_vl_32B_ali.yaml \
#  --method-config config/methods/evi.yaml

# echo "================== Running home_renovation_interior_design benchmark with EVI method... ===================="
# python run_benchmark.py \
#  --task-config config/tasks_external/home_renovation_interior_design.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/evi.yaml

# echo "================== Running card_playlog_test benchmark with EVI method... ===================="
# python run_benchmark.py \
#   --task-config config/tasks_external/card_playlog_test.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/evi.yaml

# echo "================== Running cartoon_entertainment_companion benchmark with EVI method... ===================="
# python run_benchmark.py \
#   --task-config config/tasks_external/cartoon_entertainment_companion.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/evi.yaml

# echo "================== Running multi_scene_visual_case_archive_assistant benchmark with EVI method... ===================="
# python run_benchmark.py \
#   --task-config config/tasks_external/multi_scene_visual_case_archive_assistant.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/evi.yaml

#echo "================== Running outdoor_navigation_route_memory_assistant benchmark with EVI method... ===================="
#python run_benchmark.py \
#  --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
#  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#  --method-config config/methods/evi.yaml 
#python run_benchmark.py \
#  --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
#  --model-config config/models/qwen3_vl_32B.yaml \
#  --method-config config/methods/evi.yaml 


# echo "================== Running personal_health_dashboard_assistant benchmark with EVI method... ===================="
# python run_benchmark.py \
#   --task-config config/tasks_external/personal_health_dashboard_assistant.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/evi.yaml

# echo "================== Running social_chat_memory_test benchmark with EVI method... ===================="
# python run_benchmark.py \
#   --task-config config/tasks_external/social_chat_memory_test.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/evi.yaml





# python run_benchmark.py \
#   --task-config config/tasks_external/brand_memory_test.yaml \
#   --model-config config/models/qwen3_vl_8b_openrouter.yaml \
#   --method-config config/methods/evi.yaml
#  --max-questions 1
