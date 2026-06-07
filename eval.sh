export OPENAI_API_KEY=sk-vutycwckjxdohudkuuvlixqpuyzrjrhgtgptdjsikngrjiok

echo $OPENAI_API_KEY

python run_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
  --method-config config/methods/full_context_multimodal.yaml
