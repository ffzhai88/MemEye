# MEMLENS evaluation with MemEye

The runner evaluates every converted MEMLENS item as an independent memory
history. `question_date` is prepended to the question used by both retrieval and
answer generation. Outputs are stored under
`runs/MEMLENS/<timestamp>_<model-config>_<method-config>/`.

## Smoke test without API judging

```bash
python run_memlens_suite.py \
  --method-config config/methods/question_only.yaml \
  --model-config config/models/qwen3_vl_8b_openrouter.yaml \
  --max-questions 2 \
  --skip-judge
```

## Full run and official OpenAI-compatible judge

Set credentials outside the command line so they are not saved in shell history
or experiment metadata:

```bash
export OPENAI_API_KEY=<key>
export OPENAI_BASE_URL=<compatible-api-base-url>
export MEMLENS_JUDGE_MODEL=<judge-model-name>

python run_memlens_suite.py \
  --method-config config/methods/evi_compact_card_episode_image_rerank.yaml \
  --model-config config/models/qwen3_vl_8b_openrouter.yaml
```

On PowerShell, use `$env:OPENAI_API_KEY="..."` (and likewise for the other
variables). The key value is never written to the run configuration or log.

To resume an interrupted run, pass its existing directory:

```bash
python run_memlens_suite.py \
  --run-dir runs/MEMLENS/<existing-run> \
  --method-config config/methods/evi_compact_card_episode_image_rerank.yaml \
  --model-config config/models/qwen3_vl_8b_openrouter.yaml
```

Completed question IDs in `predictions.jsonl` are skipped. Failures are appended
to `errors.jsonl` and retried on the next invocation.

## Artifacts

- `suite_config.json`: reproducibility settings, with only API-key availability.
- `memlens_run.log`: per-question generation progress and errors.
- `predictions.jsonl`: incremental predictions, local metrics, token/latency data,
  retrieval context, clue recall, answer-session recall, and method runtime data.
- `metrics.json`: overall and type/subtype grouped local metrics; after judging it
  also embeds official metrics.
- `official_judge_input.json`: official MEMLENS judge schema.
- `official_judge.log`: streamed official evaluator output.
- `official_judge/judge_metrics.json`: primary official score aggregation.
- `official_judge/judge_details.json`: question-level official judgments.
- `official_judge/judge_metrics.json.jsonl`: official judge resume cache.
- `evi_debug.log`: EVI trace when an EVI method is selected.
