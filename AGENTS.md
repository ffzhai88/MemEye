# MemEye Agent Guide

## Project Purpose

MemEye is a Python benchmark/evaluation framework for multimodal long-term agent memory. It evaluates whether memory methods preserve and use visual evidence across multi-session dialogues.

The project is research-oriented, not an application server. Most work falls into one of these categories:

- Add or modify a memory method.
- Add or modify benchmark tasks and dataset path registration.
- Change scoring, metrics, or judge prompts.
- Run controlled comparisons across models and methods.
- Analyze results under the MemEye X/Y taxonomy.

The README is the public-facing overview. This file is the practical guide for agents working in the repository.

## Repository Map

- `run_benchmark.py`: main single-run entry point.
- `run_matrix.py`: model x method comparison entry point.
- `run_retrieval_benchmark.py`: retrieval-only evaluation for one task; it never calls final QA.
- `run_retrieval_suite.py`: retrieval-only evaluation and aggregation over multiple tasks.
- `run_joint_retrieval_suite.py`: runs the MemEye retrieval suite and converted MEMLENS subset under one experiment directory with a shared log/status/summary.
- `analyze_retrieval_comparison.py`: compares two retrieval suites using clue-round metrics and writes per-dataset/per-question deltas.
- `analyze_episode_oracle.py`: replays an EVI episode suite with annotated clue-session filtering to measure session-routing headroom and diagnose routing, expansion-budget, and fusion failures.
- `analyze_episode_directory_v2.py`: compares v2 packet-score and v3 session-card rankings, reciprocal-rank fusion, image-session ranking, session-length bias, and fixed-pipeline replay.
- `analyze_session_card_ablation.py`: re-embeds saved Card sections without VLM calls to compare identity-only, identity-plus-distinctive-evidence, and full-Card session retrieval and replay.
- `analyze_multifacet_fusion.py`: replays visual-corroborated best-source ranking from saved multifacet traces without embedding or model calls.
- `analyze_selective_vlm_verification.py`: starts from saved Anchor-only/Raw-MM mean-rank candidates, checks only the Anchor/Raw-MM Top-K symmetric difference against raw rounds with an OpenAI-compatible VLM, and writes resumable two-benchmark diagnostics.
- `score_locked_llm_judge.py`: post-hoc LLM-as-a-judge scoring for open-ended outputs.
- `register_external_data.py`: creates task configs from an external MemEye data checkout.
- `benchmark/`: core benchmark package.
- `benchmark/runner.py`: orchestrates config loading, dataset construction, method/router dispatch, scoring, and output writing.
- `benchmark/dataset.py`: loads MemEye dialogue JSON, resolves images, builds round/session/QA objects.
- `benchmark/methods.py`: method registry and shared `HistoryMethod` interface.
- `benchmark/retrieval.py`: sparse/dense/multimodal retrieval used by semantic RAG methods.
- `benchmark/embeddings.py`: text and multimodal embedding wrappers.
- `benchmark/image_retrieval.py`: shared raw-image-to-round index with persistent embeddings and SigLIP-to-local-CLIP fallback.
- `benchmark/evaluator.py`: MCQ extraction, open-ended metrics, LLM judge parsing, X/Y aggregation.
- `benchmark/matrix.py`: writes matrix summaries under `runs/<task>/matrices/`.
- `router/`: unified model routers for OpenAI-compatible APIs, Gemini, and local Qwen.
- `config/tasks/`: example task configs.
- `config/models/`: model configs.
- `config/methods/`: method configs.
- `docs/` and `assets/`: project page and paper figures.
- `tools/`: data and caption utility scripts.

Large external/baseline implementations live under directories such as `benchmark/memgpt/upstream/`, `benchmark/mirix/upstream/`, `benchmark/simplemem/upstream/`, and `benchmark/evermemos/upstream/`. Treat them as vendored code unless the task explicitly targets them.

## Execution Flow

Single benchmark run:

1. `run_benchmark.py` parses CLI arguments.
2. `benchmark.runner.run_modular_benchmark()` merges task, model, and method YAML configs.
3. `MemoryBenchmarkDataset` loads the dialogue JSON, image root, sessions, rounds, and QAs.
4. `benchmark.methods.get_method()` creates the selected memory method.
5. Non-agentic methods call `method.build_history(dataset, qa)` and then `router.answer(...)`.
6. Agentic methods implement `answer(dataset, qa, question, ...)` and own retrieval/memory/inference internally.
7. `benchmark.evaluator` scores each prediction.
8. The run writes `config.json`, `metrics.json`, and `predictions.jsonl` under `runs/`.

Important distinction:

- Non-agentic methods produce history messages and use a shared router.
- Agentic methods bypass the shared router and perform end-to-end answering themselves. Examples include `m2a`, `mma`, `evi`, and several external wrappers.

## Common Commands

Install:

```bash
conda create -n memeye python=3.10 -y
conda activate memeye
pip install -r requirements.txt
```

Register external data:

```bash
python register_external_data.py --data-root ./data --overwrite
```

Run one evaluation:

```bash
python run_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/full_context_multimodal.yaml
```

Run EVI:

```bash
python run_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/evi.yaml \
  --mode open
```

Run a small smoke test:

```bash
python run_benchmark.py \
  --task-config config/tasks/brand_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/question_only.yaml \
  --max-questions 1
```

Run matrix:

```bash
python run_matrix.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/full_context_multimodal.yaml \
  --method-config config/methods/evi.yaml
```

Post-score open outputs:

```bash
python score_locked_llm_judge.py --root runs/<model>/open --judge-model gpt-5.2
```

## Retrieval-Only Evaluation

Run retrieval-only evaluation for one task:

```bash
python run_retrieval_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/evi.yaml \
  --ks 1,3,5,10,20
```

Run retrieval-only evaluation over all registered external tasks:

```bash
python run_retrieval_suite.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --task-config config/tasks_external/card_playlog_test.yaml \
  --task-config config/tasks_external/cartoon_entertainment_companion.yaml \
  --task-config config/tasks_external/home_renovation_interior_design.yaml \
  --task-config config/tasks_external/multi_scene_visual_case_archive_assistant.yaml \
  --task-config config/tasks_external/outdoor_navigation_route_memory_assistant.yaml \
  --task-config config/tasks_external/personal_health_dashboard_assistant.yaml \
  --task-config config/tasks_external/social_chat_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/evi.yaml
```

Run the standard MemEye tasks and MEMLENS as one retrieval experiment:

```bash
bash eval_joint_retrieval.sh \
  config/models/qwen3_vl_8b_openrouter.yaml \
  config/methods/evi_retrieval_multifacet_visual_corroborated_best_source.yaml \
  runs
```

The shared directory is `runs/JOINT-retrieval/<timestamp>_<model>_<method>/`.
It contains `joint_run.log`, `joint_config.json`, `joint_status.json`,
`joint_metrics.json`, the MemEye suite under `memeye/`, and MEMLENS artifacts
under `memlens/`. Metrics remain side-by-side because the two benchmarks use
different annotation semantics.

After generating abstract candidates and running the offline Raw-MM/provenance
analyzer, selectively verify contested Top-10 rounds with a VLM:

```bash
bash eval_selective_verification.sh \
  runs/JOINT-retrieval/<abstract-candidate-run> \
  config/models/qwen3_vl_8b_openrouter.yaml
```

This stage never passes benchmark, question-type, answer, or clue labels to the
VLM. By default it verifies the Anchor/Raw-MM Top-10 symmetric difference; the
shared Anchor candidate pool remains Top-30. It uses an exact prompt/evidence
cache and question-level JSONL resume.

Compare retrieval suites at `K=10`:

```bash
python analyze_retrieval_comparison.py \
  --candidate-suite runs/retrieval/<candidate-run> \
  --baseline-suite runs/retrieval/<baseline-run> \
  --k 10
```

Analyze oracle episode-routing headroom at `K=10`:

```bash
python analyze_episode_oracle.py \
  --suite runs/retrieval/<episode-suite> \
  --k 10
```

Analyze v2 packet-directory aggregation and replay:

```bash
python analyze_episode_directory_v2.py \
  --suite runs/retrieval/<v2-suite> \
  --k 10
```

Run the saved Session Card representation ablation on a Linux/GPU environment:

```bash
python analyze_session_card_ablation.py \
  --suite runs/retrieval/<card-diagnostic-suite> \
  --k 10
```

The script reads the embedding model from the saved task config, never calls a
VLM, and caches newly computed vectors under
`<suite>/session_card_ablation_embedding_cache/`.

New v2 traces retain compact per-packet scores for every session. The analyzer
compares max, mean, Top-2 mean, and Top-3 mean aggregation, plus parameter-free
reciprocal-rank fusion with the current episode order. Legacy v2 suites contain
only the max score and remain analyzable with a reduced strategy set.

This is an offline diagnostic and does not call a model. It writes
`episode_oracle_metrics.json`, `episode_oracle_questions.jsonl`, and
`episode_oracle_report.md` into the suite. Annotated clue sessions are used only
as an evaluation oracle and never enter the retrieval method. The report includes
session MAP/nDCG/Recall@M, reranking by saved episode score fields, oracle-cardinality
Top-M, whole-session oracle expansion, parameter-free balanced oracle expansion,
and the clue-round capacity limit at the requested K.

Run the holistic episode-directory diagnostic with
`config/methods/evi_retrieval_episode_directory_diagnostic.yaml`. It builds one
cached, query-independent embedding per natural session from ordered raw dialogue
and deduplicated existing anchors. The full question ranks these entries, and the
ranking is written only to `retrieval_trace.episode_directory`; it does not alter
round retrieval, image reranking, or final QA.

Run `config/methods/evi_retrieval_episode_directory_v2_diagnostic.yaml` to
compare that holistic directory against a query-independent multi-vector
directory in the same retrieval run. V2 embeds one packet per natural round
from raw dialogue and deduplicated visual evidence, scores each session by its
best packet, and writes the complete ranking to
`retrieval_trace.episode_directory_v2`. It is also diagnostic-only.

Run `config/methods/evi_retrieval_episode_directory_card_diagnostic.yaml` for
the minimum v3 representation experiment. It adds one cached, query-independent
LLM retrieval card per natural session while retaining the unchanged v1/v2
diagnostics and online round retrieval. Cards never receive a QA and never enter
final QA context.

Retrieval-only runs do not belong under individual benchmark task directories. A single-task run writes to:

```text
runs/retrieval/<timestamp>_<model>_<method>/<task>/
```

A suite creates one shared root before processing tasks:

```text
runs/retrieval/<timestamp>_<model-config>_<method-config>/
  suite_metrics.json
  suite_runs.json
  <task-a>/retrieval_debug.log
  <task-a>/retrievals.jsonl
  <task-a>/retrieval_metrics.json
  <task-a>/evi_debug.log
  <task-b>/...
```

Use `retrievals.jsonl` for per-question rankings and `retrieval_metrics.json` for task-level Recall@K, Precision@K, Hit@K, full clue coverage, and MRR. Use `suite_metrics.json` for pooled and dataset-macro summaries. Image late-fusion runs also write component rankings per question and `component_diagnostics` (anchor recall, image recall, image-unique clue hits, and oracle-union recall) in task metrics. Retrieval suite metadata can contain an absolute path from a different machine; `analyze_retrieval_comparison.py` falls back to the local stable `runs/<task>/retrieval/<run-name>` layout when analyzing legacy suites.

## End-to-End QA Suites

Run all eight registered datasets with one model/method pair using:
sh eval_qa_suite.sh config/models/qwen3_vl_8b_openrouter.yaml config/methods/evi_compact_card_episode_image_rerank.yaml mcq runs

QA suites are separate from retrieval-only suites and write to
runs/QA/<timestamp>_<model-config>_<method-config>/. The suite root contains
suite_config.json, suite_metrics.json, and suite_runs.json. Each task
subdirectory contains qa_run.log, config.json, metrics.json,
predictions.jsonl, and, for EVI, evi_debug.log.

run_qa_suite.py accepts repeated --task-config arguments. The shell script
passes the eight standard external tasks explicitly. Card text and retrieval
diagnostics must not be injected into final QA context.

Final Semantic RAG and EVI QA routers use the same persistent exact-input cache by default. Set use_qa_cache: false in a model config to disable it, or set MEMEYE_QA_CACHE_DIR to relocate the default ~/.cache/memeye_qa_answers directory.

## Data Format

MemEye task JSONs contain:

- `multi_session_dialogues`: sessions with `session_id`, `date`, and `dialogues`.
- Each dialogue round may include `round`, `user`, `assistant`, `input_image`, `image_caption`, and image IDs.
- QA lists are read from `human-annotated QAs`, `human_annotated_qas`, or `qas`.
- QA metadata commonly includes `point`, `question`, `answer`, `options`, `session_id`, and `clue`.

Image paths are resolved in `benchmark/dataset.py` using the dialogue JSON location and optional `image_root`.

## Method Architecture

The shared method interface is `HistoryMethod` in `benchmark/methods.py`.

Non-agentic baselines:

- `question_only`: no history, tests guessability.
- `full_context_text_only`: full dialogue plus captions, no image inputs.
- `full_context_multimodal`: full dialogue with original images.
- `full_context_no_visual`: full dialogue text only, no images or captions.
- `target_session_context`: only annotated target sessions.
- `clue_only_context`: oracle clue rounds.
- `semantic_rag_text_only`: text dense retrieval.
- `semantic_rag_multimodal`: multimodal dense retrieval.
- semantic_rag_multimodal_nvembed: encoder-controlled Semantic RAG using NV-Embed-v2 for text while preserving the canonical 0.5 text/image fusion.
- `semantic_rag_dialogue_control`: retrieval-only dialogue-text control without captions.
- `semantic_rag_image_only`: retrieval-only raw-image control.
- `evi_retrieval_image_late_fusion`: EVI anchor/facet ranking fused with full-question raw-image ranking at round level.
- `evi_retrieval_image_rerank`: raw-image ranking reranks only EVI anchor candidates; image-only rounds are excluded.
- `evi_retrieval_episode_set_image_rerank`: softly fuses direct round retrieval with non-lossy session-set expansion before raw-image reranking.

Agentic or wrapped methods:

- `m2a`, `mma`, `evi`, `a_mem`, `memgpt`, `gen_agents`, `evermemos`, `reflexion`, `simplemem`, `memoryos`, `mirix`, and variants.

When adding a method, update:

- `config/methods/<name>.yaml`
- `benchmark/methods.py`
- A method implementation module under `benchmark/` if needed
- README or experiment documentation if it changes the paper-facing method set

## Scoring And Outputs

MCQ:

- `extract_choice()` in `benchmark/evaluator.py` extracts a valid option letter.
- Rotation-style MCQ is supported when `qa["options"]` is a list of rotated option dicts.
- Metrics include EM, valid choice rate, debiased EM for rotations, and position-bias summaries.

Open-ended:

- Exact match and contains are normalized string matches.
- F1 uses token-level Porter stemming.
- BLEU uses NLTK with smoothing.
- BERTScore is optional and slow on first use.
- LLM-as-a-judge is optional and configured through CLI flags or `score_locked_llm_judge.py`.

Aggregation:

- Overall metrics are written to `metrics.json`.
- X/Y/cell breakdowns are derived from QA `point` fields.
- Per-question rows are written to `predictions.jsonl`.

## Configuration Notes

Configs are split into task/model/method YAML files. `benchmark.runner.compose_modular_config()` merges them.

Runtime method config receives extra private keys:

- `_model_cfg`: selected model config.
- `_runtime_paths`: output root, output JSON, and run directory.
- `_eval_cfg`: evaluation config.

Do not rely on these private keys in public method configs, but method implementations may read them at runtime.

## Development Rules

- Prefer small, scoped changes. This is a benchmark, so reproducibility matters more than broad refactors.
- Preserve output schema unless intentionally changing downstream analysis.
- Be careful with text-only methods: they require valid `image_caption` entries for image-bearing rounds.
- Do not rewrite vendored upstream baselines unless the task explicitly targets that baseline.
- API-dependent runs need appropriate environment variables such as `OPENAI_API_KEY` and `GEMINI_API_KEY`.
- Runs may download models or metrics on first use. BERTScore and embedding models can be slow.
- The existing `CLAUDE.md` appears to contain encoding corruption in this checkout; prefer this file for agent guidance.

## Research And Paper Pointers

For paper work, the key conceptual axes are:

- Visual evidence granularity: X1 scene-level through X4 pixel-level.
- Memory reasoning depth: Y1 atomic retrieval, Y2 relational association, Y3 evolutionary synthesis.

Useful paper-facing artifacts:

- `README.md`: claims, supported methods, quick start, citation.
- `docs/` and `assets/`: figures used by the project page and paper.
- `metrics.json` / `predictions.jsonl`: source of quantitative and qualitative analysis.
- `benchmark/evi/`: current custom method likely relevant for novel method sections.

When writing claims, tie implementation behavior to logged artifacts and config values, especially retrieval top-k, modality, model, judge model, and cache settings.
