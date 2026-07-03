# QDMO-EVI Agent Guide

## Scope

This directory implements `evi`, an agentic MemEye method registered in `benchmark/methods.py` as `EVIMethod`.

The current implementation is a candidate-centric version of QDMO-EVI. It is benchmark-independent: no code branches on MemEye task names, answer types, question files, or dataset-specific file names. The method indexes generic multimodal evidence anchors, then organizes retrieved memories into natural-language candidate briefs at question time.

## Core Idea

QDMO-EVI separates long-term memory into three stages:

1. Offline evidence anchors: compact, typed, provenance-grounded anchors extracted from dialogue text and images.
2. Online candidate consolidation: broad anchor retrieval is collapsed into mostly non-overlapping memory candidates by round/image provenance.
3. Query-conditioned memory briefs: each candidate is rewritten into a concise natural-language brief before final answering.

The final answer prompt receives organized memory briefs, not raw anchor dumps and not a table. This is meant to reduce repeated evidence, preserve visual provenance, and keep the final VLM focused on candidate memories that may support, contradict, or exclude an answer.

## Files

- `system.py`: EVI orchestration: indexing, broad retrieval, candidate consolidation, memory briefing, final VLM answering, and debug tracing.
- `schemas.py`: dataclasses for `EvidenceAnchor`, `MemoryCandidate`, and `MemoryBrief`.
- `extractor.py`: task-agnostic offline visual anchor extraction.
- `indexes.py`: in-memory anchor vector index and type-aware retrieval scoring.
- `candidates.py`: collapses retrieved anchors into candidate memories and selects compact candidate anchors.
- `briefs.py`: cached VLM generation of natural-language memory briefs for each candidate.
- `vlm.py`: OpenAI-compatible VLM callable with model-aware disk cache.
- `trace.py`: console/file debug logging and structured JSON trace helpers.
- `_utils.py`: JSON extraction, retry wrappers, and image MIME helpers.

The old overlapping candidate path has been removed. Keep future changes on the candidate-brief architecture unless explicitly requested.

## Lifecycle

`benchmark.methods.EVIMethod` creates `EVISystem` and calls:

1. `process_all_sessions(dataset)` once per dataset object.
2. `answer_question(question, qa=qa, question_images=...)` for each QA.

Because EVI implements `answer(...)`, it bypasses the shared non-agentic `router.answer()` path.

## Offline Memory Construction

For each round, QDMO-EVI stores:

- A dialogue/temporal anchor from user/assistant text.
- Optional dataset-caption anchors when `use_dataset_captions: true`.
- Image-derived anchors from `extractor.extract_image_anchors(...)`.

Anchor types are generic and benchmark-independent:

- `scene`
- `text`
- `entity`
- `attribute`
- `spatial`
- `relation`
- `identity`
- `structured_visual`
- `temporal`

Every anchor keeps provenance: `session_id`, `round_id`, `date`, and optional `image_path`.

## Question-Time Pipeline

1. Use `qa["question"]` as the retrieval stem when available, so rotated MCQ options do not dominate retrieval.
2. Embed the question and retrieve a broad pool of evidence anchors from `EvidenceIndex`.
3. Consolidate anchors into memory candidates with `candidates.consolidate_candidates(...)`.
4. Fold same-round text-only anchors into image candidates instead of creating duplicate text candidates for that round.
5. Select diverse candidate anchors with `select_candidate_anchors(...)`.
6. Generate cached natural-language briefs with `briefs.generate_memory_briefs(...)`.
7. Select relevant/uncertain briefs, optionally preserving a small number of excluded briefs as negative evidence.
8. Build the final prompt from memory briefs, provenance, key evidence, and the original full question/options.
9. Call the VLM with selected images from the question and non-excluded memory briefs.

## Config

Main config: `config/methods/evi.yaml`.

Important keys:

- `text_embedding_model`: anchor retrieval embedding model.
- `raw_search_k`: broad anchor retrieval size before candidate consolidation.
- `max_candidates`: maximum candidate memories briefed per question.
- `max_candidate_anchors`: maximum selected anchors shown to the brief model per candidate.
- `max_final_briefs`: maximum memory briefs shown to the final answer model.
- `max_excluded_briefs`: maximum excluded briefs retained as negative evidence.
- `max_answer_images`: number of images passed to final answer call.
- `use_dataset_captions`: include benchmark-provided captions only for ablation/upper-information runs.
- `use_embedding_cache`: enable disk cache for text embeddings.
- `use_memory_brief_cache`: enable disk cache for candidate memory briefs.
- `evi_debug`: enable or disable EVI debug trace logging.
- `evi_debug_console`: print concise key-step logs to console.
- `evi_debug_log_path`: debug trace output path, default `logs/evi_debug.log`.
- `evi_debug_top_k`: number of retrieved anchors included in trace summaries.
- `evi_debug_prompt_chars`: maximum final-prompt preview characters written to the trace.

## Debug Trace

The default debug trace is `logs/evi_debug.log`. It records structured `[TRACE]` blocks for:

- `indexing_start`
- `indexing_done`
- `answer_start`
- `retrieved_anchors`
- `candidate_pool`
- `candidate_clue_coverage` when QA clue metadata exists
- `memory_briefs`
- `selected_memory_briefs`
- `final_answer_call`
- `answer_done`

The trace intentionally stores image paths, anchor/candidate/brief summaries, prompt previews, and raw brief-model text. It does not store base64 image payloads.

## Caches

- Anchor extraction: `EVI_ANCHOR_CACHE_DIR`, default `~/.cache/evi_anchors`.
- Text embeddings: `EVI_EMBED_CACHE_DIR`, default `~/.cache/evi_embeddings`.
- Candidate memory briefs: `EVI_MEMORY_BRIEF_CACHE_DIR`, default `~/.cache/evi_memory_briefs`.
- Raw VLM calls: `EVI_VLM_CACHE_DIR`, default `~/.cache/evi_vlm`.

Cache keys include prompt versions and relevant context. VLM cache keys include model namespace.

## Research Notes

The clean paper story is:

- Long-term multimodal memory needs more than flat retrieval over captions or single-image summaries.
- The final answering model should not receive an unfiltered pile of retrieved anchors.
- QDMO-EVI uses task-agnostic anchors, then performs query-conditioned candidate briefing to convert retrieved memories into compact evidence units.
- Candidate briefs can support, exclude, or mark uncertainty, which makes counting/comparison questions easier to debug than raw retrieval dumps.

Useful ablations:

- Flat anchor retrieval without candidate briefing.
- Candidate briefing without candidate images.
- `use_dataset_captions: true` vs `false`.
- Varying `raw_search_k`, `max_candidates`, `max_candidate_anchors`, and `max_final_briefs`.

Report answer accuracy together with candidate coverage, selected-brief traces, final prompt length, and selected image counts from EVI logs.
