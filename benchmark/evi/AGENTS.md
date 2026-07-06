# QDMO-EVI Agent Guide

## Scope

This directory implements `evi`, an agentic MemEye method registered in `benchmark/methods.py` as `EVIMethod`.

The current default pipeline is episodic-state EVI. It keeps the task-agnostic anchor construction and retrieval stack, then replaces candidate-level assertion generation with ordered episodic memory sets and cached state readout. The legacy candidate assertion pipeline is retained for ablation via `evi_pipeline: candidate_assertion`.

No code should branch on MemEye task names, question files, answer labels, or dataset-specific benchmark categories.

## Core Idea

EVI separates long-term memory use into these stages:

1. Offline evidence anchors: compact, typed, provenance-grounded anchors extracted from dialogue text and images.
2. Broad retrieval: the question stem retrieves a large pool of relevant anchors, reweighted by query-agnostic collection-level IDF discriminativeness.
3. Episodic memory sets: retrieved rounds are grouped by session and round order, with small local windows around hits.
4. Question-relevant evidence readout: each ordered memory set is inspected with its round dialogue and attached images, then converted into only answer-helpful evidence facts and uncertainties.
5. Final answer: the final prompt receives selected clean states, not raw anchor dumps, debug metadata, or table-like intermediate state.

This design is intended to preserve item-level evidence for counting while recovering temporal, spatial, comparative, and change relations that are lost by independent candidate assertions.

## Files

- `system.py`: EVI orchestration: indexing, broad retrieval, pipeline routing, episodic state answering, legacy candidate assertion answering, and debug tracing.
- `schemas.py`: dataclasses for anchors, legacy candidates/briefs, episodic memory sets, and episodic states.
- `extractor.py`: task-agnostic offline visual anchor extraction.
- `indexes.py`: in-memory anchor vector index and type-aware retrieval scoring.
- `sets.py`: builds ordered local `EpisodicMemorySet` objects from retrieved anchors using session/round provenance.
- `states.py`: cached VLM readout from an episodic memory set into itemized `EpisodicState` evidence.
- `candidates.py`: legacy candidate consolidation for `evi_pipeline: candidate_assertion`.
- `briefs.py`: legacy cached candidate assertion generation for `evi_pipeline: candidate_assertion`.
- `vlm.py`: OpenAI-compatible VLM callable with model-aware disk cache.
- `trace.py`: console/file debug logging and structured JSON trace helpers.
- `_utils.py`: JSON extraction, retry wrappers, and image MIME helpers.

## Lifecycle

`benchmark.methods.EVIMethod` creates `EVISystem` and calls:

1. `process_all_sessions(dataset)` once per dataset object.
2. `answer_question(question, qa=qa, question_images=...)` for each QA.

Because EVI implements `answer(...)`, it bypasses the shared non-agentic `router.answer()` path.

## Offline Memory Construction

For each round, EVI stores:

- A dialogue/temporal anchor from user/assistant text.
- Optional dataset-caption anchors when `use_dataset_captions: true`.
- Image-derived anchors from `extractor.extract_image_anchors(...)`.
- Round-level provenance maps: session id, date, round order, image paths, and anchors by round.

Anchor types are generic and benchmark-independent: `scene`, `text`, `entity`, `attribute`, `spatial`, `relation`, `identity`, `structured_visual`, and `temporal`.

## Default Question-Time Pipeline

1. Use `qa["question"]` as the retrieval stem when available, so rotated MCQ options do not dominate retrieval.
2. Extract short retrieval cues from the question stem, using the question's original words where possible.
3. Run fixed-budget multi-channel retrieval over the whole question and each cue, then fuse hits at the round level. A round is strengthened when multiple channels retrieve anchors from it. The default method config currently disables IDF weighting so multi-channel effects are easier to inspect.
4. Build episodic memory sets with `sets.build_episodic_memory_sets(...)`:
   - group by session,
   - add before/after round windows around retrieved hits,
   - merge nearby windows,
   - keep ordered rounds and selected anchors per round,
   - rank memory sets lexicographically by hit-round count, mean best hit-round score, then max hit-anchor score.
5. Read each set with `states.read_episodic_states(...)`.
6. Select relevant states first, then uncertain states as fallback.
7. Build a clean final prompt from extracted `answer_relevant_facts` and `uncertainties` only.
8. Call the VLM for the final answer. By default, memory images are not attached to the final answer call; images are consumed during state readout.

## Config

Main config: `config/methods/evi.yaml`.

Important keys:

- `evi_pipeline`: `episodic_state` by default; set `candidate_assertion` for the legacy pipeline.
- `text_embedding_model`: anchor retrieval embedding model.
- `raw_search_k`: total broad retrieval budget before memory organization; it is split uniformly across question/cue channels.
- `use_anchor_quality_weighting`: apply query-agnostic collection-level IDF weights during anchor retrieval; default EVI config currently sets this false for multi-channel retrieval experiments.
- `max_retrieval_cues`: maximum question-stem retrieval cues kept after cue extraction.
- `use_retrieval_cue_cache`: enable disk cache for retrieval cue extraction.
- `max_memory_sets`: maximum episodic sets read per question. Keep this moderately high when many sessions share generic visual cues, because top-k set selection can otherwise crowd out the correct episode.
- `memory_set_window_before` / `memory_set_window_after`: local round window around retrieved hits.
- `max_rounds_per_memory_set`: cap on merged set length.
- `max_state_anchors_per_round`: cap on anchors shown per round during state readout.
- `max_final_states`: maximum selected states shown to the final answer model.
- `use_state_cache`: enable disk cache for state readout.
- `use_state_images`: attach set images to state readout calls.
- `max_state_images_per_set`: cap images in each question-relevant evidence readout call.
- `use_final_memory_images`: attach selected memory images to the final answer call; default false.
- `use_dataset_captions`: include benchmark-provided captions only for ablation/upper-information runs.
- `use_embedding_cache`: enable disk cache for text embeddings.
- `use_memory_brief_cache`: enable legacy candidate brief cache.
- `evi_debug_*`: console/file debug tracing controls.

## Debug Trace

The default debug trace is `logs/evi_debug.log`. It records structured `[TRACE]` blocks for:

- `indexing_start`
- `indexing_done`
- `answer_start`
- `retrieved_anchors`
- `raw_retrieval_clue_coverage` when QA clue metadata exists
- `episodic_memory_sets`
- `episodic_set_clue_coverage` when QA clue metadata exists
- `episodic_states`
- `selected_episodic_states`
- `final_answer_call`
- `answer_done`

Legacy pipeline traces include `candidate_pool`, `candidate_clue_coverage`, `memory_briefs`, and `selected_memory_assertions`.

## Caches

- Anchor extraction: `EVI_ANCHOR_CACHE_DIR`, default `~/.cache/evi_anchors`.
- Text embeddings: `EVI_EMBED_CACHE_DIR`, default `~/.cache/evi_embeddings`.
- Episodic state readout: `EVI_STATE_CACHE_DIR`, default `~/.cache/evi_states`.
- Legacy candidate memory briefs: `EVI_MEMORY_BRIEF_CACHE_DIR`, default `~/.cache/evi_memory_briefs`.
- Raw VLM calls: `EVI_VLM_CACHE_DIR`, default `~/.cache/evi_vlm`.

State cache keys include prompt version, model namespace, question stem, memory set content, selected anchors, and image-use settings.

## Research Notes

The paper-facing story should distinguish:

- Flat RAG: retrieve anchors/chunks and answer directly.
- Legacy candidate assertion: judge each candidate independently.
- Episodic evidence EVI: retrieve anchors, organize local memory sets, extract question-relevant evidence from round dialogue/images, then answer from clean evidence facts.

Useful ablations:

- `evi_pipeline: candidate_assertion` vs `episodic_state`.
- `use_state_images: true` vs `false`.
- `use_final_memory_images: true` vs `false`.
- Varying memory set window size and `max_state_anchors_per_round`.
- `use_dataset_captions: true` vs `false`.

Report answer accuracy together with retrieval clue coverage, memory-set coverage, state readout traces, final prompt length, and selected image counts from EVI logs.
