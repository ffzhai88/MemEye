# QDMO-EVI Agent Guide

## Scope

This directory implements `evi`, an agentic MemEye method registered in `benchmark/methods.py` as `EVIMethod`.

The current implementation is QDMO-EVI: Query-Driven Memory Organization with typed Evidence anchors. It is benchmark-independent: no code branches on MemEye task names, answer types, question files, or dataset-specific file names.

## Core Idea

QDMO-EVI separates long-term memory into two stages:

1. Offline evidence anchors: compact, typed, provenance-grounded anchors extracted from dialogue text and images.
2. Online query-driven organization: broad anchor retrieval followed by dynamic evidence grouping, group-level visual verification, and final answering.

The method does not try to predict benchmark task categories. It embeds each question directly and organizes retrieved memory into coherent evidence groups using semantic affinity, provenance, evidence-type compatibility, chronology, and visual availability.

## Files

- `system.py`: QDMO-EVI orchestration: indexing, broad retrieval, memory organization, group verification, final VLM answering.
- `schemas.py`: `EvidenceAnchor` and `EvidenceGroup` dataclasses.
- `extractor.py`: task-agnostic offline visual anchor extraction.
- `indexes.py`: in-memory anchor vector index and type-aware retrieval scoring.
- `organizer.py`: query-driven memory organization into evidence groups. No LLM group description is used in the minimal implementation.
- `verifier.py`: group-level visual verification over selected group images.
- `vlm.py`: OpenAI-compatible VLM callable with model-aware disk cache.
- `trace.py`: file-based debug tracing for indexing, retrieval, grouping, verification, and final answering.

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

1. Use `qa["question"]` as retrieval stem, excluding rotated MCQ options when available.
2. Embed the question and retrieve a broad pool of evidence anchors from `EvidenceIndex`.
3. Organize anchors into query-driven evidence groups with `organizer.organize_evidence(...)`.
4. Verify selected group images with `verifier.verify_group(...)`.
5. Build the final prompt from grouped anchors, provenance, verification notes, contradictions, uncertainty, and the original full question/options.
6. Call the VLM with selected images from the question and evidence groups.

## Config

Main config: `config/methods/evi.yaml`.

Important keys:

- `text_embedding_model`: anchor retrieval embedding model.
- `raw_search_k`: broad anchor retrieval size before organization.
- `max_groups`: maximum evidence groups used for answering.
- `max_group_size`: maximum anchors per group.
- `max_group_images`: maximum images verified per group.
- `use_group_verification`: enable group-level visual verification.
- `max_answer_images`: number of images passed to final answer call.
- `use_dataset_captions`: include benchmark-provided captions only for ablation/upper-information runs.
- `use_embedding_cache`: enable disk cache for text embeddings.
- `evi_debug`: enable or disable EVI debug trace logging.
- `evi_debug_log_path`: debug trace output path, default `logs/evi_debug.log`.
- `evi_debug_top_k`: number of retrieved anchors included in trace summaries.
- `evi_debug_prompt_chars`: maximum final-prompt preview characters written to the trace.

## Debug Trace

The default debug trace is `logs/evi_debug.log`. It records structured `[TRACE]` blocks for `indexing_start`, `indexing_done`, `answer_start`, `retrieved_anchors`, `organized_groups`, `clue_coverage`, `verified_groups`, `final_answer_call`, and `answer_done`. The trace intentionally stores image paths, anchor/group summaries, and prompt previews rather than base64 image payloads.

## Caches

- Anchor extraction: `EVI_ANCHOR_CACHE_DIR`, default `~/.cache/evi_anchors`.
- Text embeddings: `EVI_EMBED_CACHE_DIR`, default `~/.cache/evi_embeddings`.
- Group verification: `EVI_GROUP_VERIFY_CACHE_DIR`, default `~/.cache/evi_group_verification`.
- Raw VLM calls: `EVI_VLM_CACHE_DIR`, default `~/.cache/evi_vlm`.

Cache keys include prompt versions and relevant context. VLM cache keys include model namespace.

## Research Notes

The clean paper story is:

- Long-term multimodal memory needs more than flat retrieval over captions or single-image summaries.
- Questions often require dynamically organizing multiple related memory fragments, not only selecting one round.
- QDMO-EVI introduces query-driven memory organization: retrieved anchors become structured evidence groups with provenance, temporal context, and visual verification.

Useful ablations:

- Flat anchor retrieval without organization.
- Organization without group visual verification.
- `use_dataset_captions: true` vs `false`.
- varying `max_groups`, `max_group_size`, and `raw_search_k`.

Report answer accuracy together with clue/evidence round coverage from EVI logs.






