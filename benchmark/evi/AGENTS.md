# QDMO-EVI Agent Guide

## Scope

This directory implements `evi`, an agentic MemEye method registered in `benchmark/methods.py` as `EVIMethod`.

The current default pipeline is episodic-state EVI. It keeps the task-agnostic anchor construction and retrieval stack, then replaces candidate-level assertion generation with ordered episodic memory sets and cached state readout. The legacy candidate assertion pipeline is retained for ablation via `evi_pipeline: candidate_assertion`.

No code should branch on MemEye task names, question files, answer labels, or dataset-specific benchmark categories.

## Core Idea

EVI separates long-term memory use into these stages:

1. Offline evidence anchors: compact, typed, provenance-grounded anchors extracted from dialogue text and images.
2. Broad retrieval: the question stem retrieves a large pool of relevant anchors.
3. Episodic memory sets: retrieved rounds are grouped by session and round order, with small local windows around hits.
4. Episodic state readout: each ordered memory set is converted into a clean, itemized state with per-round facts, relations, changes, and uncertainties.
5. Final answer: the final prompt receives selected clean states, not raw anchor dumps, debug metadata, or table-like intermediate state.

This design is intended to preserve item-level evidence for counting while recovering temporal, spatial, comparative, and change relations that are lost by independent candidate assertions.

## Files

- `system.py`: EVI orchestration: indexing, broad retrieval, pipeline routing, episodic state answering, legacy candidate assertion answering, and debug tracing.
- `schemas.py`: dataclasses for anchors, legacy candidates/briefs, episodic memory sets, and episodic states.
- `extractor.py`: task-agnostic offline visual anchor extraction.
- `indexes.py`: in-memory anchor vector index and type-aware retrieval scoring.
- `facet_multimodal.py`: query-local calibration and early round-level fusion of dialogue anchors, visual anchors, and raw-image retrieval for each facet.
- `raw_multimodal.py`: fixed-candidate raw dialogue/image scoring and equal-weight mean-rank fusion with the primary EVI ranking.
- `episode_retrieval.py`: pure session-set merge, member expansion, and direct/episode reciprocal-rank fusion.
- `episode_directory.py`: diagnostic query-independent session indexes: one holistic vector per session (v1) and one packet vector per natural round with max-packet session scoring (v2).
- `episode_cards.py`: cached query-independent LLM session retrieval cards (v3); cards are retrieval representations only and are not QA evidence.
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
2. Embed the question stem and retrieve a broad pool of evidence anchors from `EvidenceIndex`.
3. Build episodic memory sets with `sets.build_episodic_memory_sets(...)`:
   - group by session,
   - add before/after round windows around retrieved hits,
   - merge nearby windows,
   - keep ordered rounds and selected anchors per round.
4. Read each set with `states.read_episodic_states(...)`.
5. Select relevant states first, then uncertain states as fallback.
6. Build a clean final prompt from `memory_items`, `observations`, `relations`, `changes`, `answer_relevant_facts`, and `uncertainties`.
7. Call the VLM for the final answer. By default, memory images are not attached to the final answer call; images are consumed during state readout.

## Config

Main config: `config/methods/evi.yaml`.

Important keys:

- `evi_pipeline`: `episodic_state` by default; set `candidate_assertion` for the legacy pipeline.
- `text_embedding_model`: anchor retrieval embedding model.
- `raw_search_k`: broad anchor retrieval size before memory organization.
- `max_memory_sets`: maximum episodic sets read per question. Keep this moderately high when many sessions share generic visual cues, because top-k set selection can otherwise crowd out the correct episode.
- `memory_set_window_before` / `memory_set_window_after`: local round window around retrieved hits.
- `max_rounds_per_memory_set`: cap on merged set length.
- `max_state_anchors_per_round`: cap on anchors shown per round during state readout.
- `max_final_states`: maximum selected states shown to the final answer model.
- `use_state_cache`: enable disk cache for state readout.
- `use_state_images`: attach set images to state readout calls.
- `max_state_images_per_set`: cap images in each state readout call.
- `use_final_memory_images`: attach selected memory images to the final answer call; default false.
- `use_dataset_captions`: include benchmark-provided captions only for ablation/upper-information runs.
- `use_embedding_cache`: enable disk cache for text embeddings.
- `include_session_markers`: prefix the first selected turn of each natural session with its session id/date in final QA history only; selector and retrieval inputs remain unchanged.
- `evi_use_raw_image_retrieval`: enable independent full-question text-to-image retrieval.
- `evi_facet_round_scorer`: `anchor` preserves legacy retrieval; `multimodal_anchor_early_fusion` queries every modality per facet, combines the two visual views, then combines text and visual branches before facet consensus.
- `evi_facet_round_scorer: visual_corroborated_best_source` retains independent dialogue and visual candidate budgets; raw images rerank only visual-anchor candidates before the existing cross-facet best-source consensus.
- `evi_image_round_search_k`: candidate depth for both EVI and raw-image rankings before fusion.
- `evi_image_round_fusion`: `reciprocal_rank` uses the anchor/image union; `anchor_candidate_reciprocal_rank` reranks only anchor candidates.
- `evi_apply_raw_multimodal_candidate_rank_fusion`: rescore the fixed EVI Top-K with full-question raw dialogue/image evidence and fuse EVI/Raw-MM ranks by their parameter-free arithmetic mean.
- `evi_raw_multimodal_candidate_k`: fixed EVI candidate depth for Raw-MM calibration; the paper-facing configuration uses 30.
- `evi_raw_multimodal_text_weight` / `evi_raw_multimodal_image_weight`: raw-evidence score weights before Raw-MM ranking; the canonical setting is 0.5/0.5 with missing image score zero.
- `evi_use_episode_set_retrieval`: add a soft session-set retrieval path without filtering direct round candidates.
- `evi_episode_search_k`: session-set depth per facet.
- `evi_episode_round_search_k`: maximum expanded episode-round path depth before direct/episode fusion.
- `evi_enable_episode_directory_diagnostic`: build and query the holistic session directory without applying its ranking.
- `evi_episode_directory_search_k`: directory ranking depth; zero preserves the complete ranking for MAP/nDCG.
- `evi_enable_episode_directory_packet_diagnostic`: build and query the v2 round-packet directory without applying its ranking.
- `evi_episode_directory_packet_search_k`: v2 ranking depth; zero preserves the complete session ranking.
- `evi_enable_episode_directory_card_diagnostic`: build and rank cached v3 session cards without applying the ranking.
- `evi_episode_directory_card_search_k`: v3 ranking depth; zero preserves the complete session ranking.
- `use_episode_directory_card_cache`: cache cards independently of raw VLM-call caching.
- `config/methods/evi_retrieval_episode_set_image_rerank.yaml`: retrieval-only direct + episode-set + raw-image corroboration ablation.
- `config/methods/evi_retrieval_episode_directory_diagnostic.yaml`: unchanged final retrieval plus diagnostic holistic directory ranking.
- `config/methods/evi_retrieval_episode_directory_v2_diagnostic.yaml`: runs the unchanged retrieval with both v1 holistic and v2 round-packet directory traces.
- `config/methods/evi_retrieval_episode_directory_card_diagnostic.yaml`: adds v3 session-card traces while leaving online retrieval unchanged.
- `config/methods/evi_retrieval_multifacet_multimodal.yaml`: dataset-agnostic multifacet multimodal retrieval with no question-type router or post-hoc image reranking.
- `config/methods/evi_retrieval_multifacet_visual_corroborated_best_source.yaml`: provenance-gated visual corroboration followed by facet-local best-source selection.
- `config/methods/evi_retrieval_multifacet_raw_multimodal_rank_fusion.yaml`: the same EVI candidate generator followed by fixed-pool raw multimodal scoring and EVI/Raw-MM mean-rank fusion.
- `config/methods/evi_retrieval_multifacet_abstract_candidates.yaml`: broad anchor-only Top-50 traces for offline provenance-verification replay; original images never affect its online candidate ranking.
- `use_image_embedding_cache`: cache raw-image and image-query embeddings under `~/.cache/memeye/raw_image_embeddings` by default.
- `multimodal_clip_fallback_model`: local CLIP fallback used when SigLIP loading or encoding fails.
- `use_memory_brief_cache`: enable legacy candidate brief cache.
- `evi_debug_*`: console/file debug tracing controls.

## Debug Trace

The default debug trace is `<run_dir>/evi_debug.log` when runtime paths are available. It records structured `[TRACE]` blocks for:

- `indexing_start`
- `indexing_done`
- `answer_start`
- `retrieved_anchors`
- `raw_retrieval_clue_coverage` when QA clue metadata exists
- `episode_set_retrieval` with per-facet witness rounds, expanded members, and direct/episode fusion
- `episode_directory_retrieval` with full-question holistic session ranking and diagnostic witness anchors
- `episode_directory_v2_retrieval` with max-packet session ranking and the best witness round per session
- `episode_directory_v3_retrieval` with full-question ranking over cached query-independent session cards
- `facet_multimodal_round_scoring` with calibrated source scores and the per-facet round ranking
- New v2 traces also contain compact `packet_scores` and natural `packet_round_ids` for offline aggregation and exact expansion replay.
- clue coverage for `direct_anchor_top10`, `episode_path_top10`, `direct_episode_fused_top10`, and `final_retrieval_top10`; each trace includes both exact round coverage and target-session coverage
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
- Session retrieval cards: `EVI_EPISODE_CARD_CACHE_DIR`, default `~/.cache/evi_episode_cards`.

Session-card cache keys include separate prompt and postprocessing versions. The
postprocessor canonicalizes local model IDs such as `R1` to full provenance IDs
before completeness checks, so formatting variation does not trigger duplicate
raw-dialogue fallback.

State cache keys include prompt version, model namespace, question stem, memory set content, selected anchors, and image-use settings.

## Retrieval-Only Episode-Set Path

The optional episode-set path uses the dataset's natural session boundary as a non-lossy memory node. It does not summarize sessions and does not call another LLM. For each retrieval facet, the session score is witnessed by the best matching member anchor, so different facets may be grounded by different rounds in one episode. Ranked episode members are expanded and fused with the unchanged direct round path using reciprocal ranks. Raw-image retrieval then corroborates the combined semantic candidates exactly as in the image-rerank ablation.

Keep this path soft: direct candidates remain eligible, and episode membership must never hard-filter the global round ranking. Retrieval-only artifacts expose `episode_ranked_round_ids` and `direct_episode_fused_round_ids` for component-level analysis.

Use `analyze_episode_oracle.py` on an episode-set retrieval suite before adding a
stronger session directory. It keeps only annotated clue sessions, preserves their
saved relative order, and replays the existing member expansion, reciprocal-rank
fusion, and image reranking. Its result is an oracle-assisted headroom estimate,
not a strict mathematical upper bound. The report separates absent sessions,
episode-path budget misses, intra-session misses, and fusion displacement. It also
compares current Top-M selection, whole-session oracle expansion, and round-robin
balanced oracle expansion under the same total round budget. Treat these as
counterfactual diagnostics, not deployable retrieval policies.

Use `analyze_session_card_ablation.py` on a completed v3 Card suite to compare
the saved full Card against `Episode identity` and `Episode identity +
Distinctive evidence`. It re-embeds only saved query-independent text, performs
no VLM calls, and replays current/packet/Card RRF variants through the saved
round expansion and image-reranking pipeline. Run it where the configured text
embedding model is available; the analysis cache is stored inside the suite.
## Compact Card Online Path

config/methods/evi_compact_card_episode_image_rerank.yaml applies the cached
query-independent session Card online. Only Episode identity and Distinctive
evidence are embedded. The complete Card remains a provenance/debug artifact.

The full-question compact-Card session ranking is fused with the existing
episode-set session ranking using parameter-free reciprocal ranks. The fusion
only reorders sessions already supported by the episode-set path. It then reuses
the unchanged session expansion, direct-round fusion, raw-image candidate
reranking, and Top-10 raw multimodal QA path. Never send Card text to final QA.

The trace records compact-Card session ranks and
episode_compact_card_fusion_rows; the existing final retrieval clue-coverage
trace remains the end-to-end retrieval diagnostic.

## Research Notes

The paper-facing story should distinguish:

- Flat RAG: retrieve anchors/chunks and answer directly.
- Legacy candidate assertion: judge each candidate independently.
- Episodic-state EVI: retrieve anchors, reconstruct ordered local memory state, then answer from clean itemized states.

Useful ablations:

- `evi_pipeline: candidate_assertion` vs `episodic_state`.
- `use_state_images: true` vs `false`.
- `use_final_memory_images: true` vs `false`.
- Varying memory set window size and `max_state_anchors_per_round`.
- `use_dataset_captions: true` vs `false`.

Report answer accuracy together with retrieval clue coverage, memory-set coverage, state readout traces, final prompt length, and selected image counts from EVI logs.
