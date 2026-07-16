from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from router import GeminiAPIRouter, OpenAIAPIRouter, QwenLocalRouter

from ..dataset import history_from_round_ids
from ._utils import extract_json
from .briefs import generate_memory_briefs
from .candidates import consolidate_candidates
from .extractor import extract_image_anchors
from .indexes import EvidenceIndex, embed_text, normalize_type
from .schemas import EpisodicState, EvidenceAnchor, MemoryBrief
from .sets import build_episodic_memory_sets, build_session_memory_sets, build_session_memory_sets_from_candidates
from .states import read_episodic_states
from .trace import (
    anchors_summary,
    briefs_summary,
    candidates_summary,
    memory_sets_summary,
    setup_evi_debug_logging,
    states_summary,
    trace_json,
)
from .vlm import VLMCallable, make_vlm_callable

log = logging.getLogger(__name__)

FINAL_ANSWER_SYSTEM_PROMPT = """You are answering a multimodal long-term memory question.
Use the selected memory evidence as the primary evidence.
The evidence may be reconstructed episodic states or selected factual assertions.
Use attached images only to resolve uncertainty.
Be concise and grounded in the selected evidence.
If the question is multiple-choice, answer with ONLY the option letter.
"""
FACET_PROMPT_VERSION = "retrieval_facets_v3b_dedup_locator"
SCOPE_INTERPRETATION_PROMPT_VERSION = "question_scope_brief_v1"
EVIDENCE_ORGANIZER_PROMPT_VERSION = "session_round_evidence_v4_complete_scope_labels"

FACET_EXTRACTION_SYSTEM_PROMPT = """You extract retrieval facets for a multimodal long-term memory system.

The goal is only retrieval. Do not answer the question.
A retrieval facet is a self-contained memory locator phrase, not a keyword list.
Keep modifiers attached to the objects/events they describe.
If a phrase only makes sense because of another phrase, merge them into one facet.
Include both sides of comparisons when the question compares memories.
Include collection/scope phrases when the question asks over a set.
Use original nouns, names, visual descriptions, dates, order words, and quoted labels when possible.

Return ONLY valid JSON:
{
  "facets": ["retrieval phrase 1", "retrieval phrase 2"]
}

Guidelines:
- Extract 2-4 facets unless the question is extremely simple.
- Each facet should be independently searchable as a memory locator.
- Do not create answer choices or solve the question.
- Do not split compound visual or temporal descriptions into isolated words.
- Do not output one facet that is mostly contained inside another facet; keep the more specific locator.
- Avoid generic standalone words such as scene, image, item, later, compare, before, after, count.

Examples:
Question: "Which item was on the red object near the wooden table in the last photo?"
Bad facets: ["red", "object", "table", "last photo"]
Good facets: ["red object near the wooden table in the last photo"]

Question: "After the rainy street scene with a person holding an umbrella, what vehicle appeared at the corner?"
Bad facets: ["rainy", "street", "umbrella", "rainy street scene with a person holding an umbrella", "vehicle", "corner"]
Good facets: ["rainy street scene with a person holding an umbrella", "vehicle at the corner after that rainy street scene"]

Question: "Between the first meeting with the blue notebook and the later meeting with the white folder, which one had the round wall clock?"
Bad facets: ["first", "meeting", "blue notebook", "white folder", "clock"]
Good facets: ["first meeting with the blue notebook", "later meeting with the white folder", "round wall clock in one of the meetings"]

Question: "How did the wooden shelf change between its earlier image and the most recent image?"
Bad facets: ["wooden shelf", "wooden shelf in the earlier image", "wooden shelf in the most recent image", "last image"]
Good facets: ["wooden shelf in the earlier image", "wooden shelf in the most recent image"]

Question: "Which cue belongs to the scene with the green bag on the left of the gray suitcase rather than the scene with the black backpack?"
Bad facets: ["left", "green bag", "gray suitcase", "black backpack"]
Good facets: ["scene with the green bag on the left of the gray suitcase", "scene with the black backpack"]
"""

SCOPE_INTERPRETATION_SYSTEM_PROMPT = """You interpret the information scope of a question for a multimodal long-term memory system.

The scope description will later be used to judge whether retrieved memory rounds belong to the same requested entity, collection, event, comparison, or time span.

Describe only the boundary implied by the question:
- what entity, collection, event, comparison, or time span the answer must concern;
- what information may help identify or disambiguate that requested scope;
- what superficially similar information must not be mixed into that scope.

Do not inspect or refer to memory rounds. Do not retrieve evidence. Do not answer the question, count, compare, select an option, or introduce facts that are not stated in the question.

Return only a concise natural-language scope brief in 1-3 sentences. Do not use JSON, headings, scores, or bullet lists.
"""

EVIDENCE_ORGANIZER_SYSTEM_PROMPT = f"""You annotate retrieved multimodal memory rounds for a later QA model.

Version: {EVIDENCE_ORGANIZER_PROMPT_VERSION}

You are not answering the question.
You MUST return exactly one decision for every candidate round id given in the user message. Never omit a candidate round.

When a scope interpretation is provided, treat it as the fixed boundary for this question. Decide membership in that scope independently from whether a round satisfies the question's final property, condition, comparison, or answer criterion.
- keep: the round belongs to the requested scope and directly provides answer evidence.
- weak_keep: the round belongs to the requested scope but is contextual, uncertain, or provides negative evidence. A member of the requested scope that does not satisfy the queried property MUST be weak_keep, not drop.
- drop: the round is clearly outside the requested entity, collection, event, comparison, or time span. Visual or lexical similarity alone does not make it in scope.

Use the attached image as primary visual evidence when available. Dialogue and captions may help, but do not replace visual inspection.
Do not choose an answer or perform the final counting, comparison, or option selection.
Do not mention options, scores, retrieval facets, or confidence.
Keep notes concrete, concise, and grounded in that round.

Return ONLY valid JSON:
{{
  "round_notes": [
    {{
      "round_id": "ROUND_ID",
      "decision": "keep",
      "note": "This round can provide ..."
    }}
  ]
}}

Allowed decision values: keep, weak_keep, drop.
"""

class EVISystem:
    """QDMO-EVI over typed evidence anchors with episodic-state readout."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._cfg = cfg
        self._model_cfg = dict(cfg.get("_model_cfg", {}))
        self._debug_log_path = setup_evi_debug_logging(cfg)

        self._index = EvidenceIndex()
        self._vlm: Optional[VLMCallable] = None
        self._embedder: Optional[Any] = None
        self._answer_routers: Dict[str, Any] = {}

        self._round_order: List[str] = []
        self._round_session: Dict[str, str] = {}
        self._round_date: Dict[str, str] = {}
        self._round_text: Dict[str, str] = {}
        self._round_images: Dict[str, List[str]] = {}
        self._round_anchors: Dict[str, List[EvidenceAnchor]] = {}
        self._session_rounds: Dict[str, List[str]] = {}
        self._current_dataset: Optional[Any] = None
        self._last_context_round_ids: List[str] = []

        self._pipeline = str(cfg.get("evi_pipeline", "episodic_state") or "episodic_state").strip().lower()
        self._raw_search_k = int(cfg.get("raw_search_k", 120))
        self._max_candidates = int(cfg.get("max_candidates", 20))
        self._max_candidate_anchors = int(cfg.get("max_candidate_anchors", 8))
        self._max_final_briefs = int(cfg.get("max_final_briefs", 10))
        self._max_excluded_briefs = int(cfg.get("max_excluded_briefs", 3))
        self._max_answer_images = int(cfg.get("max_answer_images", 12))
        self._use_final_memory_images = self._as_bool(cfg.get("use_final_memory_images"), False)
        self._use_dataset_captions = self._as_bool(cfg.get("use_dataset_captions"), False)
        self._use_embedding_cache = self._as_bool(cfg.get("use_embedding_cache"), True)
        self._use_memory_brief_cache = self._as_bool(cfg.get("use_memory_brief_cache"), True)
        self._max_memory_sets = int(cfg.get("max_memory_sets", 8))
        self._memory_set_window_before = int(cfg.get("memory_set_window_before", 1))
        self._memory_set_window_after = int(cfg.get("memory_set_window_after", 1))
        self._max_rounds_per_memory_set = int(cfg.get("max_rounds_per_memory_set", 5))
        self._max_state_anchors_per_round = int(cfg.get("max_state_anchors_per_round", 6))
        self._max_final_states = int(cfg.get("max_final_states", 4))
        self._max_selected_rounds = int(cfg.get("max_selected_rounds", 10))
        self._min_selected_rounds = int(cfg.get("min_selected_rounds", 1))
        self._round_selector_max_new_tokens = int(cfg.get("round_selector_max_new_tokens", 512))
        self._facet_max_facets = int(cfg.get("facet_max_facets", 4))
        self._facet_search_k = int(cfg.get("facet_search_k", 30))
        self._facet_round_fusion = str(
            cfg.get("facet_round_fusion", "max_similarity_times_reciprocal_rank_consensus")
            or "max_similarity_times_reciprocal_rank_consensus"
        ).strip().lower()
        valid_facet_round_fusions = {
            "max_similarity_times_best_source_rank_consensus",
            "max_similarity_times_reciprocal_rank_consensus",
            "source_facet_reciprocal_rank_consensus",
        }
        if self._facet_round_fusion not in valid_facet_round_fusions:
            raise ValueError(
                f"Unsupported facet_round_fusion={self._facet_round_fusion!r}; "
                f"expected one of {sorted(valid_facet_round_fusions)}"
            )

        self._facet_full_question_weight = float(cfg.get("facet_full_question_weight", 1.0))
        self._use_retrieval_facets = self._as_bool(cfg.get("evi_use_retrieval_facets"), True)
        self._use_image_anchors = self._as_bool(cfg.get("evi_use_image_anchors"), True)
        self._retrieval_only = self._as_bool(cfg.get("evi_retrieval_only"), False)
        self._use_evidence_organizer = self._as_bool(cfg.get("evi_use_evidence_organizer"), True)
        self._final_include_evidence_images = self._as_bool(cfg.get("evi_final_include_evidence_images"), True)
        self._final_max_rounds = int(cfg.get("evi_final_max_rounds", 10))
        self._organizer_max_rounds_per_session = int(cfg.get("evi_organizer_max_rounds_per_session", 0) or 0)
        self._use_scope_interpretation = self._as_bool(cfg.get("evi_use_scope_interpretation"), True)
        self._apply_scope_to_organizer = self._as_bool(cfg.get("evi_apply_scope_to_organizer"), True)
        self._use_scope_cache = self._as_bool(cfg.get("use_scope_cache"), True)

        self._use_facet_cache = self._as_bool(cfg.get("use_facet_cache"), True)
        self._use_round_selection_cache = self._as_bool(cfg.get("use_round_selection_cache"), True)
        self._use_state_cache = self._as_bool(cfg.get("use_state_cache"), True)
        self._use_state_images = self._as_bool(cfg.get("use_state_images"), True)
        self._max_state_images_per_set = int(cfg.get("max_state_images_per_set", 4))
        self._debug_top_k = int(cfg.get("evi_debug_top_k", 20))
        self._debug_prompt_chars = int(cfg.get("evi_debug_prompt_chars", 12000))
        self._embed_cache_namespace = "uninitialized"
        self._vlm_result_namespace = "uninitialized"

        self._initialized = False
        if self._debug_log_path:
            log.info("EVI debug log: %s", self._debug_log_path)

    def _ensure(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        vlm_cfg = dict(self._model_cfg)
        if self._cfg.get("vlm_max_new_tokens"):
            vlm_cfg["max_new_tokens"] = int(self._cfg.get("vlm_max_new_tokens"))
        if self._cfg.get("vlm_timeout"):
            vlm_cfg["timeout"] = int(self._cfg.get("vlm_timeout"))
        self._vlm_result_namespace = json.dumps(
            {
                "provider": vlm_cfg.get("provider", "openai_api"),
                "model": vlm_cfg.get("model", "gpt-4o"),
                "base_url": vlm_cfg.get("base_url", "https://api.openai.com/v1"),
                "max_new_tokens": int(vlm_cfg.get("max_new_tokens", 1024) or 1024),
                "timeout": int(vlm_cfg.get("timeout", 90) or 90),
            },
            sort_keys=True,
            default=str,
        )
        self._vlm = make_vlm_callable(vlm_cfg)

        from ..embeddings import TextEmbedder

        embed_model = (
            self._cfg.get("text_embedding_model")
            or self._model_cfg.get("text_embedding_model")
            or TextEmbedder.DEFAULT_MODEL
        )
        embed_kwargs = dict(
            self._cfg.get("text_embedding_kwargs")
            or self._model_cfg.get("text_embedding_kwargs")
            or {}
        )
        self._embed_cache_namespace = json.dumps(
            {"model": embed_model, "kwargs": embed_kwargs},
            sort_keys=True,
            default=str,
        )
        self._embedder = TextEmbedder(embed_model, **embed_kwargs)
        if self._embedder.is_available:
            log.info("Embedder loaded: %s", embed_model)
        else:
            log.error(
                "Embedder unavailable: %s. Evidence anchors will be empty; check embedding dependencies.",
                embed_model,
            )
            self._embedder = None

    def _embed(self, text: str) -> List[float]:
        return embed_text(
            text,
            self._embedder,
            cache_namespace=self._embed_cache_namespace,
            use_cache=self._use_embedding_cache,
        )

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _as_image_list(value: Optional[Any]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(path) for path in value if str(path)]
        return [str(value)] if str(value) else []

    def _make_anchor(
        self,
        *,
        session_id: str,
        round_id: str,
        date: str,
        evidence_type: str,
        text: str,
        image_path: Optional[str] = None,
        subject: str = "",
        predicate: str = "",
        object: str = "",
        region: str = "",
        confidence: float = 1.0,
        suffix: str = "",
    ) -> Optional[EvidenceAnchor]:
        text = str(text or "").strip()
        if not text:
            return None
        evidence_type = normalize_type(evidence_type)
        anchor_id = f"{round_id}:{evidence_type}:{len(self._index)}"
        if suffix:
            anchor_id += f":{suffix}"
        vector = self._embed(text)
        if not vector:
            return None
        return EvidenceAnchor(
            id=anchor_id,
            session_id=session_id,
            round_id=round_id,
            date=date,
            image_path=image_path,
            evidence_type=evidence_type,
            text=text,
            subject=subject,
            predicate=predicate,
            object=object,
            region=region,
            confidence=confidence,
            vector=vector,
        )


    def _add_anchor(self, anchor: EvidenceAnchor) -> None:
        self._index.add(anchor)
        self._round_anchors.setdefault(anchor.round_id, []).append(anchor)

    def process_all_sessions(self, dataset: Any) -> None:
        """Build task-agnostic typed evidence anchors for all sessions."""
        self._ensure()
        if dataset is not None:
            self._current_dataset = dataset
        if self._vlm is None:
            raise RuntimeError("VLM not initialized")

        log.info("Processing sessions for QDMO-EVI evidence anchors ...")
        trace_json(log, "indexing_start", {
            "pipeline": self._pipeline,
            "raw_search_k": self._raw_search_k,
            "max_candidates": self._max_candidates,
            "max_candidate_anchors": self._max_candidate_anchors,
            "max_final_briefs": self._max_final_briefs,
            "max_answer_images": self._max_answer_images,
            "use_final_memory_images": self._use_final_memory_images,
            "use_dataset_captions": self._use_dataset_captions,
            "use_embedding_cache": self._use_embedding_cache,
            "use_memory_brief_cache": self._use_memory_brief_cache,
            "max_memory_sets": self._max_memory_sets,
            "memory_set_window_before": self._memory_set_window_before,
            "memory_set_window_after": self._memory_set_window_after,
            "max_rounds_per_memory_set": self._max_rounds_per_memory_set,
            "max_state_anchors_per_round": self._max_state_anchors_per_round,
            "max_final_states": self._max_final_states,
            "max_selected_rounds": self._max_selected_rounds,
            "min_selected_rounds": self._min_selected_rounds,
            "round_selector_max_new_tokens": self._round_selector_max_new_tokens,
            "facet_max_facets": self._facet_max_facets,
            "facet_search_k": self._facet_search_k,
            "facet_round_fusion": self._facet_round_fusion,
            "facet_full_question_weight": self._facet_full_question_weight,
            "evi_use_retrieval_facets": self._use_retrieval_facets,
            "evi_use_image_anchors": self._use_image_anchors,
            "evi_retrieval_only": self._retrieval_only,
            "evi_use_evidence_organizer": self._use_evidence_organizer,
            "evi_final_include_evidence_images": self._final_include_evidence_images,
            "evi_final_max_rounds": self._final_max_rounds,
            "evi_organizer_max_rounds_per_session": self._organizer_max_rounds_per_session,
            "evi_use_scope_interpretation": self._use_scope_interpretation,
            "evi_apply_scope_to_organizer": self._apply_scope_to_organizer,
            "use_scope_cache": self._use_scope_cache,
            "scope_interpretation_prompt_version": SCOPE_INTERPRETATION_PROMPT_VERSION,
            "evidence_organizer_prompt_version": EVIDENCE_ORGANIZER_PROMPT_VERSION,
            "use_facet_cache": self._use_facet_cache,
            "use_round_selection_cache": self._use_round_selection_cache,
            "use_state_cache": self._use_state_cache,
            "use_state_images": self._use_state_images,
            "max_state_images_per_set": self._max_state_images_per_set,
        })

        for sid in dataset.session_order():
            session = dataset.get_session(sid)
            date = str(session.get("date", "")).strip() or "unknown"
            prior_rounds_text: List[str] = []
            self._session_rounds.setdefault(sid, [])
            log.info(
                "[INDEX] session=%s date=%s rounds=%d",
                sid,
                date,
                len(session.get("dialogues", []) or []),
            )

            for dialogue in session.get("dialogues", []):
                rid = str(dialogue.get("round", "")).strip()
                if not rid:
                    continue
                rp = dataset.rounds.get(rid, {})
                user_text = str(rp.get("user", "")).strip()
                assistant_text = str(rp.get("assistant", "")).strip()
                if not user_text and not assistant_text and not (rp.get("images") or []):
                    continue

                round_text = f"User: {user_text}\nAssistant: {assistant_text}".strip()
                self._round_order.append(rid)
                self._session_rounds.setdefault(sid, []).append(rid)
                self._round_session[rid] = sid
                self._round_date[rid] = date
                self._round_text[rid] = round_text
                raw_images = list(rp.get("images", []) or [])
                self._round_images[rid] = [str(path) for path in raw_images if os.path.isfile(str(path))]

                dialogue_anchor = self._make_anchor(
                    session_id=sid,
                    round_id=rid,
                    date=date,
                    evidence_type="temporal",
                    text=f"Round {rid} on {date}. {round_text}",
                )
                if dialogue_anchor is not None:
                    self._add_anchor(dialogue_anchor)

                if self._use_dataset_captions:
                    raw = rp.get("raw", {}) or {}
                    for cidx, caption in enumerate(raw.get("image_caption", []) or []):
                        caption_anchor = self._make_anchor(
                            session_id=sid,
                            round_id=rid,
                            date=date,
                            evidence_type="scene",
                            text=f"Dataset image caption: {caption}",
                            suffix=f"caption{cidx}",
                        )
                        if caption_anchor is not None:
                            self._add_anchor(caption_anchor)

                if self._use_image_anchors:
                    images = list(self._round_images.get(rid, []))
                    for img_idx, image_path in enumerate(images):
                        if not os.path.isfile(image_path):
                            log.warning("[INDEX] image file not found, skipping: %s", image_path)
                            continue
                        raw_anchors = extract_image_anchors(
                            image_path=image_path,
                            round_text=round_text,
                            prior_rounds_text="\n---\n".join(prior_rounds_text),
                            vlm_callable=self._vlm,
                            cache_namespace=self._vlm_result_namespace,
                        )
                        log.debug("[INDEX] round=%s image=%s extracted_anchors=%d", rid, image_path, len(raw_anchors))
                        for aidx, raw_anchor in enumerate(raw_anchors):
                            anchor = self._make_anchor(
                                session_id=sid,
                                round_id=rid,
                                date=date,
                                image_path=image_path,
                                evidence_type=str(raw_anchor.get("evidence_type", "scene")),
                                text=str(raw_anchor.get("text", "")),
                                subject=str(raw_anchor.get("subject", "")),
                                predicate=str(raw_anchor.get("predicate", "")),
                                object=str(raw_anchor.get("object", "")),
                                region=str(raw_anchor.get("region", "")),
                                confidence=float(raw_anchor.get("confidence", 1.0)),
                                suffix=f"img{img_idx}_{aidx}",
                            )
                            if anchor is not None:
                                self._add_anchor(anchor)

                prior_rounds_text.append(round_text)

        type_counts: Dict[str, int] = {}
        for anchor in self._index.anchors:
            type_counts[anchor.evidence_type] = type_counts.get(anchor.evidence_type, 0) + 1
        log.info(
            "QDMO-EVI indexing done: %d anchors across %d rounds. Types: %s",
            len(self._index),
            len(self._round_order),
            ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items())),
        )
        trace_json(log, "indexing_done", {
            "num_anchors": len(self._index),
            "num_rounds": len(self._round_order),
            "type_counts": dict(sorted(type_counts.items())),
            "sample_anchors": anchors_summary(self._index.anchors, max_items=self._debug_top_k),
        })

    def _select_final_briefs(self, briefs: List[MemoryBrief]) -> List[MemoryBrief]:
        relevant = [b for b in briefs if b.relevance == "relevant"]
        uncertain = [b for b in briefs if b.relevance == "uncertain"]
        relevant.sort(key=lambda b: (b.confidence, b.score), reverse=True)
        uncertain.sort(key=lambda b: (b.confidence, b.score), reverse=True)

        selected = relevant[: self._max_final_briefs]
        if selected:
            return selected
        return uncertain[: self._max_final_briefs]
    def _build_final_prompt(self, question: str, briefs: List[MemoryBrief]) -> str:
        lines: List[str] = []
        lines.append("Evidence assertions:")
        if not briefs:
            lines.append("No selected memory evidence was available.")
        for idx, brief in enumerate(briefs, start=1):
            assertion = " ".join(str(brief.brief or "").split())
            lines.append(f"{idx}. Memory from {brief.round_id} on {brief.date}: {assertion}")
            facts = [" ".join(str(item).split()) for item in brief.key_evidence if str(item).strip()]
            if facts:
                lines.append("   Observed facts: " + "; ".join(facts[:3]))
        lines.append("")
        lines.append("Question:")
        lines.append(str(question or ""))
        lines.append("")
        #lines.append("Answer using only the evidence assertions and attached images. Do not use excluded or unselected memories.")
        return "\n".join(lines)
    def _answer_images(self, briefs: List[MemoryBrief], question_images: Optional[List[str]]) -> List[str]:
        images: List[str] = []
        seen: set[str] = set()
        for path in self._as_image_list(question_images):
            if path not in seen:
                images.append(path)
                seen.add(path)
            if len(images) >= self._max_answer_images:
                return images
        if not self._use_final_memory_images:
            return images
        for brief in briefs:
            if brief.relevance == "excluded":
                continue
            for path in brief.image_paths:
                if path and path not in seen:
                    images.append(path)
                    seen.add(path)
                if len(images) >= self._max_answer_images:
                    return images
        return images

    def _log_raw_clue_coverage(self, qa: Optional[Dict[str, Any]], retrieved: List[EvidenceAnchor]) -> None:
        clue_rounds = (qa or {}).get("clue", [])
        if not clue_rounds:
            return
        retrieved_rounds = {anchor.round_id for anchor in retrieved}
        hits = [rid for rid in clue_rounds if rid in retrieved_rounds]
        misses = [rid for rid in clue_rounds if rid not in set(hits)]
        ranks: Dict[str, int] = {}
        for idx, anchor in enumerate(retrieved, start=1):
            ranks.setdefault(anchor.round_id, idx)
        hit_ranks = {rid: ranks.get(rid) for rid in hits}
        log.info(
            "QDMO-EVI raw retrieval clue coverage: %d/%d hits=%s misses=%s ranks=%s",
            len(hits),
            len(clue_rounds),
            hits,
            misses,
            hit_ranks,
        )
        trace_json(log, "raw_retrieval_clue_coverage", {
            "num_hits": len(hits),
            "num_clues": len(clue_rounds),
            "hits": hits,
            "misses": misses,
            "hit_ranks": hit_ranks,
        })

    def _log_final_state_clue_coverage(self, qa: Optional[Dict[str, Any]], states: List[EpisodicState]) -> None:
        clue_rounds = (qa or {}).get("clue", [])
        if not clue_rounds:
            return
        state_rounds = {rid for state in states for rid in state.round_ids}
        hits = [rid for rid in clue_rounds if rid in state_rounds]
        misses = [rid for rid in clue_rounds if rid not in set(hits)]
        per_state_rounds = {
            state.set_id: list(state.round_ids)
            for state in states
        }
        log.info(
            "QDMO-EVI final state clue coverage: %d/%d hits=%s misses=%s state_rounds=%s",
            len(hits),
            len(clue_rounds),
            hits,
            misses,
            per_state_rounds,
        )
        trace_json(log, "final_state_clue_coverage", {
            "num_hits": len(hits),
            "num_clues": len(clue_rounds),
            "hits": hits,
            "misses": misses,
            "state_rounds": per_state_rounds,
        })

    def answer_question(
        self,
        question: str,
        qa: Optional[Dict[str, Any]] = None,
        question_images: Optional[List[str]] = None,
        dataset: Optional[Any] = None,
    ) -> str:
        self._ensure()
        if dataset is not None:
            self._current_dataset = dataset
        if self._vlm is None:
            raise RuntimeError("VLM not initialized")

        question_stem = str((qa or {}).get("question", "")).strip() or str(question or "")
        log.info("############## QDMO-EVI answering question: %s", question_stem)
        trace_json(log, "answer_start", {
            "question_stem": question_stem,
            "question_full": str(question or ""),
            "question_images": self._as_image_list(question_images),
            "qa_id": (qa or {}).get("id") or (qa or {}).get("question_id"),
            "clue_rounds": (qa or {}).get("clue", []),
        })

        if not self._index.anchors:
            log.error("QDMO-EVI index is empty; falling back to question-only answer with raw images.")
            images = self._as_image_list(question_images)[: self._max_answer_images]
            return self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, question, images)

        if self._pipeline == "faceted_topk":
            return self._answer_with_faceted_topk(question, question_stem, qa, question_images)

        query_vec = self._embed(question_stem)
        if not query_vec:
            log.warning("QDMO-EVI query embedding is empty; answering with question images only")
            images = self._as_image_list(question_images)[: self._max_answer_images]
            return self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, question, images)

        for anchor in self._index.anchors:
            anchor.score = 0.0
        retrieved = self._index.search(query_vec, top_k=self._raw_search_k)
        log.info("QDMO-EVI retrieved anchors=%d", len(retrieved))
        for idx, anchor in enumerate(retrieved[: self._debug_top_k]):
            log.info(
                "  anchor[%02d] id=%s round=%s type=%s score=%.4f text=%s",
                idx + 1,
                anchor.id,
                anchor.round_id,
                anchor.evidence_type,
                anchor.score or 0.0,
                anchor.text.replace("\n", " ")[:160],
            )
        if len(retrieved) > self._debug_top_k:
            log.info("  ... and %d more retrieved anchors", len(retrieved) - self._debug_top_k)
        trace_json(log, "retrieved_anchors", {
            "num_retrieved": len(retrieved),
            "top_anchors": anchors_summary(retrieved, max_items=min(self._debug_top_k, len(retrieved))),
        })
        self._log_raw_clue_coverage(qa, retrieved)

        if self._pipeline == "candidate_assertion":
            return self._answer_with_candidate_assertions(question, question_stem, qa, question_images, retrieved)
        if self._pipeline == "consolidated_topk":
            return self._answer_with_consolidated_topk(question, question_stem, qa, question_images, retrieved)
        if self._pipeline == "session_round_selection":
            return self._answer_with_session_round_selection(question, question_stem, qa, question_images, retrieved)
        return self._answer_with_episodic_states(question, question_stem, qa, question_images, retrieved)

    def _answer_with_candidate_assertions(
        self,
        question: str,
        question_stem: str,
        qa: Optional[Dict[str, Any]],
        question_images: Optional[List[str]],
        retrieved: List[EvidenceAnchor],
    ) -> str:
        candidates = consolidate_candidates(
            retrieved,
            round_text=self._round_text,
            max_candidates=self._max_candidates,
            max_candidate_anchors=self._max_candidate_anchors,
        )
        trace_json(log, "candidate_pool", {
            "num_candidates": len(candidates),
            "candidates": candidates_summary(candidates, max_items=self._max_candidates),
        })

        clue_rounds = (qa or {}).get("clue", [])
        if clue_rounds:
            candidate_rounds = {candidate.round_id for candidate in candidates}
            hits = [rid for rid in clue_rounds if rid in candidate_rounds]
            misses = [rid for rid in clue_rounds if rid not in set(hits)]
            log.info(
                "QDMO-EVI candidate clue coverage: %d/%d hits=%s misses=%s",
                len(hits),
                len(clue_rounds),
                hits,
                misses,
            )
            trace_json(log, "candidate_clue_coverage", {
                "num_hits": len(hits),
                "num_clues": len(clue_rounds),
                "hits": hits,
                "misses": misses,
            })

        briefs = generate_memory_briefs(
            question_stem,
            candidates,
            self._vlm,
            use_cache=self._use_memory_brief_cache,
            cache_namespace=self._vlm_result_namespace,
        )
        trace_json(log, "memory_briefs", {
            "num_briefs": len(briefs),
            "briefs": briefs_summary(briefs, max_items=self._max_candidates),
        })

        final_briefs = self._select_final_briefs(briefs)
        self._set_last_context_round_ids([brief.round_id for brief in final_briefs])
        log.info("QDMO-EVI selected final evidence assertions=%d", len(final_briefs))
        for idx, brief in enumerate(final_briefs, start=1):
            log.info(
                "  final_assertion[%02d] candidate=%s relevance=%s confidence=%.2f round=%s",
                idx,
                brief.candidate_id,
                brief.relevance,
                brief.confidence,
                brief.round_id,
            )
        trace_json(log, "selected_memory_assertions", {
            "num_selected": len(final_briefs),
            "assertions": briefs_summary(final_briefs, max_items=self._max_final_briefs),
        })

        prompt = self._build_final_prompt(question, final_briefs)
        images = self._answer_images(final_briefs, question_images)
        return self._call_final_answer(prompt, images, "candidate_assertion")

    def _select_final_states(self, states: List[EpisodicState]) -> List[EpisodicState]:
        relevant = [state for state in states if state.relevance == "relevant"]
        uncertain = [state for state in states if state.relevance == "uncertain"]
        relevant.sort(key=lambda s: (s.score, s.confidence), reverse=True)
        uncertain.sort(key=lambda s: (s.score, s.confidence), reverse=True)
        selected = relevant[: self._max_final_states]
        if selected:
            return selected
        return uncertain[: self._max_final_states]

    def _build_state_final_prompt(self, question: str, states: List[EpisodicState]) -> str:
        lines: List[str] = []
        lines.append("Reconstructed episodic memory states:")
        if not states:
            lines.append("No selected episodic memory state was available.")
        for idx, state in enumerate(states, start=1):
            lines.append(
                f"{idx}. Memory state from session {state.session_id} on {state.date}; "
                f"rounds {' -> '.join(state.round_ids)}."
            )
            if state.memory_items:
                lines.append("   Memory items:")
                for item in state.memory_items[:12]:
                    lines.append("   - " + " ".join(str(item).split()))
            if state.observations:
                lines.append("   Observations:")
                for item in state.observations[:6]:
                    lines.append("   - " + " ".join(str(item).split()))
            if state.relations or state.changes:
                lines.append("   Relations and changes:")
                for item in (state.relations + state.changes)[:8]:
                    lines.append("   - " + " ".join(str(item).split()))
            if state.answer_relevant_facts:
                lines.append("   Answer-relevant facts:")
                for item in state.answer_relevant_facts[:8]:
                    lines.append("   - " + " ".join(str(item).split()))
            if state.uncertainties:
                lines.append("   Uncertainties:")
                for item in state.uncertainties[:4]:
                    lines.append("   - " + " ".join(str(item).split()))
        lines.append("")
        lines.append("Question:")
        lines.append(str(question or ""))
        lines.append("")
        #lines.append("Answer using only the reconstructed memory states. If the question is multiple-choice, answer with only the option letter.")
        return "\n".join(lines)

    def _answer_images_for_states(self, states: List[EpisodicState], question_images: Optional[List[str]]) -> List[str]:
        images: List[str] = []
        seen: set[str] = set()
        for path in self._as_image_list(question_images):
            if path not in seen:
                images.append(path)
                seen.add(path)
            if len(images) >= self._max_answer_images:
                return images
        if not self._use_final_memory_images:
            return images
        for state in states:
            if state.relevance == "excluded":
                continue
            for path in state.image_paths:
                if path and path not in seen:
                    images.append(path)
                    seen.add(path)
                if len(images) >= self._max_answer_images:
                    return images
        return images

    def _load_answer_system_prompt(self, mode: str = "open") -> str:
        prompt_dir = Path(__file__).resolve().parents[1] / "prompt"
        if mode not in {"open", "mcq"}:
            mode = "open"
        mode_file = prompt_dir / f"sys_prompt_{mode}.txt"
        if mode_file.exists():
            return mode_file.read_text(encoding="utf-8").strip()
        fallback = prompt_dir / "sys_prompt.txt"
        return fallback.read_text(encoding="utf-8").strip() if fallback.exists() else ""

    def _get_answer_router(self, mode: str) -> Any:
        if mode in self._answer_routers:
            return self._answer_routers[mode]
        model_cfg = dict(self._model_cfg or {})
        system_prompt = self._load_answer_system_prompt(mode)
        provider = str(model_cfg.get("provider", "openai_api")).strip()
        if provider == "qwen_local":
            router = QwenLocalRouter(
                model_path=str(model_cfg["model_path"]),
                max_new_tokens=int(model_cfg.get("round_selector_max_new_tokens", 512)),
                system_prompt=system_prompt,
                max_time=model_cfg.get("max_time", 25),
            )
        elif provider == "openai_api":
            router = OpenAIAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "OPENAI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://api.openai.com/v1")),
                max_new_tokens=int(model_cfg.get("round_selector_max_new_tokens", 512)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
        elif provider == "gemini_api":
            router = GeminiAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "GEMINI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")),
                max_new_tokens=int(model_cfg.get("round_selector_max_new_tokens", 512)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unsupported provider for EVI final answer: {provider}")
        self._answer_routers[mode] = router
        return router


    def _get_round_selector_router(self) -> Any:
        key = "__round_selector__"
        if key in self._answer_routers:
            return self._answer_routers[key]
        model_cfg = dict(self._model_cfg or {})
        system_prompt = (
            "You select memory round ids for a multimodal long-term memory system. "
            "Use the provided candidate dialogue and attached images only to decide which rounds should be passed to a later answer model. "
            "Do not answer the user question. Return only valid JSON."
        )
        provider = str(model_cfg.get("provider", "openai_api")).strip()
        if provider == "qwen_local":
            router = QwenLocalRouter(
                model_path=str(model_cfg["model_path"]),
                max_new_tokens=int(model_cfg.get("round_selector_max_new_tokens", 512)),
                system_prompt=system_prompt,
                max_time=model_cfg.get("max_time", 25),
            )
        elif provider == "openai_api":
            router = OpenAIAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "OPENAI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://api.openai.com/v1")),
                max_new_tokens=int(model_cfg.get("round_selector_max_new_tokens", 512)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
        elif provider == "gemini_api":
            router = GeminiAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "GEMINI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")),
                max_new_tokens=int(model_cfg.get("round_selector_max_new_tokens", 512)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unsupported provider for EVI round selector: {provider}")
        self._answer_routers[key] = router
        return router

    def _round_selection_cache_dir(self) -> Path:
        raw = os.environ.get("EVI_ROUND_SELECTION_CACHE_DIR")
        path = Path(raw).expanduser() if raw else Path.home() / ".cache" / "evi_round_selection"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _round_selection_cache_key(self, question_stem: str, memory_sets: List[Any]) -> str:
        payload = {
            "version": "round_selection_router_raw_v2",
            "cache_namespace": self._vlm_result_namespace,
            "question_stem": question_stem,
            "max_selected_rounds": self._max_selected_rounds,
            "min_selected_rounds": self._min_selected_rounds,
            "round_selector_max_new_tokens": self._round_selector_max_new_tokens,
            "memory_sets": [
                {
                    "set_id": memory_set.id,
                    "session_id": memory_set.session_id,
                    "round_ids": list(memory_set.round_ids),
                    "round_text": {rid: memory_set.round_text.get(rid, "") for rid in memory_set.round_ids},
                    "round_images": {rid: list(memory_set.round_images.get(rid, [])) for rid in memory_set.round_ids},
                }
                for memory_set in memory_sets
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    def _read_round_selection_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._use_round_selection_cache:
            return None
        path = self._round_selection_cache_dir() / f"{key}.json"
        if not path.exists():
            log.info("  [ROUND SELECT CACHE] MISS key=%s", key)
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            log.info("  [ROUND SELECT CACHE] HIT key=%s selected=%s", key, data.get("selected_round_ids"))
            return data if isinstance(data, dict) else None
        except Exception as exc:
            log.warning("  [ROUND SELECT CACHE] read failed key=%s error=%s", key, exc)
            return None

    def _write_round_selection_cache(self, key: str, payload: Dict[str, Any]) -> None:
        if not self._use_round_selection_cache:
            return
        path = self._round_selection_cache_dir() / f"{key}.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            log.info("  [ROUND SELECT CACHE] WRITE key=%s selected=%s", key, payload.get("selected_round_ids"))
        except Exception as exc:
            log.warning("  [ROUND SELECT CACHE] write failed key=%s error=%s", key, exc)

    def _selection_prompt(self, question_stem: str, memory_sets: List[Any]) -> str:
        candidate_round_ids = [rid for memory_set in memory_sets for rid in memory_set.round_ids]
        lines: List[str] = []
        lines.append("Select the memory rounds that should be passed to the final multimodal answer model.")
        lines.append("The candidate dialogue and images are provided above as conversation history.")
        lines.append("Return ONLY valid JSON with selected_round_ids as a list of candidate round ids.")
        lines.append("Do not answer the question and do not choose a multiple-choice option.")
        lines.append(f"Select at most {self._max_selected_rounds} rounds. Prefer enough rounds to preserve all evidence needed by the later answer model.")
        lines.append("")
        lines.append("Question stem:")
        lines.append(str(question_stem or ""))
        lines.append("")
        lines.append("Candidate round ids:")
        for rid in candidate_round_ids:
            lines.append(f"- {rid}")
        lines.append("")
        lines.append('JSON schema: {"selected_round_ids": ["ROUND_ID"], "notes": {"ROUND_ID": "short reason"}}')
        return "\n".join(lines)
    def _fallback_selected_rounds(self, memory_sets: List[Any], limit: int) -> List[str]:
        scored: Dict[str, float] = {}
        order: Dict[str, int] = {}
        cursor = 0
        for memory_set in memory_sets:
            for rid in memory_set.round_ids:
                order.setdefault(rid, cursor)
                cursor += 1
                anchor_scores = [anchor.score or 0.0 for anchor in memory_set.round_anchors.get(rid, [])]
                score = max(anchor_scores, default=memory_set.score or 0.0)
                scored[rid] = max(scored.get(rid, 0.0), score)
        ranked = sorted(scored, key=lambda rid: (scored[rid], -order.get(rid, 0)), reverse=True)
        return ranked[:limit]

    def _select_rounds_from_session_memory_sets(
        self,
        question_stem: str,
        memory_sets: List[Any],
        dataset: Any,
    ) -> List[str]:
        valid_rounds = {rid for memory_set in memory_sets for rid in memory_set.round_ids}
        if not valid_rounds:
            return []
        cache_key = self._round_selection_cache_key(question_stem, memory_sets)
        cached = self._read_round_selection_cache(cache_key)
        if cached is not None:
            selected: List[str] = []
            seen: set[str] = set()
            for item in cached.get("selected_round_ids", []):
                rid = str(item).strip()
                if rid in valid_rounds and rid not in seen:
                    selected.append(rid)
                    seen.add(rid)
                if len(selected) >= self._max_selected_rounds:
                    break
            if selected:
                trace_json(log, "selected_round_ids", {
                    "selected_round_ids": selected,
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "raw_response": cached.get("raw_response", ""),
                })
                log.info("QDMO-EVI selected round ids=%s cache_hit=True", selected)
                return selected
            log.warning("  [ROUND SELECT CACHE] invalid cached selection key=%s", cache_key)

        prompt = self._selection_prompt(question_stem, memory_sets)
        prompt_preview = prompt[: self._debug_prompt_chars]
        if len(prompt) > self._debug_prompt_chars:
            prompt_preview += f"\n... [truncated with {len(prompt) - self._debug_prompt_chars} more chars]"
        log.info("QDMO-EVI round selector prompt preview:\n%s", prompt_preview)
        trace_json(log, "round_selector_prompt", {"prompt_chars": len(prompt), "prompt_preview": prompt_preview})

        candidate_round_ids = [rid for memory_set in memory_sets for rid in memory_set.round_ids]
        selector_history = self._build_semantic_style_history(dataset, candidate_round_ids)
        selector_preview = [
            {
                "role": item.get("role"),
                "round_id": item.get("round_id"),
                "text": str(item.get("text", ""))[:500],
                "images": item.get("images", []),
            }
            for item in selector_history
        ]
        trace_json(log, "round_selector_history", {
            "candidate_round_ids": candidate_round_ids,
            "history_turns": len(selector_history),
            "history_preview": selector_preview,
        })
        log.info("QDMO-EVI round selector history turns=%d candidate_rounds=%s", len(selector_history), candidate_round_ids)
        for idx, item in enumerate(selector_preview, start=1):
            log.info(
                "  selector_history[%02d] role=%s round=%s images=%s text=%s",
                idx,
                item.get("role"),
                item.get("round_id"),
                item.get("images"),
                " ".join(str(item.get("text", "")).split())[:500],
            )

        selector_router = self._get_round_selector_router()
        raw = selector_router.answer(selector_history, prompt, question_images=[])
        log.info("QDMO-EVI round selector raw response: %s", str(raw).replace("\n", " ")[:20000])
        parsed = extract_json(raw or "") or {}
        selected_raw = parsed.get("selected_round_ids", []) if isinstance(parsed, dict) else []
        selected: List[str] = []
        seen: set[str] = set()
        if isinstance(selected_raw, list):
            for item in selected_raw:
                rid = str(item).strip()
                if rid in valid_rounds and rid not in seen:
                    selected.append(rid)
                    seen.add(rid)
                if len(selected) >= self._max_selected_rounds:
                    break
        if len(selected) < max(1, self._min_selected_rounds):
            fallback = self._fallback_selected_rounds(memory_sets, self._max_selected_rounds)
            for rid in fallback:
                if rid not in seen:
                    selected.append(rid)
                    seen.add(rid)
                if len(selected) >= self._max_selected_rounds:
                    break
            log.warning("QDMO-EVI round selector used fallback/top-up: selected=%s", selected)
        cache_payload = {
            "selected_round_ids": selected,
            "raw_response": raw,
            "question_stem": question_stem,
            "cache_key": cache_key,
        }
        self._write_round_selection_cache(cache_key, cache_payload)
        trace_json(log, "selected_round_ids", {"selected_round_ids": selected, "raw_response": raw, "cache_key": cache_key, "cache_hit": False})
        log.info("QDMO-EVI selected round ids=%s cache_hit=False", selected)
        return selected

    def _log_selected_round_clue_coverage(self, qa: Optional[Dict[str, Any]], selected_round_ids: List[str], label: str) -> None:
        clue_rounds = (qa or {}).get("clue", [])
        if not clue_rounds:
            return
        selected_set = set(selected_round_ids)
        hits = [rid for rid in clue_rounds if rid in selected_set]
        misses = [rid for rid in clue_rounds if rid not in selected_set]
        log.info(
            "QDMO-EVI %s clue coverage: %d/%d hits=%s misses=%s selected=%s",
            label,
            len(hits),
            len(clue_rounds),
            hits,
            misses,
            selected_round_ids,
        )
        trace_json(log, f"{label}_clue_coverage", {
            "num_hits": len(hits),
            "num_clues": len(clue_rounds),
            "hits": hits,
            "misses": misses,
            "selected_round_ids": selected_round_ids,
        })

    def _set_last_context_round_ids(self, round_ids: List[str]) -> None:
        out: List[str] = []
        seen: set[str] = set()
        for rid in round_ids:
            value = str(rid or "").strip()
            if not value or value in seen:
                continue
            out.append(value)
            seen.add(value)
        self._last_context_round_ids = out

    @property
    def last_context_round_ids(self) -> List[str]:
        return list(self._last_context_round_ids)
    def _build_semantic_style_history(self, dataset: Any, selected_round_ids: List[str]) -> List[Dict[str, Any]]:
        allowed = set(selected_round_ids)
        history: List[Dict[str, Any]] = []
        for session_id in dataset.session_order():
            history.extend(
                history_from_round_ids(
                    dataset.get_session(session_id),
                    dataset.rounds,
                    allowed,
                    modality="multimodal",
                )
            )
        return history

    def _question_with_image_caption(self, qa: Optional[Dict[str, Any]], question: str) -> str:
        query = str(question or "").strip()
        if not qa:
            return query
        image_caption = qa.get("image_caption")
        if not image_caption:
            return query
        if isinstance(image_caption, list):
            caption_text = " ".join(str(item).strip() for item in image_caption if str(item).strip())
        else:
            caption_text = str(image_caption).strip()
        return f"{query}\nquestion image caption: {caption_text}" if caption_text else query

    def _facet_cache_dir(self) -> Path:
        raw = self._cfg.get("facet_cache_dir") or os.environ.get("EVI_FACET_CACHE_DIR")
        path = Path(raw).expanduser() if raw else Path.home() / ".cache" / "evi_facets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _facet_cache_key(self, question_stem: str) -> str:
        payload = {
            "version": FACET_PROMPT_VERSION,
            "cache_namespace": self._vlm_result_namespace,
            "question_stem": question_stem,
            "max_facets": self._facet_max_facets,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    @staticmethod
    def _normalize_facet_text(value: str) -> str:
        chars = []
        for ch in str(value or "").lower():
            chars.append(ch if ch.isalnum() else " ")
        return " ".join("".join(chars).split())

    @classmethod
    def _facet_is_contained(cls, shorter: str, longer: str) -> bool:
        short_norm = cls._normalize_facet_text(shorter)
        long_norm = cls._normalize_facet_text(longer)
        if not short_norm or not long_norm:
            return False
        if short_norm == long_norm:
            return True
        if short_norm in long_norm:
            return True
        short_tokens = [tok for tok in short_norm.split() if len(tok) > 2]
        long_tokens = set(tok for tok in long_norm.split() if len(tok) > 2)
        if len(short_tokens) < 2:
            return False
        overlap = sum(1 for tok in short_tokens if tok in long_tokens)
        return overlap / max(1, len(short_tokens)) >= 0.85 and len(short_norm) + 8 <= len(long_norm)

    def _dedupe_facets(self, facets: List[str]) -> List[str]:
        deduped: List[str] = []
        for facet in facets:
            replaced = False
            drop = False
            for idx, existing in enumerate(list(deduped)):
                if self._facet_is_contained(facet, existing):
                    log.info("QDMO-EVI dropped redundant retrieval facet=%s covered_by=%s", facet, existing)
                    drop = True
                    break
                if self._facet_is_contained(existing, facet):
                    log.info("QDMO-EVI replaced redundant retrieval facet=%s with=%s", existing, facet)
                    deduped[idx] = facet
                    replaced = True
                    break
            if not drop and not replaced:
                deduped.append(facet)
        return deduped

    def _clean_facets(self, question_stem: str, values: Any) -> List[str]:
        facets: List[str] = []
        seen: set[str] = set()
        if isinstance(values, list):
            raw_values = values
        else:
            raw_values = []
        for item in raw_values:
            facet = " ".join(str(item or "").strip().split())
            if not facet:
                continue
            key = self._normalize_facet_text(facet)
            if key in seen:
                continue
            seen.add(key)
            facets.append(facet)
        facets = self._dedupe_facets(facets)
        facets = facets[: self._facet_max_facets]
        if not facets:
            facets = [question_stem]
        return facets
    def _extract_retrieval_facets(self, question_stem: str) -> List[str]:
        cache_key = self._facet_cache_key(question_stem)
        cache_file = self._facet_cache_dir() / f"{cache_key}.json"
        if self._use_facet_cache and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                facets = self._clean_facets(question_stem, data.get("facets", []))
                log.info("QDMO-EVI retrieval facets cache_hit=True key=%s version=%s facets=%s", cache_key, FACET_PROMPT_VERSION, facets)
                trace_json(log, "retrieval_facets", {"cache_hit": True, "cache_key": cache_key, "prompt_version": FACET_PROMPT_VERSION, "facets": facets})
                return facets
            except Exception as exc:
                log.warning("QDMO-EVI retrieval facet cache read failed key=%s error=%s", cache_key, exc)

        user_text = "Question stem:\n" + str(question_stem or "")
        raw = self._vlm(FACET_EXTRACTION_SYSTEM_PROMPT, user_text, []) if self._vlm is not None else ""
        parsed = extract_json(raw or "") or {}
        facets = self._clean_facets(question_stem, parsed.get("facets", []))
        log.info("QDMO-EVI retrieval facets cache_hit=False key=%s version=%s facets=%s", cache_key, FACET_PROMPT_VERSION, facets)
        log.debug("[FACET RAW] key=%s raw=%s", cache_key, str(raw).replace("\n", " ")[:4000])
        trace_json(log, "retrieval_facets", {
            "cache_hit": False,
            "cache_key": cache_key,
            "facets": facets,
            "raw_response": raw,
        })
        if self._use_facet_cache:
            try:
                cache_file.write_text(
                    json.dumps(
                        {
                            "version": FACET_PROMPT_VERSION,
                            "cache_namespace": self._vlm_result_namespace,
                            "question_stem": question_stem,
                            "facets": facets,
                            "raw_response": raw,
                            "cache_key": cache_key,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("QDMO-EVI retrieval facet cache write failed key=%s error=%s", cache_key, exc)
        return facets

    def _scope_cache_dir(self) -> Path:
        raw = self._cfg.get("scope_cache_dir") or os.environ.get("EVI_SCOPE_CACHE_DIR")
        path = Path(raw).expanduser() if raw else Path.home() / ".cache" / "evi_scope_interpretation"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scope_cache_key(self, question_stem: str, question_images: List[str]) -> str:
        payload = {
            "version": SCOPE_INTERPRETATION_PROMPT_VERSION,
            "cache_namespace": self._vlm_result_namespace,
            "question_stem": question_stem,
            "question_images": question_images,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

    @staticmethod
    def _clean_scope_brief(value: Any) -> str:
        brief = " ".join(str(value or "").split())
        if brief.lower().startswith("scope brief:"):
            brief = brief[len("scope brief:"):].strip()
        return brief[:1600]

    def _interpret_question_scope(
        self,
        question_stem: str,
        question_images: Optional[List[str]],
    ) -> str:
        if not self._use_scope_interpretation:
            log.info("QDMO-EVI scope interpretation disabled")
            return ""

        images = self._as_image_list(question_images)[: self._max_answer_images]
        cache_key = self._scope_cache_key(question_stem, images)
        cache_file = self._scope_cache_dir() / f"{cache_key}.json"
        if self._use_scope_cache and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                brief = self._clean_scope_brief(data.get("scope_brief", ""))
                if brief:
                    log.info(
                        "QDMO-EVI scope interpretation cache_hit=True key=%s version=%s brief=%s",
                        cache_key,
                        SCOPE_INTERPRETATION_PROMPT_VERSION,
                        brief,
                    )
                    trace_json(log, "scope_interpretation", {
                        "cache_hit": True,
                        "cache_key": cache_key,
                        "prompt_version": SCOPE_INTERPRETATION_PROMPT_VERSION,
                        "question_stem": question_stem,
                        "question_images": images,
                        "scope_brief": brief,
                    })
                    return brief
            except Exception as exc:
                log.warning("QDMO-EVI scope cache read failed key=%s error=%s", cache_key, exc)

        user_text = "Question:\n" + str(question_stem or "")
        raw = self._vlm(SCOPE_INTERPRETATION_SYSTEM_PROMPT, user_text, images) if self._vlm is not None else ""
        brief = self._clean_scope_brief(raw)
        log.info(
            "QDMO-EVI scope interpretation cache_hit=False key=%s version=%s brief=%s",
            cache_key,
            SCOPE_INTERPRETATION_PROMPT_VERSION,
            brief,
        )
        trace_json(log, "scope_interpretation", {
            "cache_hit": False,
            "cache_key": cache_key,
            "prompt_version": SCOPE_INTERPRETATION_PROMPT_VERSION,
            "question_stem": question_stem,
            "question_images": images,
            "scope_brief": brief,
            "raw_response": raw,
        })
        if self._use_scope_cache:
            try:
                cache_file.write_text(
                    json.dumps(
                        {
                            "version": SCOPE_INTERPRETATION_PROMPT_VERSION,
                            "cache_namespace": self._vlm_result_namespace,
                            "question_stem": question_stem,
                            "question_images": images,
                            "scope_brief": brief,
                            "raw_response": raw,
                            "cache_key": cache_key,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("QDMO-EVI scope cache write failed key=%s error=%s", cache_key, exc)
        return brief

    def _search_rounds_for_text(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        query_vec = self._embed(query_text)
        if not query_vec:
            return []

        out: List[Dict[str, Any]] = []
        source_aware = self._facet_round_fusion in {
            "max_similarity_times_best_source_rank_consensus",
            "source_facet_reciprocal_rank_consensus",
        }
        for hit in self._index.search_rounds(
            query_vec,
            top_k=top_k,
            source_aware=source_aware,
        ):
            best_anchor = hit.get("best_anchor")
            if not isinstance(best_anchor, EvidenceAnchor):
                continue
            score = float(hit.get("score", 0.0))
            source_scores = {
                str(source): float(value)
                for source, value in dict(hit.get("source_scores", {})).items()
            }
            source_anchors = {
                str(source): replace(anchor, score=float(source_scores.get(source, 0.0)))
                for source, anchor in dict(hit.get("source_anchors", {})).items()
                if isinstance(anchor, EvidenceAnchor)
            }
            source_ranks = {
                str(source): int(rank)
                for source, rank in dict(hit.get("source_ranks", {})).items()
            }
            out.append(
                {
                    "round_id": str(hit.get("round_id", best_anchor.round_id)),
                    "session_id": str(hit.get("session_id", best_anchor.session_id)),
                    "date": str(hit.get("date", best_anchor.date)),
                    "score": score,
                    "best_anchor": replace(best_anchor, score=score),
                    "source_scores": source_scores,
                    "source_ranks": source_ranks,
                    "source_anchors": source_anchors,
                }
            )
        return out

    def _merge_facet_rounds(
        self,
        facet_results: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        round_data: Dict[str, Dict[str, Any]] = {}
        for facet_idx, (facet, round_hits) in enumerate(facet_results):
            facet_key = f"f{facet_idx}"
            for rank, hit in enumerate(round_hits, start=1):
                anchor = hit.get("best_anchor")
                if not isinstance(anchor, EvidenceAnchor):
                    continue
                rid = str(hit["round_id"])
                score = float(hit.get("score", 0.0))
                item = round_data.setdefault(
                    rid,
                    {
                        "round_id": rid,
                        "session_id": str(hit["session_id"]),
                        "date": str(hit["date"]),
                        "max_score": 0.0,
                        "score_sum": 0.0,
                        "matched_facets": set(),
                        "facet_scores": {},
                        "facet_ranks": {},
                        "source_scores": {"dialogue": 0.0, "visual": 0.0},
                        "source_facet_ranks": {},
                        "source_facet_scores": {},
                        "best_source_facet_ranks": {},
                        "best_source_by_facet": {},
                        "top_anchors": [],
                    },
                )
                item["max_score"] = max(float(item["max_score"]), score)
                item["score_sum"] = float(item["score_sum"]) + score
                item["matched_facets"].add(facet_key)
                current = item["facet_scores"].get(facet, 0.0)
                if score > current:
                    item["facet_scores"][facet] = score
                    item["facet_ranks"][facet] = rank
                for source, source_score in dict(hit.get("source_scores", {})).items():
                    item["source_scores"][source] = max(
                        float(item["source_scores"].get(source, 0.0)),
                        float(source_score),
                    )
                for source, source_rank in dict(hit.get("source_ranks", {})).items():
                    source_facet_key = f"{facet_key}:{source}"
                    item["source_facet_ranks"][source_facet_key] = int(source_rank)
                    item["source_facet_scores"][source_facet_key] = float(
                        dict(hit.get("source_scores", {})).get(source, 0.0)
                    )
                source_ranks = dict(hit.get("source_ranks", {}))
                source_scores = dict(hit.get("source_scores", {}))
                if source_ranks:
                    best_source, best_source_rank = min(
                        source_ranks.items(),
                        key=lambda pair: (
                            int(pair[1]),
                            -float(source_scores.get(pair[0], 0.0)),
                            str(pair[0]),
                        ),
                    )
                    item["best_source_facet_ranks"][facet_key] = int(best_source_rank)
                    item["best_source_by_facet"][facet_key] = str(best_source)
                if len(item["top_anchors"]) < self._max_candidate_anchors:
                    item["top_anchors"].append(
                        {
                            "facet": facet,
                            "rank": rank,
                            "score": score,
                            "type": anchor.evidence_type,
                            "text": anchor.text,
                        }
                    )

        merged: List[Dict[str, Any]] = []
        for item in round_data.values():
            matched_count = len(item["matched_facets"])
            if self._facet_round_fusion == "source_facet_reciprocal_rank_consensus":
                consensus_score = sum(
                    1.0 / max(1, int(rank))
                    for rank in item["source_facet_ranks"].values()
                )
                final_score = consensus_score
            elif self._facet_round_fusion == "max_similarity_times_best_source_rank_consensus":
                consensus_score = sum(
                    1.0 / max(1, int(rank))
                    for rank in item["best_source_facet_ranks"].values()
                )
                final_score = float(item["max_score"]) * consensus_score
            else:
                consensus_score = sum(
                    1.0 / max(1, int(rank))
                    for rank in item["facet_ranks"].values()
                )
                final_score = float(item["max_score"]) * consensus_score
            item["matched_facets"] = matched_count
            item["consensus_score"] = consensus_score
            item["score"] = final_score
            merged.append(item)
        if self._facet_round_fusion == "source_facet_reciprocal_rank_consensus":
            merged.sort(
                key=lambda item: (
                    float(item["score"]),
                    float(item["max_score"]),
                    int(item["matched_facets"]),
                ),
                reverse=True,
            )
        else:
            merged.sort(
                key=lambda item: (
                    float(item["score"]),
                    float(item["consensus_score"]),
                    int(item["matched_facets"]),
                    float(item["max_score"]),
                ),
                reverse=True,
            )
        return merged

    def _group_round_ids_by_session(self, round_ids: List[str]) -> List[Tuple[str, List[str]]]:
        grouped: Dict[str, List[str]] = {}
        seen: set[str] = set()
        for rid in round_ids:
            value = str(rid or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            sid = self._round_session.get(value, "unknown")
            grouped.setdefault(sid, []).append(value)
        ordered: List[Tuple[str, List[str]]] = []
        session_order = list(self._session_rounds.keys())
        for sid in session_order:
            if sid in grouped:
                ordered.append((sid, grouped[sid]))
        for sid, ids in grouped.items():
            if sid not in session_order:
                ordered.append((sid, ids))
        return ordered

    def _build_organizer_user_text(
        self,
        *,
        session_id: str,
        round_ids: List[str],
        question_stem: str,
        scope_brief: str,
        dataset: Any,
    ) -> Tuple[str, List[str]]:
        lines: List[str] = []
        images: List[str] = []
        lines.append(f"Question stem, without answer options:\n{question_stem}")
        if scope_brief:
            lines.append(f"\nScope interpretation:\n{scope_brief}")
        lines.append(f"\nSession: {session_id}")
        lines.append("Candidate rounds from this session:")
        for rid in round_ids:
            rp = dataset.rounds.get(rid, {}) if dataset is not None else {}
            user_text = " ".join(str(rp.get("user", "")).split())
            assistant_text = " ".join(str(rp.get("assistant", "")).split())
            date = self._round_date.get(rid, "")
            round_images = [path for path in self._round_images.get(rid, []) if path]
            lines.append(f"\nRound {rid} date={date} images={len(round_images)}")
            if user_text:
                lines.append(f"User: {user_text}")
            if assistant_text:
                lines.append(f"Assistant: {assistant_text}")
            raw = rp.get("raw", {}) or {}
            captions = raw.get("image_caption", []) or []
            if captions:
                caption_text = "; ".join(" ".join(str(c).split()) for c in captions if str(c).strip())
                if caption_text:
                    lines.append(f"Dataset captions, if any: {caption_text}")
            if round_images:
                for img_idx, path in enumerate(round_images, start=1):
                    lines.append(f"Attached image order marker: {rid} image {img_idx}")
                    images.append(path)
        lines.append("\nOutput JSON only. Return exactly one round_notes entry for every candidate round listed above. Never omit a candidate round. Use drop only for rounds outside the scope; use weak_keep for in-scope negative or contextual evidence.")
        return "\n".join(lines), images

    def _clean_organized_evidence(
        self,
        *,
        session_id: str,
        candidate_round_ids: List[str],
        parsed: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        allowed = {str(rid) for rid in candidate_round_ids}
        order = {str(rid): idx for idx, rid in enumerate(candidate_round_ids)}
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        items = parsed.get("round_notes", []) if isinstance(parsed, dict) else []
        if not isinstance(items, list):
            return out
        for item in items:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("round_id", "")).strip()
            if rid not in allowed or rid in seen:
                continue
            decision = str(item.get("decision", "weak_keep") or "weak_keep").strip().lower()
            if decision not in {"keep", "weak_keep", "drop"}:
                decision = "weak_keep"
            note = " ".join(str(item.get("note", "") or "").split())
            out.append({"session_id": session_id, "round_id": rid, "decision": decision, "note": note})
            seen.add(rid)
        out.sort(key=lambda item: order.get(str(item.get("round_id")), 10**9))
        return out

    def _organize_session_evidence(
        self,
        *,
        question_stem: str,
        scope_brief: str,
        dataset: Any,
        candidate_round_ids: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not self._use_evidence_organizer:
            selected = candidate_round_ids[: self._final_max_rounds]
            return [], selected
        all_evidence: List[Dict[str, Any]] = []
        for session_id, session_round_ids in self._group_round_ids_by_session(candidate_round_ids):
            round_ids = list(session_round_ids)
            if self._organizer_max_rounds_per_session > 0:
                round_ids = round_ids[: self._organizer_max_rounds_per_session]
            user_text, images = self._build_organizer_user_text(
                session_id=session_id,
                round_ids=round_ids,
                question_stem=question_stem,
                scope_brief=scope_brief,
                dataset=dataset,
            )
            log.info(
                "QDMO-EVI evidence organizer input session=%s rounds=%s images=%d prompt_chars=%d",
                session_id,
                round_ids,
                len(images),
                len(user_text),
            )
            log.info("QDMO-EVI evidence organizer prompt session=%s:\n%s", session_id, user_text[: self._debug_prompt_chars])
            trace_json(log, "evidence_organizer_input", {
                "prompt_version": EVIDENCE_ORGANIZER_PROMPT_VERSION,
                "session_id": session_id,
                "round_ids": round_ids,
                "images": images,
                "scope_brief": scope_brief,
                "prompt_preview": user_text[: self._debug_prompt_chars],
                "with_options": False,
            })
            raw = self._vlm(EVIDENCE_ORGANIZER_SYSTEM_PROMPT, user_text, images) if self._vlm is not None else ""
            parsed = extract_json(raw or "") or {}
            session_evidence = self._clean_organized_evidence(
                session_id=session_id,
                candidate_round_ids=round_ids,
                parsed=parsed,
            )
            returned_round_ids = {str(item.get("round_id", "")) for item in session_evidence}
            missing_round_ids = [rid for rid in round_ids if rid not in returned_round_ids]
            if missing_round_ids:
                log.warning(
                    "QDMO-EVI evidence organizer omitted required round decisions session=%s missing=%s",
                    session_id,
                    missing_round_ids,
                )
            log.info(
                "QDMO-EVI evidence organizer output session=%s notes=%s raw=%s",
                session_id,
                [{"round_id": item["round_id"], "decision": item["decision"], "note": item.get("note", "")[:200]} for item in session_evidence],
                str(raw).replace("\n", " ")[:4000],
            )
            trace_json(log, "evidence_organizer_output", {
                "prompt_version": EVIDENCE_ORGANIZER_PROMPT_VERSION,
                "session_id": session_id,
                "round_ids": round_ids,
                "missing_round_ids": missing_round_ids,
                "raw_response": raw,
                "parsed_evidence": session_evidence,
            })
            all_evidence.extend(session_evidence)

        notes_by_round = {str(item.get("round_id")): item for item in all_evidence}
        selected_round_ids: List[str] = []
        dropped_round_ids: List[str] = []
        for rid in candidate_round_ids:
            item = notes_by_round.get(rid)
            if item is not None and item.get("decision") == "drop":
                dropped_round_ids.append(rid)
                continue
            selected_round_ids.append(rid)
            if len(selected_round_ids) >= self._final_max_rounds:
                break
        if len(selected_round_ids) < max(1, self._min_selected_rounds):
            selected_round_ids = candidate_round_ids[: self._final_max_rounds]
            log.warning("QDMO-EVI weak evidence filtering left too few rounds; fallback selected_rounds=%s", selected_round_ids)
        log.info(
            "QDMO-EVI weak evidence filtering selected=%s dropped=%s final_max_rounds=%d",
            selected_round_ids,
            dropped_round_ids,
            self._final_max_rounds,
        )
        trace_json(log, "weak_evidence_filtering", {
            "candidate_round_ids": candidate_round_ids,
            "selected_round_ids": selected_round_ids,
            "dropped_round_ids": dropped_round_ids,
            "notes": all_evidence,
            "final_max_rounds": self._final_max_rounds,
        })
        return all_evidence, selected_round_ids

    def _build_augmented_semantic_history(
        self,
        dataset: Any,
        evidence_items: List[Dict[str, Any]],
        selected_round_ids: List[str],
    ) -> List[Dict[str, Any]]:
        notes_by_round: Dict[str, Dict[str, Any]] = {}
        for item in evidence_items:
            rid = str(item.get("round_id", "")).strip()
            if rid and item.get("decision") != "drop":
                notes_by_round[rid] = item
        allowed = set(selected_round_ids)
        history: List[Dict[str, Any]] = []
        for session_id in dataset.session_order():
            session_history = history_from_round_ids(
                dataset.get_session(session_id),
                dataset.rounds,
                allowed,
                modality="multimodal",
            )
            for msg in session_history:
                rid = str(msg.get("round_id", "")).strip()
                if msg.get("role") == "user" and rid in notes_by_round:
                    note = " ".join(str(notes_by_round[rid].get("note", "") or "").split())
                    if note:
                        new_msg = dict(msg)
                        base_text = str(new_msg.get("text", "")).strip()
                        new_msg["text"] = (base_text + "\n\nMemory evidence note for the current question:\n" + note).strip()
                        msg = new_msg
                history.append(msg)
        return history

    def _retrieve_faceted_round_candidates(
        self,
        question_stem: str,
        qa: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Retrieve and merge anchor hits into a ranked round list without QA-side filtering."""
        extracted_facets = self._extract_retrieval_facets(question_stem) if self._use_retrieval_facets else []
        if not self._use_retrieval_facets:
            log.info("QDMO-EVI retrieval facets disabled; using the full question as the only query")
        facets: List[Tuple[str, float]] = [(question_stem, self._facet_full_question_weight)]
        seen = {question_stem.lower()}
        for facet in extracted_facets:
            key = facet.lower()
            if key in seen:
                continue
            facets.append((facet, 1.0))
            seen.add(key)

        facet_results: List[Tuple[str, List[Dict[str, Any]]]] = []
        all_retrieved: List[EvidenceAnchor] = []
        for idx, (facet, weight) in enumerate(facets, start=1):
            round_hits = self._search_rounds_for_text(facet, self._facet_search_k)
            if weight != 1.0:
                for hit in round_hits:
                    hit["score"] = float(hit["score"]) * weight
                    hit["best_anchor"].score = float(hit["score"])
                    hit["source_scores"] = {
                        source: float(value) * weight
                        for source, value in hit["source_scores"].items()
                    }
            facet_results.append((facet, round_hits))
            all_retrieved.extend(hit["best_anchor"] for hit in round_hits)
            source_candidate_counts = {
                source: sum(
                    1 for hit in round_hits
                    if source in dict(hit.get("source_ranks", {}))
                )
                for source in ("dialogue", "visual")
            }
            log.info(
                "QDMO-EVI facet[%02d] query=%s unique_rounds=%d source_candidates=%s",
                idx,
                facet,
                len(round_hits),
                source_candidate_counts,
            )
            for rank, hit in enumerate(round_hits[: self._debug_top_k], start=1):
                anchor = hit["best_anchor"]
                log.info(
                    "  facet[%02d] round[%02d] round=%s type=%s score=%.4f dialogue=%.4f visual=%.4f source_ranks=%s text=%s",
                    idx,
                    rank,
                    anchor.round_id,
                    anchor.evidence_type,
                    float(hit["score"]),
                    float(hit["source_scores"].get("dialogue", 0.0)),
                    float(hit["source_scores"].get("visual", 0.0)),
                    hit.get("source_ranks", {}),
                    anchor.text.replace("\n", " ")[:180],
                )
            trace_json(
                log,
                "facet_retrieval",
                {
                    "facet_index": idx,
                    "facet": facet,
                    "weight": weight,
                    "num_unique_rounds": len(round_hits),
                    "source_candidate_counts": source_candidate_counts,
                    "top_rounds": [
                        {
                            "round_id": hit["round_id"],
                            "score": round(float(hit["score"]), 6),
                            "source_scores": {
                                source: round(float(value), 6)
                                for source, value in hit["source_scores"].items()
                            },
                            "source_ranks": {
                                source: int(rank)
                                for source, rank in hit.get("source_ranks", {}).items()
                            },
                            "best_anchor": anchors_summary([hit["best_anchor"]], max_items=1),
                        }
                        for hit in round_hits[: self._debug_top_k]
                    ],
                },
            )
        self._log_raw_clue_coverage(qa, all_retrieved)
        merged = self._merge_facet_rounds(facet_results)
        ranked_rounds = [str(item["round_id"]) for item in merged[: self._max_candidates]]
        if len(ranked_rounds) < max(1, self._min_selected_rounds):
            seen_rounds = set(ranked_rounds)
            for rid in self._round_order:
                if rid not in seen_rounds:
                    ranked_rounds.append(rid)
                    seen_rounds.add(rid)
                if len(ranked_rounds) >= self._max_candidates:
                    break
            log.warning("QDMO-EVI faceted retrieval used candidate fallback/top-up: candidates=%s", ranked_rounds)

        round_trace = [
            {
                "round_id": item["round_id"],
                "session_id": item["session_id"],
                "score": round(float(item["score"]), 6),
                "max_score": round(float(item["max_score"]), 6),
                "consensus_score": round(float(item["consensus_score"]), 6),
                "matched_facets": item["matched_facets"],
                "facet_scores": {key: round(float(value), 6) for key, value in item["facet_scores"].items()},
                "facet_ranks": {key: int(value) for key, value in item["facet_ranks"].items()},
                "source_scores": {key: round(float(value), 6) for key, value in item["source_scores"].items()},
                "source_facet_ranks": {
                    key: int(value) for key, value in item["source_facet_ranks"].items()
                },
                "source_facet_scores": {
                    key: round(float(value), 6) for key, value in item["source_facet_scores"].items()
                },
                "best_source_facet_ranks": {
                    key: int(value) for key, value in item["best_source_facet_ranks"].items()
                },
                "best_source_by_facet": dict(item["best_source_by_facet"]),
                "top_anchors": item["top_anchors"][: self._max_candidate_anchors],
            }
            for item in merged[: self._max_candidates]
        ]
        trace = {
            "question_stem": question_stem,
            "facet_round_fusion": self._facet_round_fusion,
            "facets": [{"text": facet, "weight": weight} for facet, weight in facets],
            "ranked_round_ids": ranked_rounds,
            "rounds": round_trace,
        }
        trace_json(log, "faceted_round_merge", {"num_rounds": len(merged), "rounds": round_trace})
        log.info("QDMO-EVI faceted round merge candidates=%d", len(merged))
        for rank, item in enumerate(round_trace, start=1):
            log.info(
                "  faceted_round[%02d] round=%s session=%s score=%.4f max=%.4f consensus=%.4f matched_facets=%s facet_ranks=%s source_facet_ranks=%s best_source_ranks=%s best_sources=%s source_scores=%s facet_scores=%s",
                rank,
                item["round_id"],
                item["session_id"],
                item["score"],
                item["max_score"],
                item["consensus_score"],
                item["matched_facets"],
                item["facet_ranks"],
                item["source_facet_ranks"],
                item["best_source_facet_ranks"],
                item["best_source_by_facet"],
                {key: round(float(value), 4) for key, value in item["source_scores"].items()},
                {key: round(float(value), 4) for key, value in item["facet_scores"].items()},
            )
        self._log_selected_round_clue_coverage(qa, ranked_rounds, "faceted_candidate_round")
        return ranked_rounds, trace

    def retrieve_faceted_rounds(
        self,
        qa: Dict[str, Any],
        dataset: Optional[Any] = None,
        limit: int = 0,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Return ranked rounds for the anchor-plus-facet retrieval ablation only.

        This method intentionally does not invoke scope interpretation, evidence
        organization, history construction, or final QA.
        """
        self._ensure()
        if dataset is not None:
            self._current_dataset = dataset
        if self._current_dataset is None:
            raise RuntimeError("EVI retrieval requires a dataset")
        if not self._index.anchors:
            self.process_all_sessions(self._current_dataset)

        question_stem = str(qa.get("question", "")).strip()
        if not question_stem:
            return [], {"question_stem": "", "facets": [], "ranked_round_ids": [], "rounds": []}
        ranked_round_ids, trace = self._retrieve_faceted_round_candidates(question_stem, qa)
        final_limit = int(limit) if limit > 0 else self._max_candidates
        selected_round_ids = ranked_round_ids[:final_limit]
        trace["selected_round_ids"] = selected_round_ids
        trace["retrieval_only"] = True
        self._set_last_context_round_ids(selected_round_ids)
        self._log_selected_round_clue_coverage(qa, selected_round_ids, "faceted_retrieval_only")
        log.info("QDMO-EVI faceted retrieval-only selected_rounds=%s", selected_round_ids)
        return selected_round_ids, trace
    def _answer_with_faceted_topk(
        self,
        question: str,
        question_stem: str,
        qa: Optional[Dict[str, Any]],
        question_images: Optional[List[str]],
    ) -> str:
        dataset = self._current_dataset
        if dataset is None:
            log.warning("QDMO-EVI faceted_topk requires dataset; falling back to question-only answer")
            images = self._as_image_list(question_images)[: self._max_answer_images]
            return self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, question, images) if self._vlm is not None else ""

        scope_brief = ""
        organizer_scope_brief = ""
        if self._retrieval_only:
            log.info("QDMO-EVI retrieval-only ablation: scope interpretation and evidence organizer are disabled")
        else:
            scope_brief = self._interpret_question_scope(question_stem, question_images)
            organizer_scope_brief = scope_brief if self._apply_scope_to_organizer else ""
            log.info(
                "QDMO-EVI scope organizer_apply=%s scope_brief=%s",
                self._apply_scope_to_organizer,
                organizer_scope_brief or "<not passed to organizer>",
            )
        candidate_round_ids, _retrieval_trace = self._retrieve_faceted_round_candidates(question_stem, qa)

        if self._retrieval_only:
            evidence_items = []
            selected_round_ids = candidate_round_ids[: self._final_max_rounds]
            self._log_selected_round_clue_coverage(qa, selected_round_ids, "faceted_top10_raw")
        else:
            evidence_items, selected_round_ids = self._organize_session_evidence(
                question_stem=question_stem,
                scope_brief=organizer_scope_brief,
                dataset=dataset,
                candidate_round_ids=candidate_round_ids,
            )
            self._log_selected_round_clue_coverage(qa, selected_round_ids, "evidence_organizer_output_round")
        self._set_last_context_round_ids(selected_round_ids)
        history = (
            self._build_semantic_style_history(dataset, selected_round_ids)
            if self._retrieval_only
            else self._build_augmented_semantic_history(dataset, evidence_items, selected_round_ids)
        )
        history_preview = [
            {
                "role": item.get("role"),
                "round_id": item.get("round_id"),
                "text": str(item.get("text", ""))[:1000],
                "images": item.get("images", []),
            }
            for item in history
        ]
        trace_json(log, "organized_evidence_history", {
            "candidate_round_ids": candidate_round_ids,
            "retrieval_only": self._retrieval_only,
            "scope_brief": scope_brief,
            "scope_applied_to_organizer": self._apply_scope_to_organizer and not self._retrieval_only,
            "selected_round_ids": selected_round_ids,
            "evidence_items": evidence_items,
            "history_turns": len(history),
            "history_preview": history_preview,
            "final_include_evidence_images": True,
        })
        log.info(
            "QDMO-EVI final history mode=%s turns=%d selected_rounds=%s include_images=%s",
            "faceted_top10_raw" if self._retrieval_only else "organized_evidence",
            len(history),
            selected_round_ids,
            True,
        )
        for idx, item in enumerate(history_preview, start=1):
            log.info(
                "  organized_evidence_history[%02d] role=%s round=%s images=%s text=%s",
                idx,
                item.get("role"),
                item.get("round_id"),
                item.get("images"),
                " ".join(str(item.get("text", "")).split())[:1000],
            )

        mode = "mcq" if isinstance((qa or {}).get("options"), (dict, list)) and bool((qa or {}).get("options")) else "open"
        router = self._get_answer_router(mode)
        query = self._question_with_image_caption(qa, question)
        qa_images = self._as_image_list(question_images)[: self._max_answer_images]
        final_prompt_preview = "Final question:\n" + query + "\n\nEvidence history:\n" + "\n".join(
            f"[{idx}] round={item.get('round_id')} images={len(item.get('images', []) or [])}\n{item.get('text', '')}"
            for idx, item in enumerate(history, start=1)
        )
        log.info("QDMO-EVI final QA prompt preview:\n%s", final_prompt_preview[: self._debug_prompt_chars])
        log.info("QDMO-EVI organized-evidence answer call mode=%s question_images=%s", mode, qa_images)
        trace_json(log, "final_qa_prompt", {
            "mode": mode,
            "question": query,
            "question_images": qa_images,
            "history_preview": history_preview,
            "prompt_preview": final_prompt_preview[: self._debug_prompt_chars],
        })
        answer = router.answer(history, query, question_images=qa_images)
        trace_json(log, "answer_done", {
            "pipeline": "faceted_topk",
            "retrieval_only": self._retrieval_only,
            "evidence_organizer_enabled": self._use_evidence_organizer,
            "evidence_organizer_effective": self._use_evidence_organizer and not self._retrieval_only,
            "answer": answer,
            "candidate_round_ids": candidate_round_ids,
            "selected_round_ids": selected_round_ids,
            "history_turns": len(history),
        })
        log.info("QDMO-EVI answer returned length=%d text=%s", len(str(answer)), str(answer).replace("\n", " ")[:1000])
        return answer
    def _answer_with_consolidated_topk(
        self,
        question: str,
        question_stem: str,
        qa: Optional[Dict[str, Any]],
        question_images: Optional[List[str]],
        retrieved: List[EvidenceAnchor],
    ) -> str:
        dataset = self._current_dataset
        if dataset is None:
            log.warning("QDMO-EVI consolidated_topk requires dataset; falling back to episodic_state pipeline")
            return self._answer_with_episodic_states(question, question_stem, qa, question_images, retrieved)

        candidates = consolidate_candidates(
            retrieved,
            round_text=self._round_text,
            max_candidates=self._max_candidates,
            max_candidate_anchors=self._max_candidate_anchors,
        )
        trace_json(log, "consolidated_topk_candidate_pool", {
            "num_candidates": len(candidates),
            "candidates": candidates_summary(candidates, max_items=self._max_candidates),
        })
        candidate_round_ids = [candidate.round_id for candidate in candidates]
        self._log_selected_round_clue_coverage(qa, candidate_round_ids, "consolidated_candidate_round")

        selected_round_ids: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            rid = candidate.round_id
            if rid in seen:
                continue
            selected_round_ids.append(rid)
            seen.add(rid)
            if len(selected_round_ids) >= self._max_selected_rounds:
                break
        if len(selected_round_ids) < max(1, self._min_selected_rounds):
            for rid in self._round_order:
                if rid not in seen:
                    selected_round_ids.append(rid)
                    seen.add(rid)
                if len(selected_round_ids) >= self._max_selected_rounds:
                    break
            log.warning("QDMO-EVI consolidated_topk used fallback/top-up: selected=%s", selected_round_ids)

        self._log_selected_round_clue_coverage(qa, selected_round_ids, "selected_round")
        self._set_last_context_round_ids(selected_round_ids)

        history = self._build_semantic_style_history(dataset, selected_round_ids)
        history_preview = [
            {
                "role": item.get("role"),
                "round_id": item.get("round_id"),
                "text": str(item.get("text", ""))[:500],
                "images": item.get("images", []),
            }
            for item in history
        ]
        trace_json(log, "consolidated_topk_history", {
            "selected_round_ids": selected_round_ids,
            "history_turns": len(history),
            "history_preview": history_preview,
        })
        log.info("QDMO-EVI consolidated-topk final history turns=%d selected_rounds=%s", len(history), selected_round_ids)
        for idx, item in enumerate(history_preview, start=1):
            log.info(
                "  consolidated_topk_history[%02d] role=%s round=%s images=%s text=%s",
                idx,
                item.get("role"),
                item.get("round_id"),
                item.get("images"),
                " ".join(str(item.get("text", "")).split())[:500],
            )

        mode = "mcq" if isinstance((qa or {}).get("options"), (dict, list)) and bool((qa or {}).get("options")) else "open"
        router = self._get_answer_router(mode)
        query = self._question_with_image_caption(qa, question)
        qa_images = self._as_image_list(question_images)[: self._max_answer_images]
        log.info("QDMO-EVI consolidated-topk answer call mode=%s question_images=%s", mode, qa_images)
        answer = router.answer(history, query, question_images=qa_images)
        trace_json(log, "answer_done", {
            "pipeline": "consolidated_topk",
            "answer": answer,
            "selected_round_ids": selected_round_ids,
            "history_turns": len(history),
        })
        log.info("QDMO-EVI answer returned length=%d text=%s", len(str(answer)), str(answer).replace("\n", " ")[:1000])
        return answer

    def _answer_with_session_round_selection(
        self,
        question: str,
        question_stem: str,
        qa: Optional[Dict[str, Any]],
        question_images: Optional[List[str]],
        retrieved: List[EvidenceAnchor],
    ) -> str:
        dataset = self._current_dataset
        if dataset is None:
            log.warning("QDMO-EVI session_round_selection requires dataset; falling back to episodic_state pipeline")
            return self._answer_with_episodic_states(question, question_stem, qa, question_images, retrieved)

        candidates = consolidate_candidates(
            retrieved,
            round_text=self._round_text,
            max_candidates=self._max_candidates,
            max_candidate_anchors=self._max_candidate_anchors,
        )
        trace_json(log, "session_round_candidate_pool", {
            "num_candidates": len(candidates),
            "candidates": candidates_summary(candidates, max_items=self._max_candidates),
        })
        candidate_round_ids = [candidate.round_id for candidate in candidates]
        self._log_selected_round_clue_coverage(qa, candidate_round_ids, "consolidated_candidate_round")

        memory_sets = build_session_memory_sets_from_candidates(
            candidates,
            session_rounds=self._session_rounds,
        )
        trace_json(log, "session_memory_sets", {
            "num_sets": len(memory_sets),
            "memory_sets": memory_sets_summary(memory_sets, max_items=self._max_memory_sets),
        })
        session_round_ids = [rid for memory_set in memory_sets for rid in memory_set.round_ids]
        self._log_selected_round_clue_coverage(qa, session_round_ids, "session_candidate_round")

        selected_round_ids = self._select_rounds_from_session_memory_sets(question_stem, memory_sets, dataset)
        self._log_selected_round_clue_coverage(qa, selected_round_ids, "llm_selected_round")
        self._set_last_context_round_ids(selected_round_ids)

        history = self._build_semantic_style_history(dataset, selected_round_ids)
        history_preview = [
            {
                "role": item.get("role"),
                "round_id": item.get("round_id"),
                "text": str(item.get("text", ""))[:500],
                "images": item.get("images", []),
            }
            for item in history
        ]
        trace_json(log, "semantic_style_history", {
            "selected_round_ids": selected_round_ids,
            "history_turns": len(history),
            "history_preview": history_preview,
        })
        log.info("QDMO-EVI semantic-style final history turns=%d selected_rounds=%s", len(history), selected_round_ids)

        for idx, item in enumerate(history_preview, start=1):
            log.info(
                "  semantic_history[%02d] role=%s round=%s images=%s text=%s",
                idx,
                item.get("role"),
                item.get("round_id"),
                item.get("images"),
                " ".join(str(item.get("text", "")).split())[:500],
            )
        mode = "mcq" if isinstance((qa or {}).get("options"), (dict, list)) and bool((qa or {}).get("options")) else "open"
        router = self._get_answer_router(mode)
        query = self._question_with_image_caption(qa, question)
        qa_images = self._as_image_list(question_images)[: self._max_answer_images]
        log.info("QDMO-EVI semantic-style answer call mode=%s question_images=%s", mode, qa_images)
        answer = router.answer(history, query, question_images=qa_images)
        trace_json(log, "answer_done", {
            "pipeline": "session_round_selection",
            "answer": answer,
            "selected_round_ids": selected_round_ids,
            "history_turns": len(history),
        })
        log.info("QDMO-EVI answer returned length=%d text=%s", len(str(answer)), str(answer).replace("\n", " ")[:1000])
        return answer

    def _answer_with_episodic_states(
        self,
        question: str,
        question_stem: str,
        qa: Optional[Dict[str, Any]],
        question_images: Optional[List[str]],
        retrieved: List[EvidenceAnchor],
    ) -> str:
        memory_sets = build_episodic_memory_sets(
            retrieved,
            session_rounds=self._session_rounds,
            round_text=self._round_text,
            round_images=self._round_images,
            round_anchors=self._round_anchors,
            max_sets=self._max_memory_sets,
            window_before=self._memory_set_window_before,
            window_after=self._memory_set_window_after,
            max_rounds_per_set=self._max_rounds_per_memory_set,
            max_anchors_per_round=self._max_state_anchors_per_round,
        )
        trace_json(log, "episodic_memory_sets", {
            "num_sets": len(memory_sets),
            "memory_sets": memory_sets_summary(memory_sets, max_items=self._max_memory_sets),
        })

        clue_rounds = (qa or {}).get("clue", [])
        if clue_rounds:
            set_rounds = {rid for memory_set in memory_sets for rid in memory_set.round_ids}
            hits = [rid for rid in clue_rounds if rid in set_rounds]
            misses = [rid for rid in clue_rounds if rid not in set(hits)]
            log.info(
                "QDMO-EVI episodic set clue coverage: %d/%d hits=%s misses=%s",
                len(hits),
                len(clue_rounds),
                hits,
                misses,
            )
            trace_json(log, "episodic_set_clue_coverage", {
                "num_hits": len(hits),
                "num_clues": len(clue_rounds),
                "hits": hits,
                "misses": misses,
            })

        states = read_episodic_states(
            question_stem,
            memory_sets,
            self._vlm,
            use_cache=self._use_state_cache,
            cache_namespace=self._vlm_result_namespace,
            use_images=self._use_state_images,
            max_images=self._max_state_images_per_set,
            max_prompt_chars=self._debug_prompt_chars,
        )
        trace_json(log, "episodic_states", {
            "num_states": len(states),
            "states": states_summary(states, max_items=self._max_memory_sets),
        })

        final_states = self._select_final_states(states)
        self._set_last_context_round_ids([rid for state in final_states for rid in state.round_ids])
        log.info("QDMO-EVI selected final episodic states=%d", len(final_states))
        for idx, state in enumerate(final_states, start=1):
            log.info(
                "  final_state[%02d] set=%s relevance=%s confidence=%.2f rounds=%s",
                idx,
                state.set_id,
                state.relevance,
                state.confidence,
                " -> ".join(state.round_ids),
            )
        self._log_final_state_clue_coverage(qa, final_states)
        trace_json(log, "selected_episodic_states", {
            "num_selected": len(final_states),
            "states": states_summary(final_states, max_items=self._max_final_states),
        })

        prompt = self._build_state_final_prompt(question, final_states)
        images = self._answer_images_for_states(final_states, question_images)
        return self._call_final_answer(prompt, images, "episodic_state")

    def _call_final_answer(self, prompt: str, images: List[str], pipeline: str) -> str:
        prompt_preview = prompt[:self._debug_prompt_chars]
        if len(prompt) > self._debug_prompt_chars:
            prompt_preview = f"{prompt_preview}\n... [truncated with {len(prompt) - self._debug_prompt_chars} more chars]"
        log.info("QDMO-EVI final prompt preview:\n%s", prompt_preview)
        log.info(
            "QDMO-EVI answer images=%s use_final_memory_images=%s pipeline=%s",
            images,
            self._use_final_memory_images,
            pipeline,
        )
        trace_json(log, "final_answer_call", {
            "pipeline": pipeline,
            "prompt_chars": len(prompt),
            "prompt_preview": prompt_preview,
            "images": images,
            "use_final_memory_images": self._use_final_memory_images,
        }, max_chars=self._debug_prompt_chars + 4000)
        log.info("QDMO-EVI final prompt=%d chars, images=%d", len(prompt), len(images))
        answer = self._vlm("", prompt, images)
        trace_json(log, "answer_done", {"pipeline": pipeline, "answer": answer})
        log.info("QDMO-EVI answer returned length=%d text=%s", len(str(answer)), str(answer).replace("\n", " ")[:1000])
        return answer

    @property
    def num_indexed(self) -> int:
        return len(self._index)
