from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .briefs import generate_memory_briefs
from .candidates import consolidate_candidates
from .extractor import extract_image_anchors
from .indexes import EvidenceIndex, embed_text, normalize_type
from .schemas import EvidenceAnchor, MemoryBrief
from .trace import anchors_summary, briefs_summary, candidates_summary, setup_evi_debug_logging, trace_json
from .vlm import VLMCallable, make_vlm_callable

log = logging.getLogger(__name__)

FINAL_ANSWER_SYSTEM_PROMPT = """You are answering a multimodal long-term memory question.
Use the evidence assertions as the primary evidence.
The assertions are selected factual observations, not reasoning traces.
Use attached images only to resolve uncertainty.
Be concise and grounded in the assertions.
If the question is multiple-choice, answer with ONLY the option letter.
"""


class EVISystem:
    """QDMO-EVI: typed evidence anchors with candidate-centric memory briefs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._cfg = cfg
        self._model_cfg = dict(cfg.get("_model_cfg", {}))
        self._debug_log_path = setup_evi_debug_logging(cfg)

        self._index = EvidenceIndex()
        self._vlm: Optional[VLMCallable] = None
        self._embedder: Optional[Any] = None

        self._round_order: List[str] = []
        self._round_session: Dict[str, str] = {}
        self._round_date: Dict[str, str] = {}
        self._round_text: Dict[str, str] = {}

        self._raw_search_k = int(cfg.get("raw_search_k", 120))
        self._max_candidates = int(cfg.get("max_candidates", 16))
        self._max_candidate_anchors = int(cfg.get("max_candidate_anchors", 8))
        self._max_final_briefs = int(cfg.get("max_final_briefs", 10))
        self._max_excluded_briefs = int(cfg.get("max_excluded_briefs", 3))
        self._max_answer_images = int(cfg.get("max_answer_images", 12))
        self._use_dataset_captions = self._as_bool(cfg.get("use_dataset_captions"), False)
        self._use_embedding_cache = self._as_bool(cfg.get("use_embedding_cache"), True)
        self._use_memory_brief_cache = self._as_bool(cfg.get("use_memory_brief_cache"), True)
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

    def process_all_sessions(self, dataset: Any) -> None:
        """Build task-agnostic typed evidence anchors for all sessions."""
        self._ensure()
        if self._vlm is None:
            raise RuntimeError("VLM not initialized")

        log.info("Processing sessions for QDMO-EVI evidence anchors ...")
        trace_json(log, "indexing_start", {
            "raw_search_k": self._raw_search_k,
            "max_candidates": self._max_candidates,
            "max_candidate_anchors": self._max_candidate_anchors,
            "max_final_briefs": self._max_final_briefs,
            "max_answer_images": self._max_answer_images,
            "use_dataset_captions": self._use_dataset_captions,
            "use_embedding_cache": self._use_embedding_cache,
            "use_memory_brief_cache": self._use_memory_brief_cache,
        })

        for sid in dataset.session_order():
            session = dataset.get_session(sid)
            date = str(session.get("date", "")).strip() or "unknown"
            prior_rounds_text: List[str] = []
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
                self._round_session[rid] = sid
                self._round_date[rid] = date
                self._round_text[rid] = round_text

                dialogue_anchor = self._make_anchor(
                    session_id=sid,
                    round_id=rid,
                    date=date,
                    evidence_type="temporal",
                    text=f"Round {rid} on {date}. {round_text}",
                )
                if dialogue_anchor is not None:
                    self._index.add(dialogue_anchor)

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
                            self._index.add(caption_anchor)

                images = list(rp.get("images", []) or [])
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
                            self._index.add(anchor)

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
        lines.append("Answer using only the evidence assertions and attached images. Do not use excluded or unselected memories.")
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

    def answer_question(
        self,
        question: str,
        qa: Optional[Dict[str, Any]] = None,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure()
        if self._vlm is None:
            raise RuntimeError("VLM not initialized")

        question_stem = str((qa or {}).get("question", "")).strip() or str(question or "")
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

        query_vec = self._embed(question_stem)
        if not query_vec:
            log.warning("QDMO-EVI query embedding is empty; answering with question images only")
            images = self._as_image_list(question_images)[: self._max_answer_images]
            return self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, question, images)

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
        prompt_preview = prompt[:self._debug_prompt_chars]
        if len(prompt) > self._debug_prompt_chars:
            prompt_preview = f"{prompt_preview}\n... [truncated with {len(prompt) - self._debug_prompt_chars} more chars]"
        log.info("QDMO-EVI final prompt preview:\n%s", prompt_preview)
        log.info("QDMO-EVI answer images=%s", images)
        trace_json(log, "final_answer_call", {
            "prompt_chars": len(prompt),
            "prompt_preview": prompt_preview,
            "images": images,
        }, max_chars=self._debug_prompt_chars + 4000)
        log.info("QDMO-EVI final prompt=%d chars, images=%d", len(prompt), len(images))

        answer = self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, prompt, images)
        trace_json(log, "answer_done", {"answer": answer})
        log.info("QDMO-EVI answer returned length=%d", len(str(answer)))
        return answer

    @property
    def num_indexed(self) -> int:
        return len(self._index)
