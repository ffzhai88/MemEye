from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .extractor import extract_image_anchors
from .indexes import EvidenceIndex, embed_text, normalize_type
from .organizer import organize_evidence
from .schemas import EvidenceAnchor, EvidenceGroup
from .trace import anchors_summary, groups_summary, setup_evi_debug_logging, trace_json
from .verifier import verify_group
from .vlm import VLMCallable, make_vlm_callable

log = logging.getLogger(__name__)

FINAL_ANSWER_SYSTEM_PROMPT = """You are answering a multimodal long-term memory benchmark question.
Use the organized evidence groups, verified visual evidence, provenance, chronology, and images.
For multiple-choice questions, answer with ONLY the option letter unless the question explicitly asks otherwise.
Be concise and grounded in the retrieved evidence.
"""


class EVISystem:
    """QDMO-EVI: typed evidence anchors with query-driven memory organization."""

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
        self._round_images: Dict[str, List[str]] = defaultdict(list)

        self._raw_search_k = int(cfg.get("raw_search_k", 120))
        self._max_groups = int(cfg.get("max_groups", 6))
        self._max_group_size = int(cfg.get("max_group_size", 12))
        self._max_group_images = int(cfg.get("max_group_images", 4))
        self._max_answer_images = int(cfg.get("max_answer_images", 20))
        self._use_group_verification = self._as_bool(cfg.get("use_group_verification"), True)
        self._use_dataset_captions = self._as_bool(cfg.get("use_dataset_captions"), False)
        self._debug_top_k = int(cfg.get("evi_debug_top_k", 20))
        self._debug_prompt_chars = int(cfg.get("evi_debug_prompt_chars", 12000))
        self._use_embedding_cache = self._as_bool(cfg.get("use_embedding_cache"), True)
        self._embed_cache_namespace = "uninitialized"

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
        self._embed_cache_namespace = json.dumps({"model": embed_model, "kwargs": embed_kwargs}, sort_keys=True, default=str)
        self._embedder = TextEmbedder(embed_model, **embed_kwargs)
        if self._embedder.is_available:
            log.info("Embedder loaded: %s", embed_model)
        else:
            log.warning("Embedder unavailable: %s", embed_model)
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
            "max_groups": self._max_groups,
            "max_group_size": self._max_group_size,
            "max_group_images": self._max_group_images,
            "max_answer_images": self._max_answer_images,
            "use_group_verification": self._use_group_verification,
            "use_dataset_captions": self._use_dataset_captions,
            "use_embedding_cache": self._use_embedding_cache,
        })

        for sid in dataset.session_order():
            session = dataset.get_session(sid)
            date = str(session.get("date", "")).strip() or "unknown"
            prior_rounds_text: List[str] = []
            log.debug(
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
                    self._round_images[rid].append(image_path)
                    raw_anchors = extract_image_anchors(
                        image_path=image_path,
                        round_text=round_text,
                        prior_rounds_text="\n---\n".join(prior_rounds_text),
                        vlm_callable=self._vlm,
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

        type_counts: Dict[str, int] = defaultdict(int)
        for anchor in self._index.anchors:
            type_counts[anchor.evidence_type] += 1
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

    def _verify_groups(self, question_stem: str, groups: List[EvidenceGroup]) -> List[EvidenceGroup]:
        if not self._use_group_verification or self._vlm is None:
            return groups
        verified: List[EvidenceGroup] = []
        for group in groups:
            verified.append(verify_group(question_stem, group, self._vlm))
        return verified

    def _build_final_prompt(
        self,
        question: str,
        groups: List[EvidenceGroup],
    ) -> str:
        lines: List[str] = []
        lines.append("Query-driven organized evidence groups:")
        for group in groups:
            lines.append(f"=== {group.id} score={group.score:.4f} confidence={group.confidence:.2f} ===")
            lines.append(f"Label: {group.group_label}")
            lines.append(f"Hypothesis: {group.group_hypothesis}")
            if group.needed_visual_checks:
                lines.append("Visual checks:")
                for check in group.needed_visual_checks:
                    lines.append(f"- {check}")
            lines.append("Anchors:")
            for anchor in group.anchors[:16]:
                loc = f" region={anchor.region}" if anchor.region else ""
                lines.append(
                    f"- id={anchor.id} session={anchor.session_id} round={anchor.round_id} "
                    f"date={anchor.date} type={anchor.evidence_type}{loc}: {anchor.text}"
                )
            if group.verified_evidence:
                lines.append("Verified visual evidence:")
                for item in group.verified_evidence:
                    lines.append(f"- {item}")
            if group.contradictions:
                lines.append("Contradictions:")
                for item in group.contradictions:
                    lines.append(f"- {item}")
            if group.missing_evidence:
                lines.append("Missing or uncertain evidence:")
                for item in group.missing_evidence:
                    lines.append(f"- {item}")
            lines.append("")

        lines.append("Question:")
        lines.append(question)
        return "\n".join(lines)

    def _answer_images(
        self,
        groups: List[EvidenceGroup],
        question_images: Optional[List[str]],
    ) -> List[str]:
        images: List[str] = []
        seen: set[str] = set()
        for path in self._as_image_list(question_images):
            if path not in seen:
                images.append(path)
                seen.add(path)
        for group in sorted(groups, key=lambda g: g.score, reverse=True):
            for path in group.image_paths:
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

        query_vec = self._embed(question_stem)
        if not query_vec:
            log.warning("QDMO-EVI query embedding is empty")
            return ""

        retrieved = self._index.search(query_vec, top_k=self._raw_search_k)
        trace_json(log, "retrieved_anchors", {
            "num_retrieved": len(retrieved),
            "top_anchors": anchors_summary(retrieved, max_items=self._debug_top_k),
        })

        groups = organize_evidence(
            question=question_stem,
            query_vec=query_vec,
            anchors=retrieved,
            round_order=self._round_order,
            max_groups=self._max_groups,
            max_group_size=self._max_group_size,
            max_group_images=self._max_group_images,
        )
        log.info(
            "QDMO-EVI answer: retrieved=%d anchors, groups=%d",
            len(retrieved),
            len(groups),
        )
        trace_json(log, "organized_groups", {
            "num_groups": len(groups),
            "groups": groups_summary(groups, max_groups=self._max_groups),
        })

        clue_rounds = (qa or {}).get("clue", [])
        if clue_rounds:
            group_rounds = {anchor.round_id for group in groups for anchor in group.anchors}
            hits = [rid for rid in clue_rounds if rid in group_rounds]
            log.info("QDMO-EVI clue coverage: %d/%d %s", len(hits), len(clue_rounds), hits)
            trace_json(log, "clue_coverage", {
                "num_hits": len(hits),
                "num_clues": len(clue_rounds),
                "hits": hits,
                "misses": [rid for rid in clue_rounds if rid not in set(hits)],
            })

        groups = self._verify_groups(question_stem, groups)
        trace_json(log, "verified_groups", {
            "num_groups": len(groups),
            "groups": groups_summary(groups, max_groups=self._max_groups),
        })

        prompt = self._build_final_prompt(question, groups)
        images = self._answer_images(groups, question_images)
        trace_json(log, "final_answer_call", {
            "prompt_chars": len(prompt),
            "prompt_preview": prompt[:self._debug_prompt_chars],
            "images": images,
        }, max_chars=self._debug_prompt_chars + 4000)
        log.info("QDMO-EVI final prompt=%d chars, images=%d", len(prompt), len(images))

        answer = self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, prompt, images)
        trace_json(log, "answer_done", {"answer": answer})
        return answer

    @property
    def num_indexed(self) -> int:
        return len(self._index)
