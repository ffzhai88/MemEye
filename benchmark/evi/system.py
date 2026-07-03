from __future__ import annotations

import json
import logging
import os
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
Be concise and grounded in the retrieved evidence.
If the question is multiple-choice, answer with ONLY the option letter.
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
            log.error("Embedder unavailable: %s — ALL evidence anchors will be EMPTY, "
                       "answer_question() will return '' with no evidence. "
                       "Check the embedding model name and dependencies.", embed_model)
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
            t = anchor.evidence_type
            type_counts[t] = type_counts.get(t, 0) + 1
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
        """
        回答单个问题的主流程。

        该函数负责：
          1. 提取问题干并构造 Trace 信息；
          2. 检查 index 是否可用，并在失败时回退到仅使用问题图像；
          3. 计算问题 embedding，用于检索相关 evidence anchors；
          4. 对检索结果进行 evidence 组织、校验和 prompt 构造；
          5. 调用 VLM 生成最终回答。
        """
        self._ensure()
        if self._vlm is None:
            raise RuntimeError("VLM not initialized")

        # 优先使用 QA 中的标准问题文本作为 query，如果没有则退回到原始 question。
        question_stem = str((qa or {}).get("question", "")).strip() or str(question or "")
        trace_json(log, "answer_start", {
            "question_stem": question_stem,
            "question_full": str(question or ""),
            "question_images": self._as_image_list(question_images),
            "qa_id": (qa or {}).get("id") or (qa or {}).get("question_id"),
            "clue_rounds": (qa or {}).get("clue", []),
        })

        # 如果 index 为空，说明 indexing 过程出现问题，此时直接回退到只用问题图像的回答。
        if not self._index.anchors:
            log.error("QDMO-EVI index is empty — the embedder may have failed during indexing. "
                       "Falling back to question-only answer with raw images.")
            images = self._as_image_list(question_images)
            return self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, question, images)

        # 生成问题向量，用于检索与问题最相关的证据 anchors。
        query_vec = self._embed(question_stem)
        if not query_vec:
            log.warning("QDMO-EVI query embedding is empty — cannot retrieve evidence, "
                        "answering with question images only")
            images = self._as_image_list(question_images)
            return self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, question, images)

        # 检索 top_k 个与问题最相关的 evidence anchors。
        retrieved = self._index.search(query_vec, top_k=self._raw_search_k)
        log.info("QDMO-EVI retrieved anchors=%d", len(retrieved))
        for idx, anchor in enumerate(retrieved[: self._debug_top_k]):
            log.info(
                "  [%02d] id=%s round=%s type=%s score=%.4f text=%s",
                idx + 1,
                anchor.id,
                anchor.round_id,
                anchor.evidence_type,
                anchor.score or 0.0,
                anchor.text.replace("\n", " ")[:120],
            )
        if len(retrieved) > self._debug_top_k:
            log.info("  ... and %d more retrieved anchors", len(retrieved) - self._debug_top_k)
        trace_json(log, "retrieved_anchors", {
            "num_retrieved": len(retrieved),
            "top_anchors": anchors_summary(retrieved, max_items=min(self._debug_top_k, len(retrieved))),
        })

        # 将检索到的 anchors 组织成若干个 evidence group，便于后续模型在回答时
        # 参考结构化证据而不是无序大堆 raw anchors。
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
        log.info("QDMO-EVI organized groups count=%d", len(groups))
        for group in groups:
            log.info(
                "  %s score=%.4f members=%d images=%d label=%s",
                group.id,
                group.score,
                len(group.anchors),
                len(group.image_paths),
                group.group_label.replace("\n", " ")[:100],
            )
            log.info("    hypothesis=%s", group.group_hypothesis.replace("\n", " ")[:200])
        trace_json(log, "organized_groups", {
            "num_groups": len(groups),
            "groups": groups_summary(groups, max_groups=self._max_groups),
        })

        # 如果 QA 提供 clue 轮次，则检查组织后的 groups 是否覆盖了这些 clue。
        clue_rounds = (qa or {}).get("clue", [])
        if clue_rounds:
            group_rounds = {anchor.round_id for group in groups for anchor in group.anchors}
            hits = [rid for rid in clue_rounds if rid in group_rounds]
            misses = [rid for rid in clue_rounds if rid not in set(hits)]
            log.info(
                "QDMO-EVI clue coverage: %d/%d hits=%s misses=%s",
                len(hits),
                len(clue_rounds),
                hits,
                misses,
            )
            trace_json(log, "clue_coverage", {
                "num_hits": len(hits),
                "num_clues": len(clue_rounds),
                "hits": hits,
                "misses": misses,
            })

        # 对组织后的 evidence groups 进行视觉验证，补充 verified_evidence/contradictions/missing_evidence。
        verified_groups = self._verify_groups(question_stem, groups)
        if verified_groups is not groups:
            log.info("QDMO-EVI verified groups changed after verification")
        log.info("QDMO-EVI verified groups count=%d", len(verified_groups))
        for group in verified_groups:
            log.info(
                "  verified %s score=%.4f members=%d images=%d",
                group.id,
                group.score,
                len(group.anchors),
                len(group.image_paths),
            )
            if group.verified_evidence:
                log.info("    verified_evidence=%s", group.verified_evidence)
            if group.contradictions:
                log.info("    contradictions=%s", group.contradictions)
            if group.missing_evidence:
                log.info("    missing_evidence=%s", group.missing_evidence)
        trace_json(log, "verified_groups", {
            "num_groups": len(verified_groups),
            "groups": groups_summary(verified_groups, max_groups=self._max_groups),
        })

        # 根据验证后的 groups 构造最终 prompt 和 answer images。
        prompt = self._build_final_prompt(question, verified_groups)
        images = self._answer_images(verified_groups, question_images)
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

        # 最终调用 VLM 得到回答，并记录返回内容。
        answer = self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, prompt, images)
        trace_json(log, "answer_done", {"answer": answer})
        log.info("QDMO-EVI answer returned length=%d", len(str(answer)))
        return answer

    @property
    def num_indexed(self) -> int:
        return len(self._index)
