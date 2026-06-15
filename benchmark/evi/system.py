"""
EVI: System orchestrator.
Two-phase lifecycle (matching the M2A/MMA pattern):
  1. process_all_sessions(dataset) — extract + index
  2. answer_question(question, qa) — route + reason
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .extractor import extract_with_vlm
from .indexes import AnchorIndex, GistIndex, TagIndex
from .schemas import GistEntry, TagEntry
from .tracks import run_track_a, step1_coarse_retrieval, step2_llm_routing, step3_vlm_grounding
from .vlm import VLMCallable, make_vlm_callable

log = logging.getLogger(__name__)


class EVISystem:
    """Evidence-grounded Visual Indexing system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}

        # VLM callable for extraction + grounding
        self._vlm_callable: Optional[VLMCallable] = None

        # Text-only LLM callable for routing (shares VLM, no images)
        self._text_callable: Optional[Callable[[str], str]] = None

        # Model config passed from benchmark runner
        self._model_cfg = dict(cfg.get("_model_cfg", {}))

        # Indexes
        self.gist_index = GistIndex()
        self.tag_index = TagIndex()
        self.anchor_index = AnchorIndex()
        self.image_index: Dict[str, str] = {}

        # Text embedder for gist free_text
        self._text_embedder: Optional[Any] = None

        # Runtime
        self._extracted_count: int = 0
        self._total_rounds: int = 0
        self._initialized = False

    # ---- lazy init ----

    def _ensure_vlm(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Create VLM callable from model config
        self._vlm_callable = make_vlm_callable(self._model_cfg)

        # Create text-only callable from the same VLM (just pass no images)
        vlm = self._vlm_callable

        def _text_callable(prompt: str) -> str:
            return vlm("", prompt, [])

        self._text_callable = _text_callable

    def _ensure_embedder(self) -> Any:
        if self._text_embedder is not None:
            return self._text_embedder
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "all-MiniLM-L6-v2"
            self._text_embedder = SentenceTransformer(model_name)
        except ImportError:
            log.warning("sentence-transformers not available")
            self._text_embedder = None
        return self._text_embedder

    # ---- Phase 1 ----

    def process_all_sessions(self, dataset: Any) -> None:
        """Process all sessions: extract evidence from every image-bearing round."""
        self._ensure_vlm()
        self._ensure_embedder()

        vlm = self._vlm_callable
        if vlm is None:
            raise RuntimeError("VLM callable not initialized")

        session_ids = list(dataset.session_order())
        self._total_rounds = sum(
            len(dataset.get_session(sid).get("dialogues", []))
            for sid in session_ids
        )

        for _, session_id in enumerate(session_ids, start=1):
            session = dataset.get_session(session_id)
            date = str(session.get("date", "")).strip() or "unknown"
            dialogues = session.get("dialogues", [])

            for dialogue in dialogues:
                round_id = dialogue.get("round", "")
                if not round_id:
                    continue

                round_payload = dataset.rounds.get(round_id, {})
                images: List[str] = round_payload.get("images", []) or []
                if not images:
                    continue

                user_text = str(round_payload.get("user", "")).strip()
                assistant_text = str(round_payload.get("assistant", "")).strip()

                for img_path in images:
                    result = extract_with_vlm(
                        image_path=img_path,
                        user_text=user_text,
                        assistant_text=assistant_text,
                        vlm_callable=vlm,
                    )
                    if not result.gist.free_text:
                        log.warning("Empty extraction for %s, skipping", round_id)
                        continue

                    result.round_id = round_id

                    # Gist Index
                    attrs = result.gist.scene_attributes
                    self.gist_index.put(round_id, GistEntry(
                        round_id=round_id,
                        free_text=result.gist.free_text,
                        scene_attributes={
                            "background_color": attrs.background_color,
                            "lighting": attrs.lighting,
                            "setting": attrs.setting,
                            "mood": attrs.mood,
                        },
                        timestamp=f"{date}::{round_id}",
                    ))

                    # Tag Index
                    for tag in result.tags:
                        self.tag_index.add(tag.noun, TagEntry(
                            round_id=round_id,
                            color=tag.color,
                            position=tag.position,
                            count=tag.count,
                        ))

                    # Anchor Index
                    for label in result.anchors.explicit_labels:
                        key = label.lower().replace(" ", "_")
                        self.anchor_index.add(key, round_id)

                    # Image Index
                    self.image_index[round_id] = img_path

                    self._extracted_count += 1

                if (self._extracted_count % 20) == 0:
                    log.info(
                        "[EVI] Extracted %d / ~%d image rounds",
                        self._extracted_count,
                        min(self._total_rounds, self._extracted_count + 100),
                    )

        log.info(
            "[EVI] Done. %d image rounds extracted, %d gist, %d tag entries, %d anchor keys",
            self._extracted_count,
            len(self.gist_index),
            len(self.tag_index),
            len(self.anchor_index),
        )

    # ---- Phase 2 ----

    def answer_question(
        self,
        question: str,
        qa: Optional[Dict[str, Any]] = None,
        question_images: Optional[List[str]] = None,
    ) -> str:
        """Route question through Track A or B.

        Args:
            question: The question text.
            qa: QA metadata (unused, kept for API compatibility).
            question_images: Question-level images (unused, handled by VLM grounding).
        """
        _ = qa  # unused, kept for API compat
        _ = question_images
        self._ensure_vlm()

        vlm = self._vlm_callable
        text_fn = self._text_callable
        if vlm is None or text_fn is None:
            raise RuntimeError("VLM callable not initialized")

        # Track A attempt (zero cost, aggregate on indexes)
        result = run_track_a(question, self.gist_index, self.tag_index, self.anchor_index)
        if result.confidence >= 0.7:
            return result.answer

        # Track B: three-step pipeline
        embedder = self._ensure_embedder()
        directory = step1_coarse_retrieval(
            question=question,
            gist_index=self.gist_index,
            tag_index=self.tag_index,
            anchor_index=self.anchor_index,
            text_embedder=embedder,
            top_k=15,
        )

        if not directory.entries:
            return ""

        selected = step2_llm_routing(
            question=question,
            directory=directory,
            text_callable=text_fn,
            top_n=3,
        )

        answer = step3_vlm_grounding(
            question=question,
            round_ids=selected,
            gist_index=self.gist_index,
            image_index=self.image_index,
            vlm_callable=vlm,
        )
        return answer

    @property
    def num_extracted(self) -> int:
        return self._extracted_count
