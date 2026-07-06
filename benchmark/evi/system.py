from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

from .briefs import generate_memory_briefs
from .candidates import consolidate_candidates
from .cues import extract_retrieval_cues
from .extractor import extract_image_anchors
from .image_index import ImageIndex
from .indexes import EvidenceIndex, embed_text, normalize_type
from .schemas import EpisodicState, EvidenceAnchor, MemoryBrief
from .sets import build_episodic_memory_sets
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
The evidence may be question-grounded episodic evidence or selected factual assertions.
Use grounded question cues to decide which episode matches each description in the question.
Do not choose an option merely because similar wording appears in an observed fact without a matching grounded cue.
Use attached images only to resolve uncertainty.
Be concise and grounded in the selected evidence.
If the question is multiple-choice, answer with ONLY the option letter.
"""


class EVISystem:
    """QDMO-EVI over typed evidence anchors with episodic-state readout."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._cfg = cfg
        self._model_cfg = dict(cfg.get("_model_cfg", {}))
        self._debug_log_path = setup_evi_debug_logging(cfg)

        self._use_anchor_quality_weighting = self._as_bool(cfg.get("use_anchor_quality_weighting"), True)
        self._index = EvidenceIndex(use_quality_weighting=self._use_anchor_quality_weighting)
        self._vlm: Optional[VLMCallable] = None
        self._embedder: Optional[Any] = None
        self._mm_embedder: Optional[Any] = None
        self._image_index: Optional[ImageIndex] = None

        self._round_order: List[str] = []
        self._round_session: Dict[str, str] = {}
        self._round_date: Dict[str, str] = {}
        self._round_text: Dict[str, str] = {}
        self._round_images: Dict[str, List[str]] = {}
        self._round_anchors: Dict[str, List[EvidenceAnchor]] = {}
        self._session_rounds: Dict[str, List[str]] = {}

        self._pipeline = str(cfg.get("evi_pipeline", "episodic_state") or "episodic_state").strip().lower()
        self._raw_search_k = int(cfg.get("raw_search_k", 120))
        self._max_candidates = int(cfg.get("max_candidates", 16))
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
        self._use_state_cache = self._as_bool(cfg.get("use_state_cache"), True)
        self._use_state_images = self._as_bool(cfg.get("use_state_images"), True)
        self._max_state_images_per_set = int(cfg.get("max_state_images_per_set", 4))
        self._max_retrieval_cues = int(cfg.get("max_retrieval_cues", 4))
        self._use_retrieval_cue_cache = self._as_bool(cfg.get("use_retrieval_cue_cache"), True)
        self._use_image_retrieval = self._as_bool(cfg.get("use_image_retrieval"), True)
        self._use_image_embedding_cache = self._as_bool(cfg.get("use_image_embedding_cache"), True)
        self._multimodal_embedding_model = str(cfg.get("multimodal_embedding_model", "siglip2-base-patch16-384"))
        self._debug_top_k = int(cfg.get("evi_debug_top_k", 20))
        self._debug_prompt_chars = int(cfg.get("evi_debug_prompt_chars", 12000))
        self._embed_cache_namespace = "uninitialized"
        self._image_embed_cache_namespace = "uninitialized"
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

        self._image_embed_cache_namespace = json.dumps(
            {"model": self._multimodal_embedding_model},
            sort_keys=True,
            default=str,
        )
        if self._use_image_retrieval:
            from ..embeddings import get_multimodal_embedder

            try:
                self._mm_embedder = get_multimodal_embedder(self._multimodal_embedding_model)
            except Exception as exc:
                log.warning("Image retrieval embedder unavailable: %s", exc)
                self._mm_embedder = None
            if self._mm_embedder is None:
                log.warning("QDMO-EVI image retrieval disabled because no multimodal embedder is available")
                self._use_image_retrieval = False
            else:
                self._image_index = ImageIndex(
                    self._mm_embedder,
                    cache_namespace=self._image_embed_cache_namespace,
                    use_cache=self._use_image_embedding_cache,
                )
                log.info("Image retrieval embedder loaded: %s", self._multimodal_embedding_model)

    def _embed(self, text: str) -> List[float]:
        return embed_text(
            text,
            self._embedder,
            cache_namespace=self._embed_cache_namespace,
            use_cache=self._use_embedding_cache,
        )

    def _reset_retrieval_scores(self) -> None:
        for anchor in self._index.anchors:
            anchor.score = 0.0
            anchor.raw_score = 0.0
            anchor.retrieval_channels = []
            anchor.channel_scores = {}
            anchor.round_fused_score = 0.0

    @staticmethod
    def _fuse_channel_scores(scores: List[float]) -> float:
        clean = sorted([float(score) for score in scores if float(score or 0.0) > 0.0], reverse=True)
        if not clean:
            return 0.0
        if len(clean) == 1:
            return clean[0]
        max_score = clean[0]
        non_max_mean = sum(clean[1:]) / max(1, len(clean) - 1)
        return max_score + non_max_mean / len(clean)

    def _log_channel_clue_coverage(
        self,
        qa: Optional[Dict[str, Any]],
        channel_rounds: Dict[str, Set[str]],
    ) -> None:
        clue_rounds = (qa or {}).get("clue", [])
        if not clue_rounds:
            return
        coverage: Dict[str, Dict[str, Any]] = {}
        for channel_id, rounds in channel_rounds.items():
            hits = [rid for rid in clue_rounds if rid in rounds]
            misses = [rid for rid in clue_rounds if rid not in set(hits)]
            coverage[channel_id] = {
                "num_hits": len(hits),
                "num_clues": len(clue_rounds),
                "hits": hits,
                "misses": misses,
            }
            log.info(
                "QDMO-EVI channel clue coverage channel=%s: %d/%d hits=%s misses=%s",
                channel_id,
                len(hits),
                len(clue_rounds),
                hits,
                misses,
            )
        unions = {
            "text_channels_union": set(),
            "image_channels_union": set(),
            "cue_channels_union": set(),
        }
        for channel_id, rounds in channel_rounds.items():
            if channel_id.startswith("image_"):
                unions["image_channels_union"].update(rounds)
            else:
                unions["text_channels_union"].update(rounds)
            if channel_id.startswith("cue_") or channel_id.startswith("image_cue_"):
                unions["cue_channels_union"].update(rounds)
        for union_name, rounds in unions.items():
            if not rounds:
                continue
            hits = [rid for rid in clue_rounds if rid in rounds]
            misses = [rid for rid in clue_rounds if rid not in set(hits)]
            coverage[union_name] = {
                "num_hits": len(hits),
                "num_clues": len(clue_rounds),
                "hits": hits,
                "misses": misses,
            }
            log.info(
                "QDMO-EVI %s clue coverage: %d/%d hits=%s misses=%s",
                union_name,
                len(hits),
                len(clue_rounds),
                hits,
                misses,
            )
        trace_json(log, "multi_channel_clue_coverage", coverage)

    def _multi_channel_retrieve(self, question_stem: str, qa: Optional[Dict[str, Any]]) -> List[EvidenceAnchor]:
        cues = extract_retrieval_cues(
            question_stem,
            self._vlm,
            max_cues=self._max_retrieval_cues,
            use_cache=self._use_retrieval_cue_cache,
            cache_namespace=self._vlm_result_namespace,
        )
        base_channels: List[Dict[str, str]] = [{"id": "question", "text": str(question_stem or "").strip()}]
        seen_texts = {base_channels[0]["text"].lower()}
        for idx, cue in enumerate(cues, start=1):
            cue_text = " ".join(str(cue or "").split())
            if not cue_text:
                continue
            key = cue_text.lower()
            if key in seen_texts:
                continue
            base_channels.append({"id": f"cue_{idx}", "text": cue_text})
            seen_texts.add(key)

        channels: List[Dict[str, str]] = []
        for base in base_channels:
            if base["text"]:
                channels.append({"id": base["id"], "text": base["text"], "kind": "text"})
        image_channels_enabled = self._use_image_retrieval and self._image_index is not None and len(self._image_index) > 0
        if image_channels_enabled:
            for base in base_channels:
                if base["text"]:
                    channels.append({"id": f"image_{base['id']}", "text": base["text"], "kind": "image"})
        elif self._use_image_retrieval:
            log.info("QDMO-EVI image retrieval requested but image index is empty or unavailable")

        channels = [channel for channel in channels if channel["text"]]
        if not channels:
            return []
        per_channel_k = max(1, (self._raw_search_k + len(channels) - 1) // len(channels))
        log.info(
            "QDMO-EVI multi-channel retrieval: channels=%d per_channel_k=%d raw_search_k=%d cues=%s image_channels=%s",
            len(channels),
            per_channel_k,
            self._raw_search_k,
            [channel["text"] for channel in base_channels if channel["id"] != "question"],
            image_channels_enabled,
        )

        self._reset_retrieval_scores()
        channel_order = {channel["id"]: idx for idx, channel in enumerate(channels)}
        round_channel_best: Dict[str, Dict[str, float]] = {}
        round_anchor_ids: Dict[str, Set[str]] = {}
        anchor_by_id: Dict[str, EvidenceAnchor] = {}
        anchor_channel_scores: Dict[str, Dict[str, float]] = {}
        anchor_channel_raw_scores: Dict[str, Dict[str, float]] = {}
        channel_rounds: Dict[str, Set[str]] = {}
        channel_summaries: List[Dict[str, Any]] = []

        def add_hit(anchor: EvidenceAnchor, channel_id: str, channel_score: float, raw_score: float) -> None:
            if channel_score <= 0.0:
                return
            anchor_by_id[anchor.id] = anchor
            round_anchor_ids.setdefault(anchor.round_id, set()).add(anchor.id)
            best_by_channel = round_channel_best.setdefault(anchor.round_id, {})
            if channel_score > best_by_channel.get(channel_id, 0.0):
                best_by_channel[channel_id] = channel_score
            per_anchor = anchor_channel_scores.setdefault(anchor.id, {})
            if channel_score > per_anchor.get(channel_id, 0.0):
                per_anchor[channel_id] = channel_score
            per_anchor_raw = anchor_channel_raw_scores.setdefault(anchor.id, {})
            if raw_score > per_anchor_raw.get(channel_id, 0.0):
                per_anchor_raw[channel_id] = raw_score

        for channel in channels:
            channel_id = channel["id"]
            channel_text = channel["text"]
            channel_kind = channel.get("kind", "text")
            if channel_kind == "image":
                if self._image_index is None:
                    channel_rounds[channel_id] = set()
                    continue
                image_hits = self._image_index.search(channel_text, top_k=per_channel_k)
                channel_rounds[channel_id] = {hit.round_id for hit in image_hits}
                channel_summaries.append(
                    {
                        "channel_id": channel_id,
                        "channel_text": channel_text,
                        "channel_kind": channel_kind,
                        "num_results": len(image_hits),
                        "top_images": [
                            {
                                "round_id": hit.round_id,
                                "image_path": hit.image_path,
                                "raw_score": round(float(hit.score), 6),
                                "rank_score": round(float(hit.rank_score), 6),
                                "rank": hit.rank,
                            }
                            for hit in image_hits[: self._debug_top_k]
                        ],
                    }
                )
                log.info(
                    "QDMO-EVI image_channel=%s text=%s retrieved=%d",
                    channel_id,
                    channel_text,
                    len(image_hits),
                )
                for hit in image_hits:
                    log.info(
                        "  image_hit[%s:%02d] round=%s rank_score=%.4f raw=%.4f image=%s",
                        channel_id,
                        hit.rank,
                        hit.round_id,
                        hit.rank_score,
                        hit.score,
                        hit.image_path,
                    )
                for hit in image_hits:
                    anchor = EvidenceAnchor(
                        id=f"image_hit::{channel_id}::{hit.rank}::{hit.round_id}",
                        session_id=self._round_session.get(hit.round_id, ""),
                        round_id=hit.round_id,
                        date=self._round_date.get(hit.round_id, ""),
                        evidence_type="scene",
                        text=(
                            f"Image retrieval hit for channel '{channel_id}' using query '{channel_text}'. "
                            f"Matched image: {os.path.basename(hit.image_path)}."
                        ),
                        image_path=hit.image_path,
                        confidence=1.0,
                        vector=[],
                    )
                    add_hit(anchor, channel_id, float(hit.rank_score), float(hit.score))
                continue

            query_vec = self._embed(channel_text)
            if not query_vec:
                log.warning("QDMO-EVI retrieval channel has empty embedding channel=%s text=%s", channel_id, channel_text)
                channel_rounds[channel_id] = set()
                continue
            results = self._index.search(query_vec, top_k=per_channel_k)
            channel_rounds[channel_id] = {anchor.round_id for anchor in results}
            channel_summaries.append(
                {
                    "channel_id": channel_id,
                    "channel_text": channel_text,
                    "channel_kind": channel_kind,
                    "num_results": len(results),
                    "top_anchors": anchors_summary(results, max_items=min(self._debug_top_k, len(results))),
                }
            )
            log.info(
                "QDMO-EVI text_channel=%s text=%s retrieved=%d",
                channel_id,
                channel_text,
                len(results),
            )
            for rank, anchor in enumerate(results, start=1):
                rank_score = 1.0 / rank
                log.info(
                    "  channel_anchor[%s:%02d] round=%s type=%s rank_score=%.4f raw=%.4f text=%s",
                    channel_id,
                    rank,
                    anchor.round_id,
                    anchor.evidence_type,
                    rank_score,
                    anchor.score or 0.0,
                    anchor.text.replace("\n", " ")[:160],
                )
            for rank, anchor in enumerate(results, start=1):
                raw_score = float(anchor.score or 0.0)
                if raw_score <= 0.0:
                    continue
                add_hit(anchor, channel_id, 1.0 / rank, raw_score)

        self._log_channel_clue_coverage(qa, channel_rounds)
        self._reset_retrieval_scores()

        fused_rounds = []
        for round_id, scores_by_channel in round_channel_best.items():
            fused = self._fuse_channel_scores(list(scores_by_channel.values()))
            fused_rounds.append((round_id, fused, scores_by_channel))
        fused_rounds.sort(
            key=lambda item: (item[1], len(item[2]), max(item[2].values()) if item[2] else 0.0),
            reverse=True,
        )

        merged: List[EvidenceAnchor] = []
        seen_anchor_ids: Set[str] = set()
        fused_round_trace: List[Dict[str, Any]] = []
        for round_id, fused_score, scores_by_channel in fused_rounds:
            fused_round_trace.append(
                {
                    "round_id": round_id,
                    "fused_score": round(float(fused_score), 6),
                    "channels": {
                        channel_id: round(float(score), 6)
                        for channel_id, score in sorted(scores_by_channel.items(), key=lambda item: channel_order.get(item[0], 999))
                    },
                }
            )
            anchor_ids = list(round_anchor_ids.get(round_id, set()))
            anchor_ids.sort(
                key=lambda aid: (
                    max(anchor_channel_scores.get(aid, {}).values(), default=0.0),
                    len(anchor_channel_scores.get(aid, {})),
                    max(anchor_channel_raw_scores.get(aid, {}).values(), default=0.0),
                ),
                reverse=True,
            )
            for anchor_id in anchor_ids[: self._max_state_anchors_per_round]:
                if anchor_id in seen_anchor_ids:
                    continue
                anchor = anchor_by_id.get(anchor_id)
                if anchor is None:
                    continue
                channel_raw_scores = anchor_channel_raw_scores.get(anchor_id, {})
                anchor.score = fused_score
                anchor.raw_score = max(channel_raw_scores.values(), default=0.0)
                anchor.round_fused_score = fused_score
                anchor.channel_scores = {
                    channel_id: float(scores_by_channel[channel_id])
                    for channel_id in sorted(scores_by_channel, key=lambda item: channel_order.get(item, 999))
                }
                anchor.retrieval_channels = list(anchor.channel_scores.keys())
                merged.append(anchor)
                seen_anchor_ids.add(anchor_id)
                if len(merged) >= self._raw_search_k:
                    break
            if len(merged) >= self._raw_search_k:
                break

        merged.sort(
            key=lambda anchor: (
                anchor.score or 0.0,
                len(anchor.retrieval_channels),
                anchor.raw_score or 0.0,
            ),
            reverse=True,
        )
        log.info(
            "QDMO-EVI multi-channel merged anchors=%d fused_rounds=%d",
            len(merged),
            len(fused_rounds),
        )
        for idx, anchor in enumerate(merged[: self._debug_top_k], start=1):
            log.info(
                "  merged_anchor[%02d] round=%s fused=%.4f raw=%.4f channels=%s channel_scores=%s text=%s",
                idx,
                anchor.round_id,
                anchor.score or 0.0,
                anchor.raw_score or 0.0,
                anchor.retrieval_channels,
                {key: round(value, 4) for key, value in anchor.channel_scores.items()},
                anchor.text.replace("\n", " ")[:180],
            )
        trace_json(log, "multi_channel_retrieval", {
            "question_stem": question_stem,
            "retrieval_cues": cues,
            "channels": channels,
            "per_channel_k": per_channel_k,
            "channel_results": channel_summaries,
            "fused_rounds": fused_round_trace[: self._debug_top_k],
            "num_fused_rounds": len(fused_rounds),
            "num_merged_anchors": len(merged),
            "image_channels_enabled": image_channels_enabled,
            "merged_anchors": anchors_summary(merged, max_items=min(self._debug_top_k, len(merged))),
        }, max_chars=self._debug_prompt_chars + 4000)
        return merged[: self._raw_search_k]
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
        if self._vlm is None:
            raise RuntimeError("VLM not initialized")

        log.info("Processing sessions for QDMO-EVI evidence anchors ...")
        trace_json(log, "indexing_start", {
            "pipeline": self._pipeline,
            "raw_search_k": self._raw_search_k,
            "use_anchor_quality_weighting": self._use_anchor_quality_weighting,
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
            "use_state_cache": self._use_state_cache,
            "use_state_images": self._use_state_images,
            "max_state_images_per_set": self._max_state_images_per_set,
            "max_retrieval_cues": self._max_retrieval_cues,
            "use_retrieval_cue_cache": self._use_retrieval_cue_cache,
            "use_image_retrieval": self._use_image_retrieval,
            "use_image_embedding_cache": self._use_image_embedding_cache,
            "multimodal_embedding_model": self._multimodal_embedding_model,
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

                images = list(self._round_images.get(rid, []))
                for img_idx, image_path in enumerate(images):
                    if not os.path.isfile(image_path):
                        log.warning("[INDEX] image file not found, skipping: %s", image_path)
                        continue
                    if self._use_image_retrieval and self._image_index is not None:
                        self._image_index.add(rid, image_path)
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

        self._index.finalize()

        type_counts: Dict[str, int] = {}
        for anchor in self._index.anchors:
            type_counts[anchor.evidence_type] = type_counts.get(anchor.evidence_type, 0) + 1
        log.info(
            "QDMO-EVI indexing done: %d anchors across %d rounds. Types: %s",
            len(self._index),
            len(self._round_order),
            ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items())),
        )
        if self._use_image_retrieval:
            log.info(
                "QDMO-EVI image index done: %d memory images",
                len(self._image_index) if self._image_index is not None else 0,
            )
        trace_json(log, "indexing_done", {
            "num_anchors": len(self._index),
            "num_rounds": len(self._round_order),
            "type_counts": dict(sorted(type_counts.items())),
            "indexed_memory_images": len(self._image_index) if self._image_index is not None else 0,
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
            "QDMO-EVI selected final states clue coverage for final QA: %d/%d hits=%s misses=%s state_rounds=%s",
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

        retrieved = self._multi_channel_retrieve(question_stem, qa)
        if not retrieved:
            log.warning("QDMO-EVI retrieval returned no anchors; answering with question images only")
            images = self._as_image_list(question_images)[: self._max_answer_images]
            return self._vlm(FINAL_ANSWER_SYSTEM_PROMPT, question, images)

        log.info("QDMO-EVI retrieved anchors=%d", len(retrieved))
        for idx, anchor in enumerate(retrieved[: self._debug_top_k]):
            log.info(
                "  anchor[%02d] id=%s round=%s type=%s fused=%.4f raw=%.4f channels=%s channel_scores=%s text=%s",
                idx + 1,
                anchor.id,
                anchor.round_id,
                anchor.evidence_type,
                anchor.score or 0.0,
                anchor.raw_score or 0.0,
                getattr(anchor, "retrieval_channels", []),
                {key: round(value, 4) for key, value in (getattr(anchor, "channel_scores", {}) or {}).items()},
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
        lines.append("Question-grounded episodic evidence:")
        if not states:
            lines.append("No selected episodic evidence was available.")
        for idx, state in enumerate(states, start=1):
            lines.append(
                f"{idx}. Evidence from session {state.session_id} on {state.date}; "
                f"rounds {' -> '.join(state.round_ids)}."
            )
            cues = [" ".join(str(item).split()) for item in state.grounded_cues if str(item).strip()]
            if cues:
                lines.append("   Grounded question cues:")
                for item in cues[:8]:
                    lines.append("   - " + item)
            else:
                lines.append("   Grounded question cues: none")
            facts = [" ".join(str(item).split()) for item in state.observed_facts if str(item).strip()]
            if facts:
                lines.append("   Observed facts:")
                for item in facts[:12]:
                    lines.append("   - " + item)
            else:
                lines.append("   Observed facts: none")
            uncertainties = [" ".join(str(item).split()) for item in state.uncertainties if str(item).strip()]
            if uncertainties:
                lines.append("   Uncertainties:")
                for item in uncertainties[:3]:
                    lines.append("   - " + item)
        lines.append("")
        lines.append("Use grounded question cues to decide which episode corresponds to each description in the question. Use observed facts from the matching episode to answer. Ignore observed facts from episodes without matching grounded cues.")
        lines.append("")
        lines.append("Question:")
        lines.append(str(question or ""))
        lines.append("")
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

    def _readout_question_context(self, question: str, question_stem: str) -> str:
        lines: List[str] = []
        for line in str(question or question_stem or "").splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("answer with only"):
                continue
            if lowered in {"do not explain.", "do not explain"}:
                continue
            lines.append(line)
        context = "\n".join(lines).strip()
        return context or str(question_stem or question or "")

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
            question_context=self._readout_question_context(question, question_stem),
        )
        trace_json(log, "episodic_states", {
            "num_states": len(states),
            "states": states_summary(states, max_items=self._max_memory_sets),
        })

        final_states = self._select_final_states(states)
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
        log.info("QDMO-EVI final prompt full:\n%s", prompt)
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
        log.info("QDMO-EVI answer returned length=%d", len(str(answer)))
        return answer

    @property
    def num_indexed(self) -> int:
        return len(self._index)
