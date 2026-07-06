from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import EvidenceAnchor, EpisodicMemorySet, EpisodicState, MemoryBrief, MemoryCandidate

_DEFAULT_LOG_NAME = "evi_debug.log"
_HANDLER_MARK = "_evi_debug_file"
_CONSOLE_MARK = "_evi_debug_console"


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def setup_evi_debug_logging(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    cfg = config or {}
    enabled = as_bool(cfg.get("evi_debug"), True)
    if not enabled:
        return None

    raw_path = (
        cfg.get("evi_debug_log_path")
        or cfg.get("debug_log_path")
        or os.environ.get("EVI_DEBUG_LOG_PATH")
    )
    if raw_path:
        path = Path(str(raw_path)).expanduser()
    else:
        run_dir = (cfg.get("_runtime_paths") or {}).get("run_dir")
        path = Path(str(run_dir)).expanduser() / _DEFAULT_LOG_NAME if run_dir else Path("logs") / _DEFAULT_LOG_NAME
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("benchmark.evi")
    logger.setLevel(logging.DEBUG)
    has_file_handler = False
    has_console_handler = False
    for handler in list(logger.handlers):
        handler_path = getattr(handler, _HANDLER_MARK, None)
        if handler_path and handler_path != str(path):
            logger.removeHandler(handler)
            handler.close()
            continue
        if handler_path == str(path):
            has_file_handler = True
        if getattr(handler, _CONSOLE_MARK, False):
            has_console_handler = True
    if not has_file_handler:
        mode = "a" if as_bool(cfg.get("evi_debug_append"), True) else "w"
        file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
        setattr(file_handler, _HANDLER_MARK, str(path))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(file_handler)

    if as_bool(cfg.get("evi_debug_console"), True) and not has_console_handler:
        console_handler = logging.StreamHandler()
        setattr(console_handler, _CONSOLE_MARK, True)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("[EVI] %(message)s"))
        logger.addHandler(console_handler)

    logger.propagate = False
    return str(path)


def _shorten(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... <truncated {len(value) - max_chars} chars>"


def anchor_summary(anchor: EvidenceAnchor, max_text_chars: int = 260) -> Dict[str, Any]:
    return {
        "id": anchor.id,
        "score": round(float(anchor.score or 0.0), 6),
        "raw_score": round(float(anchor.raw_score or 0.0), 6),
        "quality_weight": round(float(anchor.quality_weight or 1.0), 4),
        "discriminativeness_weight": round(float(anchor.discriminativeness_weight or 1.0), 4),
        "retrieval_channels": list(getattr(anchor, "retrieval_channels", []) or []),
        "channel_scores": {
            key: round(float(value or 0.0), 6)
            for key, value in (getattr(anchor, "channel_scores", {}) or {}).items()
        },
        "round_fused_score": round(float(getattr(anchor, "round_fused_score", 0.0) or 0.0), 6),
        "session_id": anchor.session_id,
        "round_id": anchor.round_id,
        "date": anchor.date,
        "type": anchor.evidence_type,
        "image_path": anchor.image_path,
        "region": anchor.region,
        "confidence": round(float(anchor.confidence or 0.0), 4),
        "text": _shorten(anchor.text, max_text_chars),
    }


def candidate_summary(candidate: MemoryCandidate, max_anchors: int = 8) -> Dict[str, Any]:
    return {
        "id": candidate.id,
        "score": round(float(candidate.score or 0.0), 6),
        "session_id": candidate.session_id,
        "round_id": candidate.round_id,
        "date": candidate.date,
        "image_paths": list(candidate.image_paths),
        "round_text": _shorten(candidate.round_text, 900),
        "num_anchors": len(candidate.anchors),
        "num_selected_anchors": len(candidate.selected_anchors),
        "selected_anchors": [anchor_summary(anchor) for anchor in candidate.selected_anchors[:max_anchors]],
    }


def brief_summary(brief: MemoryBrief, max_text_chars: int = 900) -> Dict[str, Any]:
    return {
        "candidate_id": brief.candidate_id,
        "score": round(float(brief.score or 0.0), 6),
        "session_id": brief.session_id,
        "round_id": brief.round_id,
        "date": brief.date,
        "image_paths": list(brief.image_paths),
        "relevance": brief.relevance,
        "confidence": round(float(brief.confidence or 0.0), 4),
        "brief": _shorten(brief.brief, max_text_chars),
        "key_evidence": list(brief.key_evidence),
    }


def memory_set_summary(memory_set: EpisodicMemorySet, max_anchors_per_round: int = 6) -> Dict[str, Any]:
    return {
        "id": memory_set.id,
        "score": round(float(memory_set.score or 0.0), 6),
        "hit_round_count": int(memory_set.hit_round_count or 0),
        "max_anchor_score": round(float(memory_set.max_anchor_score or 0.0), 6),
        "session_id": memory_set.session_id,
        "date": memory_set.date,
        "round_ids": list(memory_set.round_ids),
        "retrieved_anchors": [anchor_summary(anchor) for anchor in memory_set.retrieved_anchors[:max_anchors_per_round]],
        "rounds": [
            {
                "round_id": rid,
                "round_text": _shorten(memory_set.round_text.get(rid, ""), 700),
                "image_paths": list(memory_set.round_images.get(rid, [])),
                "anchors": [
                    anchor_summary(anchor)
                    for anchor in memory_set.round_anchors.get(rid, [])[:max_anchors_per_round]
                ],
            }
            for rid in memory_set.round_ids
        ],
    }


def state_summary(state: EpisodicState, max_text_chars: int = 900) -> Dict[str, Any]:
    return {
        "set_id": state.set_id,
        "score": round(float(state.score or 0.0), 6),
        "session_id": state.session_id,
        "date": state.date,
        "round_ids": list(state.round_ids),
        "image_paths": list(state.image_paths),
        "relevance": state.relevance,
        "grounded_cues": [_shorten(item, max_text_chars) for item in state.grounded_cues],
        "observed_facts": [_shorten(item, max_text_chars) for item in state.observed_facts],
        "confidence": round(float(state.confidence or 0.0), 4),
        "memory_items": [_shorten(item, max_text_chars) for item in state.memory_items],
        "observations": [_shorten(item, max_text_chars) for item in state.observations],
        "relations": [_shorten(item, max_text_chars) for item in state.relations],
        "changes": [_shorten(item, max_text_chars) for item in state.changes],
        "answer_relevant_facts": [_shorten(item, max_text_chars) for item in state.answer_relevant_facts],
        "uncertainties": [_shorten(item, max_text_chars) for item in state.uncertainties],
    }




def anchors_summary(
    anchors: Iterable[EvidenceAnchor],
    max_items: int = 20,
    max_text_chars: int = 260,
) -> List[Dict[str, Any]]:
    return [anchor_summary(anchor, max_text_chars) for anchor in list(anchors)[:max_items]]


def candidates_summary(
    candidates: Iterable[MemoryCandidate],
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    return [candidate_summary(candidate) for candidate in list(candidates)[:max_items]]


def briefs_summary(
    briefs: Iterable[MemoryBrief],
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    return [brief_summary(brief) for brief in list(briefs)[:max_items]]


def memory_sets_summary(
    memory_sets: Iterable[EpisodicMemorySet],
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    return [memory_set_summary(memory_set) for memory_set in list(memory_sets)[:max_items]]


def states_summary(
    states: Iterable[EpisodicState],
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    return [state_summary(state) for state in list(states)[:max_items]]


def trace_json(logger: logging.Logger, title: str, payload: Dict[str, Any], max_chars: int = 12000) -> None:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        text = json.dumps({"trace_error": str(exc), "payload_repr": repr(payload)}, ensure_ascii=False, indent=2)
    logger.debug("[TRACE] %s\n%s", title, _shorten(text, max_chars))
