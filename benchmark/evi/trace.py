from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import EvidenceAnchor, MemoryBrief, MemoryCandidate

_DEFAULT_LOG_PATH = "logs/evi_debug.log"
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
        or _DEFAULT_LOG_PATH
    )
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("benchmark.evi")
    logger.setLevel(logging.DEBUG)
    has_file_handler = False
    has_console_handler = False
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARK, None) == str(path):
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


def trace_json(logger: logging.Logger, title: str, payload: Dict[str, Any], max_chars: int = 12000) -> None:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        text = json.dumps({"trace_error": str(exc), "payload_repr": repr(payload)}, ensure_ascii=False, indent=2)
    logger.debug("[TRACE] %s\n%s", title, _shorten(text, max_chars))
