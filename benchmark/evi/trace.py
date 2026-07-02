from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import EvidenceAnchor, EvidenceGroup

_DEFAULT_LOG_PATH = "logs/evi_debug.log"
_HANDLER_MARK = "_evi_debug_file"


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
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARK, None) == str(path):
            return str(path)

    mode = "a" if as_bool(cfg.get("evi_debug_append"), True) else "w"
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    setattr(handler, _HANDLER_MARK, str(path))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
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


def group_summary(group: EvidenceGroup, max_anchors: int = 16, max_text_chars: int = 220) -> Dict[str, Any]:
    return {
        "id": group.id,
        "score": round(float(group.score or 0.0), 6),
        "confidence": round(float(group.confidence or 0.0), 4),
        "seed_anchor_id": group.seed_anchor_id,
        "label": group.group_label,
        "hypothesis": group.group_hypothesis,
        "image_paths": list(group.image_paths),
        "visual_checks": list(group.needed_visual_checks),
        "verified_evidence": list(group.verified_evidence),
        "contradictions": list(group.contradictions),
        "missing_evidence": list(group.missing_evidence),
        "anchors": [anchor_summary(anchor, max_text_chars) for anchor in group.anchors[:max_anchors]],
        "num_anchors": len(group.anchors),
    }


def anchors_summary(
    anchors: Iterable[EvidenceAnchor],
    max_items: int = 20,
    max_text_chars: int = 260,
) -> List[Dict[str, Any]]:
    return [anchor_summary(anchor, max_text_chars) for anchor in list(anchors)[:max_items]]


def groups_summary(
    groups: Iterable[EvidenceGroup],
    max_groups: int = 8,
    max_anchors: int = 16,
    max_text_chars: int = 220,
) -> List[Dict[str, Any]]:
    return [group_summary(group, max_anchors, max_text_chars) for group in list(groups)[:max_groups]]


def trace_json(logger: logging.Logger, title: str, payload: Dict[str, Any], max_chars: int = 12000) -> None:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        text = json.dumps({"trace_error": str(exc), "payload_repr": repr(payload)}, ensure_ascii=False, indent=2)
    logger.debug("[TRACE] %s\n%s", title, _shorten(text, max_chars))
