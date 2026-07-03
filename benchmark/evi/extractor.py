from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .vlm import VLMCallable

from ._utils import extract_json, retry_vlm_call

log = logging.getLogger(__name__)

_PROMPT_VERSION = "anchor_v2_generic_schema"
_CACHE_DIR: Optional[str] = None

ANCHOR_EXTRACTION_PROMPT = """You are building long-term visual memory for a multimodal agent.

Given an image and local dialogue context, extract compact evidence anchors that may help future questions retrieve this image.
These anchors are for retrieval, not final answering. Do not try to exhaust every possible detail.

Return ONLY valid JSON in this format:
{
  "anchors": [
    {
      "evidence_type": "scene|text|entity|attribute|spatial|relation|identity|structured_visual|temporal",
      "text": "self-contained retrieval text",
      "subject": "optional subject",
      "predicate": "optional predicate",
      "object": "optional object",
      "region": "optional visual region such as top-left/lower-middle/center",
      "confidence": 0.0
    }
  ]
}

Evidence type meanings:
- scene: overall visual setting or high-level image content.
- text: readable words, labels, numbers, names, messages, or symbols.
- entity: salient people, objects, places, documents, interfaces, diagrams, or visual elements.
- attribute: color, count, material, texture, shape, size, or other visible properties.
- spatial: position, layout, direction, containment, or proximity.
- relation: interaction or relationship between two or more entities.
- identity: stable cues for recurring or visually similar entities.
- structured_visual: organized visual information such as tables, charts, forms, panels, diagrams, maps, or dense layouts.
- temporal: local state, change, step, update, or sequence implied by the image/context.

Guidelines:
- Include high-value anchors that make the image retrievable later without seeing it.
- Include visible text, symbols, numbers, labels, and names when present.
- Include salient entities and their distinctive attributes.
- Include spatial relations and entity relations when visually clear.
- Include stable identity cues for recurring people, objects, characters, documents, or interfaces.
- Include structured visual cues when the image contains organized information rather than a natural scene.
- Include local dialogue context only when it explains what the image represents.
- Limit to 12-20 high-value anchors.
- Each anchor text must be self-contained and searchable without seeing the image.
"""


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_ANCHOR_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_anchors"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _cache_key(image_path: str, round_text: str, prior_rounds_text: str, cache_namespace: str) -> str:
    raw = json.dumps(
        {
            "version": _PROMPT_VERSION,
            "cache_namespace": cache_namespace,
            "image_path": image_path,
            "round_text": round_text,
            "prior_rounds_text": prior_rounds_text[-4000:],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _context_text(round_text: str, prior_rounds_text: str) -> str:
    parts: List[str] = []
    if prior_rounds_text:
        parts.append("Prior local session context:\n" + prior_rounds_text[-4000:])
    parts.append("Current round context:\n" + round_text)
    return "\n\n".join(parts)


def _coerce_anchor(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text", "")).strip()
    if not text:
        subject = str(raw.get("subject", "")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        obj = str(raw.get("object", "")).strip()
        text = " :: ".join(part for part in (subject, predicate, obj) if part)
    if not text:
        return None
    try:
        confidence = float(raw.get("confidence", 1.0))
    except Exception:
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "evidence_type": str(raw.get("evidence_type", "scene")).strip() or "scene",
        "text": text,
        "subject": str(raw.get("subject", "")).strip(),
        "predicate": str(raw.get("predicate", "")).strip(),
        "object": str(raw.get("object", "")).strip(),
        "region": str(raw.get("region", "")).strip(),
        "confidence": confidence,
    }


def extract_image_anchors(
    image_path: str,
    round_text: str,
    prior_rounds_text: str,
    vlm_callable: "VLMCallable",
    use_cache: bool = True,
    cache_namespace: str = "default",
) -> List[Dict[str, Any]]:
    """Extract task-agnostic retrieval anchors from one image."""
    key = _cache_key(image_path, round_text, prior_rounds_text, cache_namespace)
    cache_file = Path(_cache_dir()) / f"{key}.json"
    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            anchors = data.get("anchors", []) if isinstance(data, dict) else []
            out = [a for a in (_coerce_anchor(x) for x in anchors) if a]
            if out:
                log.info("  [ANCHORS] Cache HIT for %s (%d anchors)", image_path, len(out))
                return out
        except Exception:
            pass

    user_text = _context_text(round_text, prior_rounds_text)
    log.info("  [ANCHORS] Extracting anchors for %s", image_path)
    raw = retry_vlm_call(
        lambda: vlm_callable(ANCHOR_EXTRACTION_PROMPT, user_text, [image_path]),
        label=f"anchor-extract {Path(image_path).name}",
    )
    parsed = extract_json(raw or "") or {}
    raw_anchors = parsed.get("anchors", []) if isinstance(parsed, dict) else []
    anchors = [a for a in (_coerce_anchor(x) for x in raw_anchors) if a]

    if use_cache:
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "version": _PROMPT_VERSION,
                        "cache_namespace": cache_namespace,
                        "image_path": image_path,
                        "anchors": anchors,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("  [ANCHORS] Cache write failed: %s", exc)

    log.info("  [ANCHORS] Extracted %d anchors", len(anchors))
    return anchors
