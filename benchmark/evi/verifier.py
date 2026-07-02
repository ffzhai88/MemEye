from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .schemas import EvidenceGroup

if TYPE_CHECKING:
    from .vlm import VLMCallable

log = logging.getLogger(__name__)

_PROMPT_VERSION = "group_verify_v2_generic_schema"
_CACHE_DIR: Optional[str] = None

GROUP_VERIFICATION_PROMPT = """You are verifying an evidence group from long-term visual memory.

A memory system retrieved and organized evidence anchors into a group for the current question.
Inspect the attached image(s) and verify the group's visual evidence. Focus on the requested visual checks.
Do not answer the final multiple-choice question and do not choose an option letter.

Return ONLY valid JSON:
{
  "verified_evidence": ["visual fact verified from the images/context"],
  "contradictions": ["evidence that conflicts with the group hypothesis, if any"],
  "missing_evidence": ["important evidence that is not visible or remains uncertain, if any"],
  "confidence": 0.0
}

Rules:
- Ground every verified item in the attached image(s) or provided memory anchors.
- Prefer precise visual facts: text, counts, colors, spatial relations, identity cues, state changes, and structured visual values.
- If multiple images are attached, compare them when the question requires comparison or temporal reasoning.
- Keep the output concise but sufficient for final reasoning.
"""


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_GROUP_VERIFY_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_group_verification"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _cache_key(question_stem: str, group: EvidenceGroup) -> str:
    raw = json.dumps(
        {
            "version": _PROMPT_VERSION,
            "question": question_stem,
            "group_id": group.id,
            "seed": group.seed_anchor_id,
            "anchors": [(a.id, a.text, a.evidence_type, a.round_id, a.image_path) for a in group.anchors],
            "checks": group.needed_visual_checks,
            "images": group.image_paths,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _extract_json(text: str) -> Optional[dict]:
    import re

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def verify_group(
    question_stem: str,
    group: EvidenceGroup,
    vlm_callable: "VLMCallable",
    use_cache: bool = True,
) -> EvidenceGroup:
    """Verify one query-organized evidence group against its raw images."""
    if not group.image_paths:
        group.missing_evidence.append("No raw image was available for this evidence group.")
        return group

    key = _cache_key(question_stem, group)
    cache_file = Path(_cache_dir()) / f"{key}.json"
    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            group.verified_evidence = _as_str_list(data.get("verified_evidence", []))
            group.contradictions = _as_str_list(data.get("contradictions", []))
            group.missing_evidence = _as_str_list(data.get("missing_evidence", []))
            group.confidence = float(data.get("confidence", 0.0) or 0.0)
            return group
        except Exception:
            pass

    anchor_lines = []
    for anchor in group.anchors[:16]:
        loc = f" region={anchor.region}" if anchor.region else ""
        anchor_lines.append(
            f"- id={anchor.id} round={anchor.round_id} type={anchor.evidence_type}{loc}: {anchor.text}"
        )
    checks = "\n".join(f"- {c}" for c in group.needed_visual_checks)
    user_text = f"""Question stem:
{question_stem}

Evidence group label:
{group.group_label}

Group hypothesis:
{group.group_hypothesis}

Requested visual checks:
{checks}

Evidence anchors:
{chr(10).join(anchor_lines)}
"""
    log.info("  [GROUP VERIFY] %s images=%d anchors=%d", group.id, len(group.image_paths), len(group.anchors))
    raw = vlm_callable(GROUP_VERIFICATION_PROMPT, user_text, group.image_paths)
    if not raw:
        group.missing_evidence.append("Group verification returned an empty response.")
        group.confidence = 0.0
        return group
    parsed = _extract_json(raw or "") or {}
    group.verified_evidence = _as_str_list(parsed.get("verified_evidence", []))
    group.contradictions = _as_str_list(parsed.get("contradictions", []))
    group.missing_evidence = _as_str_list(parsed.get("missing_evidence", []))
    try:
        group.confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
    except Exception:
        group.confidence = 0.0

    if use_cache:
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "verified_evidence": group.verified_evidence,
                        "contradictions": group.contradictions,
                        "missing_evidence": group.missing_evidence,
                        "confidence": group.confidence,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("  [GROUP VERIFY] Cache write failed: %s", exc)
    return group



