from __future__ import annotations

import hashlib
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from .schemas import EvidenceAnchor, MemoryCandidate

log = logging.getLogger(__name__)


def _candidate_key(anchor: EvidenceAnchor) -> Tuple[str, str]:
    return (anchor.round_id, anchor.image_path or "")


def _candidate_id(round_id: str, image_path: Optional[str]) -> str:
    if not image_path:
        return f"candidate::{round_id}::text"
    suffix = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:10]
    return f"candidate::{round_id}::{suffix}"


def _dedupe_text_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())[:180]


def _sort_anchors(anchors: Iterable[EvidenceAnchor]) -> List[EvidenceAnchor]:
    return sorted(list(anchors), key=lambda a: a.score or 0.0, reverse=True)


def candidate_score(anchors: List[EvidenceAnchor]) -> float:
    if not anchors:
        return 0.0
    ordered = _sort_anchors(anchors)
    max_score = ordered[0].score or 0.0
    top_scores = [a.score or 0.0 for a in ordered[:4]]
    top_mean = sum(top_scores) / max(1, len(top_scores))
    type_bonus = min(0.12, 0.02 * len({a.evidence_type for a in anchors}))
    density_bonus = min(0.08, 0.01 * max(0, len(anchors) - 1))
    return max_score + 0.45 * top_mean + type_bonus + density_bonus


def select_candidate_anchors(
    anchors: Iterable[EvidenceAnchor],
    max_anchors: int = 8,
) -> List[EvidenceAnchor]:
    """Select a compact, diverse anchor set for one memory candidate."""
    ordered = _sort_anchors(anchors)
    selected: List[EvidenceAnchor] = []
    seen_texts: set[str] = set()
    seen_types: set[str] = set()

    def add(anchor: EvidenceAnchor) -> bool:
        if len(selected) >= max_anchors:
            return False
        key = _dedupe_text_key(anchor.text)
        if key and key in seen_texts:
            return False
        selected.append(anchor)
        seen_texts.add(key)
        seen_types.add(anchor.evidence_type)
        return True

    # First pass: keep one strong anchor per evidence type.
    for anchor in ordered:
        if anchor.evidence_type in seen_types:
            continue
        add(anchor)
        if len(selected) >= max_anchors:
            return selected

    # Second pass: fill remaining slots by score after text dedupe.
    for anchor in ordered:
        if anchor in selected:
            continue
        add(anchor)
        if len(selected) >= max_anchors:
            break
    return selected


def _make_candidate(
    *,
    round_id: str,
    image_path: Optional[str],
    anchors: List[EvidenceAnchor],
    round_text: Dict[str, str],
    max_candidate_anchors: int,
) -> Optional[MemoryCandidate]:
    if not anchors:
        return None
    ordered = _sort_anchors(anchors)
    first = ordered[0]
    images = [image_path] if image_path else []
    score = candidate_score(ordered)
    selected = select_candidate_anchors(ordered, max_anchors=max_candidate_anchors)
    return MemoryCandidate(
        id=_candidate_id(round_id, image_path),
        session_id=first.session_id,
        round_id=round_id,
        date=first.date,
        image_path=image_path,
        image_paths=images,
        round_text=round_text.get(round_id, ""),
        anchors=ordered,
        selected_anchors=selected,
        score=score,
    )


def consolidate_candidates(
    anchors: List[EvidenceAnchor],
    round_text: Dict[str, str],
    max_candidates: int = 12,
    max_candidate_anchors: int = 8,
) -> List[MemoryCandidate]:
    """Collapse retrieved anchors into mostly non-overlapping memory candidates.

    Image-bearing anchors define the candidate when a round has visual evidence.
    Text-only anchors from the same round are folded into those image candidates as
    local context instead of becoming a duplicate standalone candidate.
    """
    image_buckets: Dict[Tuple[str, str], List[EvidenceAnchor]] = {}
    text_buckets: Dict[str, List[EvidenceAnchor]] = {}
    for anchor in anchors:
        if anchor.image_path:
            image_buckets.setdefault(_candidate_key(anchor), []).append(anchor)
        else:
            text_buckets.setdefault(anchor.round_id, []).append(anchor)

    candidates: List[MemoryCandidate] = []
    rounds_with_image_candidates: set[str] = set()
    image_candidate_count = 0
    text_candidate_count = 0

    for (round_id, image_path), image_anchors in image_buckets.items():
        # Keep the visual candidate as the main unit, with same-round text as context.
        combined = list(image_anchors) + list(text_buckets.get(round_id, []))
        candidate = _make_candidate(
            round_id=round_id,
            image_path=image_path or None,
            anchors=combined,
            round_text=round_text,
            max_candidate_anchors=max_candidate_anchors,
        )
        if candidate is not None:
            candidates.append(candidate)
            rounds_with_image_candidates.add(round_id)
            image_candidate_count += 1

    for round_id, text_anchors in text_buckets.items():
        if round_id in rounds_with_image_candidates:
            continue
        candidate = _make_candidate(
            round_id=round_id,
            image_path=None,
            anchors=list(text_anchors),
            round_text=round_text,
            max_candidate_anchors=max_candidate_anchors,
        )
        if candidate is not None:
            candidates.append(candidate)
            text_candidate_count += 1

    candidates.sort(key=lambda c: c.score, reverse=True)
    out = candidates[:max_candidates]
    log.info(
        "EVI candidates consolidated: anchors=%d image_candidates=%d text_candidates=%d kept=%d",
        len(anchors),
        image_candidate_count,
        text_candidate_count,
        len(out),
    )
    for idx, candidate in enumerate(out):
        log.info(
            "  candidate[%02d] id=%s round=%s score=%.4f anchors=%d selected=%d images=%d",
            idx + 1,
            candidate.id,
            candidate.round_id,
            candidate.score,
            len(candidate.anchors),
            len(candidate.selected_anchors),
            len(candidate.image_paths),
        )
    return out
