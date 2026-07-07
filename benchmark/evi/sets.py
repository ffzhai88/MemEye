from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from .schemas import EpisodicMemorySet, EvidenceAnchor

log = logging.getLogger(__name__)


def _dedupe_anchor_key(anchor: EvidenceAnchor) -> Tuple[str, str]:
    return (anchor.id, " ".join(str(anchor.text or "").lower().split())[:180])


def _sort_anchors(anchors: Iterable[EvidenceAnchor]) -> List[EvidenceAnchor]:
    return sorted(list(anchors), key=lambda a: (a.score or 0.0, a.confidence or 0.0), reverse=True)


def _select_round_anchors(
    round_id: str,
    retrieved_by_round: Dict[str, List[EvidenceAnchor]],
    all_by_round: Dict[str, List[EvidenceAnchor]],
    max_anchors: int,
) -> List[EvidenceAnchor]:
    selected: List[EvidenceAnchor] = []
    seen: set[Tuple[str, str]] = set()

    def add(anchor: EvidenceAnchor) -> None:
        if len(selected) >= max_anchors:
            return
        key = _dedupe_anchor_key(anchor)
        if key in seen:
            return
        selected.append(anchor)
        seen.add(key)

    retrieved = _sort_anchors(retrieved_by_round.get(round_id, []))
    all_anchors = _sort_anchors(all_by_round.get(round_id, []))

    for anchor in retrieved:
        add(anchor)
    if len(selected) >= max_anchors:
        return selected

    seen_types = {anchor.evidence_type for anchor in selected}
    for anchor in all_anchors:
        if anchor.evidence_type in seen_types:
            continue
        before = len(selected)
        add(anchor)
        if len(selected) > before:
            seen_types.add(anchor.evidence_type)
        if len(selected) >= max_anchors:
            return selected

    for anchor in all_anchors:
        add(anchor)
        if len(selected) >= max_anchors:
            break
    return selected


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_episodic_memory_sets(
    retrieved: List[EvidenceAnchor],
    *,
    session_rounds: Dict[str, List[str]],
    round_text: Dict[str, str],
    round_images: Dict[str, List[str]],
    round_anchors: Dict[str, List[EvidenceAnchor]],
    max_sets: int = 4,
    window_before: int = 1,
    window_after: int = 1,
    max_rounds_per_set: int = 5,
    max_anchors_per_round: int = 6,
) -> List[EpisodicMemorySet]:
    """Assemble query hits into ordered local episodes."""
    retrieved_by_round: Dict[str, List[EvidenceAnchor]] = {}
    for anchor in retrieved:
        retrieved_by_round.setdefault(anchor.round_id, []).append(anchor)

    hit_indices: Dict[str, List[int]] = {}
    for sid, rounds in session_rounds.items():
        pos = {rid: idx for idx, rid in enumerate(rounds)}
        for rid in retrieved_by_round:
            if rid in pos:
                hit_indices.setdefault(sid, []).append(pos[rid])

    sets: List[EpisodicMemorySet] = []
    for sid, indices in hit_indices.items():
        rounds = session_rounds.get(sid, [])
        if not rounds:
            continue
        intervals = []
        for idx in sorted(set(indices)):
            start = max(0, idx - max(0, window_before))
            end = min(len(rounds) - 1, idx + max(0, window_after))
            intervals.append((start, end))

        for start, end in _merge_intervals(intervals):
            if max_rounds_per_set > 0 and end - start + 1 > max_rounds_per_set:
                scored = []
                for idx in range(start, end + 1):
                    rid = rounds[idx]
                    score = max((a.score or 0.0 for a in retrieved_by_round.get(rid, [])), default=0.0)
                    scored.append((score, idx))
                _, center = max(scored, key=lambda item: item[0])
                half = max(0, max_rounds_per_set // 2)
                start = max(0, center - half)
                end = min(len(rounds) - 1, start + max_rounds_per_set - 1)
                start = max(0, end - max_rounds_per_set + 1)

            set_rounds = rounds[start : end + 1]
            set_retrieved = [a for rid in set_rounds for a in retrieved_by_round.get(rid, [])]
            if not set_retrieved:
                continue
            selected_by_round = {
                rid: _select_round_anchors(rid, retrieved_by_round, round_anchors, max_anchors_per_round)
                for rid in set_rounds
            }
            ordered_scores = sorted((a.score or 0.0 for a in set_retrieved), reverse=True)
            top3 = ordered_scores[:3]
            max_score = ordered_scores[0] if ordered_scores else 0.0
            top3_mean = sum(top3) / max(1, len(top3))
            hit_round_count = len({anchor.round_id for anchor in set_retrieved})
            type_diversity = len({anchor.evidence_type for anchor in set_retrieved})
            score = (
                1.20 * max_score
                + 0.50 * top3_mean
                + 0.035 * hit_round_count
                + 0.01 * min(type_diversity, 6)
            )
            date = set_retrieved[0].date if set_retrieved else ""
            set_id = f"episode::{sid}::{set_rounds[0]}..{set_rounds[-1]}"
            sets.append(
                EpisodicMemorySet(
                    id=set_id,
                    session_id=sid,
                    date=date,
                    round_ids=set_rounds,
                    round_text={rid: round_text.get(rid, "") for rid in set_rounds},
                    round_images={rid: list(round_images.get(rid, [])) for rid in set_rounds},
                    round_anchors=selected_by_round,
                    retrieved_anchors=_sort_anchors(set_retrieved),
                    score=score,
                )
            )

    sets.sort(key=lambda item: item.score, reverse=True)
    out = sets[:max_sets]
    log.info("EVI episodic memory sets built: retrieved=%d kept=%d", len(retrieved), len(out))
    for idx, memory_set in enumerate(out, start=1):
        log.info(
            "  memory_set[%02d] id=%s session=%s rounds=%s score=%.4f retrieved=%d",
            idx,
            memory_set.id,
            memory_set.session_id,
            " -> ".join(memory_set.round_ids),
            memory_set.score,
            len(memory_set.retrieved_anchors),
        )
    return out


def build_session_memory_sets(
    retrieved: List[EvidenceAnchor],
    *,
    session_rounds: Dict[str, List[str]],
    round_text: Dict[str, str],
    round_images: Dict[str, List[str]],
    round_anchors: Dict[str, List[EvidenceAnchor]],
    max_rounds: int = 20,
    max_anchors_per_round: int = 6,
) -> List[EpisodicMemorySet]:
    """Group top retrieved rounds by session without neighbor expansion."""
    retrieved_by_round: Dict[str, List[EvidenceAnchor]] = {}
    ordered_rounds: List[str] = []
    seen_rounds: set[str] = set()
    for anchor in retrieved:
        retrieved_by_round.setdefault(anchor.round_id, []).append(anchor)
        if anchor.round_id not in seen_rounds:
            ordered_rounds.append(anchor.round_id)
            seen_rounds.add(anchor.round_id)
        if len(ordered_rounds) >= max_rounds:
            break

    kept_rounds = set(ordered_rounds)
    sets: List[EpisodicMemorySet] = []
    for sid, rounds in session_rounds.items():
        set_rounds = [rid for rid in rounds if rid in kept_rounds]
        if not set_rounds:
            continue
        set_retrieved = [a for rid in set_rounds for a in retrieved_by_round.get(rid, [])]
        selected_by_round = {
            rid: _select_round_anchors(rid, retrieved_by_round, round_anchors, max_anchors_per_round)
            for rid in set_rounds
        }
        best_scores = [
            max((a.score or 0.0 for a in retrieved_by_round.get(rid, [])), default=0.0)
            for rid in set_rounds
        ]
        score = sum(best_scores) / max(1, len(best_scores))
        date = set_retrieved[0].date if set_retrieved else ""
        set_id = f"session::{sid}"
        sets.append(
            EpisodicMemorySet(
                id=set_id,
                session_id=sid,
                date=date,
                round_ids=set_rounds,
                round_text={rid: round_text.get(rid, "") for rid in set_rounds},
                round_images={rid: list(round_images.get(rid, [])) for rid in set_rounds},
                round_anchors=selected_by_round,
                retrieved_anchors=_sort_anchors(set_retrieved),
                score=score,
            )
        )

    rank_by_round = {rid: idx for idx, rid in enumerate(ordered_rounds)}
    sets.sort(key=lambda item: min(rank_by_round.get(rid, 10**9) for rid in item.round_ids))
    log.info("EVI session memory sets built: rounds=%d sets=%d", len(ordered_rounds), len(sets))
    for idx, memory_set in enumerate(sets, start=1):
        log.info(
            "  session_memory_set[%02d] id=%s session=%s rounds=%s score=%.4f retrieved=%d",
            idx,
            memory_set.id,
            memory_set.session_id,
            " -> ".join(memory_set.round_ids),
            memory_set.score,
            len(memory_set.retrieved_anchors),
        )
    return sets
