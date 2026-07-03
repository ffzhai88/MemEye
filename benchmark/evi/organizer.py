from __future__ import annotations

from typing import Dict, List, Set

from .indexes import cosine
from .schemas import EvidenceAnchor, EvidenceGroup


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


_COMPATIBLE_TYPES: set[tuple[str, str]] = {
    _pair_key("identity", "attribute"),
    _pair_key("identity", "entity"),
    _pair_key("entity", "attribute"),
    _pair_key("entity", "spatial"),
    _pair_key("entity", "relation"),
    _pair_key("text", "structured_visual"),
    _pair_key("text", "attribute"),
    _pair_key("temporal", "identity"),
    _pair_key("temporal", "relation"),
    _pair_key("scene", "entity"),
    _pair_key("scene", "spatial"),
    _pair_key("structured_visual", "spatial"),
}


def _temporal_index(round_order: List[str]) -> Dict[str, int]:
    return {rid: idx for idx, rid in enumerate(round_order)}


def provenance_bonus(a: EvidenceAnchor, b: EvidenceAnchor, round_order: List[str]) -> float:
    bonus = 0.0
    if a.round_id == b.round_id:
        bonus += 0.25
    if a.image_path and b.image_path and a.image_path == b.image_path:
        bonus += 0.25
    if a.session_id and a.session_id == b.session_id:
        bonus += 0.15
    order = _temporal_index(round_order)
    ia = order.get(a.round_id)
    ib = order.get(b.round_id)
    if ia is not None and ib is not None:
        gap = abs(ia - ib)
        if gap == 1:
            bonus += 0.08
        elif gap <= 3:
            bonus += 0.04
    return bonus


def type_compatibility(a: EvidenceAnchor, b: EvidenceAnchor) -> float:
    bonus = 0.0
    if a.evidence_type == b.evidence_type:
        bonus += 0.08
    if _pair_key(a.evidence_type, b.evidence_type) in _COMPATIBLE_TYPES:
        bonus += 0.12
    return bonus


def interaction_score(
    query_vec: List[float],
    a: EvidenceAnchor,
    b: EvidenceAnchor,
    round_order: List[str],
) -> float:
    query_rel_a = max(0.0, a.score or cosine(query_vec, a.vector))
    query_rel_b = max(0.0, b.score or cosine(query_vec, b.vector))
    semantic = max(0.0, cosine(a.vector, b.vector))
    provenance = provenance_bonus(a, b, round_order)
    type_bonus = type_compatibility(a, b)
    return query_rel_a * query_rel_b * (0.65 * semantic + provenance + type_bonus)


def _collect_images(anchors: List[EvidenceAnchor], max_images: int) -> List[str]:
    images: List[str] = []
    seen: Set[str] = set()
    for anchor in anchors:
        if anchor.image_path and anchor.image_path not in seen:
            images.append(anchor.image_path)
            seen.add(anchor.image_path)
        if len(images) >= max_images:
            break
    return images


def _auto_label(seed: EvidenceAnchor, members: List[EvidenceAnchor]) -> str:
    types = sorted({a.evidence_type for a in members})
    text = seed.text.strip().replace("\n", " ")
    if len(text) > 90:
        text = text[:87] + "..."
    return f"{seed.evidence_type} seed with {len(members)} anchors ({', '.join(types)}): {text}"


def _generate_hypothesis(seed: EvidenceAnchor, members: List[EvidenceAnchor]) -> str:
    """Generate a meaningful evidence-group hypothesis from the group content."""
    types = sorted({a.evidence_type for a in members})
    sessions = {a.session_id for a in members if a.session_id}
    rounds = sorted({a.round_id for a in members})
    has_images = any(a.image_path for a in members)

    parts: list[str] = []
    if len(rounds) > 1:
        parts.append(f"evidence spanning {len(rounds)} rounds")
    elif rounds:
        parts.append("evidence from round " + rounds[0])

    if len(sessions) > 1:
        parts.append(f"across {len(sessions)} sessions")
    elif sessions:
        parts.append(f"in session {next(iter(sessions))}")

    parts.append(f"types={{{','.join(types)}}}")
    if has_images:
        parts.append("with visual evidence")

    seed_text = seed.text.strip().replace("\n", " ")[:100]
    parts.append(f'seed: "{seed_text}"')

    hypothesis = " | ".join(parts)
    return f"This group contains {hypothesis}."


def _visual_checks(group: EvidenceGroup) -> List[str]:
    types = {anchor.evidence_type for anchor in group.anchors}
    checks: List[str] = []
    if types.intersection({"text", "structured_visual"}):
        checks.append("verify readable text, symbols, numbers, labels, and organized visual information represented by this group")
    if types.intersection({"spatial", "relation"}):
        checks.append("verify spatial positions and entity/layout relationships represented by this group")
    if types.intersection({"entity", "attribute"}):
        checks.append("verify entity presence, counts, colors, material, shape, and fine visual attributes represented by this group")
    if "identity" in types:
        checks.append("verify stable identity cues and whether visually similar entities are same or distinct")
    if "temporal" in types:
        checks.append("verify state, change, or sequence evidence represented by this group")
    if not checks:
        checks.append("verify the visual evidence represented by this group")
    return checks[:6]


def organize_evidence(
    question: str,
    query_vec: List[float],
    anchors: List[EvidenceAnchor],
    round_order: List[str],
    max_groups: int = 6,
    max_group_size: int = 12,
    max_group_images: int = 4,
) -> List[EvidenceGroup]:
    """Greedy query-driven memory organization over retrieved anchors."""
    if not anchors:
        return []

    seeds = sorted(anchors, key=lambda a: a.score, reverse=True)[: max_groups * 2]
    used_seeds: Set[str] = set()
    groups: List[EvidenceGroup] = []

    for seed in seeds:
        if seed.id in used_seeds:
            continue
        candidates: List[tuple[float, EvidenceAnchor]] = []
        for other in anchors:
            if other.id == seed.id:
                continue
            score = interaction_score(query_vec, seed, other, round_order)
            if score > 0:
                candidates.append((score, other))
        candidates.sort(key=lambda item: item[0], reverse=True)
        members = [seed] + [a for _, a in candidates[: max(0, max_group_size - 1)]]
        if len(members) == 1 and seed.score <= 0:
            continue
        group_score = seed.score + sum(s for s, _ in candidates[: max(0, max_group_size - 1)]) / max(1, len(members))
        group = EvidenceGroup(
            id=f"group_{len(groups)}",
            seed_anchor_id=seed.id,
            anchors=members,
            score=group_score,
            image_paths=_collect_images(members, max_group_images),
        )
        group.group_label = _auto_label(seed, members)
        group.group_hypothesis = _generate_hypothesis(seed, members)
        group.needed_visual_checks = _visual_checks(group)
        groups.append(group)
        used_seeds.add(seed.id)
        if len(groups) >= max_groups:
            break

    groups.sort(key=lambda g: g.score, reverse=True)
    return groups
