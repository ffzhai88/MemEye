from __future__ import annotations

import logging
from typing import Dict, List, Set

from .indexes import cosine
from .schemas import EvidenceAnchor, EvidenceGroup

log = logging.getLogger(__name__)


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
    """
    计算两个 anchor 之间的交互得分。

    这个分数用于衡量一个 anchor 是否适合作为另一个 seed 的支持证据。
    它结合了三个核心部分：
      1. query 相关性：a 和 b 各自与问题向量的相似度，保证两者都与查询有关；
      2. 语义相似度：a 和 b 之间的 embedding 相似度，表示它们是否在语义上互补；
      3. provenance 与类型兼容性：考虑同轮、同图、同会话、时间接近以及证据类型之间的
         兼容性，奖励更可靠的组合。

    最终得分公式：
      query_rel_a * query_rel_b * (0.65 * semantic + provenance + type_bonus)

    解释：
      - query_rel_a 和 query_rel_b 是两个 anchor 各自与问题的相关度，保证这组证据整体与问题相关；
      - semantic 是 anchor 之间的内容相似度；
      - provenance 提高来自相近上下文的证据对的权重；
      - type_bonus 提高语义类型兼容的证据对的权重。

    这样设计的目的是：只有当两个 anchor 都与问题相关且它们之间有较强语义/上下文关联时，
    它们才会得到高交互分。
    """
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
    """
    基于查询驱动的贪心 evidence 组织函数。

    这个函数接收检索到的 evidence anchors，并把它们组织成若干个
    相互支持的 evidence group，方便后续生成回答时按主题聚合证据。

    组织流程:
    1. 选取候选 seed：按 anchor 与查询的相关度得分排序，取前 `max_groups * 2`
       个最相关的 anchor 作为 seed 候选。
    2. 对每个 seed，计算它与其他 anchor 的交互得分 `interaction_score`，
       交互得分综合考虑：
         - seed 与 query 的关系强度
         - other 与 query 的关系强度
         - seed 与 other 之间的语义相似度
         - provenance 奖励（同一轮、同一图像、同一会话、时间临近）
         - 类型兼容性奖励
    3. 选取和 seed 交互得分最高的若干 anchors，构成该 seed 的 group 成员。
       最多选取 `max_group_size - 1` 个 supporting anchors，加上 seed 自身。
    4. 如果组里只有一个弱 seed（score<=0），则认为该组支持不足并直接丢弃。
    5. 组分数由 seed score 和成员交互得分平均值组成，表示该 evidence group 的
       质量和相关性。
    6. 为每个组生成元信息：label、hypothesis、visual checks，以及图片路径列表。
    7. 使用贪心策略：每个 seed 只能构成一个 group，且最多返回 `max_groups` 个组。
    8. 最终按组分数降序返回 group 列表。

    参数:
        question: 问题文本，主要用于语义上下文，当前函数本身不直接使用。
        query_vec: 问题的 embedding 向量。
        anchors: 已检索到的 evidence anchors，通常按 query 相关度排序。
        round_order: 对话轮次的时间顺序，用于计算 temporal provenance。
        max_groups: 最多保留多少个 evidence group。
        max_group_size: 每个 group 最多包含多少个 anchor。
        max_group_images: group 中最多保留多少张图片路径。

    返回:
        按组分数降序排序的 EvidenceGroup 列表。
    """
    if not anchors:
        return []

    # Seed candidate 由与 query 最相关的 anchor 组成，允许的数量为 max_groups * 2。
    seeds = sorted(anchors, key=lambda a: a.score, reverse=True)[: max_groups * 2]
    used_seeds: Set[str] = set()
    groups: List[EvidenceGroup] = []

    for seed in seeds:
        if seed.id in used_seeds:
            continue

        # 计算该 seed 与所有其他 anchor 的交互得分。
        candidates: List[tuple[float, EvidenceAnchor]] = []
        for other in anchors:
            if other.id == seed.id:
                continue
            score = interaction_score(query_vec, seed, other, round_order)
            if score > 0:
                candidates.append((score, other))

        # 选取交互得分最高的支持 anchor 作为 group 成员。
        candidates.sort(key=lambda item: item[0], reverse=True)
        members = [seed] + [a for _, a in candidates[: max(0, max_group_size - 1)]]

        # 如果组里只有一个 anchor 且 seed 相关度不高，则认为该组不足以构成有价值的 evidence。
        if len(members) == 1 and seed.score <= 0:
            continue

        # 组得分等于 seed 得分加上 supporting anchor 平均交互得分。
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
