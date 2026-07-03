from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .schemas import EvidenceGroup
from ._utils import extract_json, retry_vlm_call

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
    """
    对单个 evidence group 进行可视化证据校验。

    该函数的主要作用是：
    1. 检查 group 是否已经包含图片，如果没有则直接标记为缺失图像；
    2. 先尝试读取本地缓存，避免重复调用大模型；
    3. 构造校验 prompt，将问题、group 元信息、需要关注的 visual checks、
       以及 evidence anchors 作为输入；
    4. 调用 VLM 观察 raw image，并返回结构化 JSON 结果；
    5. 解析 JSON 并填充 group 的 verified_evidence、contradictions、missing_evidence、confidence。

    参数:
        question_stem: 问题干，用于提示模型当前验证的上下文。
        group: 需要校验的 evidence group，对象中应包含 anchors、image_paths 等字段。
        vlm_callable: 可调用的视觉语言模型接口，用于执行图像+文本校验。
        use_cache: 是否使用本地缓存，以减少重复验证开销。

    返回:
        更新后的 EvidenceGroup，包含校验后的 verified_evidence、contradictions、missing_evidence、confidence。
    """
    if not group.image_paths:
        group.missing_evidence.append("No raw image was available for this evidence group.")
        return group

    # 先尝试缓存，避免对同样的 group 重复请求 VLM。
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

    # 构造用于 VLM 的 prompt 输入，包含问题、group label、group hypothesis、
    # 需要验证的视觉检查项以及前 16 个 evidence anchors。
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
    raw = retry_vlm_call(
        lambda: vlm_callable(GROUP_VERIFICATION_PROMPT, user_text, group.image_paths),
        label=f"group-verify {group.id}",
    )

    if not raw:
        group.missing_evidence.append("Group verification returned an empty response.")
        group.confidence = 0.0
        return group

    # 从 VLM 输出中抽取 JSON，支持模型返回带噪音文本。
    parsed = extract_json(raw or "") or {}
    group.verified_evidence = _as_str_list(parsed.get("verified_evidence", []))
    group.contradictions = _as_str_list(parsed.get("contradictions", []))
    group.missing_evidence = _as_str_list(parsed.get("missing_evidence", []))
    try:
        group.confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
    except Exception:
        group.confidence = 0.0

    # 将验证结果写入缓存，方便后续复用同一 evidence group 的校验结果。
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



