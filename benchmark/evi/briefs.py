from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ._utils import extract_json, retry_vlm_call
from .schemas import EvidenceAnchor, MemoryBrief, MemoryCandidate

if TYPE_CHECKING:
    from .vlm import VLMCallable

log = logging.getLogger(__name__)

_PROMPT_VERSION = "memory_assertion_v2"
_CACHE_DIR: Optional[str] = None

MEMORY_ASSERTION_SYSTEM_PROMPT = """You are converting one retrieved multimodal memory into a clean evidence assertion.

Given one candidate memory and the current question, decide whether the memory should be used, then write a factual assertion about what the memory itself shows or says.
Do not answer the final question. Do not choose a multiple-choice option.
Do not explain how the evidence supports the answer.

Return ONLY valid JSON:
{
  "relevance": "relevant|excluded|uncertain",
  "brief": "one concise factual assertion about this memory, without reasoning about the final answer",
  "key_evidence": ["short visible or textual fact grounded in this memory"],
  "confidence": 0.0
}

Guidelines:
- The brief is an evidence assertion, not a rationale. Avoid phrases like "contributes to the count", "supports the answer", "irrelevant to the question", or "therefore".
- For relevant memories, state only the observed facts needed later, such as identity, attributes, count, spatial relation, date, or change.
- For excluded memories, still keep the brief factual: state the observed mismatch, not why it helps answer.
- Mark relevance="excluded" when this memory matches surface words but is not actually about the queried entity, condition, time, or relation.
- Mark relevance="uncertain" when the memory may matter but the evidence is insufficient or visually ambiguous.
- Ground claims in the provided anchors and attached image when available.
"""


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_MEMORY_BRIEF_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_memory_briefs"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _shorten(text: object, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... <truncated {len(value) - max_chars} chars>"


def _cache_key(question_stem: str, candidate: MemoryCandidate, cache_namespace: str) -> str:
    raw = json.dumps(
        {
            "version": _PROMPT_VERSION,
            "cache_namespace": cache_namespace,
            "question": question_stem,
            "candidate_id": candidate.id,
            "round_id": candidate.round_id,
            "image_paths": candidate.image_paths,
            "anchors": [
                (a.id, a.evidence_type, a.text, a.region, round(float(a.score or 0.0), 6))
                for a in candidate.selected_anchors
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _as_str_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _clean_relevance(value: object) -> str:
    raw = str(value or "uncertain").strip().lower()
    if raw in {"relevant", "excluded", "uncertain"}:
        return raw
    if raw in {"yes", "support", "supports"}:
        return "relevant"
    if raw in {"no", "irrelevant", "exclude"}:
        return "excluded"
    return "uncertain"


def _format_anchor(anchor: EvidenceAnchor) -> str:
    loc = f" region={anchor.region}" if anchor.region else ""
    return (
        f"- id={anchor.id} type={anchor.evidence_type}{loc} "
        f"score={anchor.score:.4f}: {anchor.text}"
    )


def _candidate_prompt(question_stem: str, candidate: MemoryCandidate) -> str:
    anchor_lines = "\n".join(_format_anchor(anchor) for anchor in candidate.selected_anchors)
    round_text = candidate.round_text.strip()
    if len(round_text) > 1200:
        round_text = round_text[:1200] + "..."
    image_lines = "\n".join(f"- {path}" for path in candidate.image_paths) or "- none"
    return f"""Question stem:
{question_stem}

Candidate memory:
- candidate_id: {candidate.id}
- session_id: {candidate.session_id}
- round_id: {candidate.round_id}
- date: {candidate.date}
- retrieval_score: {candidate.score:.4f}

Candidate image paths:
{image_lines}

Local dialogue context:
{round_text or '(none)'}

Selected evidence anchors:
{anchor_lines or '(none)'}
"""


def make_fallback_brief(question_stem: str, candidate: MemoryCandidate, reason: str) -> MemoryBrief:
    evidence = [anchor.text for anchor in candidate.selected_anchors[:4]]
    anchor_text = "; ".join(evidence) if evidence else "no selected evidence anchors"
    brief = (
        f"This memory from {candidate.round_id} has uncertain usable evidence because {reason}. "
        f"Available observed facts: {anchor_text}."
    )
    return MemoryBrief(
        candidate_id=candidate.id,
        session_id=candidate.session_id,
        round_id=candidate.round_id,
        date=candidate.date,
        image_paths=list(candidate.image_paths),
        relevance="uncertain",
        brief=brief,
        key_evidence=evidence,
        confidence=0.0,
        score=candidate.score,
    )


def generate_memory_brief(
    question_stem: str,
    candidate: MemoryCandidate,
    vlm_callable: "VLMCallable",
    use_cache: bool = True,
    cache_namespace: str = "default",
) -> MemoryBrief:
    cache_file: Optional[Path] = None
    if use_cache:
        key = _cache_key(question_stem, candidate, cache_namespace)
        cache_file = Path(_cache_dir()) / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                brief = str(data.get("brief", "")).strip()
                if brief:
                    log.info("  [ASSERTION CACHE] HIT candidate=%s", candidate.id)
                    return MemoryBrief(
                        candidate_id=candidate.id,
                        session_id=candidate.session_id,
                        round_id=candidate.round_id,
                        date=candidate.date,
                        image_paths=list(candidate.image_paths),
                        relevance=_clean_relevance(data.get("relevance")),
                        brief=brief,
                        key_evidence=_as_str_list(data.get("key_evidence", [])),
                        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
                        score=candidate.score,
                    )
            except Exception as exc:
                log.warning("  [ASSERTION CACHE] read failed candidate=%s error=%s", candidate.id, exc)

    user_text = _candidate_prompt(question_stem, candidate)
    log.info(
        "  [ASSERTION] candidate=%s round=%s anchors=%d images=%d",
        candidate.id,
        candidate.round_id,
        len(candidate.selected_anchors),
        len(candidate.image_paths),
    )
    log.debug("[ASSERTION PROMPT] candidate=%s\n%s", candidate.id, _shorten(user_text, 8000))
    raw = retry_vlm_call(
        lambda: vlm_callable(MEMORY_ASSERTION_SYSTEM_PROMPT, user_text, candidate.image_paths),
        label=f"memory-assertion {candidate.id}",
    )
    log.debug("[ASSERTION RAW] candidate=%s\n%s", candidate.id, _shorten(raw, 8000))
    if not raw:
        log.warning("  [ASSERTION] empty response candidate=%s", candidate.id)
        return make_fallback_brief(question_stem, candidate, "the brief model returned an empty response")

    parsed = extract_json(raw) or {}
    log.debug("[ASSERTION PARSED] candidate=%s data=%s", candidate.id, _shorten(parsed, 4000))
    brief_text = str(parsed.get("brief", "")).strip()
    if not brief_text:
        log.warning("  [ASSERTION] invalid JSON/empty brief candidate=%s", candidate.id)
        return make_fallback_brief(question_stem, candidate, "the brief model did not return a usable brief")

    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
    except Exception:
        confidence = 0.0
    brief = MemoryBrief(
        candidate_id=candidate.id,
        session_id=candidate.session_id,
        round_id=candidate.round_id,
        date=candidate.date,
        image_paths=list(candidate.image_paths),
        relevance=_clean_relevance(parsed.get("relevance")),
        brief=brief_text,
        key_evidence=_as_str_list(parsed.get("key_evidence", [])),
        confidence=confidence,
        score=candidate.score,
    )

    if use_cache and cache_file is not None:
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "version": _PROMPT_VERSION,
                        "cache_namespace": cache_namespace,
                        "candidate_id": candidate.id,
                        "relevance": brief.relevance,
                        "brief": brief.brief,
                        "key_evidence": brief.key_evidence,
                        "confidence": brief.confidence,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("  [ASSERTION] Cache write failed: %s", exc)
    return brief


def generate_memory_briefs(
    question_stem: str,
    candidates: List[MemoryCandidate],
    vlm_callable: "VLMCallable",
    use_cache: bool = True,
    cache_namespace: str = "default",
) -> List[MemoryBrief]:
    briefs: List[MemoryBrief] = []
    for candidate in candidates:
        briefs.append(
            generate_memory_brief(
                question_stem,
                candidate,
                vlm_callable,
                use_cache=use_cache,
                cache_namespace=cache_namespace,
            )
        )
    log.info("EVI memory assertions generated: %d", len(briefs))
    for idx, brief in enumerate(briefs):
        log.info(
            "  assertion[%02d] candidate=%s relevance=%s confidence=%.2f round=%s",
            idx + 1,
            brief.candidate_id,
            brief.relevance,
            brief.confidence,
            brief.round_id,
        )
        log.info("    %s", brief.brief.replace("\n", " ")[:260])
    return briefs
