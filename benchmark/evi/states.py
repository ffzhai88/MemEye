from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ._utils import extract_json, retry_vlm_call
from .schemas import EpisodicMemorySet, EpisodicState

if TYPE_CHECKING:
    from .vlm import VLMCallable

log = logging.getLogger(__name__)

_PROMPT_VERSION = "question_aligned_episode_evidence_v4"
_CACHE_DIR: Optional[str] = None

EPISODIC_STATE_SYSTEM_PROMPT = """Inspect one ordered memory episode for a multimodal memory agent.

Input: a question with options, several dialogue rounds, and attached images.
Do not answer the question or choose an option.

First decide how this episode aligns with the question's referring descriptions.
Then extract concrete facts from this episode under that alignment.

Return ONLY this JSON object:
{
  "relevance": "relevant|uncertain|excluded",
  "episode_alignment": "which object/event/episode in the question this memory set appears to match; use 'unmatched' if none",
  "evidence_facts": [
    {"round_id": "...", "fact": "concrete visual/dialogue fact", "source": "image|dialogue|both", "role": "alignment|answer_evidence|context|uncertain"}
  ],
  "uncertainties": ["only if a relevant detail is ambiguous or missing"],
  "confidence": 0.0
}

Rules:
- Options clarify what information may matter, but they are not evidence.
- Do not copy option wording as a fact unless that detail is directly visible or stated in this episode.
- If the episode matches only a non-answer referent, contrast referent, or distractor, say that in episode_alignment.
- Do not turn facts from a mismatched episode into answer_evidence.
- Use relevance="excluded" when the episode is unrelated or only matches distractor wording.
- If related, keep facts short, concrete, and separated by round.
- Use dialogue as context, but ground visual claims in images.
"""


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_STATE_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_states"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _shorten(text: object, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... <truncated {len(value) - max_chars} chars>"


def _as_str_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _evidence_fact_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return _as_str_list(value)
    out: List[str] = []
    for item in value:
        if isinstance(item, dict):
            rid = str(item.get("round_id", "")).strip()
            source = str(item.get("source", "")).strip()
            role = str(item.get("role", item.get("question_role", ""))).strip()
            fact = str(item.get("fact", "")).strip()
            if not fact:
                continue
            prefix_parts = [part for part in (rid, source, role) if part]
            prefix = "/".join(prefix_parts)
            out.append(f"{prefix}: {fact}" if prefix else fact)
        else:
            text = str(item).strip()
            if text:
                out.append(text)
    return out


def _clean_relevance(value: object) -> str:
    raw = str(value or "uncertain").strip().lower()
    if raw in {"relevant", "excluded", "uncertain"}:
        return raw
    if raw in {"yes", "support", "supports"}:
        return "relevant"
    if raw in {"no", "irrelevant", "exclude"}:
        return "excluded"
    return "uncertain"



def state_image_paths(memory_set: EpisodicMemorySet, max_images: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for rid in memory_set.round_ids:
        for path in memory_set.round_images.get(rid, []):
            if path and path not in seen:
                out.append(path)
                seen.add(path)
            if len(out) >= max_images:
                return out
    return out


def _memory_set_prompt(
    question_stem: str,
    memory_set: EpisodicMemorySet,
    attached_images: List[str],
    max_prompt_chars: int,
    question_context: Optional[str] = None,
) -> str:
    attached = set(attached_images)
    question_text = str(question_context or question_stem or "")
    lines: List[str] = []
    lines.append("Question to answer, including options if provided:")
    lines.append(question_text)
    if question_stem and question_text.strip() != str(question_stem).strip():
        lines.append("")
        lines.append("Question stem used for retrieval:")
        lines.append(str(question_stem))
    lines.append("")
    lines.append(f"Memory set: {memory_set.id}")
    lines.append(f"Session: {memory_set.session_id}")
    lines.append(f"Date: {memory_set.date}")
    lines.append("Round order: " + " -> ".join(memory_set.round_ids))
    lines.append("")
    lines.append("Ordered rounds with attached images:")
    for rid in memory_set.round_ids:
        lines.append(f"[{rid}]")
        text = " ".join(str(memory_set.round_text.get(rid, "")).split())
        if text:
            lines.append("Dialogue: " + _shorten(text, 900))
        round_images = [path for path in memory_set.round_images.get(rid, []) if path in attached]
        if round_images:
            lines.append("Attached images: " + "; ".join(round_images))
        else:
            lines.append("Attached images: none")
        lines.append("")
    prompt = "\n".join(lines)
    return _shorten(prompt, max_prompt_chars)


def _cache_key(
    question_stem: str,
    memory_set: EpisodicMemorySet,
    image_paths: List[str],
    cache_namespace: str,
    question_context: Optional[str] = None,
) -> str:
    raw = json.dumps(
        {
            "version": _PROMPT_VERSION,
            "cache_namespace": cache_namespace,
            "question_stem": question_stem,
            "question_context": question_context or question_stem,
            "set_id": memory_set.id,
            "round_ids": memory_set.round_ids,
            "images": image_paths,
            "round_text": {rid: memory_set.round_text.get(rid, "") for rid in memory_set.round_ids},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def make_fallback_state(memory_set: EpisodicMemorySet, reason: str) -> EpisodicState:
    return EpisodicState(
        set_id=memory_set.id,
        session_id=memory_set.session_id,
        date=memory_set.date,
        round_ids=list(memory_set.round_ids),
        image_paths=state_image_paths(memory_set, 99),
        relevance="uncertain",
        episode_alignment="unknown; readout unavailable",
        answer_relevant_facts=[],
        uncertainties=[f"Episode evidence readout was not available because {reason}."],
        confidence=0.0,
        score=memory_set.score,
    )


def _log_state_result(prefix: str, state: EpisodicState) -> None:
    log.info(
        "%s set=%s relevance=%s confidence=%.2f alignment=%s rounds=%s facts=%d uncertainties=%d",
        prefix,
        state.set_id,
        state.relevance,
        state.confidence,
        state.episode_alignment or "unspecified",
        " -> ".join(state.round_ids),
        len(state.answer_relevant_facts),
        len(state.uncertainties),
    )
    if state.answer_relevant_facts:
        for idx, fact in enumerate(state.answer_relevant_facts, start=1):
            log.info("%s fact[%02d]: %s", prefix, idx, " ".join(str(fact).split()))
    else:
        log.info("%s facts: <none>", prefix)
    if state.uncertainties:
        for idx, item in enumerate(state.uncertainties, start=1):
            log.info("%s uncertainty[%02d]: %s", prefix, idx, " ".join(str(item).split()))


def read_episodic_state(
    question_stem: str,
    memory_set: EpisodicMemorySet,
    vlm_callable: "VLMCallable",
    *,
    use_cache: bool = True,
    cache_namespace: str = "default",
    use_images: bool = True,
    max_images: int = 4,
    max_prompt_chars: int = 10000,
    question_context: Optional[str] = None,
) -> EpisodicState:
    images = state_image_paths(memory_set, max_images) if use_images else []
    cache_file: Optional[Path] = None
    if use_cache:
        key = _cache_key(question_stem, memory_set, images, cache_namespace, question_context)
        cache_file = Path(_cache_dir()) / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                log.info("  [STATE CACHE] HIT set=%s", memory_set.id)
                state = EpisodicState(
                    set_id=memory_set.id,
                    session_id=memory_set.session_id,
                    date=memory_set.date,
                    round_ids=list(memory_set.round_ids),
                    image_paths=list(images),
                    relevance=_clean_relevance(data.get("relevance")),
                    episode_alignment=str(data.get("episode_alignment", "")).strip(),
                    answer_relevant_facts=_evidence_fact_list(data.get("evidence_facts", data.get("answer_relevant_facts", []))),
                    uncertainties=_as_str_list(data.get("uncertainties", [])),
                    confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
                    score=memory_set.score,
                )
                _log_state_result("  [STATE CACHE RESULT]", state)
                return state
            except Exception as exc:
                log.warning("  [STATE CACHE] read failed set=%s error=%s", memory_set.id, exc)

    user_text = _memory_set_prompt(question_stem, memory_set, images, max_prompt_chars, question_context)
    log.info(
        "  [EVIDENCE READOUT] set=%s rounds=%s images=%d",
        memory_set.id,
        " -> ".join(memory_set.round_ids),
        len(images),
    )
    log.debug("[STATE PROMPT] set=%s\n%s", memory_set.id, _shorten(user_text, max_prompt_chars))
    raw = retry_vlm_call(
        lambda: vlm_callable(EPISODIC_STATE_SYSTEM_PROMPT, user_text, images),
        label=f"episode-evidence {memory_set.id}",
    )
    log.debug("[STATE RAW] set=%s\n%s", memory_set.id, _shorten(raw, 8000))
    if not raw:
        return make_fallback_state(memory_set, "the state model returned an empty response")

    parsed = extract_json(raw) or {}
    if not isinstance(parsed, dict):
        return make_fallback_state(memory_set, "the state model did not return JSON")
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
    except Exception:
        confidence = 0.0
    state = EpisodicState(
        set_id=memory_set.id,
        session_id=memory_set.session_id,
        date=memory_set.date,
        round_ids=list(memory_set.round_ids),
        image_paths=list(images),
        relevance=_clean_relevance(parsed.get("relevance")),
        episode_alignment=str(parsed.get("episode_alignment", "")).strip(),
        answer_relevant_facts=_evidence_fact_list(parsed.get("evidence_facts", parsed.get("answer_relevant_facts", []))),
        uncertainties=_as_str_list(parsed.get("uncertainties", [])),
        confidence=confidence,
        score=memory_set.score,
    )
    if not state.answer_relevant_facts and not state.uncertainties and state.relevance != "excluded":
        log.warning(
            "  [EVIDENCE READOUT] no usable fields set=%s parsed_keys=%s raw=%s",
            memory_set.id,
            sorted(parsed.keys()),
            _shorten(raw, 1000),
        )
        state = make_fallback_state(memory_set, "the state model returned no usable state fields")

    is_fallback_state = (
        not state.answer_relevant_facts
        and any("readout was not available" in item for item in state.uncertainties)
    )
    if use_cache and cache_file is not None and not is_fallback_state:
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "version": _PROMPT_VERSION,
                        "cache_namespace": cache_namespace,
                        "set_id": memory_set.id,
                        "relevance": state.relevance,
                        "episode_alignment": state.episode_alignment,
                        "evidence_facts": state.answer_relevant_facts,
                        "uncertainties": state.uncertainties,
                        "confidence": state.confidence,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("  [STATE] Cache write failed: %s", exc)
    elif is_fallback_state:
        log.info("  [STATE CACHE] skip fallback state set=%s", memory_set.id)
    _log_state_result("  [EVIDENCE READOUT RESULT]", state)
    return state


def read_episodic_states(
    question_stem: str,
    memory_sets: List[EpisodicMemorySet],
    vlm_callable: "VLMCallable",
    *,
    use_cache: bool = True,
    cache_namespace: str = "default",
    use_images: bool = True,
    max_images: int = 4,
    max_prompt_chars: int = 10000,
    question_context: Optional[str] = None,
) -> List[EpisodicState]:
    states: List[EpisodicState] = []
    for memory_set in memory_sets:
        states.append(
            read_episodic_state(
                question_stem,
                memory_set,
                vlm_callable,
                use_cache=use_cache,
                cache_namespace=cache_namespace,
                use_images=use_images,
                max_images=max_images,
                max_prompt_chars=max_prompt_chars,
                question_context=question_context,
            )
        )
    log.info("EVI episodic evidence readouts generated: %d", len(states))
    for idx, state in enumerate(states, start=1):
        log.info(
            "  evidence_readout[%02d] set=%s relevance=%s confidence=%.2f rounds=%s facts=%d uncertainties=%d",
            idx,
            state.set_id,
            state.relevance,
            state.confidence,
            " -> ".join(state.round_ids),
            len(state.answer_relevant_facts),
            len(state.uncertainties),
        )
    return states
