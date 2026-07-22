"""Selective VLM verification for retrieved raw memory evidence.

The verifier is deliberately downstream of retrieval. It does not generate a
new claim, route by dataset or question type, or replace the base ranking. It
only checks whether an individual raw dialogue round can contribute evidence
towards the question. The conservative reranker demotes a candidate only when
the verifier returns a parseable, high-confidence ``not_useful`` verdict.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from ._utils import extract_json


PROMPT_VERSION = "selective-evidence-utility-v1"


SYSTEM_PROMPT = """You are checking one retrieved memory round for evidence utility.

Do not answer the user's question. Decide only whether this one past round can
contribute evidence toward answering it. A useful round need not answer the
whole question by itself. It may supply one fact, one item in a count or total,
one side or boundary of a comparison, an earlier or later state, an update, or
information that disambiguates another memory. Older, partial, and even
contradictory evidence can still be useful when the question requires history,
change, comparison, or aggregation.

The stored memory descriptions are retrieval hints, not ground truth. Judge the
original dialogue and images as authoritative. A weak description does not make
strong raw evidence irrelevant. Likewise, seeing the same object in an image is
not by itself evidence for relations such as preferred, current, replaced,
owned, or chosen; those relations need support from the dialogue or image.

Use not_useful only when the raw round clearly contributes no relevant fact. If
the evidence is ambiguous or might become useful together with other rounds,
choose possibly_useful. Be conservative because a mistaken rejection can remove
necessary memory evidence."""


@dataclass(frozen=True)
class VerificationVerdict:
    anchor_grounding: str
    evidence_utility: str
    confidence: str
    supported_facet_indices: Tuple[int, ...] = ()
    dialogue_grounding: str = ""
    visual_grounding: str = ""
    reason: str = ""
    parse_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["supported_facet_indices"] = list(self.supported_facet_indices)
        return result


def _clean_text(value: Any, max_chars: int = 6000) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{text[:head]} [middle omitted] {text[-tail:]}"


def _anchor_lines(anchors: Sequence[Any], limit: int = 5) -> List[str]:
    lines: List[str] = []
    for anchor in list(anchors)[:limit]:
        if isinstance(anchor, str):
            value = anchor.strip()
            if value:
                lines.append(value)
            continue
        if not isinstance(anchor, Mapping):
            continue
        text = str(
            anchor.get("text")
            or anchor.get("anchor")
            or anchor.get("description")
            or anchor.get("content")
            or ""
        ).strip()
        if not text:
            continue
        source = str(
            anchor.get("source") or anchor.get("kind") or anchor.get("type") or ""
        ).strip()
        line = f"{text} ({source})" if source else text
        lines.append(_clean_text(line, 1000))
    return lines


def build_verification_prompt(
    *,
    question: str,
    question_date: str,
    facets: Sequence[str],
    session_id: str,
    round_id: str,
    session_date: str,
    user_text: str,
    assistant_text: str,
    anchors: Sequence[Any],
    image_count: int,
) -> str:
    """Build a narrative prompt; only the requested response is structured."""

    facet_lines = [
        f"({i}) {_clean_text(f, 600)}"
        for i, f in enumerate(facets, 1)
        if str(f).strip()
    ]
    anchor_lines = _anchor_lines(anchors)
    facets_text = (
        "; ".join(facet_lines)
        if facet_lines
        else "no separate reading beyond the original question"
    )
    anchors_text = (
        "; ".join(f'"{line}"' for line in anchor_lines)
        if anchor_lines
        else "no stored description, so the raw round must stand on its own"
    )
    date_note = question_date or "not provided"
    memory_date = session_date or "not provided"
    visual_note = (
        f"This round has {image_count} original image(s), supplied after the text."
        if image_count
        else "This round has no original image. Judge it from the dialogue alone."
    )

    return f"""We are retrieving past evidence for this user question:

{_clean_text(question, 2500)}

The question date is {date_note}. While searching, the retriever considered several overlapping readings of the question, in this order: {facets_text}.

It proposed round {round_id} from session {session_id}, dated {memory_date}, because the stored memory descriptions included {anchors_text}.

Now inspect the original memory rather than trusting those descriptions.

In that round, the user said:
{_clean_text(user_text)}

The assistant replied:
{_clean_text(assistant_text)}

{visual_note}

Decide whether this raw round can contribute evidence toward the question. Do not require it to answer the entire question on its own, and do not answer the question.

Return only one JSON object with exactly these fields:
{{
  "anchor_grounding": "supported | partial | unsupported | uncertain",
  "evidence_utility": "useful | possibly_useful | not_useful",
  "confidence": "high | medium | low",
  "supported_facet_indices": [1],
  "dialogue_grounding": "a short exact quote from the dialogue, or an empty string",
  "visual_grounding": "a brief description of relevant visible evidence, or an empty string",
  "reason": "one concise sentence"
}}"""


def _enum(value: Any, allowed: Sequence[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "partially_supported": "partial",
        "maybe_useful": "possibly_useful",
        "possibly_relevant": "possibly_useful",
        "irrelevant": "not_useful",
        "not_relevant": "not_useful",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in allowed else fallback


def _normalize_quote(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def parse_verdict(raw_response: str, dialogue_text: str = "") -> VerificationVerdict:
    """Parse verifier output fail-closed: malformed output can never demote."""

    parsed = extract_json(raw_response or "")
    if not isinstance(parsed, Mapping):
        return VerificationVerdict(
            anchor_grounding="uncertain",
            evidence_utility="possibly_useful",
            confidence="low",
            reason="Verifier response could not be parsed; original ranking is preserved.",
            parse_valid=False,
        )

    anchor = _enum(parsed.get("anchor_grounding"), ("supported", "partial", "unsupported", "uncertain"), "uncertain")
    utility = _enum(parsed.get("evidence_utility"), ("useful", "possibly_useful", "not_useful"), "possibly_useful")
    confidence = _enum(parsed.get("confidence"), ("high", "medium", "low"), "low")

    facet_indices: List[int] = []
    raw_indices = parsed.get("supported_facet_indices")
    if isinstance(raw_indices, list):
        for value in raw_indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index > 0 and index not in facet_indices:
                facet_indices.append(index)

    dialogue_grounding = str(parsed.get("dialogue_grounding") or "").strip()
    if dialogue_grounding and _normalize_quote(dialogue_grounding) not in _normalize_quote(dialogue_text):
        dialogue_grounding = ""

    required_valid = (
        str(parsed.get("anchor_grounding") or "").strip() != ""
        and str(parsed.get("evidence_utility") or "").strip() != ""
        and str(parsed.get("confidence") or "").strip() != ""
    )
    return VerificationVerdict(
        anchor_grounding=anchor,
        evidence_utility=utility,
        confidence=confidence,
        supported_facet_indices=tuple(facet_indices),
        dialogue_grounding=dialogue_grounding,
        visual_grounding=str(parsed.get("visual_grounding") or "").strip(),
        reason=_clean_text(parsed.get("reason"), 800),
        parse_valid=bool(required_valid),
    )


def verification_candidate_ids(
    abstract_ranking: Sequence[str],
    raw_ranking: Sequence[str],
    top_k: int = 10,
    order_ranking: Sequence[str] = (),
) -> List[str]:
    """Return candidates with conflicting Top-K inclusion decisions.

    Selection is exactly the symmetric difference between abstract Anchor and
    Raw-MM Top-K sets. ``order_ranking`` only makes API/log order deterministic;
    it cannot add or remove a candidate.
    """

    abstract_top = set(map(str, list(abstract_ranking)[:top_k]))
    raw_top = set(map(str, list(raw_ranking)[:top_k]))
    contested = (abstract_top ^ raw_top) - {""}
    ordered: List[str] = []
    for ranking in (order_ranking, abstract_ranking, raw_ranking):
        for round_id in ranking:
            rid = str(round_id)
            if rid in contested and rid not in ordered:
                ordered.append(rid)
    return ordered


def _is_high_confidence_rejection(verdict: Mapping[str, Any]) -> bool:
    return (
        bool(verdict.get("parse_valid", True))
        and verdict.get("evidence_utility") == "not_useful"
        and verdict.get("confidence") == "high"
    )


def conservative_rerank(
    base_ranking: Sequence[str],
    abstract_ranking: Sequence[str],
    raw_ranking: Sequence[str],
    verdicts: Mapping[str, Mapping[str, Any]],
    top_k: int = 10,
) -> Tuple[List[str], Dict[str, Any]]:
    """Resolve Top-K disagreement without admitting consensus-excluded rounds.

    The shared Anchor/Raw-MM Top-K intersection is frozen in. Remaining slots
    are filled only from their symmetric difference, with parse-valid,
    high-confidence rejections considered last. If every alternative is
    rejected, rejected contested candidates are used as a deterministic
    fallback so the retrieval budget remains exactly Top-K.
    """

    base = list(dict.fromkeys(map(str, base_ranking)))
    abstract_top = set(map(str, list(abstract_ranking)[:top_k]))
    raw_top = set(map(str, list(raw_ranking)[:top_k]))
    union = abstract_top | raw_top
    stable_in = abstract_top & raw_top
    contested = abstract_top ^ raw_top
    rejected = {
        rid
        for rid in contested
        if rid in verdicts and _is_high_confidence_rejection(verdicts[rid])
    }
    accepted = contested - rejected

    # Mean-rank orders candidates but cannot change the eligible Top-K set.
    order = list(dict.fromkeys(
        base
        + list(map(str, abstract_ranking))
        + list(map(str, raw_ranking))
    ))
    selected = set(stable_in)
    for pool in (accepted, rejected):
        for rid in order:
            if len(selected) >= min(top_k, len(union)):
                break
            if rid in pool:
                selected.add(rid)

    final_top = [rid for rid in order if rid in selected]
    final_ranking = final_top + [rid for rid in base if rid not in selected]
    base_top = set(base[:top_k])
    excluded_consensus = [rid for rid in base if rid not in union]
    rejected_outside_top = [rid for rid in order if rid in rejected and rid not in selected]
    demoted = [rid for rid in order if rid in rejected and rid in base_top and rid not in selected]
    promoted = [rid for rid in final_top if rid not in base_top]
    return final_ranking, {
        "policy": "top_k_union_closed_high_confidence_rejection_demotion",
        "top_k": top_k,
        "stable_in_round_ids": [rid for rid in order if rid in stable_in],
        "contested_round_ids": [rid for rid in order if rid in contested],
        "high_confidence_rejected_round_ids": [rid for rid in order if rid in rejected],
        "rejected_outside_top_k": rejected_outside_top,
        "consensus_excluded_round_ids": excluded_consensus,
        "final_top_k_round_ids": final_top,
        "demoted_round_ids": demoted,
        "promoted_round_ids": promoted,
        "demoted_count": len(demoted),
    }


class SelectiveEvidenceVerifier:
    """Cached VLM wrapper for one-round evidence-utility decisions."""

    def __init__(
        self,
        vlm: Callable[[str, str, List[str]], str],
        cache_dir: Path,
        model_namespace: str,
    ) -> None:
        self.vlm = vlm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_namespace = model_namespace
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, user_prompt: str, image_paths: Sequence[str]) -> str:
        image_state: List[Dict[str, Any]] = []
        for value in image_paths:
            path = Path(value)
            try:
                stat = path.stat()
                image_state.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            except OSError:
                image_state.append({"path": str(path), "missing": True})
        payload = {
            "prompt_version": PROMPT_VERSION,
            "model": self.model_namespace,
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "images": image_state,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify(self, user_prompt: str, image_paths: Sequence[str], dialogue_text: str) -> Dict[str, Any]:
        key = self._cache_key(user_prompt, image_paths)
        cache_path = self.cache_dir / f"{key}.json"
        raw_response = ""
        cache_hit = False
        error = ""

        with self._lock:
            if cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    raw_response = str(cached.get("raw_response") or "")
                    cache_hit = True
                    self.cache_hits += 1
                except (OSError, json.JSONDecodeError):
                    cache_hit = False

        if not cache_hit:
            try:
                raw_response = self.vlm(SYSTEM_PROMPT, user_prompt, list(image_paths)) or ""
            except Exception as exc:  # API failure must preserve the base rank.
                error = f"{type(exc).__name__}: {exc}"
                raw_response = ""
            if not raw_response and not error:
                error = "empty_response"
            with self._lock:
                self.cache_misses += 1
                # Transient failures and empty generations must remain retryable.
                if raw_response:
                    payload = {
                        "prompt_version": PROMPT_VERSION,
                        "model_namespace": self.model_namespace,
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "image_paths": list(image_paths),
                        "raw_response": raw_response,
                        "error": error,
                    }
                    cache_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

        verdict = parse_verdict(raw_response, dialogue_text=dialogue_text)
        return {
            "verdict": verdict.to_dict(),
            "raw_response": raw_response,
            "cache_hit": cache_hit,
            "cache_key": key,
            "error": error,
        }
