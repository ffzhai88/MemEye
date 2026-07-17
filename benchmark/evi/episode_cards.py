from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .schemas import EvidenceAnchor
from .vlm import VLMCallable

SESSION_CARD_PROMPT_VERSION = "session_retrieval_card_v1"
SESSION_CARD_SYSTEM_PROMPT = """You build a query-independent retrieval card for one long-term memory session.

Summarize only the supplied session records. Preserve concrete episode identity, named entities, visible text, distinctive objects, attributes, places, actions, and ordered progression. Do not anticipate future questions or invent details.

Use this plain-text structure:
Episode identity: one compact description of what makes this session distinct.
Ordered progression:
- ROUND_ID: the concrete information contributed by that round.
Distinctive evidence: a compact list of discriminative details across the session.

Mention every supplied ROUND_ID exactly once under Ordered progression. Return only the retrieval card."""


@dataclass
class SessionRetrievalCard:
    session_id: str
    date: str
    round_ids: List[str]
    text: str
    cache_hit: bool
    missing_round_ids: List[str]
    source_chars: int


def _normalized(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def _cache_dir() -> Path:
    path = Path(os.environ.get("EVI_EPISODE_CARD_CACHE_DIR", Path.home() / ".cache" / "evi_episode_cards"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _round_source(round_id: str, round_text: Dict[str, str], anchors_by_round: Dict[str, List[EvidenceAnchor]]) -> str:
    dialogue = str(round_text.get(round_id, "") or "").strip()
    dialogue_key = _normalized(dialogue)
    lines = [f"[{round_id}]", "Raw dialogue:", dialogue or "(none)"]
    evidence: List[str] = []
    seen = set()
    for anchor in anchors_by_round.get(round_id, []):
        key = _normalized(anchor.text)
        if not key or key in seen:
            continue
        seen.add(key)
        if not anchor.image_path and dialogue_key and dialogue_key in key:
            continue
        source = "visual" if anchor.image_path else "dialogue"
        evidence.append(f"- [{source}] {' '.join(str(anchor.text).split())}")
    if evidence:
        lines.append("Existing memory evidence:")
        lines.extend(evidence)
    return "\n".join(lines)


def build_session_retrieval_card(session_id: str, date: str, round_ids: Iterable[str], round_text: Dict[str, str], anchors_by_round: Dict[str, List[EvidenceAnchor]], vlm_callable: VLMCallable, cache_namespace: str, use_cache: bool = True) -> SessionRetrievalCard:
    ordered_rounds = [str(value) for value in round_ids]
    user_text = f"Session: {session_id}\nDate: {date}\n\n" + "\n\n".join(
        _round_source(rid, round_text, anchors_by_round) for rid in ordered_rounds
    )
    raw_key = json.dumps({"version": SESSION_CARD_PROMPT_VERSION, "cache_namespace": cache_namespace, "session_id": session_id, "date": date, "round_ids": ordered_rounds, "source": user_text}, ensure_ascii=False, sort_keys=True)
    cache_file = _cache_dir() / f"{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:40]}.json"
    cache_hit = False
    card_text = ""
    if use_cache and cache_file.exists():
        try:
            card_text = str(json.loads(cache_file.read_text(encoding="utf-8")).get("card", "") or "").strip()
            cache_hit = bool(card_text)
        except Exception:
            card_text = ""
    if not card_text:
        card_text = str(vlm_callable(SESSION_CARD_SYSTEM_PROMPT, user_text, []) or "").strip()
        if card_text.startswith("```") and card_text.endswith("```"):
            card_text = "\n".join(card_text.splitlines()[1:-1]).strip()
    missing = [
        rid
        for rid in ordered_rounds
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(rid)}(?![A-Za-z0-9_])", card_text)
        is None
    ]
    if missing:
        fallback = ["Uncovered round records:"]
        for rid in missing:
            compact = " ".join(str(round_text.get(rid, "") or "").split())
            fallback.append(f"- {rid}: {compact or '(no dialogue text; see indexed memory evidence)'}")
        card_text = (card_text + "\n\n" + "\n".join(fallback)).strip()
    if not card_text:
        card_text = f"Episode identity: Session {session_id} on {date}."
    if use_cache and not cache_hit:
        cache_file.write_text(json.dumps({"version": SESSION_CARD_PROMPT_VERSION, "card": card_text}, ensure_ascii=False, indent=2), encoding="utf-8")
    return SessionRetrievalCard(session_id, date, ordered_rounds, card_text, cache_hit, missing, len(user_text))
