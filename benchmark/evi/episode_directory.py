from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

from .indexes import cosine
from .schemas import EvidenceAnchor


def _normalized_text(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


@dataclass
class EpisodeDirectoryEntry:
    session_id: str
    date: str
    round_ids: List[str]
    text: str
    vector: List[float]
    anchor_count: int


class EpisodeDirectoryIndex:
    """Query-independent holistic embeddings for natural dialogue sessions."""

    def __init__(self) -> None:
        self._entries: List[EpisodeDirectoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[EpisodeDirectoryEntry]:
        return self._entries

    def add(self, entry: EpisodeDirectoryEntry) -> None:
        if entry.vector:
            self._entries.append(entry)

    def search(
        self,
        query_vec: List[float],
        anchors_by_round: Dict[str, List[EvidenceAnchor]],
        top_k: int = 0,
    ) -> List[Dict[str, object]]:
        ranked: List[Dict[str, object]] = []
        for entry in self._entries:
            witness: EvidenceAnchor | None = None
            witness_score = float("-inf")
            for round_id in entry.round_ids:
                for anchor in anchors_by_round.get(round_id, []):
                    score = cosine(query_vec, anchor.vector)
                    if score > witness_score:
                        witness = anchor
                        witness_score = score
            ranked.append(
                {
                    "session_id": entry.session_id,
                    "date": entry.date,
                    "score": cosine(query_vec, entry.vector),
                    "round_ids": list(entry.round_ids),
                    "directory_text_chars": len(entry.text),
                    "anchor_count": entry.anchor_count,
                    "witness_round_id": witness.round_id if witness is not None else "",
                    "witness_text": witness.text if witness is not None else "",
                    "witness_score": witness_score if witness is not None else None,
                }
            )
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["session_id"])))
        return ranked[:top_k] if top_k > 0 else ranked


def build_episode_directory_entry(
    session_id: str,
    date: str,
    round_ids: Iterable[str],
    round_text: Dict[str, str],
    anchors_by_round: Dict[str, List[EvidenceAnchor]],
    embed: Callable[[str], List[float]],
) -> EpisodeDirectoryEntry:
    ordered_rounds = [str(round_id) for round_id in round_ids]
    lines = [f"Session {session_id} on {date}."]
    anchor_count = 0
    for round_id in ordered_rounds:
        lines.append("")
        lines.append(f"Round {round_id}")
        dialogue = str(round_text.get(round_id, "") or "").strip()
        if dialogue:
            lines.append("Dialogue:")
            lines.append(dialogue)

        unique_anchors: List[EvidenceAnchor] = []
        seen_text = set()
        for anchor in anchors_by_round.get(round_id, []):
            key = _normalized_text(anchor.text)
            if not key or key in seen_text:
                continue
            seen_text.add(key)
            unique_anchors.append(anchor)
        if unique_anchors:
            lines.append("Memory evidence:")
            for anchor in unique_anchors:
                source = "visual" if anchor.image_path else "dialogue"
                text = " ".join(str(anchor.text).split())
                lines.append(f"- [{source}/{anchor.evidence_type}] {text}")
            anchor_count += len(unique_anchors)

    text = "\n".join(lines).strip()
    return EpisodeDirectoryEntry(
        session_id=session_id,
        date=date,
        round_ids=ordered_rounds,
        text=text,
        vector=embed(text),
        anchor_count=anchor_count,
    )
