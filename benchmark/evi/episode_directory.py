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


@dataclass
class EpisodeDirectoryPacket:
    session_id: str
    date: str
    round_id: str
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


class EpisodeDirectoryPacketIndex:
    """Multi-vector session directory with one query-independent packet per round."""

    def __init__(self) -> None:
        self._packets_by_session: Dict[str, List[EpisodeDirectoryPacket]] = {}

    def __len__(self) -> int:
        return len(self._packets_by_session)

    @property
    def packet_count(self) -> int:
        return sum(len(values) for values in self._packets_by_session.values())

    def extend(self, packets: Iterable[EpisodeDirectoryPacket]) -> None:
        for packet in packets:
            if not packet.vector:
                continue
            self._packets_by_session.setdefault(packet.session_id, []).append(packet)

    def search(
        self,
        query_vec: List[float],
        top_k: int = 0,
    ) -> List[Dict[str, object]]:
        ranked: List[Dict[str, object]] = []
        for session_id, packets in self._packets_by_session.items():
            scored = [
                (cosine(query_vec, packet.vector), packet)
                for packet in packets
            ]
            score, best_packet = max(
                scored,
                key=lambda item: item[0],
            )
            ranked.append(
                {
                    "session_id": session_id,
                    "date": best_packet.date,
                    "score": score,
                    "packet_count": len(packets),
                    "best_packet_round_id": best_packet.round_id,
                    "best_packet_text": best_packet.text,
                    "best_packet_text_chars": len(best_packet.text),
                    "best_packet_anchor_count": best_packet.anchor_count,
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


def build_episode_directory_packets(
    session_id: str,
    date: str,
    round_ids: Iterable[str],
    round_text: Dict[str, str],
    anchors_by_round: Dict[str, List[EvidenceAnchor]],
    embed: Callable[[str], List[float]],
) -> List[EpisodeDirectoryPacket]:
    packets: List[EpisodeDirectoryPacket] = []
    for round_id_value in round_ids:
        round_id = str(round_id_value)
        dialogue = str(round_text.get(round_id, "") or "").strip()
        dialogue_key = _normalized_text(dialogue)
        lines = [f"Session {session_id} on {date}.", f"Round {round_id}"]
        if dialogue:
            lines.extend(["Dialogue:", dialogue])

        evidence_lines: List[str] = []
        seen_text = set()
        for anchor in anchors_by_round.get(round_id, []):
            key = _normalized_text(anchor.text)
            if not key or key in seen_text:
                continue
            seen_text.add(key)
            is_wrapped_dialogue = (
                not anchor.image_path
                and dialogue_key
                and dialogue_key in key
            )
            if is_wrapped_dialogue:
                continue
            source = "visual" if anchor.image_path else "dialogue"
            text = " ".join(str(anchor.text).split())
            evidence_lines.append(f"- [{source}/{anchor.evidence_type}] {text}")
        if evidence_lines:
            lines.append("Memory evidence:")
            lines.extend(evidence_lines)

        text = "\n".join(lines).strip()
        packets.append(
            EpisodeDirectoryPacket(
                session_id=session_id,
                date=date,
                round_id=round_id,
                text=text,
                vector=embed(text),
                anchor_count=len(evidence_lines),
            )
        )
    return packets
