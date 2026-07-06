from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvidenceAnchor:
    """A compact, typed visual or dialogue memory unit."""

    id: str
    session_id: str
    round_id: str
    date: str
    evidence_type: str
    text: str
    vector: List[float] = field(default_factory=list)
    image_path: Optional[str] = None
    subject: str = ""
    predicate: str = ""
    object: str = ""
    region: str = ""
    confidence: float = 1.0
    score: float = 0.0
    raw_score: float = 0.0
    quality_weight: float = 1.0
    discriminativeness_weight: float = 1.0


@dataclass
class MemoryCandidate:
    """A non-overlapping candidate episodic memory assembled from retrieved anchors."""

    id: str
    session_id: str
    round_id: str
    date: str
    image_path: Optional[str]
    image_paths: List[str]
    round_text: str
    anchors: List[EvidenceAnchor]
    score: float
    selected_anchors: List[EvidenceAnchor] = field(default_factory=list)


@dataclass
class MemoryBrief:
    """A query-conditioned natural-language brief for one memory candidate."""

    candidate_id: str
    session_id: str
    round_id: str
    date: str
    image_paths: List[str]
    relevance: str
    brief: str
    key_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    score: float = 0.0


@dataclass
class EpisodicMemorySet:
    """A local, ordered episode assembled from retrieved anchor hits."""

    id: str
    session_id: str
    date: str
    round_ids: List[str]
    round_text: Dict[str, str]
    round_images: Dict[str, List[str]]
    round_anchors: Dict[str, List[EvidenceAnchor]]
    retrieved_anchors: List[EvidenceAnchor]
    score: float
    hit_round_count: int = 0
    max_anchor_score: float = 0.0


@dataclass
class EpisodicState:
    """A query-conditioned readout over one ordered episodic memory set."""

    set_id: str
    session_id: str
    date: str
    round_ids: List[str]
    image_paths: List[str]
    relevance: str
    grounded_cues: List[str] = field(default_factory=list)
    observed_facts: List[str] = field(default_factory=list)
    memory_items: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)
    answer_relevant_facts: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    confidence: float = 0.0
    score: float = 0.0
