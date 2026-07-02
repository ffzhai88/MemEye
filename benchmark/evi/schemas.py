from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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


@dataclass
class EvidenceGroup:
    """A query-conditioned group of mutually supporting evidence anchors."""

    id: str
    seed_anchor_id: str
    anchors: List[EvidenceAnchor]
    score: float
    group_label: str = ""
    group_hypothesis: str = ""
    needed_visual_checks: List[str] = field(default_factory=list)
    verified_evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    confidence: float = 0.0
