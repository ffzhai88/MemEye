"""
EVI: Data structures for evidence extraction and retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Phase 1: extracted data
# ---------------------------------------------------------------------------

@dataclass
class SceneAttributes:
    background_color: str = ""
    lighting: str = ""
    setting: str = ""
    mood: str = ""


@dataclass
class GistData:
    free_text: str = ""
    scene_attributes: SceneAttributes = field(default_factory=SceneAttributes)


@dataclass
class TagData:
    noun: str = ""
    color: str = ""
    position: str = ""
    count: Optional[int] = None


@dataclass
class AnchorData:
    explicit_labels: List[str] = field(default_factory=list)
    explicit_topic: str = ""
    user_intent: str = ""


@dataclass
class ExtractionResult:
    """Output of one VLM call per round."""
    round_id: str
    image_path: str
    gist: GistData = field(default_factory=GistData)
    tags: List[TagData] = field(default_factory=list)
    anchors: AnchorData = field(default_factory=AnchorData)


# ---------------------------------------------------------------------------
# Index entries (flat, stored in dicts)
# ---------------------------------------------------------------------------

@dataclass
class GistEntry:
    round_id: str
    free_text: str
    scene_attributes: Dict[str, str]  # flat dict for easy lookup
    timestamp: str
    _embedding: Optional[List[float]] = None  # lazy computed


@dataclass
class TagEntry:
    round_id: str
    color: str = ""
    position: str = ""
    count: Optional[int] = None


@dataclass
class AnchorEntry:
    keyword: str
    round_ids: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# Phase 2: retrieval and reasoning
# ---------------------------------------------------------------------------

@dataclass
class RichDirectoryEntry:
    round_id: str
    free_text: str
    scene_attributes: Dict[str, str]
    tags: List[str]  # formatted like "noun(color,position)"
    anchors: List[str]


@dataclass
class RichDirectory:
    entries: List[RichDirectoryEntry] = field(default_factory=list)


@dataclass
class AggregateResult:
    answer: str = ""
    confidence: float = 0.0
    candidates: List[str] = field(default_factory=list)
