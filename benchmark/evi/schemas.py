"""
EVI v2: Data structures for multi-vector temporal indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DialogueNode:
    """A conversation round. Indexed as one embedding vector."""
    round_id: str
    session_id: str
    text: str               # round_text + image_caption
    image_path: Optional[str] = None
    caption: str = ""
    timestamp: str = ""


@dataclass
class ImageNode:
    """
    One image described from multiple angles.
    - image_descs: free-form descriptions generated from the image alone (VLM decides count)
    - context_desc: description generated from conversation context (user + assistant + prior rounds)
    Each description becomes a separate embedding vector.
    """
    round_id: str
    session_id: str
    image_path: str
    image_descs: List[str] = field(default_factory=list)   # from image-only call
    context_desc: str = ""                                  # from context-aware call
    timestamp: str = ""


@dataclass
class VectorRecord:
    """One embedding vector in the index, referencing its source."""
    id: str                 # unique: "dialogue_{rid}" or "image_{rid}_{type}"
    round_id: str
    session_id: str
    text: str               # the text that was embedded
    vector: List[float]
    node_type: str          # "dialogue" | "image_visual" | "image_context" | "image_session"
    image_path: Optional[str] = None
    score: float = 0.0      # set during retrieval


@dataclass
class RetrievalResult:
    """Final context block sent to VLM."""
    ordered_context: str     # temporally sorted text with explicit ordering language
    image_paths: List[str]   # original images for selected rounds
