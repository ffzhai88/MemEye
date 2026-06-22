"""
EVI v2: Data structures for multi-vector temporal indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Fact:
    """
    知识图谱中的一条边（三元组）。

    每个图片的所有 Fact 以 image_name 为中心节点，构成一个微型知识图谱。
    subject 可以是 image_name（图片级事实）或图中的实体名（实体/关系事实）。
    """
    subject: str
    predicate: str
    object: str

    def to_text(self) -> str:
        """面向检索的表示：保留结构化分隔符，便于精确匹配。"""
        return f"{self.subject} :: {self.predicate} :: {self.object}"

    @staticmethod
    def from_dict(d: dict) -> "Fact":
        return Fact(
            subject=str(d.get("subject", "")),
            predicate=str(d.get("predicate", "")),
            object=str(d.get("object", "")),
        )


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
    One image represented as a micro knowledge graph.

    Each image is described from three angles, forming a structured
    representation centered around image_name (the root node of the micro-KG).

    - image_name: short semantic name capturing the image's identity/type
      (from context-aware VLM call)
    - image_descs: free-form descriptions from the image alone (VLM decides count)
      → indexed as "image_visual" vectors
    - context_desc: description of the image's role in conversation
      (from context-aware call) → indexed as "image_context" vector
    - facts: structured factual triples that form the micro-KG edges
      (from fact extraction call) → indexed as "image_fact" vectors

    VectorRecord node_type mapping:
      "image_name"    — the micro-KG root node name, for name-level matching
      "image_visual"  — free-form visual description (image-only)
      "image_context" — context-aware description
      "image_fact"    — individual factual triple
    """
    round_id: str
    session_id: str
    image_path: str
    image_name: str = ""                                        # from context-aware call
    image_descs: List[str] = field(default_factory=list)        # from image-only call
    context_desc: str = ""                                       # from context-aware call
    facts: List[Fact] = field(default_factory=list)              # from fact extraction call
    timestamp: str = ""


@dataclass
class SessionNode:
    """A session-level summary for LLM directory retrieval.
    The VLM generates a comprehensive summary of what the session is about,
    which is embedded and used for coarse-grained session lookup.
    """
    session_id: str
    summary: str               # VLM-generated session summary
    date: str = ""
    timestamp: str = ""


@dataclass
class VectorRecord:
    """One embedding vector in the index, referencing its source."""
    id: str                 # unique: "dialogue_{rid}" / "image_{rid}_{type}" / "session_{sid}"
    round_id: str
    session_id: str
    text: str               # the text that was embedded
    vector: List[float]
    node_type: str          # "dialogue" | "image_visual" | "image_context" | "image_name" | "image_fact" | "session"
    image_path: Optional[str] = None
    score: float = 0.0      # set during retrieval


@dataclass
class RetrievalResult:
    """Final context block sent to VLM."""
    ordered_context: str     # temporally sorted text with explicit ordering language
    image_paths: List[str]   # original images for selected rounds
