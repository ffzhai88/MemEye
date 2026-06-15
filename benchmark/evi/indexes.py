"""
EVI v2: In-memory vector index for multi-modal search.
"""

from __future__ import annotations

import math
import re
import string
from typing import Any, FrozenSet, List, Optional, Tuple

from .schemas import VectorRecord

# Lightweight English stop word set (no external dependency on nltk corpus download).
_STOP_WORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "been", "be", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "would", "could", "should", "might", "may", "must",
    "shall", "will", "can", "need", "dare", "ought", "used", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "as", "until", "while", "it", "its", "itself", "they",
    "them", "their", "themselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "i", "me", "my",
    "mine", "myself",
})

_PUNCTUATION_PATTERN: re.Pattern = re.compile(r"[{}]".format(re.escape(string.punctuation)))


def _clean_text(text: str) -> str:
    """Remove punctuation, lower-case, and drop English stop words.

    Returns a cleaned string with tokens joined by a single space.
    """
    text = text.lower()
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


class VectorIndex:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self):
        self._records: List[VectorRecord] = []

    def add(self, rec: VectorRecord) -> None:
        self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records)

    def search(
        self,
        query_vec: List[float],
        top_k: int = 15,
        node_types: Optional[List[str]] = None,
    ) -> List[VectorRecord]:
        """Search by cosine similarity. Optionally filter by node_type."""
        scored: List[Tuple[float, int]] = []
        for idx, rec in enumerate(self._records):
            if node_types and rec.node_type not in node_types:
                continue
            score = _cosine(query_vec, rec.vector)
            if score > 0.2:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            rec = self._records[idx]
            rec.score = score
            results.append(rec)
        return results

    def get_by_type(self, node_type: str) -> List[VectorRecord]:
        return [r for r in self._records if r.node_type == node_type]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_text(text: str, embedder: Any) -> List[float]:
    """Embed text using various embedder interfaces.

    The input text is cleaned before embedding:
    lower-cased, punctuation removed, English stop words dropped.
    """
    if embedder is None:
        return []
    text = _clean_text(text)
    # Try common interfaces: .encode() (sentence-transformers),
    # .embed_query() (chromadb/langchain), or callable
    for method_name in ("encode", "embed_query", "__call__"):
        try:
            fn = getattr(embedder, method_name, embedder if method_name == "__call__" else None)
            if fn is None:
                continue
            result = fn([text] if method_name == "__call__" else text)
            if hasattr(result, "tolist"):
                return result.tolist()
            if isinstance(result, (list, tuple)):
                return list(result)
            return result
        except Exception:
            continue
    return []
