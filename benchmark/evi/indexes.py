from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, FrozenSet, Iterable, List, Optional, Tuple

from .schemas import EvidenceAnchor

log = logging.getLogger(__name__)

_STOP_WORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "been", "be", "being", "have", "has", "had", "do", "does", "did", "would",
    "could", "should", "might", "may", "must", "will", "can", "to", "of", "in",
    "on", "with", "at", "by", "from", "into", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "few", "more", "most", "other", "some", "no", "nor", "not",
    "only", "own", "same", "too", "very", "it", "its", "they", "them", "their",
    "he", "him", "his", "she", "her", "we", "us", "our", "you", "your", "i",
    "me", "my",
})

_TYPE_ALIASES = {
    "dialogue": "temporal",
    "event": "temporal",
    "state": "temporal",
    "ocr": "text",
    "ui": "structured_visual",
    "chart": "structured_visual",
    "chart_ui": "structured_visual",
    "table": "structured_visual",
    "diagram": "structured_visual",
    "visual": "scene",
    "object": "entity",
}

_VALID_TYPES = {
    "scene",
    "text",
    "entity",
    "attribute",
    "spatial",
    "relation",
    "identity",
    "structured_visual",
    "temporal",
}

_EMBED_CACHE_DIR: Optional[str] = None
_EMBED_METHOD_CACHE: dict[int, str] = {}  # id(embedder) -> method_name


def _embed_cache_dir() -> str:
    global _EMBED_CACHE_DIR
    if _EMBED_CACHE_DIR is None:
        _EMBED_CACHE_DIR = os.environ.get(
            "EVI_EMBED_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_embeddings"),
        )
        os.makedirs(_EMBED_CACHE_DIR, exist_ok=True)
    return _EMBED_CACHE_DIR


def _embed_cache_key(text: str, namespace: str) -> str:
    raw = json.dumps(
        {"version": "embed_v1", "namespace": namespace, "text": text},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def clean_text(text: str) -> str:
    """Normalize text for embedding: lowercase and remove stop words, preserving punctuation."""
    text = str(text).lower()
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return " ".join(tokens)


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def normalize_type(evidence_type: str) -> str:
    raw = str(evidence_type or "scene").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    normalized = _TYPE_ALIASES.get(raw, raw)
    return normalized if normalized in _VALID_TYPES else "scene"


def _resolve_embed_method(embedder: Any) -> Optional[str]:
    """Probe the embedder object once and cache which call interface it supports."""
    eid = id(embedder)
    cached = _EMBED_METHOD_CACHE.get(eid)
    if cached:
        return cached

    for method_name in ("embed_query", "encode", "__call__"):
        try:
            if method_name == "__call__":
                test_result = embedder(["probe"])
            else:
                fn = getattr(embedder, method_name, None)
                if fn is None:
                    continue
                test_result = fn("probe")

            if hasattr(test_result, "tolist"):
                test_result = test_result.tolist()
            if isinstance(test_result, (list, tuple)) and len(test_result) > 0:
                _EMBED_METHOD_CACHE[eid] = method_name
                log.debug("Embed method resolved: %s for embedder %d", method_name, eid)
                return method_name
        except Exception:
            continue

    log.warning("Could not resolve an embedding method for embedder %d", eid)
    return None


def _call_embed(embedder: Any, text: str, method: str) -> List[float]:
    """Call the embedder using the pre-resolved method."""
    if method == "__call__":
        result = embedder([text])
    elif method == "embed_query":
        result = embedder.embed_query(text)
    elif method == "encode":
        result = embedder.encode(text)
    else:
        return []

    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, (list, tuple)):
        if result and isinstance(result[0], (list, tuple)):
            return list(result[0])
        return list(result)
    return []


def embed_text(
    text: str,
    embedder: Any,
    cache_namespace: str = "default",
    use_cache: bool = True,
) -> List[float]:
    """Embed text through the repository TextEmbedder-compatible interface."""
    if embedder is None:
        return []
    text = clean_text(text)
    if not text:
        return []

    cache_file: Optional[Path] = None
    if use_cache:
        cache_file = Path(_embed_cache_dir()) / f"{_embed_cache_key(text, cache_namespace)}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                vector = data.get("vector", []) if isinstance(data, dict) else []
                if isinstance(vector, list) and vector:
                    return [float(x) for x in vector]
            except Exception:
                pass

    method = _resolve_embed_method(embedder)
    if method is None:
        return []

    vector = _call_embed(embedder, text, method)
    if vector:
        try:
            vector = [float(x) for x in vector]
        except Exception:
            vector = []

    if vector and use_cache and cache_file is not None:
        try:
            cache_file.write_text(
                json.dumps({"namespace": cache_namespace, "text": text, "vector": vector}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
    return vector


class EvidenceIndex:
    """Simple in-memory vector index for EvidenceAnchor objects."""

    def __init__(self) -> None:
        self._anchors: List[EvidenceAnchor] = []

    def add(self, anchor: EvidenceAnchor) -> None:
        anchor.evidence_type = normalize_type(anchor.evidence_type)
        self._anchors.append(anchor)

    def extend(self, anchors: Iterable[EvidenceAnchor]) -> None:
        for anchor in anchors:
            self.add(anchor)

    def __len__(self) -> int:
        return len(self._anchors)

    @property
    def anchors(self) -> List[EvidenceAnchor]:
        return self._anchors

    def search(
        self,
        query_vec: List[float],
        top_k: int = 60,
        session_ids: Optional[set[str]] = None,
    ) -> List[EvidenceAnchor]:
        results: List[Tuple[float, int]] = []
        for idx, anchor in enumerate(self._anchors):
            if session_ids is not None and anchor.session_id not in session_ids:
                continue
            score = cosine(query_vec, anchor.vector)
            if score <= 0:
                continue
            results.append((score, idx))
        results.sort(key=lambda item: item[0], reverse=True)

        out: List[EvidenceAnchor] = []
        for score, idx in results[:top_k]:
            anchor = self._anchors[idx]
            anchor.score = score
            out.append(anchor)
        return out
