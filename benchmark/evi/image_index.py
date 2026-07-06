from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .indexes import cosine

log = logging.getLogger(__name__)

_PROMPT_VERSION = "image_index_v1"
_CACHE_DIR: Optional[str] = None


@dataclass
class ImageHit:
    round_id: str
    image_path: str
    score: float
    rank_score: float
    rank: int


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_IMAGE_EMBED_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_image_embeddings"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _cache_key(kind: str, value: str, cache_namespace: str) -> str:
    raw = json.dumps(
        {
            "version": _PROMPT_VERSION,
            "kind": kind,
            "cache_namespace": cache_namespace,
            "value": value,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _read_cached_vector(kind: str, value: str, cache_namespace: str) -> Optional[List[float]]:
    path = Path(_cache_dir()) / f"{_cache_key(kind, value, cache_namespace)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        vector = data.get("vector", []) if isinstance(data, dict) else []
        if isinstance(vector, list) and vector:
            return [float(item) for item in vector]
    except Exception as exc:
        log.warning("EVI image embedding cache read failed kind=%s value=%s error=%s", kind, value, exc)
    return None


def _write_cached_vector(kind: str, value: str, cache_namespace: str, vector: List[float]) -> None:
    path = Path(_cache_dir()) / f"{_cache_key(kind, value, cache_namespace)}.json"
    try:
        path.write_text(
            json.dumps(
                {
                    "version": _PROMPT_VERSION,
                    "kind": kind,
                    "cache_namespace": cache_namespace,
                    "value": value,
                    "vector": vector,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("EVI image embedding cache write failed kind=%s value=%s error=%s", kind, value, exc)


def _embed_text_cached(
    embedder: Any,
    text: str,
    *,
    cache_namespace: str,
    use_cache: bool,
) -> List[float]:
    text = str(text or "").strip()
    if not text:
        return []
    if use_cache:
        cached = _read_cached_vector("text", text, cache_namespace)
        if cached:
            return cached
    vector = embedder.embed_text(text)
    vector = [float(item) for item in vector] if vector else []
    if vector and use_cache:
        _write_cached_vector("text", text, cache_namespace, vector)
    return vector


def _embed_image_cached(
    embedder: Any,
    image_path: str,
    *,
    cache_namespace: str,
    use_cache: bool,
) -> List[float]:
    image_path = str(Path(image_path).resolve())
    if use_cache:
        cached = _read_cached_vector("image", image_path, cache_namespace)
        if cached:
            return cached
    vector = embedder.embed_image(image_path)
    vector = [float(item) for item in vector] if vector else []
    if vector and use_cache:
        _write_cached_vector("image", image_path, cache_namespace, vector)
    return vector


class ImageIndex:
    """Text-to-image retrieval over memory images in a shared multimodal embedding space."""

    def __init__(
        self,
        embedder: Any,
        *,
        cache_namespace: str,
        use_cache: bool = True,
    ) -> None:
        self._embedder = embedder
        self._cache_namespace = cache_namespace
        self._use_cache = use_cache
        self._items: List[Tuple[str, str, List[float]]] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, round_id: str, image_path: str) -> None:
        if not round_id or not image_path or not os.path.isfile(image_path):
            return
        try:
            vector = _embed_image_cached(
                self._embedder,
                image_path,
                cache_namespace=self._cache_namespace,
                use_cache=self._use_cache,
            )
        except Exception as exc:
            log.warning("EVI image embedding failed round=%s image=%s error=%s", round_id, image_path, exc)
            return
        if vector:
            self._items.append((round_id, image_path, vector))

    def search(self, query_text: str, top_k: int) -> List[ImageHit]:
        query_text = str(query_text or "").strip()
        top_k = max(1, int(top_k or 1))
        if not query_text or not self._items:
            return []
        try:
            query_vec = _embed_text_cached(
                self._embedder,
                query_text,
                cache_namespace=self._cache_namespace,
                use_cache=self._use_cache,
            )
        except Exception as exc:
            log.warning("EVI image query embedding failed text=%s error=%s", query_text, exc)
            return []
        if not query_vec:
            return []

        scored: List[Tuple[float, str, str]] = []
        for round_id, image_path, image_vec in self._items:
            score = cosine(query_vec, image_vec)
            if score > 0:
                scored.append((score, round_id, image_path))
        scored.sort(key=lambda item: item[0], reverse=True)

        hits: List[ImageHit] = []
        for rank, (score, round_id, image_path) in enumerate(scored[:top_k], start=1):
            hits.append(
                ImageHit(
                    round_id=round_id,
                    image_path=image_path,
                    score=score,
                    rank_score=1.0 / rank,
                    rank=rank,
                )
            )
        return hits
