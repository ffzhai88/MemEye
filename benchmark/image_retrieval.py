"""Shared raw-image retrieval for Semantic-RAG and EVI.

The index embeds original round images and searches them with a text query.  A
round with multiple images receives its best image score.  SigLIP is preferred;
if model loading, indexing, or query embedding fails, the complete index is
rebuilt with local CLIP so vector spaces are never mixed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import LocalCLIPEmbedder, MultimodalEmbedder


log = logging.getLogger(__name__)

_SHORT_NAME_MAP = {
    "siglip2-base-patch16-384": "google/siglip2-base-patch16-384",
    "siglip-so400m-patch14-384": "google/siglip-so400m-patch14-384",
}


def _cosine(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _valid_vector(vector: List[float]) -> bool:
    return bool(vector) and all(math.isfinite(value) for value in vector) and any(
        value != 0.0 for value in vector
    )


class RawImageRoundIndex:
    """Persistent text-to-image index collapsed to round-level rankings."""

    CACHE_VERSION = "raw_image_round_index_v1"

    def __init__(self, dataset: Any, config: Dict[str, Any]) -> None:
        self.dataset = dataset
        self.config = config
        preferred = str(config.get("multimodal_embedding_model", "siglip2-base-patch16-384"))
        self.preferred_model = _SHORT_NAME_MAP.get(preferred, preferred)
        self.clip_model = str(
            config.get("multimodal_clip_fallback_model", LocalCLIPEmbedder.DEFAULT_MODEL)
        )
        self.clip_local_files_only = self._as_bool(
            config.get("multimodal_clip_local_files_only"), True
        )
        self.use_cache = self._as_bool(config.get("use_image_embedding_cache"), True)
        cache_raw = str(config.get("image_embedding_cache_dir", "")).strip()
        self.cache_dir = (
            Path(cache_raw).expanduser()
            if cache_raw
            else Path.home() / ".cache" / "memeye" / "raw_image_embeddings"
        )
        if self.use_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning("Raw image embedding cache disabled: %s", exc)
                self.use_cache = False

        self.embedder: Any = None
        self.backend = ""
        self.model_name = ""
        self.fallback_reason = ""
        self.image_rows: List[Tuple[str, str, List[float]]] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self._activate_siglip_or_clip()

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _activate_siglip_or_clip(self) -> None:
        try:
            candidate = MultimodalEmbedder(self.preferred_model)
            if not candidate.is_available:
                raise RuntimeError(f"SigLIP model unavailable: {self.preferred_model}")
            self.embedder = candidate
            self.backend = "siglip"
            self.model_name = self.preferred_model
            self._build()
            return
        except Exception as exc:
            self._activate_clip(exc)

    def _activate_clip(self, reason: Exception) -> None:
        self.fallback_reason = f"{type(reason).__name__}: {reason}"
        log.warning(
            "Raw image retrieval SigLIP failed; falling back to local CLIP. "
            "siglip=%s clip=%s reason=%s",
            self.preferred_model,
            self.clip_model,
            self.fallback_reason,
        )
        candidate = LocalCLIPEmbedder(
            self.clip_model, local_files_only=self.clip_local_files_only
        )
        if not candidate.is_available:
            raise RuntimeError(
                "Raw image retrieval requires SigLIP or local CLIP, but both failed. "
                f"SigLIP error: {self.fallback_reason}"
            )
        self.embedder = candidate
        self.backend = "clip"
        self.model_name = self.clip_model
        self.image_rows = []
        self.cache_hits = 0
        self.cache_misses = 0
        self._build()

    def _image_cache_path(self, image_path: str) -> Path:
        path = Path(image_path).resolve()
        stat = path.stat()
        payload = {
            "version": self.CACHE_VERSION,
            "backend": self.backend,
            "model": self.model_name,
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / "images" / f"{key}.json"

    def _query_cache_path(self, text: str) -> Path:
        payload = {
            "version": self.CACHE_VERSION,
            "backend": self.backend,
            "model": self.model_name,
            "text": text,
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / "queries" / f"{key}.json"

    @staticmethod
    def _read_vector(path: Path) -> Optional[List[float]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            vector = payload.get("vector")
            if isinstance(vector, list) and vector:
                parsed = [float(value) for value in vector]
                return parsed if _valid_vector(parsed) else None
        except Exception:
            return None
        return None

    @staticmethod
    def _write_vector(path: Path, vector: List[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"vector": vector}), encoding="utf-8")
        tmp.replace(path)

    def _embed_image(self, image_path: str) -> List[float]:
        cache_path = self._image_cache_path(image_path)
        if self.use_cache and cache_path.exists():
            vector = self._read_vector(cache_path)
            if vector:
                self.cache_hits += 1
                return vector
        vector = self.embedder.embed_image(image_path)
        if not _valid_vector(vector):
            raise RuntimeError(f"Invalid image embedding for {image_path}")
        self.cache_misses += 1
        if self.use_cache:
            try:
                self._write_vector(cache_path, vector)
            except OSError as exc:
                log.warning("Raw image embedding cache write failed path=%s error=%s", cache_path, exc)
        return vector

    def _embed_query(self, text: str) -> List[float]:
        cache_path = self._query_cache_path(text)
        if self.use_cache and cache_path.exists():
            vector = self._read_vector(cache_path)
            if vector:
                return vector
        vector = self.embedder.embed_text(text)
        if not _valid_vector(vector):
            raise RuntimeError("Invalid text-to-image query embedding")
        if self.use_cache:
            try:
                self._write_vector(cache_path, vector)
            except OSError as exc:
                log.warning("Raw image embedding cache write failed path=%s error=%s", cache_path, exc)
        return vector

    def _build(self) -> None:
        rows: List[Tuple[str, str, List[float]]] = []
        for round_id, payload in self.dataset.rounds.items():
            for image_path in list(payload.get("images", []) or []):
                rows.append((str(round_id), str(image_path), self._embed_image(str(image_path))))
        self.image_rows = rows
        log.info(
            "Raw image round index built backend=%s model=%s images=%d cache_hits=%d "
            "cache_misses=%d fallback_reason=%s",
            self.backend,
            self.model_name,
            len(rows),
            self.cache_hits,
            self.cache_misses,
            self.fallback_reason or "<none>",
        )

    def search(self, query_text: str, top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return unique rounds ranked by their best matching original image."""
        try:
            return self._search(query_text, top_k)
        except Exception as exc:
            if self.backend != "siglip":
                raise
            self._activate_clip(exc)
            return self._search(query_text, top_k)

    def _search(self, query_text: str, top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_vec = self._embed_query(query_text)
        by_round: Dict[str, Dict[str, Any]] = {}
        for round_id, image_path, image_vec in self.image_rows:
            score = _cosine(query_vec, image_vec)
            current = by_round.get(round_id)
            if current is None or score > float(current["score"]):
                by_round[round_id] = {
                    "round_id": round_id,
                    "score": score,
                    "best_image_path": image_path,
                }
        ranked = sorted(by_round.values(), key=lambda item: (-float(item["score"]), item["round_id"]))
        selected = ranked[: max(0, int(top_k))]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        metadata = self.metadata()
        metadata["indexed_round_count"] = len(by_round)
        return selected, metadata

    def metadata(self) -> Dict[str, Any]:
        return {
            "image_embedding_backend": self.backend,
            "image_embedding_model": self.model_name,
            "siglip_preferred_model": self.preferred_model,
            "clip_fallback_model": self.clip_model,
            "clip_local_files_only": self.clip_local_files_only,
            "fallback_used": self.backend == "clip",
            "fallback_reason": self.fallback_reason,
            "indexed_image_count": len(self.image_rows),
            "image_embedding_cache_enabled": self.use_cache,
            "image_embedding_cache_hits": self.cache_hits,
            "image_embedding_cache_misses": self.cache_misses,
        }
