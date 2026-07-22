"""Raw-evidence calibration over a fixed EVI round candidate pool.

EVI supplies both the candidates and their primary ranking.  This module
independently scores the same rounds against the full question using raw
dialogue and original images, then combines the EVI and raw-evidence rankings
with an equal-weight mean rank.  It never receives evaluation annotations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..dataset import build_round_retrieval_text


RAW_TEXT_CACHE_VERSION = "evi_raw_multimodal_text_v1"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _rank_scores(scores: Mapping[str, float]) -> List[str]:
    return sorted(scores, key=lambda round_id: (-float(scores[round_id]), round_id))


def mean_rank_fuse(
    primary_ranking: Sequence[str], secondary_ranking: Sequence[str]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Fuse two rankings over the same fixed candidates without score weights."""
    primary = list(dict.fromkeys(str(value) for value in primary_ranking if str(value)))
    allowed = set(primary)
    secondary = list(
        dict.fromkeys(str(value) for value in secondary_ranking if str(value) in allowed)
    )
    secondary_seen = set(secondary)
    secondary.extend(value for value in primary if value not in secondary_seen)
    primary_ranks = {round_id: rank for rank, round_id in enumerate(primary, start=1)}
    secondary_ranks = {round_id: rank for rank, round_id in enumerate(secondary, start=1)}
    rows = [
        {
            "round_id": round_id,
            "evi_rank": primary_ranks[round_id],
            "raw_multimodal_rank": secondary_ranks[round_id],
            "mean_rank": 0.5 * (
                primary_ranks[round_id] + secondary_ranks[round_id]
            ),
        }
        for round_id in primary
    ]
    rows.sort(
        key=lambda item: (
            float(item["mean_rank"]),
            int(item["evi_rank"]),
            str(item["round_id"]),
        )
    )
    for final_rank, item in enumerate(rows, start=1):
        item["final_rank"] = final_rank
    return [str(item["round_id"]) for item in rows], rows


class RawMultimodalCandidateReranker:
    """Cached raw dialogue/image scorer for a single dataset instance."""

    def __init__(
        self,
        dataset: Any,
        text_embedder: Any,
        image_index: Any,
        config: Mapping[str, Any],
        embedding_namespace: str,
    ) -> None:
        self.dataset = dataset
        self.text_embedder = text_embedder
        self.image_index = image_index
        self.embedding_namespace = embedding_namespace
        self.text_weight = float(config.get("evi_raw_multimodal_text_weight", 0.5))
        self.image_weight = float(config.get("evi_raw_multimodal_image_weight", 0.5))
        if self.text_weight < 0 or self.image_weight < 0:
            raise ValueError("Raw multimodal weights must be non-negative")
        if self.text_weight == 0 and self.image_weight == 0:
            raise ValueError("At least one raw multimodal weight must be positive")
        self.use_cache = self._as_bool(
            config.get("use_raw_multimodal_text_embedding_cache"), True
        )
        configured_cache = str(
            config.get("raw_multimodal_text_embedding_cache_dir", "") or ""
        ).strip()
        self.cache_dir = Path(
            configured_cache
            or os.environ.get(
                "EVI_RAW_MM_EMBED_CACHE_DIR",
                str(Path.home() / ".cache" / "evi_raw_multimodal_embeddings"),
            )
        ).expanduser()
        if self.use_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.use_cache = False
        self.cache_hits = 0
        self.cache_misses = 0
        self.round_texts: Dict[str, str] = {
            str(round_id): build_round_retrieval_text(payload, modality="multimodal")
            for round_id, payload in dataset.rounds.items()
        }
        self.round_vectors = self._embed_many(
            list(self.round_texts.values()), role="document"
        )
        self.vector_by_round = dict(zip(self.round_texts, self.round_vectors))

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _cache_path(self, text: str, role: str) -> Path:
        payload = json.dumps(
            {
                "version": RAW_TEXT_CACHE_VERSION,
                "namespace": self.embedding_namespace,
                "role": role,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / role / f"{digest}.json"

    @staticmethod
    def _read_vector(path: Path) -> List[float]:
        try:
            vector = json.loads(path.read_text(encoding="utf-8")).get("vector", [])
            return [float(value) for value in vector] if vector else []
        except Exception:
            return []

    @staticmethod
    def _write_vector(path: Path, vector: Sequence[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"vector": list(vector)}), encoding="utf-8")
        temporary.replace(path)

    def _embed_many(self, texts: Sequence[str], role: str) -> List[List[float]]:
        output: List[List[float]] = [[] for _ in texts]
        missing: List[int] = []
        for index, text in enumerate(texts):
            path = self._cache_path(text, role)
            vector = self._read_vector(path) if self.use_cache and path.exists() else []
            if vector:
                self.cache_hits += 1
                output[index] = vector
            else:
                missing.append(index)
        if missing:
            missing_texts = [texts[index] for index in missing]
            vectors = (
                [self.text_embedder.embed_query(text) for text in missing_texts]
                if role == "query"
                else self.text_embedder.embed_batch(missing_texts)
            )
            for index, vector in zip(missing, vectors):
                parsed = [float(value) for value in vector]
                output[index] = parsed
                self.cache_misses += 1
                if self.use_cache and parsed:
                    try:
                        self._write_vector(self._cache_path(texts[index], role), parsed)
                    except OSError:
                        pass
        return output

    def rerank(
        self, question: str, evi_ranked_round_ids: Sequence[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        candidates = list(
            dict.fromkeys(
                str(round_id)
                for round_id in evi_ranked_round_ids
                if str(round_id) in self.round_texts
            )
        )
        if not candidates:
            return [], {"enabled": True, "candidate_count": 0, "rows": []}
        query_vector = self._embed_many([question], role="query")[0]
        image_hits, image_metadata = self.image_index.search(
            question, len(self.dataset.rounds)
        )
        image_by_round = {
            str(item["round_id"]): dict(item) for item in image_hits
        }
        score_rows: Dict[str, Dict[str, Any]] = {}
        raw_scores: Dict[str, float] = {}
        for round_id in candidates:
            text_score = _cosine(query_vector, self.vector_by_round.get(round_id, []))
            image_hit = image_by_round.get(round_id)
            image_score = float(image_hit["score"]) if image_hit is not None else 0.0
            raw_score = self.text_weight * text_score + self.image_weight * image_score
            raw_scores[round_id] = raw_score
            score_rows[round_id] = {
                "text_score": text_score,
                "image_score": image_score,
                "raw_multimodal_score": raw_score,
                "has_image": image_hit is not None,
                "best_image_path": (
                    str(image_hit.get("best_image_path", "")) if image_hit else ""
                ),
            }
        raw_ranking = _rank_scores(raw_scores)
        fused_ranking, fusion_rows = mean_rank_fuse(candidates, raw_ranking)
        for item in fusion_rows:
            item.update(score_rows[str(item["round_id"])])
        trace = {
            "enabled": True,
            "candidate_policy": "fixed_evi_top_k",
            "fusion": "equal_weight_mean_rank",
            "missing_image_policy": "zero_image_score",
            "candidate_count": len(candidates),
            "text_weight": self.text_weight,
            "image_weight": self.image_weight,
            "evi_ranked_round_ids": candidates,
            "raw_multimodal_ranked_round_ids": raw_ranking,
            "fused_ranked_round_ids": fused_ranking,
            "rows": fusion_rows,
            "text_embedding_cache": {
                "enabled": self.use_cache,
                "path": str(self.cache_dir),
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            },
            **image_metadata,
        }
        return fused_ranking, trace
