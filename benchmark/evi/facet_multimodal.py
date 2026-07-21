"""Facet-conditioned multimodal scoring over provenance-bound memory rounds.

This module deliberately contains no dataset, task-name, question-type, or clue
logic. It supports both the early-mean ablation and a provenance-gated policy
that corroborates visual anchors with raw images before facet-local source
selection.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .schemas import EvidenceAnchor


def empirical_midrank_percentiles(
    values: Mapping[str, float],
) -> Dict[str, float]:
    """Return deterministic empirical midrank percentiles in the unit interval.

    A tied group receives (# lower + 0.5 * # equal) / N. This makes an
    uninformative all-tied source neutral (0.5) rather than arbitrarily ordering
    rounds by identifier.
    """
    if not values:
        return {}
    groups: Dict[float, List[str]] = defaultdict(list)
    for key, value in values.items():
        groups[float(value)].append(str(key))

    total = len(values)
    lower = 0
    out: Dict[str, float] = {}
    for value in sorted(groups):
        keys = sorted(groups[value])
        percentile = (lower + 0.5 * len(keys)) / total
        for key in keys:
            out[key] = float(percentile)
        lower += len(keys)
    return out


def _available_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def _descending_ranks(values: Mapping[str, float]) -> Dict[str, int]:
    ordered = sorted(values, key=lambda key: (-float(values[key]), str(key)))
    return {key: rank for rank, key in enumerate(ordered, start=1)}


def score_visual_corroborated_best_source_rounds(
    anchor_rows: List[Dict[str, Any]],
    image_hits: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rank dialogue and visually corroborated evidence as alternative paths.

    Raw images may reorder rounds that already have a provenance-bound visual
    anchor, but cannot introduce image-only evidence. Each source retains its
    own Top-K candidate budget; the better source rank is exposed to the
    existing cross-facet best-source consensus.
    """
    anchor_by_round = {str(row["round_id"]): row for row in anchor_rows}
    image_by_round = {str(row["round_id"]): row for row in image_hits}

    dialogue_raw = {
        rid: float(row["source_scores"]["dialogue"])
        for rid, row in anchor_by_round.items()
        if "dialogue" in dict(row.get("source_scores", {}))
    }
    visual_anchor_raw = {
        rid: float(row["source_scores"]["visual"])
        for rid, row in anchor_by_round.items()
        if "visual" in dict(row.get("source_scores", {}))
    }
    raw_image_raw = {
        rid: float(row.get("score", 0.0)) for rid, row in image_by_round.items()
    }
    dialogue_cal = empirical_midrank_percentiles(dialogue_raw)
    visual_anchor_cal = empirical_midrank_percentiles(visual_anchor_raw)
    raw_image_cal = empirical_midrank_percentiles(raw_image_raw)
    dialogue_ranks = _descending_ranks(dialogue_raw)
    visual_anchor_ranks = _descending_ranks(visual_anchor_raw)
    raw_image_ranks = _descending_ranks(raw_image_raw)

    visual_corroboration_scores = {
        rid: (1.0 / visual_anchor_ranks[rid])
        + (1.0 / raw_image_ranks[rid] if rid in raw_image_ranks else 0.0)
        for rid in visual_anchor_ranks
    }
    corroborated_visual_ranks = _descending_ranks(visual_corroboration_scores)
    source_k = max(0, int(top_k))
    dialogue_candidates = {
        rid for rid, rank in dialogue_ranks.items() if source_k <= 0 or rank <= source_k
    }
    visual_candidates = {
        rid
        for rid, rank in corroborated_visual_ranks.items()
        if source_k <= 0 or rank <= source_k
    }
    candidate_ids = dialogue_candidates | visual_candidates

    ranked: List[Dict[str, Any]] = []
    for rid in candidate_ids:
        anchor_row = anchor_by_round[rid]
        source_anchors = dict(anchor_row.get("source_anchors", {}))
        source_ranks: Dict[str, int] = {}
        if rid in dialogue_candidates:
            source_ranks["dialogue"] = dialogue_ranks[rid]
        if rid in visual_candidates:
            source_ranks["visual"] = corroborated_visual_ranks[rid]
        best_source = min(
            source_ranks,
            key=lambda source: (
                source_ranks[source],
                -float(
                    dialogue_cal.get(rid, 0.0)
                    if source == "dialogue"
                    else visual_anchor_cal.get(rid, 0.0)
                ),
                source,
            ),
        )
        best_anchor = source_anchors.get(best_source)
        if not isinstance(best_anchor, EvidenceAnchor):
            continue
        semantic_strength = max(
            float(dialogue_cal.get(rid, 0.0)),
            float(visual_anchor_cal.get(rid, 0.0)),
        )
        best_source_rank = int(source_ranks[best_source])
        ranked.append(
            {
                "round_id": rid,
                "session_id": str(anchor_row.get("session_id", best_anchor.session_id)),
                "date": str(anchor_row.get("date", best_anchor.date)),
                "score": semantic_strength,
                "best_anchor": replace(best_anchor, score=semantic_strength),
                "source_scores": {
                    source: score
                    for source, score in {
                        "dialogue": dialogue_cal.get(rid),
                        "visual": visual_anchor_cal.get(rid),
                        "raw_image": raw_image_cal.get(rid),
                    }.items()
                    if score is not None
                },
                "source_ranks": source_ranks,
                "source_anchors": source_anchors,
                "multimodal_scores": {
                    "dialogue_calibrated": dialogue_cal.get(rid),
                    "visual_anchor_calibrated": visual_anchor_cal.get(rid),
                    "raw_image_calibrated": raw_image_cal.get(rid),
                    "visual_corroboration_score": visual_corroboration_scores.get(rid),
                    "semantic_strength": semantic_strength,
                },
                "multimodal_ranks": {
                    "dialogue_rank": dialogue_ranks.get(rid),
                    "visual_anchor_rank": visual_anchor_ranks.get(rid),
                    "raw_image_rank": raw_image_ranks.get(rid),
                    "corroborated_visual_rank": corroborated_visual_ranks.get(rid),
                    "best_source_rank": best_source_rank,
                },
                "best_source": best_source,
                "best_image_path": str(
                    image_by_round.get(rid, {}).get("best_image_path", "")
                ),
            }
        )

    ranked.sort(
        key=lambda item: (
            int(item["multimodal_ranks"]["best_source_rank"]),
            -float(item["score"]),
            str(item["round_id"]),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    trace = {
        "policy": "visual_corroborated_best_source",
        "score_calibration": "empirical_midrank_percentile",
        "visual_corroboration": "reciprocal_rank_sum_over_visual_anchor_candidates",
        "source_selection": "minimum_source_rank_per_facet_round",
        "candidate_policy": "dialogue_top_k_union_corroborated_visual_top_k",
        "raw_image_admission": "visual_anchor_candidates_only",
        "source_top_k": source_k,
        "num_anchor_rounds": len(anchor_by_round),
        "num_image_rounds": len(image_by_round),
        "num_dialogue_candidates": len(dialogue_candidates),
        "num_visual_candidates": len(visual_candidates),
        "num_union_candidates": len(ranked),
        "excluded_image_only_round_ids": sorted(set(image_by_round) - set(visual_anchor_ranks)),
        "rounds": [
            {
                "round_id": item["round_id"],
                "rank": item["rank"],
                "score": item["score"],
                "source_scores": dict(item["source_scores"]),
                "source_ranks": dict(item["source_ranks"]),
                "multimodal_scores": dict(item["multimodal_scores"]),
                "multimodal_ranks": dict(item["multimodal_ranks"]),
                "best_source": item["best_source"],
                "best_image_path": item["best_image_path"],
            }
            for item in ranked
        ],
    }
    return ranked, trace


def score_multimodal_facet_rounds(
    anchor_rows: List[Dict[str, Any]],
    image_hits: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build one early-fused multimodal ranking for a single query facet.

    Dialogue anchors form the text branch. VLM-derived visual anchors and raw
    image similarity are averaged into one visual branch, preventing an image
    round from receiving two independent final votes. Missing visual evidence
    is treated as unavailable rather than as negative evidence.
    """
    anchor_by_round = {str(row["round_id"]): row for row in anchor_rows}
    image_by_round = {str(row["round_id"]): row for row in image_hits}
    round_ids = sorted(set(anchor_by_round) | set(image_by_round))

    dialogue_raw = {
        rid: float(anchor_by_round[rid]["source_scores"]["dialogue"])
        for rid in round_ids
        if rid in anchor_by_round
        and "dialogue" in dict(anchor_by_round[rid].get("source_scores", {}))
    }
    visual_anchor_raw = {
        rid: float(anchor_by_round[rid]["source_scores"]["visual"])
        for rid in round_ids
        if rid in anchor_by_round
        and "visual" in dict(anchor_by_round[rid].get("source_scores", {}))
    }
    raw_image_raw = {
        rid: float(image_by_round[rid].get("score", 0.0))
        for rid in round_ids
        if rid in image_by_round
    }

    dialogue_cal = empirical_midrank_percentiles(dialogue_raw)
    visual_anchor_cal = empirical_midrank_percentiles(visual_anchor_raw)
    raw_image_cal = empirical_midrank_percentiles(raw_image_raw)

    ranked: List[Dict[str, Any]] = []
    for rid in round_ids:
        anchor_row = anchor_by_round.get(rid, {})
        source_anchors = dict(anchor_row.get("source_anchors", {}))
        dialogue_score = dialogue_cal.get(rid)
        visual_semantic_score = visual_anchor_cal.get(rid)
        visual_raw_score = raw_image_cal.get(rid)
        visual_score = _available_mean(
            [visual_semantic_score, visual_raw_score]
        )
        multimodal_score = _available_mean([dialogue_score, visual_score])
        if multimodal_score is None:
            continue

        best_anchor: Optional[EvidenceAnchor] = None
        if dialogue_score is not None or visual_semantic_score is not None:
            if (
                visual_semantic_score is not None
                and (
                    dialogue_score is None
                    or visual_semantic_score > dialogue_score
                )
            ):
                best_anchor = source_anchors.get("visual")
            else:
                best_anchor = source_anchors.get("dialogue")
        if best_anchor is None and source_anchors:
            best_anchor = next(iter(source_anchors.values()))
        if not isinstance(best_anchor, EvidenceAnchor):
            # Every usable EVI round has a provenance-bound dialogue anchor.
            # Image-only rows without one are not admitted as memory evidence.
            continue

        ranked.append(
            {
                "round_id": rid,
                "session_id": str(anchor_row.get("session_id", best_anchor.session_id)),
                "date": str(anchor_row.get("date", best_anchor.date)),
                "score": float(multimodal_score),
                "best_anchor": best_anchor,
                "source_scores": {
                    source: score
                    for source, score in {
                        "dialogue": dialogue_score,
                        "visual": visual_score,
                        "raw_image": visual_raw_score,
                    }.items()
                    if score is not None
                },
                "source_ranks": {},
                "source_anchors": source_anchors,
                "multimodal_scores": {
                    "dialogue_calibrated": dialogue_score,
                    "visual_anchor_calibrated": visual_semantic_score,
                    "raw_image_calibrated": visual_raw_score,
                    "visual_combined": visual_score,
                    "multimodal": float(multimodal_score),
                },
                "has_visual": visual_score is not None,
                "best_image_path": str(
                    image_by_round.get(rid, {}).get("best_image_path", "")
                ),
            }
        )

    ranked.sort(key=lambda item: (-float(item["score"]), str(item["round_id"])))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    selected = ranked[: max(0, int(top_k))] if top_k > 0 else ranked
    trace = {
        "score_calibration": "empirical_midrank_percentile",
        "visual_view_fusion": "available_mean",
        "modality_fusion": "available_mean",
        "candidate_unit": "round",
        "reported_source_scores": "calibrated",
        "num_anchor_rounds": len(anchor_by_round),
        "num_image_rounds": len(image_by_round),
        "num_scored_rounds": len(ranked),
        "rounds": [
            {
                "round_id": item["round_id"],
                "rank": int(item["rank"]),
                "score": float(item["score"]),
                "source_scores": dict(item["source_scores"]),
                "multimodal_scores": dict(item["multimodal_scores"]),
                "has_visual": bool(item["has_visual"]),
                "best_image_path": item["best_image_path"],
            }
            for item in ranked
        ],
    }
    return selected, trace
