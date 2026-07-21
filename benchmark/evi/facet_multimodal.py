"""Facet-conditioned multimodal scoring over provenance-bound memory rounds.

This module deliberately contains no dataset, task-name, question-type, or clue
logic. It converts source-specific similarities into comparable within-query
percentiles, combines the two visual views first, and only then combines text
and visual evidence into one score per (facet, round).
"""
from __future__ import annotations

from collections import defaultdict
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
