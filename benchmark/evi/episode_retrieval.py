from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .schemas import EvidenceAnchor


def merge_facet_sessions(
    facet_results: List[Tuple[str, List[Dict[str, Any]]]],
    facet_round_fusion: str,
) -> List[Dict[str, Any]]:
    """Merge per-facet session-set matches without losing witness rounds."""
    session_data: Dict[str, Dict[str, Any]] = {}
    for facet_idx, (facet, session_hits) in enumerate(facet_results):
        facet_key = f"f{facet_idx}"
        for rank, hit in enumerate(session_hits, start=1):
            anchor = hit.get("best_anchor")
            if not isinstance(anchor, EvidenceAnchor):
                continue
            session_id = str(hit["session_id"])
            score = float(hit.get("score", 0.0))
            item = session_data.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "date": str(hit["date"]),
                    "max_score": 0.0,
                    "matched_facets": set(),
                    "facet_scores": {},
                    "facet_ranks": {},
                    "facet_witness_rounds": {},
                    "source_scores": {"dialogue": 0.0, "visual": 0.0},
                    "source_facet_ranks": {},
                    "source_facet_scores": {},
                    "source_facet_witness_rounds": {},
                    "best_source_facet_ranks": {},
                    "best_source_by_facet": {},
                    "best_source_witness_rounds": {},
                },
            )
            item["max_score"] = max(float(item["max_score"]), score)
            item["matched_facets"].add(facet_key)
            if score > float(item["facet_scores"].get(facet, 0.0)):
                item["facet_scores"][facet] = score
                item["facet_ranks"][facet] = rank
                item["facet_witness_rounds"][facet] = anchor.round_id

            source_scores = dict(hit.get("source_scores", {}))
            source_anchors = dict(hit.get("source_anchors", {}))
            source_ranks = dict(hit.get("source_ranks", {}))
            for source, source_score in source_scores.items():
                item["source_scores"][source] = max(
                    float(item["source_scores"].get(source, 0.0)),
                    float(source_score),
                )
            for source, source_rank in source_ranks.items():
                source_facet_key = f"{facet_key}:{source}"
                item["source_facet_ranks"][source_facet_key] = int(source_rank)
                item["source_facet_scores"][source_facet_key] = float(
                    source_scores.get(source, 0.0)
                )
                source_anchor = source_anchors.get(source)
                if isinstance(source_anchor, EvidenceAnchor):
                    item["source_facet_witness_rounds"][source_facet_key] = (
                        source_anchor.round_id
                    )
            if source_ranks:
                best_source, best_source_rank = min(
                    source_ranks.items(),
                    key=lambda pair: (
                        int(pair[1]),
                        -float(source_scores.get(pair[0], 0.0)),
                        str(pair[0]),
                    ),
                )
                item["best_source_facet_ranks"][facet_key] = int(best_source_rank)
                item["best_source_by_facet"][facet_key] = str(best_source)
                witness = source_anchors.get(best_source)
                if isinstance(witness, EvidenceAnchor):
                    item["best_source_witness_rounds"][facet_key] = witness.round_id

    merged: List[Dict[str, Any]] = []
    for item in session_data.values():
        if facet_round_fusion == "source_facet_reciprocal_rank_consensus":
            consensus_score = sum(
                1.0 / max(1, int(rank))
                for rank in item["source_facet_ranks"].values()
            )
            final_score = consensus_score
        elif facet_round_fusion == "max_similarity_times_best_source_rank_consensus":
            consensus_score = sum(
                1.0 / max(1, int(rank))
                for rank in item["best_source_facet_ranks"].values()
            )
            final_score = float(item["max_score"]) * consensus_score
        else:
            consensus_score = sum(
                1.0 / max(1, int(rank))
                for rank in item["facet_ranks"].values()
            )
            final_score = float(item["max_score"]) * consensus_score
        item["matched_facets"] = len(item["matched_facets"])
        item["consensus_score"] = consensus_score
        item["score"] = final_score
        merged.append(item)

    merged.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["consensus_score"]),
            int(item["matched_facets"]),
            float(item["max_score"]),
            str(item["session_id"]),
        ),
        reverse=True,
    )
    return merged


def build_episode_round_path(
    merged_sessions: List[Dict[str, Any]],
    direct_round_ids: List[str],
    session_rounds: Dict[str, List[str]],
    limit: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Expand ranked episodes, retaining direct rank within each episode."""
    direct_ranks = {
        round_id: rank for rank, round_id in enumerate(direct_round_ids, start=1)
    }
    rows: List[Dict[str, Any]] = []
    limit = max(1, int(limit))
    for episode_rank, episode in enumerate(merged_sessions, start=1):
        session_id = str(episode["session_id"])
        members = list(session_rounds.get(session_id, []))
        member_order = {round_id: index for index, round_id in enumerate(members)}
        members.sort(
            key=lambda round_id: (
                direct_ranks.get(round_id, len(direct_ranks) + len(members) + 1),
                member_order[round_id],
            )
        )
        for local_rank, round_id in enumerate(members, start=1):
            rows.append(
                {
                    "round_id": round_id,
                    "session_id": session_id,
                    "episode_rank": episode_rank,
                    "episode_score": float(episode["score"]),
                    "local_rank": local_rank,
                    "direct_rank": direct_ranks.get(round_id),
                }
            )
            if len(rows) >= limit:
                return [str(item["round_id"]) for item in rows], rows
    return [str(item["round_id"]) for item in rows], rows


def fuse_direct_episode_rounds(
    direct_round_ids: List[str],
    episode_round_ids: List[str],
    round_session: Dict[str, str],
    pool_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Fuse direct and episode paths with parameter-free reciprocal ranks."""
    pool_size = max(1, int(pool_size))
    direct_ids = direct_round_ids[:pool_size]
    episode_ids = episode_round_ids[:pool_size]
    direct_ranks = {round_id: rank for rank, round_id in enumerate(direct_ids, start=1)}
    episode_ranks = {
        round_id: rank for rank, round_id in enumerate(episode_ids, start=1)
    }
    candidate_ids = list(dict.fromkeys(direct_ids + episode_ids))
    rows: List[Dict[str, Any]] = []
    for round_id in candidate_ids:
        direct_rank = direct_ranks.get(round_id)
        episode_rank = episode_ranks.get(round_id)
        score = (
            (1.0 / direct_rank if direct_rank is not None else 0.0)
            + (1.0 / episode_rank if episode_rank is not None else 0.0)
        )
        rows.append(
            {
                "round_id": round_id,
                "session_id": round_session.get(round_id, ""),
                "score": score,
                "direct_rank": direct_rank,
                "episode_rank": episode_rank,
                "matched_paths": int(direct_rank is not None)
                + int(episode_rank is not None),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["matched_paths"]),
            int(item["direct_rank"] or (pool_size + 1)),
            int(item["episode_rank"] or (pool_size + 1)),
            str(item["round_id"]),
        )
    )
    ranked = [str(item["round_id"]) for item in rows[:pool_size]]
    return ranked, rows
