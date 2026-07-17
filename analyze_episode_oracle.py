"""Measure retrieval headroom from oracle episode relevance filtering.

Clue session ids are used only as an evaluation oracle. The script replays the
saved episode expansion, rank fusion, and image reranking without model calls.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from benchmark.common import write_json, write_jsonl
from benchmark.retrieval_eval import summarize_retrievals


DEFAULT_K = 10
DEFAULT_SESSION_KS = (1, 3, 5, 10)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_positive_ints(raw: str) -> List[int]:
    try:
        values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Expected at least one positive integer")
    return values


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _fallback_session_id(round_id: str) -> str:
    marker = round_id.rfind(":R")
    return round_id[:marker] if marker > 0 else round_id


def _resolve_task_run_dir(entry: Dict[str, Any], suite_dir: Path) -> Path:
    task_name = str(entry["task_name"])
    local = suite_dir / task_name
    if (local / "retrievals.jsonl").exists():
        return local
    configured = Path(str(entry.get("run_dir", "")))
    if (configured / "retrievals.jsonl").exists():
        return configured
    legacy = suite_dir.parent.parent / task_name / "retrieval" / configured.name
    if (legacy / "retrievals.jsonl").exists():
        return legacy
    raise FileNotFoundError(
        f"Could not locate retrievals.jsonl for task={task_name}. "
        f"Tried {local}, {configured}, and {legacy}."
    )


def _round_session_map(episode_trace: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for episode in episode_trace.get("episodes", []) or []:
        session_id = str(episode.get("session_id", ""))
        for round_id in episode.get("member_round_ids", []) or []:
            mapping[str(round_id)] = session_id
    return mapping


def _clue_sessions(clues: List[str], round_sessions: Dict[str, str]) -> List[str]:
    return _ordered_unique(
        round_sessions.get(round_id, _fallback_session_id(round_id)) for round_id in clues
    )


def _build_episode_path(
    episodes: List[Dict[str, Any]], direct_round_ids: List[str], limit: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    direct_ranks = {
        round_id: rank for rank, round_id in enumerate(direct_round_ids, start=1)
    }
    rows: List[Dict[str, Any]] = []
    for episode_rank, episode in enumerate(episodes, start=1):
        session_id = str(episode["session_id"])
        members = [str(value) for value in episode.get("member_round_ids", []) or []]
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
                    "local_rank": local_rank,
                    "direct_rank": direct_ranks.get(round_id),
                }
            )
            if len(rows) >= limit:
                return [str(item["round_id"]) for item in rows], rows
    return [str(item["round_id"]) for item in rows], rows


def _build_balanced_episode_path(
    episodes: List[Dict[str, Any]], direct_round_ids: List[str], limit: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Allocate one round per episode per pass under the unchanged total budget."""
    direct_ranks = {
        round_id: rank for rank, round_id in enumerate(direct_round_ids, start=1)
    }
    ordered_members: List[Tuple[str, List[str]]] = []
    for episode in episodes:
        session_id = str(episode["session_id"])
        members = [str(value) for value in episode.get("member_round_ids", []) or []]
        member_order = {round_id: index for index, round_id in enumerate(members)}
        members.sort(
            key=lambda round_id: (
                direct_ranks.get(round_id, len(direct_ranks) + len(members) + 1),
                member_order[round_id],
            )
        )
        ordered_members.append((session_id, members))

    rows: List[Dict[str, Any]] = []
    local_index = 0
    while len(rows) < limit:
        added = False
        for episode_rank, (session_id, members) in enumerate(ordered_members, start=1):
            if local_index >= len(members):
                continue
            round_id = members[local_index]
            rows.append(
                {
                    "round_id": round_id,
                    "session_id": session_id,
                    "episode_rank": episode_rank,
                    "local_rank": local_index + 1,
                    "direct_rank": direct_ranks.get(round_id),
                }
            )
            added = True
            if len(rows) >= limit:
                break
        if not added:
            break
        local_index += 1
    return [str(item["round_id"]) for item in rows], rows


def _rank_episode_sessions(
    episodes: List[Dict[str, Any]], strategy: str
) -> List[Dict[str, Any]]:
    if strategy == "current":
        return list(episodes)
    fields = {
        "max_score": "max_score",
        "consensus_score": "consensus_score",
        "matched_facets": "matched_facets",
    }
    if strategy not in fields:
        raise ValueError(f"Unknown session reranking strategy: {strategy}")
    field = fields[strategy]
    indexed = list(enumerate(episodes))
    indexed.sort(
        key=lambda pair: (-float(pair[1].get(field, 0.0) or 0.0), pair[0])
    )
    return [episode for _, episode in indexed]


def _replay_episode_selection(
    episodes: List[Dict[str, Any]],
    direct_ids: List[str],
    image_trace: Dict[str, Any],
    episode_limit: int,
    balanced: bool = False,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    build_path = _build_balanced_episode_path if balanced else _build_episode_path
    episode_ids, episode_rows = build_path(episodes, direct_ids, episode_limit)
    pool_size = max(episode_limit, len(direct_ids), len(_image_ids(image_trace)), 1)
    pre_image_ids = _fuse_direct_episode(direct_ids, episode_ids, pool_size)
    return _fuse_image(pre_image_ids, image_trace), episode_ids, episode_rows


def _fuse_direct_episode(
    direct_round_ids: List[str], episode_round_ids: List[str], pool_size: int
) -> List[str]:
    direct_ids = direct_round_ids[:pool_size]
    episode_ids = episode_round_ids[:pool_size]
    direct_ranks = {round_id: rank for rank, round_id in enumerate(direct_ids, start=1)}
    episode_ranks = {round_id: rank for rank, round_id in enumerate(episode_ids, start=1)}
    rows: List[Dict[str, Any]] = []
    for round_id in _ordered_unique(direct_ids + episode_ids):
        direct_rank = direct_ranks.get(round_id)
        episode_rank = episode_ranks.get(round_id)
        matched_paths = int(direct_rank is not None) + int(episode_rank is not None)
        rows.append(
            {
                "round_id": round_id,
                "score": (1.0 / direct_rank if direct_rank else 0.0)
                + (1.0 / episode_rank if episode_rank else 0.0),
                "direct_rank": direct_rank,
                "episode_rank": episode_rank,
                "matched_paths": matched_paths,
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
    return [str(item["round_id"]) for item in rows[:pool_size]]


def _image_ids(image_trace: Dict[str, Any]) -> List[str]:
    image_rows = image_trace.get("image_rounds", []) or []
    if image_rows:
        return [
            str(item["round_id"])
            for item in sorted(image_rows, key=lambda item: int(item.get("rank", 10**9)))
        ]
    return [str(value) for value in image_trace.get("image_ranked_round_ids", []) or []]


def _fuse_image(pre_image_ids: List[str], image_trace: Dict[str, Any]) -> List[str]:
    if not image_trace.get("enabled"):
        return list(pre_image_ids)
    image_ids = _image_ids(image_trace)
    search_k = max(len(image_ids), len(image_trace.get("anchor_ranked_round_ids", []) or []), 1)
    anchor_ids = pre_image_ids[:search_k]
    anchor_ranks = {round_id: rank for rank, round_id in enumerate(anchor_ids, start=1)}
    image_ranks = {round_id: rank for rank, round_id in enumerate(image_ids, start=1)}
    anchor_only = image_trace.get("candidate_policy") == "anchor_candidates_only"
    candidate_ids = anchor_ids if anchor_only else _ordered_unique(anchor_ids + image_ids)
    rows: List[Dict[str, Any]] = []
    for round_id in candidate_ids:
        anchor_rank = anchor_ranks.get(round_id)
        image_rank = image_ranks.get(round_id)
        matched_sources = int(anchor_rank is not None) + int(image_rank is not None)
        rows.append(
            {
                "round_id": round_id,
                "score": (1.0 / anchor_rank if anchor_rank else 0.0)
                + (1.0 / image_rank if image_rank else 0.0),
                "matched_sources": matched_sources,
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["matched_sources"]),
            str(item["round_id"]),
        )
    )
    return [str(item["round_id"]) for item in rows]


def _question_stats(ranked: List[str], clues: List[str], k: int) -> Dict[str, Any]:
    selected = ranked[:k]
    selected_set = set(selected)
    hits = [round_id for round_id in clues if round_id in selected_set]
    return {
        "top_k_round_ids": selected,
        "hit_clue_round_ids": hits,
        "missed_clue_round_ids": [round_id for round_id in clues if round_id not in selected_set],
        "recall": len(hits) / len(clues) if clues else 0.0,
        "full_coverage": bool(clues) and len(hits) == len(clues),
    }


def _classify_misses(
    clues: List[str],
    final_ids: List[str],
    direct_ids: List[str],
    episode_ids: List[str],
    ranked_sessions: List[str],
    expanded_sessions: List[str],
    round_sessions: Dict[str, str],
    k: int,
) -> Dict[str, List[str]]:
    final_set = set(final_ids[:k])
    candidate_set = set(direct_ids) | set(episode_ids)
    ranked_session_set = set(ranked_sessions)
    expanded_set = set(expanded_sessions)
    output = {
        "session_routing_miss": [],
        "episode_path_budget_miss": [],
        "intra_session_miss": [],
        "fusion_displacement": [],
    }
    for round_id in clues:
        if round_id in final_set:
            continue
        session_id = round_sessions.get(round_id, _fallback_session_id(round_id))
        if round_id in candidate_set:
            output["fusion_displacement"].append(round_id)
        elif session_id not in ranked_session_set:
            output["session_routing_miss"].append(round_id)
        elif session_id in expanded_set:
            output["intra_session_miss"].append(round_id)
        else:
            output["episode_path_budget_miss"].append(round_id)
    return output


def _session_metrics(
    rows: List[Dict[str, Any]],
    ks: Iterable[int],
    ranked_field: str = "current_ranked_session_ids",
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for k in ks:
        clue_total = hit_total = 0
        recalls: List[float] = []
        hits: List[float] = []
        full: List[float] = []
        reciprocal_ranks: List[float] = []
        for row in rows:
            targets = list(row["clue_session_ids"])
            ranked = list(row[ranked_field])
            selected = set(ranked[:k])
            found = sum(1 for value in targets if value in selected)
            clue_total += len(targets)
            hit_total += found
            recalls.append(found / len(targets) if targets else 0.0)
            hits.append(1.0 if found else 0.0)
            full.append(1.0 if targets and found == len(targets) else 0.0)
            first = next(
                (rank for rank, value in enumerate(ranked[:k], start=1) if value in set(targets)),
                None,
            )
            reciprocal_ranks.append(1.0 / first if first else 0.0)
        count = len(rows)
        output[str(k)] = {
            "target_session_recall_micro": hit_total / clue_total if clue_total else 0.0,
            "target_session_recall_macro": sum(recalls) / count if count else 0.0,
            "target_session_hit_rate": sum(hits) / count if count else 0.0,
            "target_session_full_coverage_rate": sum(full) / count if count else 0.0,
            "target_session_mrr": sum(reciprocal_ranks) / count if count else 0.0,
            "num_questions": count,
            "num_target_sessions": clue_total,
        }
    return output


def _session_ranking_quality(
    rows: List[Dict[str, Any]], ranked_field: str
) -> Dict[str, Any]:
    target_total = selected_hits = 0
    recall_at_m: List[float] = []
    precision_at_m: List[float] = []
    f1_at_m: List[float] = []
    average_precisions: List[float] = []
    ndcg_values: List[float] = []
    first_ranks: List[int] = []
    last_ranks: List[int] = []
    noise_before_last: List[int] = []
    complete_rankings = 0
    for row in rows:
        targets = set(row["clue_session_ids"])
        ranked = list(row[ranked_field])
        m = len(targets)
        selected = ranked[:m]
        found = sum(1 for value in selected if value in targets)
        recall = found / m if m else 0.0
        precision = found / len(selected) if selected else 0.0
        f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
        target_total += m
        selected_hits += found
        recall_at_m.append(recall)
        precision_at_m.append(precision)
        f1_at_m.append(f1)

        relevant_seen = 0
        precision_sum = 0.0
        dcg = 0.0
        target_ranks: List[int] = []
        for rank, session_id in enumerate(ranked, start=1):
            if session_id not in targets:
                continue
            relevant_seen += 1
            target_ranks.append(rank)
            precision_sum += relevant_seen / rank
            dcg += 1.0 / math.log2(rank + 1)
        average_precisions.append(precision_sum / m if m else 0.0)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, m + 1))
        ndcg_values.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        if target_ranks:
            first_ranks.append(min(target_ranks))
        if len(target_ranks) == m and m:
            complete_rankings += 1
            last_rank = max(target_ranks)
            last_ranks.append(last_rank)
            noise_before_last.append(last_rank - m)

    count = len(rows)
    return {
        "session_recall_at_oracle_cardinality_micro": (
            selected_hits / target_total if target_total else 0.0
        ),
        "session_recall_at_oracle_cardinality_macro": (
            sum(recall_at_m) / count if count else 0.0
        ),
        "session_precision_at_oracle_cardinality_macro": (
            sum(precision_at_m) / count if count else 0.0
        ),
        "session_f1_at_oracle_cardinality_macro": (
            sum(f1_at_m) / count if count else 0.0
        ),
        "session_map": sum(average_precisions) / count if count else 0.0,
        "session_ndcg": sum(ndcg_values) / count if count else 0.0,
        "mean_first_target_rank": sum(first_ranks) / len(first_ranks) if first_ranks else None,
        "mean_last_target_rank": sum(last_ranks) / len(last_ranks) if last_ranks else None,
        "mean_irrelevant_sessions_before_last_target": (
            sum(noise_before_last) / len(noise_before_last) if noise_before_last else None
        ),
        "full_target_session_ranking_coverage_rate": (
            complete_rankings / count if count else 0.0
        ),
        "num_questions": count,
        "num_target_sessions": target_total,
    }


def _session_question_scores(
    row: Dict[str, Any], ranked_field: str
) -> Tuple[float, float]:
    targets = set(row["clue_session_ids"])
    ranked = list(row[ranked_field])
    m = len(targets)
    recall_at_m = (
        sum(1 for value in ranked[:m] if value in targets) / m if m else 0.0
    )
    relevant_seen = 0
    precision_sum = 0.0
    for rank, session_id in enumerate(ranked, start=1):
        if session_id not in targets:
            continue
        relevant_seen += 1
        precision_sum += relevant_seen / rank
    average_precision = precision_sum / m if m else 0.0
    return recall_at_m, average_precision


def _paired_session_bootstrap(
    rows: List[Dict[str, Any]],
    candidate_field: str,
    baseline_field: str,
    samples: int = 10000,
) -> Dict[str, Any]:
    recall_deltas: List[float] = []
    map_deltas: List[float] = []
    for row in rows:
        candidate_recall, candidate_ap = _session_question_scores(row, candidate_field)
        baseline_recall, baseline_ap = _session_question_scores(row, baseline_field)
        recall_deltas.append(candidate_recall - baseline_recall)
        map_deltas.append(candidate_ap - baseline_ap)
    if not rows:
        return {"num_questions": 0, "samples": samples}

    rng = random.Random(0)
    recall_samples: List[float] = []
    map_samples: List[float] = []
    count = len(rows)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        recall_samples.append(sum(recall_deltas[index] for index in indices) / count)
        map_samples.append(sum(map_deltas[index] for index in indices) / count)
    recall_samples.sort()
    map_samples.sort()

    def interval(values: List[float]) -> List[float]:
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
        return [low, high]

    return {
        "num_questions": count,
        "samples": samples,
        "seed": 0,
        "recall_at_m_delta_mean": sum(recall_deltas) / count,
        "recall_at_m_delta_ci95": interval(recall_samples),
        "map_delta_mean": sum(map_deltas) / count,
        "map_delta_ci95": interval(map_samples),
    }


def _round_capacity_metrics(rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    clue_total = sum(len(row["clue_round_ids"]) for row in rows)
    capacity_hits = sum(min(k, len(row["clue_round_ids"])) for row in rows)
    macro_limits = [
        min(k, len(row["clue_round_ids"])) / len(row["clue_round_ids"])
        for row in rows
        if row["clue_round_ids"]
    ]
    return {
        "max_clue_round_recall_micro_at_k": capacity_hits / clue_total if clue_total else 0.0,
        "max_clue_round_recall_macro_at_k": (
            sum(macro_limits) / len(macro_limits) if macro_limits else 0.0
        ),
        "full_coverage_feasible_rate": (
            sum(1 for row in rows if len(row["clue_round_ids"]) <= k) / len(rows)
            if rows else 0.0
        ),
        "num_questions_over_capacity": sum(
            1 for row in rows if len(row["clue_round_ids"]) > k
        ),
    }


def _available_session_ranking_fields(
    rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    fields = {
        "current": "current_ranked_session_ids",
        "max_score": "max_score_ranked_session_ids",
        "consensus_score": "consensus_score_ranked_session_ids",
        "matched_facets": "matched_facets_ranked_session_ids",
    }
    if rows and all(row.get("episode_directory_complete") for row in rows):
        fields["holistic_directory"] = "directory_ranked_session_ids"
    if rows and all(row.get("episode_directory_v2_complete") for row in rows):
        fields["round_packet_directory"] = "packet_directory_ranked_session_ids"
    return fields


def _aggregate_failure_counts(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for category, round_ids in row[field].items():
            counts[category] += len(round_ids)
    return dict(counts)


def _metric_block(rows: List[Dict[str, Any]], ranked_field: str, k: int) -> Dict[str, Any]:
    metric_rows = [
        {"clue_round_ids": row["clue_round_ids"], "ranked_round_ids": row[ranked_field]}
        for row in rows
    ]
    return summarize_retrievals(metric_rows, [k])["by_k"][str(k)]


def _oracle_row(task_name: str, row: Dict[str, Any], k: int) -> Optional[Dict[str, Any]]:
    trace = row.get("retrieval_trace", {}) or {}
    episode_trace = trace.get("episode_set_retrieval", {}) or {}
    if not episode_trace.get("enabled"):
        return None
    episodes = list(episode_trace.get("episodes", []) or [])
    if not episodes:
        return None

    clues = [str(value) for value in row.get("clue_round_ids", []) or []]
    direct_ids = [str(value) for value in trace.get("anchor_ranked_round_ids", []) or []]
    round_sessions = _round_session_map(episode_trace)
    clue_sessions = _clue_sessions(clues, round_sessions)
    episodes_by_session = {str(item["session_id"]): item for item in episodes}
    missing_session_data = [value for value in clue_sessions if value not in episodes_by_session]
    if missing_session_data:
        return {
            "task_name": task_name,
            "idx": row.get("idx"),
            "question_id": row.get("question_id", ""),
            "question": row.get("question", ""),
            "clue_round_ids": clues,
            "clue_session_ids": clue_sessions,
            "oracle_available": False,
            "missing_session_data": missing_session_data,
        }

    ranked_sessions = [str(item["session_id"]) for item in episodes]
    directory_trace = trace.get("episode_directory", {}) or {}
    directory_enabled = bool(directory_trace.get("enabled"))
    directory_rows = list(directory_trace.get("ranked_sessions", []) or [])
    directory_ranked_sessions = [
        str(item.get("session_id", ""))
        for item in directory_rows
        if str(item.get("session_id", ""))
    ]
    directory_expected = int(directory_trace.get("num_expected_sessions", 0) or 0)
    directory_indexed = int(directory_trace.get("num_indexed_sessions", 0) or 0)
    directory_complete = (
        directory_enabled
        and directory_expected > 0
        and directory_indexed == directory_expected
        and len(set(directory_ranked_sessions)) == directory_expected
    )
    packet_directory_trace = trace.get("episode_directory_v2", {}) or {}
    packet_directory_enabled = bool(packet_directory_trace.get("enabled"))
    packet_directory_rows = list(
        packet_directory_trace.get("ranked_sessions", []) or []
    )
    packet_directory_ranked_sessions = [
        str(item.get("session_id", ""))
        for item in packet_directory_rows
        if str(item.get("session_id", ""))
    ]
    packet_directory_expected = int(
        packet_directory_trace.get("num_expected_sessions", 0) or 0
    )
    packet_directory_indexed = int(
        packet_directory_trace.get("num_indexed_sessions", 0) or 0
    )
    packet_directory_complete = (
        packet_directory_enabled
        and packet_directory_expected > 0
        and packet_directory_indexed == packet_directory_expected
        and len(set(packet_directory_ranked_sessions)) == packet_directory_expected
    )
    target_set = set(clue_sessions)
    oracle_sessions = [episodes_by_session[value] for value in ranked_sessions if value in target_set]
    episode_limit = int(episode_trace.get("episode_round_search_k", 0) or 0)
    if episode_limit <= 0:
        episode_limit = max(len(episode_trace.get("episode_ranked_round_ids", []) or []), 1)
    image_trace = trace.get("image_fusion", {}) or {}
    oracle_final_ids, oracle_episode_ids, oracle_episode_rows = _replay_episode_selection(
        oracle_sessions, direct_ids, image_trace, episode_limit
    )
    oracle_balanced_final_ids, oracle_balanced_episode_ids, oracle_balanced_rows = (
        _replay_episode_selection(
            oracle_sessions, direct_ids, image_trace, episode_limit, balanced=True
        )
    )
    current_top_m_sessions = episodes[: len(clue_sessions)]
    current_top_m_final_ids, current_top_m_episode_ids, current_top_m_rows = (
        _replay_episode_selection(
            current_top_m_sessions, direct_ids, image_trace, episode_limit
        )
    )

    current_episode_ids = [
        str(value) for value in episode_trace.get("episode_ranked_round_ids", []) or []
    ]
    current_episode_rows = list(episode_trace.get("episode_rounds", []) or [])
    current_expanded_sessions = _ordered_unique(
        str(item.get("session_id", "")) for item in current_episode_rows
    )
    oracle_expanded_sessions = _ordered_unique(
        str(item.get("session_id", "")) for item in oracle_episode_rows
    )
    current_final_ids = [str(value) for value in row.get("ranked_round_ids", []) or []]

    trace_pre_image = [str(value) for value in trace.get("pre_image_ranked_round_ids", []) or []]
    replayed_current = _fuse_image(trace_pre_image, image_trace)
    expected_current = [str(value) for value in trace.get("ranked_round_ids", []) or []]
    compare_count = min(len(replayed_current), len(expected_current))
    replay_matches = replayed_current[:compare_count] == expected_current[:compare_count]

    current_stats = _question_stats(current_final_ids, clues, k)
    oracle_stats = _question_stats(oracle_final_ids, clues, k)
    oracle_balanced_stats = _question_stats(oracle_balanced_final_ids, clues, k)
    current_top_m_stats = _question_stats(current_top_m_final_ids, clues, k)
    reranked_sessions = {
        strategy: [str(item["session_id"]) for item in _rank_episode_sessions(episodes, strategy)]
        for strategy in ("max_score", "consensus_score", "matched_facets")
    }
    return {
        "task_name": task_name,
        "idx": row.get("idx"),
        "question_id": row.get("question_id", ""),
        "point": row.get("point"),
        "question": row.get("question", ""),
        "clue_round_ids": clues,
        "clue_session_ids": clue_sessions,
        "oracle_available": True,
        "current_ranked_session_ids": ranked_sessions,
        "episode_directory_enabled": directory_enabled,
        "episode_directory_complete": directory_complete,
        "episode_directory_expected_sessions": directory_expected,
        "episode_directory_indexed_sessions": directory_indexed,
        "directory_ranked_session_ids": directory_ranked_sessions,
        "episode_directory_rows": directory_rows,
        "episode_directory_v2_enabled": packet_directory_enabled,
        "episode_directory_v2_complete": packet_directory_complete,
        "episode_directory_v2_expected_sessions": packet_directory_expected,
        "episode_directory_v2_indexed_sessions": packet_directory_indexed,
        "episode_directory_v2_indexed_packets": int(
            packet_directory_trace.get("num_indexed_packets", 0) or 0
        ),
        "packet_directory_ranked_session_ids": packet_directory_ranked_sessions,
        "episode_directory_v2_rows": packet_directory_rows,
        "max_score_ranked_session_ids": reranked_sessions["max_score"],
        "consensus_score_ranked_session_ids": reranked_sessions["consensus_score"],
        "matched_facets_ranked_session_ids": reranked_sessions["matched_facets"],
        "current_expanded_session_ids": current_expanded_sessions,
        "current_top_m_session_ids": [
            str(item["session_id"]) for item in current_top_m_sessions
        ],
        "oracle_ranked_session_ids": [str(item["session_id"]) for item in oracle_sessions],
        "oracle_expanded_session_ids": oracle_expanded_sessions,
        "current_ranked_round_ids": current_final_ids,
        "current_top_m_ranked_round_ids": current_top_m_final_ids,
        "oracle_ranked_round_ids": oracle_final_ids,
        "oracle_balanced_ranked_round_ids": oracle_balanced_final_ids,
        "current": current_stats,
        "current_top_m": current_top_m_stats,
        "oracle": oracle_stats,
        "oracle_balanced": oracle_balanced_stats,
        "recall_delta": oracle_stats["recall"] - current_stats["recall"],
        "counterfactual_recall_deltas": {
            "current_top_m": current_top_m_stats["recall"] - current_stats["recall"],
            "oracle_whole": oracle_stats["recall"] - current_stats["recall"],
            "oracle_balanced": oracle_balanced_stats["recall"] - current_stats["recall"],
        },
        "counterfactual_episode_paths": {
            "current_top_m": current_top_m_episode_ids,
            "oracle_whole": oracle_episode_ids,
            "oracle_balanced": oracle_balanced_episode_ids,
        },
        "counterfactual_expanded_sessions": {
            "current_top_m": _ordered_unique(
                str(item.get("session_id", "")) for item in current_top_m_rows
            ),
            "oracle_whole": oracle_expanded_sessions,
            "oracle_balanced": _ordered_unique(
                str(item.get("session_id", "")) for item in oracle_balanced_rows
            ),
        },
        "current_failure_types": _classify_misses(
            clues, current_final_ids, direct_ids, current_episode_ids,
            ranked_sessions, current_expanded_sessions, round_sessions, k,
        ),
        "oracle_failure_types": _classify_misses(
            clues, oracle_final_ids, direct_ids, oracle_episode_ids,
            [str(item["session_id"]) for item in oracle_sessions],
            oracle_expanded_sessions, round_sessions, k,
        ),
        "replay_validation": {
            "matches_saved_ranking": replay_matches,
            "compared_rounds": compare_count,
        },
    }


def _dataset_result(
    task_name: str, rows: List[Dict[str, Any]], k: int, session_ks: List[int]
) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("oracle_available")]
    variant_fields = {
        "current": "current_ranked_round_ids",
        "current_top_m": "current_top_m_ranked_round_ids",
        "oracle": "oracle_ranked_round_ids",
        "oracle_balanced": "oracle_balanced_ranked_round_ids",
    }
    variants = {
        name: _metric_block(eligible, field, k)
        for name, field in variant_fields.items()
    }
    current = variants["current"]
    oracle = variants["oracle"]
    metric_keys = (
        "clue_round_recall_micro", "clue_round_recall_macro", "hit_rate",
        "full_clue_coverage_rate", "mrr",
    )
    session_fields = _available_session_ranking_fields(eligible)
    session_diagnostics = {
        name: {
            "by_k": _session_metrics(eligible, session_ks, field),
            "ranking_quality": _session_ranking_quality(eligible, field),
        }
        for name, field in session_fields.items()
    }
    directory_bootstraps = {
        name: _paired_session_bootstrap(
            eligible, field, "current_ranked_session_ids"
        )
        for name, field in session_fields.items()
        if name in {"holistic_directory", "round_packet_directory"}
    }
    return {
        "task_name": task_name,
        "num_questions": len(rows),
        "num_oracle_eligible": len(eligible),
        "num_oracle_unavailable": len(rows) - len(eligible),
        "num_directory_incomplete": sum(
            1
            for row in eligible
            if row.get("episode_directory_enabled")
            and not row.get("episode_directory_complete")
        ),
        "num_directory_v2_incomplete": sum(
            1
            for row in eligible
            if row.get("episode_directory_v2_enabled")
            and not row.get("episode_directory_v2_complete")
        ),
        "replay_mismatch_count": sum(
            1 for row in eligible if not row["replay_validation"]["matches_saved_ranking"]
        ),
        "current": current,
        "oracle": oracle,
        "retrieval_counterfactuals": variants,
        "counterfactual_deltas": {
            name: {
                key: float(metrics[key]) - float(current[key])
                for key in metric_keys
            }
            for name, metrics in variants.items()
            if name != "current"
        },
        "delta": {key: float(oracle[key]) - float(current[key]) for key in metric_keys},
        "current_session_routing": _session_metrics(eligible, session_ks),
        "session_ranking_diagnostics": session_diagnostics,
        "directory_vs_current_bootstrap": directory_bootstraps.get(
            "holistic_directory"
        ),
        "directory_bootstraps_vs_current": directory_bootstraps,
        "round_capacity_at_k": _round_capacity_metrics(eligible, k),
        "current_failure_counts": _aggregate_failure_counts(eligible, "current_failure_types"),
        "oracle_failure_counts": _aggregate_failure_counts(eligible, "oracle_failure_types"),
        "num_question_gains": sum(1 for row in eligible if row["recall_delta"] > 0.0),
        "num_question_losses": sum(1 for row in eligible if row["recall_delta"] < 0.0),
        "counterfactual_question_changes": {
            name: {
                "gains": sum(
                    1
                    for row in eligible
                    if row[name]["recall"] > row["current"]["recall"]
                ),
                "losses": sum(
                    1
                    for row in eligible
                    if row[name]["recall"] < row["current"]["recall"]
                ),
            }
            for name in ("current_top_m", "oracle", "oracle_balanced")
        },
    }


def _macro_dataset_metrics(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = (
        "clue_round_recall_micro", "clue_round_recall_macro", "hit_rate",
        "full_clue_coverage_rate", "mrr",
    )
    return {
        side: {
            key: sum(
                float(item["retrieval_counterfactuals"][side][key])
                for item in datasets
            ) / len(datasets)
            if datasets else 0.0
            for key in keys
        }
        for side in ("current", "current_top_m", "oracle", "oracle_balanced")
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    k = int(payload["k"])
    lines = [
        "# Oracle Episode Routing Analysis", "",
        f"Suite: `{payload['suite_dir']}`", f"Evaluation K: {k}", "",
        "The oracle uses annotated clue session ids only for offline diagnosis. It keeps the saved "
        "direct ranking, episode member expansion, reciprocal-rank fusion, and image reranking unchanged.",
        "", "## Dataset Results", "",
        "| Dataset | Current R@K | Oracle R@K | Delta | Current Full | Oracle Full | Session absent | Path budget | Intra-session | Fusion displacement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["datasets"]:
        fail = item["current_failure_counts"]
        lines.append(
            f"| {item['task_name']} | {item['current']['clue_round_recall_micro']:.4f} | "
            f"{item['oracle']['clue_round_recall_micro']:.4f} | "
            f"{item['delta']['clue_round_recall_micro']:+.4f} | "
            f"{item['current']['full_clue_coverage_rate']:.4f} | "
            f"{item['oracle']['full_clue_coverage_rate']:.4f} | "
            f"{fail.get('session_routing_miss', 0)} | "
            f"{fail.get('episode_path_budget_miss', 0)} | "
            f"{fail.get('intra_session_miss', 0)} | "
            f"{fail.get('fusion_displacement', 0)} |"
        )
    lines.extend([
        "", "## Retrieval Counterfactuals", "",
        "| Dataset | Current | Current Top-M | Oracle whole | Oracle balanced | Capacity max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in payload["datasets"]:
        variants = item["retrieval_counterfactuals"]
        lines.append(
            f"| {item['task_name']} | {variants['current']['clue_round_recall_micro']:.4f} | "
            f"{variants['current_top_m']['clue_round_recall_micro']:.4f} | "
            f"{variants['oracle']['clue_round_recall_micro']:.4f} | "
            f"{variants['oracle_balanced']['clue_round_recall_micro']:.4f} | "
            f"{item['round_capacity_at_k']['max_clue_round_recall_micro_at_k']:.4f} |"
        )
    lines.extend([
        "", "## Current Session Routing", "",
        "| Dataset | Hit@1 | Full@1 | Hit@3 | Full@3 | Question gains | Question losses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in payload["datasets"]:
        at1 = item["current_session_routing"]["1"]
        at3 = item["current_session_routing"]["3"]
        lines.append(
            f"| {item['task_name']} | {at1['target_session_hit_rate']:.4f} | "
            f"{at1['target_session_full_coverage_rate']:.4f} | "
            f"{at3['target_session_hit_rate']:.4f} | "
            f"{at3['target_session_full_coverage_rate']:.4f} | "
            f"{item['num_question_gains']} | {item['num_question_losses']} |"
        )
    lines.extend([
        "", "## Session Score Reranking", "",
        "Recall@M uses the annotated number of target sessions only as an evaluation cutoff.",
        "", "| Dataset | Current R@M | Max-score R@M | Consensus R@M | Matched-facets R@M | Global-dir R@M | Packet-dir R@M | Current MAP | Global-dir MAP | Packet-dir MAP | Best MAP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in payload["datasets"]:
        diagnostics = item["session_ranking_diagnostics"]
        qualities = {
            name: value["ranking_quality"] for name, value in diagnostics.items()
        }
        directory = qualities.get("holistic_directory")
        directory_recall = (
            f"{directory['session_recall_at_oracle_cardinality_micro']:.4f}"
            if directory is not None
            else "-"
        )
        directory_map = f"{directory['session_map']:.4f}" if directory is not None else "-"
        packet_directory = qualities.get("round_packet_directory")
        packet_directory_recall = (
            f"{packet_directory['session_recall_at_oracle_cardinality_micro']:.4f}"
            if packet_directory is not None
            else "-"
        )
        packet_directory_map = (
            f"{packet_directory['session_map']:.4f}"
            if packet_directory is not None
            else "-"
        )
        best_map = max(value["session_map"] for value in qualities.values())
        lines.append(
            f"| {item['task_name']} | "
            f"{qualities['current']['session_recall_at_oracle_cardinality_micro']:.4f} | "
            f"{qualities['max_score']['session_recall_at_oracle_cardinality_micro']:.4f} | "
            f"{qualities['consensus_score']['session_recall_at_oracle_cardinality_micro']:.4f} | "
            f"{qualities['matched_facets']['session_recall_at_oracle_cardinality_micro']:.4f} | "
            f"{directory_recall} | {packet_directory_recall} | "
            f"{qualities['current']['session_map']:.4f} | "
            f"{directory_map} | {packet_directory_map} | {best_map:.4f} |"
        )
    pooled = payload["pooled"]
    pooled_variants = pooled["retrieval_counterfactuals"]
    pooled_session = pooled["session_ranking_diagnostics"]
    best_session_strategy, best_session_values = max(
        pooled_session.items(),
        key=lambda item: item[1]["ranking_quality"]["session_map"],
    )
    directory_findings: List[str] = []
    directory_bootstrap = pooled.get("directory_vs_current_bootstrap")
    if directory_bootstrap:
        recall_ci = directory_bootstrap["recall_at_m_delta_ci95"]
        map_ci = directory_bootstrap["map_delta_ci95"]
        directory_findings = [
            f"- Holistic directory Recall@M delta: {directory_bootstrap['recall_at_m_delta_mean']:+.4f}, paired bootstrap 95% CI [{recall_ci[0]:+.4f}, {recall_ci[1]:+.4f}].",
            f"- Holistic directory MAP delta: {directory_bootstrap['map_delta_mean']:+.4f}, paired bootstrap 95% CI [{map_ci[0]:+.4f}, {map_ci[1]:+.4f}].",
        ]
    packet_bootstrap = pooled.get("directory_bootstraps_vs_current", {}).get(
        "round_packet_directory"
    )
    if packet_bootstrap:
        recall_ci = packet_bootstrap["recall_at_m_delta_ci95"]
        map_ci = packet_bootstrap["map_delta_ci95"]
        directory_findings.extend([
            f"- Round-packet directory Recall@M delta: {packet_bootstrap['recall_at_m_delta_mean']:+.4f}, paired bootstrap 95% CI [{recall_ci[0]:+.4f}, {recall_ci[1]:+.4f}].",
            f"- Round-packet directory MAP delta: {packet_bootstrap['map_delta_mean']:+.4f}, paired bootstrap 95% CI [{map_ci[0]:+.4f}, {map_ci[1]:+.4f}].",
        ])
    lines.extend([
        "", "## Pooled Result", "",
        f"- Current clue-round Recall@{k}: {pooled['current']['clue_round_recall_micro']:.4f}",
        f"- Current Top-M Recall@{k}: {pooled_variants['current_top_m']['clue_round_recall_micro']:.4f}",
        f"- Oracle clue-round Recall@{k}: {pooled['oracle']['clue_round_recall_micro']:.4f}",
        f"- Oracle balanced Recall@{k}: {pooled_variants['oracle_balanced']['clue_round_recall_micro']:.4f}",
        f"- Delta: {pooled['delta']['clue_round_recall_micro']:+.4f}",
        f"- Current full coverage: {pooled['current']['full_clue_coverage_rate']:.4f}",
        f"- Oracle full coverage: {pooled['oracle']['full_clue_coverage_rate']:.4f}",
        f"- Replay mismatches: {payload['replay_mismatch_count']}",
        f"- Directory-incomplete questions: {payload['num_directory_incomplete']}",
        f"- Round-packet-directory-incomplete questions: {payload['num_directory_v2_incomplete']}",
        f"- Capacity-limited maximum Recall@{k}: {pooled['round_capacity_at_k']['max_clue_round_recall_micro_at_k']:.4f}",
        f"- Questions with more than {k} clues: {pooled['round_capacity_at_k']['num_questions_over_capacity']}",
        f"- Current session Recall@M: {pooled_session['current']['ranking_quality']['session_recall_at_oracle_cardinality_micro']:.4f}",
        f"- Best existing-field session MAP: {max(value['ranking_quality']['session_map'] for value in pooled_session.values()):.4f}",
        "", "## Observed Findings", "",
        f"- Oracle cardinality alone changes Recall@{k} by {pooled['counterfactual_deltas']['current_top_m']['clue_round_recall_micro']:+.4f}; stopping at the correct number of sessions does not repair the current ordering.",
        f"- Oracle target-session filtering changes Recall@{k} by {pooled['counterfactual_deltas']['oracle']['clue_round_recall_micro']:+.4f}; suppressing irrelevant episodes has substantial headroom.",
        f"- Balanced oracle expansion changes Recall@{k} by {pooled['counterfactual_deltas']['oracle_balanced']['clue_round_recall_micro']:+.4f}; whole-session expansion wastes part of the candidate budget.",
        f"- The best saved-field reranker is `{best_session_strategy}` with MAP {best_session_values['ranking_quality']['session_map']:.4f}, versus {pooled_session['current']['ranking_quality']['session_map']:.4f} for the current ranking.",
        *directory_findings,
        "", "## Interpretation", "",
        "- Current Top-M isolates oracle stopping cardinality while retaining the current session order.",
        "- Oracle whole keeps only target sessions but expands each session completely before the next.",
        "- Oracle balanced uses the same total path budget and round scores, but allocates rounds across target sessions round-robin.",
        "- Session absent: the clue session was absent from the saved episode ranking.",
        "- Path budget: the clue session was ranked, but earlier episodes exhausted the expansion budget.",
        "- Intra-session miss: the session was expanded, but the clue round fell outside the path budget.",
        "- Fusion displacement: the clue reached a candidate path but was absent from final Top-K.",
        "- A small oracle gain means a stronger directory cannot materially improve this architecture.",
        "- Large oracle residual misses point to expansion or fusion, not directory ranking.",
        "- This is an oracle-assisted fixed-pipeline result, not a strict mathematical upper bound; fixed fusion can still cause per-question losses.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_suite(suite_dir: Path, k: int, session_ks: List[int]) -> Dict[str, Any]:
    suite_dir = suite_dir.resolve()
    suite = _load_json(suite_dir / "suite_metrics.json")
    task_entries = list(suite.get("task_runs", []) or [])
    if not task_entries:
        raise ValueError("suite_metrics.json contains no task_runs")
    all_rows: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    for entry in task_entries:
        task_name = str(entry["task_name"])
        source_rows = _load_jsonl(
            _resolve_task_run_dir(entry, suite_dir) / "retrievals.jsonl"
        )
        analyzed = [
            result for source in source_rows
            if (result := _oracle_row(task_name, source, k)) is not None
        ]
        all_rows.extend(analyzed)
        datasets.append(_dataset_result(task_name, analyzed, k, session_ks))

    eligible = [row for row in all_rows if row.get("oracle_available")]
    pooled_variant_fields = {
        "current": "current_ranked_round_ids",
        "current_top_m": "current_top_m_ranked_round_ids",
        "oracle": "oracle_ranked_round_ids",
        "oracle_balanced": "oracle_balanced_ranked_round_ids",
    }
    pooled_variants = {
        name: _metric_block(eligible, field, k)
        for name, field in pooled_variant_fields.items()
    }
    pooled_current = pooled_variants["current"]
    pooled_oracle = pooled_variants["oracle"]
    pooled_session_fields = _available_session_ranking_fields(eligible)
    metric_keys = (
        "clue_round_recall_micro", "clue_round_recall_macro", "hit_rate",
        "full_clue_coverage_rate", "mrr",
    )
    payload = {
        "suite_dir": str(suite_dir),
        "suite_git_commit": suite.get("git_commit", "unknown"),
        "oracle_definition": "Keep only annotated clue sessions, preserve their current relative order, then replay saved expansion and fusion.",
        "k": k,
        "session_k_values": session_ks,
        "num_questions": len(all_rows),
        "num_oracle_eligible": len(eligible),
        "num_oracle_unavailable": len(all_rows) - len(eligible),
        "num_directory_incomplete": sum(
            1
            for row in eligible
            if row.get("episode_directory_enabled")
            and not row.get("episode_directory_complete")
        ),
        "num_directory_v2_incomplete": sum(
            1
            for row in eligible
            if row.get("episode_directory_v2_enabled")
            and not row.get("episode_directory_v2_complete")
        ),
        "replay_mismatch_count": sum(
            1 for row in eligible if not row["replay_validation"]["matches_saved_ranking"]
        ),
        "datasets": datasets,
        "pooled": {
            "current": pooled_current,
            "oracle": pooled_oracle,
            "retrieval_counterfactuals": pooled_variants,
            "counterfactual_deltas": {
                name: {
                    key: float(metrics[key]) - float(pooled_current[key])
                    for key in metric_keys
                }
                for name, metrics in pooled_variants.items()
                if name != "current"
            },
            "delta": {
                key: float(pooled_oracle[key]) - float(pooled_current[key])
                for key in metric_keys
            },
            "current_session_routing": _session_metrics(eligible, session_ks),
            "session_ranking_diagnostics": {
                name: {
                    "by_k": _session_metrics(eligible, session_ks, field),
                    "ranking_quality": _session_ranking_quality(eligible, field),
                }
                for name, field in pooled_session_fields.items()
            },
            "directory_vs_current_bootstrap": (
                _paired_session_bootstrap(
                    eligible,
                    "directory_ranked_session_ids",
                    "current_ranked_session_ids",
                )
                if "holistic_directory" in pooled_session_fields
                else None
            ),
            "directory_bootstraps_vs_current": {
                name: _paired_session_bootstrap(
                    eligible, field, "current_ranked_session_ids"
                )
                for name, field in pooled_session_fields.items()
                if name in {"holistic_directory", "round_packet_directory"}
            },
            "round_capacity_at_k": _round_capacity_metrics(eligible, k),
            "current_failure_counts": _aggregate_failure_counts(eligible, "current_failure_types"),
            "oracle_failure_counts": _aggregate_failure_counts(eligible, "oracle_failure_types"),
        },
        "macro_over_datasets": _macro_dataset_metrics(datasets),
    }
    write_json(suite_dir / "episode_oracle_metrics.json", payload)
    write_jsonl(suite_dir / "episode_oracle_questions.jsonl", all_rows)
    _write_report(suite_dir / "episode_oracle_report.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a retrieval suite with oracle clue-session routing."
    )
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--session-ks", type=_parse_positive_ints, default=list(DEFAULT_SESSION_KS)
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    result = analyze_suite(args.suite, args.k, args.session_ks)
    print(
        "[INFO] Saved oracle episode analysis: "
        f"{Path(result['suite_dir']) / 'episode_oracle_report.md'}"
    )


if __name__ == "__main__":
    main()
