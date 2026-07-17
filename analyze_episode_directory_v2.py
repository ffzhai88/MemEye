"""Evaluate packet and session-card directories with fixed-pipeline replay.

The analysis uses clue sessions only for evaluation. It never changes retrieval
artifacts and never calls a model.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from analyze_episode_oracle import (
    _fallback_session_id,
    _question_stats,
    _replay_episode_selection,
    _resolve_task_run_dir,
)
from benchmark.common import write_json, write_jsonl


PACKET_AGGREGATIONS = ("packet_max", "packet_mean", "packet_top2", "packet_top3")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _packet_score(row: Dict[str, Any], strategy: str) -> float:
    scores = [
        float(item.get("score", 0.0))
        for item in row.get("packet_scores", []) or []
    ]
    if strategy == "packet_max":
        return float(row.get("score", scores[0] if scores else 0.0))
    if not scores:
        raise ValueError(f"{strategy} requires packet_scores in the v2 trace")
    if strategy == "packet_mean":
        selected = scores
    elif strategy == "packet_top2":
        selected = scores[:2]
    elif strategy == "packet_top3":
        selected = scores[:3]
    else:
        raise ValueError(f"Unknown packet aggregation: {strategy}")
    return statistics.fmean(selected)


def _packet_ranking(
    directory_rows: List[Dict[str, Any]], strategy: str
) -> List[str]:
    scored = [
        (str(row.get("session_id", "")), _packet_score(row, strategy))
        for row in directory_rows
        if str(row.get("session_id", ""))
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [session_id for session_id, _ in scored]


def _rrf(rankings: List[List[str]]) -> List[str]:
    positions = [
        {session_id: rank for rank, session_id in enumerate(ranking, start=1)}
        for ranking in rankings
    ]
    session_ids = set().union(*(set(ranking) for ranking in rankings))
    return sorted(
        session_ids,
        key=lambda session_id: (
            -sum(
                1.0 / position[session_id]
                for position in positions
                if session_id in position
            ),
            session_id,
        ),
    )


def _image_session_ranking(image_trace: Dict[str, Any]) -> List[str]:
    round_ids = image_trace.get("image_ranked_round_ids", []) or []
    return _ordered_unique(_fallback_session_id(str(round_id)) for round_id in round_ids)


def _session_question_metrics(
    ranking: List[str], targets: List[str]
) -> Dict[str, float]:
    positions = {session_id: rank for rank, session_id in enumerate(ranking, start=1)}
    cutoff = len(targets)
    hits = sum(positions.get(session_id, 10**9) <= cutoff for session_id in targets)
    ordered_positions = sorted(
        positions[session_id] for session_id in targets if session_id in positions
    )
    average_precision = (
        sum(
            relevant_rank / retrieved_rank
            for relevant_rank, retrieved_rank in enumerate(ordered_positions, start=1)
        )
        / cutoff
        if cutoff
        else 0.0
    )
    dcg = sum(
        1.0 / math.log2(positions[session_id] + 1.0)
        for session_id in targets
        if session_id in positions
    )
    ideal_dcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, cutoff + 1))
    return {
        "hits": float(hits),
        "targets": float(cutoff),
        "recall_at_m": hits / cutoff if cutoff else 0.0,
        "average_precision": average_precision,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _aggregate_session_metrics(
    rows: List[Dict[str, Any]], strategy: str
) -> Dict[str, float]:
    values = [row["session_metrics"][strategy] for row in rows]
    hits = sum(value["hits"] for value in values)
    targets = sum(value["targets"] for value in values)
    return {
        "recall_at_m_micro": hits / targets if targets else 0.0,
        "recall_at_m_macro": statistics.fmean(
            value["recall_at_m"] for value in values
        ),
        "map": statistics.fmean(value["average_precision"] for value in values),
        "ndcg": statistics.fmean(value["ndcg"] for value in values),
    }


def _aggregate_round_metrics(
    rows: List[Dict[str, Any]], strategy: str
) -> Dict[str, float]:
    values = [row["round_metrics"][strategy] for row in rows]
    hits = sum(len(value["hit_clue_round_ids"]) for value in values)
    clues = sum(len(row["clue_round_ids"]) for row in rows)
    return {
        "clue_round_recall_micro": hits / clues if clues else 0.0,
        "clue_round_recall_macro": statistics.fmean(
            value["recall"] for value in values
        ),
        "full_clue_coverage_rate": statistics.fmean(
            float(value["full_coverage"]) for value in values
        ),
    }


def _episode_objects(
    episode_trace: Dict[str, Any],
    directory_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    episodes = {
        str(item.get("session_id", "")): dict(item)
        for item in episode_trace.get("episodes", []) or []
        if str(item.get("session_id", ""))
    }
    synthesized = 0
    for row in directory_rows:
        session_id = str(row.get("session_id", ""))
        if not session_id or session_id in episodes:
            continue
        member_ids = [
            str(value) for value in row.get("packet_round_ids", []) or []
        ]
        if not member_ids:
            continue
        episodes[session_id] = {
            "session_id": session_id,
            "score": float(row.get("score", 0.0)),
            "member_round_ids": member_ids,
        }
        synthesized += 1
    return episodes, synthesized


def _replay(
    ranking: List[str],
    trace: Dict[str, Any],
    directory_rows: List[Dict[str, Any]],
) -> Tuple[List[str], int, int]:
    episode_trace = trace.get("episode_set_retrieval", {}) or {}
    episodes, synthesized = _episode_objects(episode_trace, directory_rows)
    ordered = [episodes[session_id] for session_id in ranking if session_id in episodes]
    missing = sum(session_id not in episodes for session_id in ranking)
    direct_ids = [
        str(value) for value in trace.get("anchor_ranked_round_ids", []) or []
    ]
    episode_limit = int(episode_trace.get("episode_round_search_k", 0) or 0)
    if episode_limit <= 0:
        episode_limit = max(
            len(episode_trace.get("episode_ranked_round_ids", []) or []), 1
        )
    final_ids, _, _ = _replay_episode_selection(
        ordered,
        direct_ids,
        trace.get("image_fusion", {}) or {},
        episode_limit,
    )
    return final_ids, synthesized, missing


def _pearson(left: List[float], right: List[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else 0.0


def _length_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    counts: List[float] = []
    scores: List[float] = []
    top_counts: List[float] = []
    for row in rows:
        directory_rows = row["directory_rows"]
        for item in directory_rows:
            counts.append(float(item.get("packet_count", 0)))
            scores.append(float(item.get("score", 0.0)))
        if directory_rows:
            top_counts.append(float(directory_rows[0].get("packet_count", 0)))
    return {
        "mean_packet_count_all_sessions": statistics.fmean(counts) if counts else 0.0,
        "mean_packet_count_top1_sessions": (
            statistics.fmean(top_counts) if top_counts else 0.0
        ),
        "packet_count_max_score_pearson": _pearson(counts, scores),
    }


def _question_row(
    task_name: str, source: Dict[str, Any], k: int
) -> Dict[str, Any]:
    trace = source.get("retrieval_trace", {}) or {}
    episode_trace = trace.get("episode_set_retrieval", {}) or {}
    directory_trace = trace.get("episode_directory_v2", {}) or {}
    directory_rows = list(directory_trace.get("ranked_sessions", []) or [])
    if not directory_trace.get("enabled") or not directory_rows:
        raise ValueError(
            f"{task_name} question {source.get('idx')} has no v2 directory trace"
        )

    clue_round_ids = [
        str(value) for value in source.get("clue_round_ids", []) or []
    ]
    clue_session_ids = _ordered_unique(
        _fallback_session_id(round_id) for round_id in clue_round_ids
    )
    current = [
        str(item.get("session_id", ""))
        for item in episode_trace.get("episodes", []) or []
        if str(item.get("session_id", ""))
    ]
    image = _image_session_ranking(trace.get("image_fusion", {}) or {})

    rankings: Dict[str, List[str]] = {"current": current, "image": image}
    rankings["packet_max"] = _packet_ranking(directory_rows, "packet_max")
    has_packet_scores = all(
        bool(item.get("packet_scores")) for item in directory_rows
    )
    if has_packet_scores:
        for strategy in PACKET_AGGREGATIONS[1:]:
            rankings[strategy] = _packet_ranking(directory_rows, strategy)

    for strategy in list(rankings):
        if not strategy.startswith("packet_"):
            continue
        rankings[f"fusion_current_{strategy}"] = _rrf(
            [current, rankings[strategy]]
        )
    rankings["fusion_current_packet_max_image"] = _rrf(
        [current, rankings["packet_max"], image]
    )
    card_trace = trace.get("episode_directory_v3", {}) or {}
    card_rows = list(card_trace.get("ranked_sessions", []) or [])
    if card_trace.get("enabled") and card_rows:
        rankings["session_card"] = [
            str(item.get("session_id", ""))
            for item in card_rows
            if str(item.get("session_id", ""))
        ]
        rankings["fusion_current_session_card"] = _rrf(
            [current, rankings["session_card"]]
        )
        rankings["fusion_current_session_card_image"] = _rrf(
            [current, rankings["session_card"], image]
        )

    session_metrics = {
        strategy: _session_question_metrics(ranking, clue_session_ids)
        for strategy, ranking in rankings.items()
    }
    round_metrics = {
        "current": _question_stats(
            [str(value) for value in source.get("ranked_round_ids", []) or []],
            clue_round_ids,
            k,
        )
    }
    replay_diagnostics: Dict[str, Dict[str, int]] = {}
    for strategy, ranking in rankings.items():
        if strategy in {"current", "image"}:
            continue
        replay_rows = card_rows if "session_card" in strategy else directory_rows
        final_ids, synthesized, missing = _replay(ranking, trace, replay_rows)
        round_metrics[strategy] = _question_stats(final_ids, clue_round_ids, k)
        replay_diagnostics[strategy] = {
            "synthesized_sessions": synthesized,
            "missing_sessions": missing,
        }

    return {
        "task_name": task_name,
        "idx": source.get("idx"),
        "question_id": source.get("question_id", ""),
        "question": source.get("question", ""),
        "clue_round_ids": clue_round_ids,
        "clue_session_ids": clue_session_ids,
        "has_complete_packet_scores": has_packet_scores,
        "has_session_card_ranking": bool(card_trace.get("enabled") and card_rows),
        "directory_rows": directory_rows,
        "session_rankings": rankings,
        "session_metrics": session_metrics,
        "round_metrics": round_metrics,
        "replay_diagnostics": replay_diagnostics,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    session_strategies = sorted(
        set.intersection(*(set(row["session_metrics"]) for row in rows))
    )
    round_strategies = sorted(
        set.intersection(*(set(row["round_metrics"]) for row in rows))
    )
    return {
        "num_questions": len(rows),
        "num_questions_with_complete_packet_scores": sum(
            row["has_complete_packet_scores"] for row in rows
        ),
        "num_questions_with_session_card_ranking": sum(
            row["has_session_card_ranking"] for row in rows
        ),
        "session_metrics": {
            strategy: _aggregate_session_metrics(rows, strategy)
            for strategy in session_strategies
        },
        "round_replay_metrics": {
            strategy: _aggregate_round_metrics(rows, strategy)
            for strategy in round_strategies
        },
        "length_diagnostics": _length_diagnostics(rows),
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    pooled = payload["pooled"]
    lines = [
        "# Episode Directory Offline Analysis",
        "",
        f"Suite: `{payload['suite_dir']}`",
        f"Evaluation K: {payload['k']}",
        "",
        "This analysis uses clue annotations only for evaluation and replays the saved",
        "expansion, direct/episode fusion, and image reranking without model calls.",
        "",
        "## Pooled Session Ranking",
        "",
        "| Strategy | Recall@M micro | Recall@M macro | MAP | nDCG |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in pooled["session_metrics"].items():
        lines.append(
            f"| {strategy} | {metrics['recall_at_m_micro']:.4f} | "
            f"{metrics['recall_at_m_macro']:.4f} | {metrics['map']:.4f} | "
            f"{metrics['ndcg']:.4f} |"
        )
    lines.extend([
        "",
        "## Fixed-Pipeline Replay",
        "",
        "| Strategy | Clue Recall micro | Clue Recall macro | Full coverage |",
        "| --- | ---: | ---: | ---: |",
    ])
    for strategy, metrics in pooled["round_replay_metrics"].items():
        lines.append(
            f"| {strategy} | {metrics['clue_round_recall_micro']:.4f} | "
            f"{metrics['clue_round_recall_macro']:.4f} | "
            f"{metrics['full_clue_coverage_rate']:.4f} |"
        )
    lines.extend([
        "",
        "## Dataset Replay",
        "",
        "| Dataset | Current | Packet max | Current + packet max | Session card | Current + card | Current + card + image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dataset in payload["datasets"]:
        metrics = dataset["round_replay_metrics"]
        value = lambda name: (
            f"{metrics[name]['clue_round_recall_micro']:.4f}"
            if name in metrics
            else "-"
        )
        lines.append(
            f"| {dataset['task_name']} | {value('current')} | "
            f"{value('packet_max')} | {value('fusion_current_packet_max')} | "
            f"{value('session_card')} | {value('fusion_current_session_card')} | "
            f"{value('fusion_current_session_card_image')} |"
        )
    lines.extend([
        "",
        "## Best Packet Aggregation By Dataset",
        "",
        "| Dataset | Best packet session rank | Recall@M micro | Best current + packet replay | Clue Recall micro |",
        "| --- | --- | ---: | --- | ---: |",
    ])
    for dataset in payload["datasets"]:
        packet_session = {
            name: metrics
            for name, metrics in dataset["session_metrics"].items()
            if name.startswith("packet_")
        }
        packet_replay = {
            name: metrics
            for name, metrics in dataset["round_replay_metrics"].items()
            if name.startswith("fusion_current_packet_")
            and not name.endswith("_image")
        }
        best_session_name, best_session = max(
            packet_session.items(),
            key=lambda item: (
                item[1]["recall_at_m_micro"],
                item[1]["map"],
                item[0],
            ),
        )
        best_replay_name, best_replay = max(
            packet_replay.items(),
            key=lambda item: (
                item[1]["clue_round_recall_micro"],
                item[1]["clue_round_recall_macro"],
                item[0],
            ),
        )
        lines.append(
            f"| {dataset['task_name']} | {best_session_name} | "
            f"{best_session['recall_at_m_micro']:.4f} | {best_replay_name} | "
            f"{best_replay['clue_round_recall_micro']:.4f} |"
        )
    length = pooled["length_diagnostics"]
    lines.extend([
        "",
        "## Length Diagnostics",
        "",
        f"- Mean packet count over all scored sessions: {length['mean_packet_count_all_sessions']:.2f}",
        f"- Mean packet count of Top-1 sessions: {length['mean_packet_count_top1_sessions']:.2f}",
        f"- Pearson correlation between packet count and max score: {length['packet_count_max_score_pearson']:.4f}",
        "",
        "| Dataset | Mean packets | Top-1 mean packets | Count/max-score correlation |",
        "| --- | ---: | ---: | ---: |",
    ])
    for dataset in payload["datasets"]:
        values = dataset["length_diagnostics"]
        lines.append(
            f"| {dataset['task_name']} | "
            f"{values['mean_packet_count_all_sessions']:.2f} | "
            f"{values['mean_packet_count_top1_sessions']:.2f} | "
            f"{values['packet_count_max_score_pearson']:.4f} |"
        )
    lines.extend([
        "",
        "Packet mean and Top-N rows appear only when every session trace contains",
        "`packet_scores`. Legacy v2 runs support packet-max analysis only.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_suite(suite_dir: Path, k: int) -> Dict[str, Any]:
    suite = _load_json(suite_dir / "suite_metrics.json")
    all_rows: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    for entry in suite.get("task_runs", []):
        task_name = str(entry["task_name"])
        task_dir = _resolve_task_run_dir(entry, suite_dir)
        rows = [
            _question_row(task_name, source, k)
            for source in _load_jsonl(task_dir / "retrievals.jsonl")
        ]
        summary = {"task_name": task_name, **_summarize(rows)}
        datasets.append(summary)
        all_rows.extend(rows)

    payload = {
        "suite_dir": str(suite_dir),
        "suite_git_commit": suite.get("git_commit", "unknown"),
        "k": k,
        "datasets": datasets,
        "pooled": _summarize(all_rows),
    }
    write_json(suite_dir / "episode_directory_v2_metrics.json", payload)
    write_jsonl(
        suite_dir / "episode_directory_v2_questions.jsonl",
        [
            {key: value for key, value in row.items() if key != "directory_rows"}
            for row in all_rows
        ],
    )
    _write_report(suite_dir / "episode_directory_v2_report.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze packet/card directory rankings and pipeline replay."
    )
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    payload = analyze_suite(args.suite.resolve(), args.k)
    print(
        "[INFO] Saved v2 directory analysis: "
        f"{Path(payload['suite_dir']) / 'episode_directory_v2_report.md'}"
    )


if __name__ == "__main__":
    main()
