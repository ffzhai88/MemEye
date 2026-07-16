"""Compare two retrieval suites using the shared clue-round metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_K = 10


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metric_delta(candidate: Dict[str, Any], baseline: Dict[str, Any], k: int) -> Dict[str, float]:
    keys = (
        "clue_round_recall_micro",
        "clue_round_recall_macro",
        "clue_round_precision_micro",
        "clue_round_precision_macro",
        "hit_rate",
        "full_clue_coverage_rate",
        "mrr",
    )
    candidate_metrics = candidate["summary"]["by_k"][str(k)]
    baseline_metrics = baseline["summary"]["by_k"][str(k)]
    return {key: float(candidate_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0)) for key in keys}


def _task_runs(suite: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(entry["task_name"]): entry for entry in suite.get("task_runs", [])}


def _question_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("question_id", "")), str(row.get("question", ""))


def _question_stats(row: Dict[str, Any], k: int) -> Dict[str, Any]:
    clues = list(row.get("clue_round_ids", []) or [])
    ranked = list(row.get("ranked_round_ids", []) or [])[:k]
    ranked_set = set(ranked)
    hits = [round_id for round_id in clues if round_id in ranked_set]
    return {
        "clue_round_ids": clues,
        "top_k_round_ids": ranked,
        "hit_clue_round_ids": hits,
        "recall": len(hits) / len(clues) if clues else 0.0,
        "full_coverage": bool(clues) and len(hits) == len(clues),
        "first_clue_rank": next(
            (index for index, round_id in enumerate(ranked, start=1) if round_id in set(clues)),
            None,
        ),
    }


def _resolve_task_run_dir(entry: Dict[str, Any], suite_dir: Path) -> Path:
    configured = Path(str(entry["run_dir"]))
    if (configured / "retrievals.jsonl").exists():
        return configured

    # Suite metadata may contain an absolute path from another machine. Retrieval
    # artifacts have a stable local layout under runs/<task>/retrieval/<run-name>.
    local_run_dir = suite_dir.parent.parent / str(entry["task_name"]) / "retrieval" / configured.name
    if (local_run_dir / "retrievals.jsonl").exists():
        return local_run_dir
    raise FileNotFoundError(
        f"Could not locate retrievals.jsonl for task={entry['task_name']}. "
        f"Tried {configured} and {local_run_dir}."
    )


def _load_task_rows(entry: Dict[str, Any], suite_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = _resolve_task_run_dir(entry, suite_dir) / "retrievals.jsonl"
    return {_question_key(row): row for row in _load_jsonl(path)}


def _format_delta(value: float) -> str:
    return f"{value:+.3f}"


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    k = payload["k"]
    lines = [
        "# Retrieval Suite Comparison",
        "",
        f"Candidate: {payload['candidate_suite']}",
        f"Baseline: {payload['baseline_suite']}",
        f"K: {k}",
        "",
        "## Dataset Deltas",
        "",
        "| Dataset | Recall micro | Full coverage | MRR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in payload["datasets"]:
        delta = item["metric_delta"]
        lines.append(
            f"| {item['task_name']} | {_format_delta(delta['clue_round_recall_micro'])} | "
            f"{_format_delta(delta['full_clue_coverage_rate'])} | {_format_delta(delta['mrr'])} |"
        )

    lines.extend(["", "## Largest Per-Question Changes", ""])
    for item in payload["datasets"]:
        lines.extend([f"### {item['task_name']}", "", "Candidate gains:"])
        gains = item["question_changes"]["gains"]
        losses = item["question_changes"]["losses"]
        if not gains:
            lines.append("- None")
        for change in gains:
            lines.append(
                f"- Recall {_format_delta(change['recall_delta'])}; candidate={change['candidate']['hit_clue_round_ids']}; "
                f"baseline={change['baseline']['hit_clue_round_ids']}. Question: {change['question']}"
            )
        lines.extend(["", "Candidate losses:"])
        if not losses:
            lines.append("- None")
        for change in losses:
            lines.append(
                f"- Recall {_format_delta(change['recall_delta'])}; candidate={change['candidate']['hit_clue_round_ids']}; "
                f"baseline={change['baseline']['hit_clue_round_ids']}. Question: {change['question']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_suites(candidate_dir: Path, baseline_dir: Path, output_dir: Path, k: int, limit: int) -> Dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    baseline_dir = baseline_dir.resolve()
    candidate = _load_json(candidate_dir / "suite_metrics.json")
    baseline = _load_json(baseline_dir / "suite_metrics.json")
    candidate_runs = _task_runs(candidate)
    baseline_runs = _task_runs(baseline)
    shared_tasks = sorted(set(candidate_runs) & set(baseline_runs))
    if not shared_tasks:
        raise ValueError("The two suites have no task names in common")

    datasets: List[Dict[str, Any]] = []
    for task_name in shared_tasks:
        candidate_entry = candidate_runs[task_name]
        baseline_entry = baseline_runs[task_name]
        candidate_rows = _load_task_rows(candidate_entry, candidate_dir)
        baseline_rows = _load_task_rows(baseline_entry, baseline_dir)
        changes: List[Dict[str, Any]] = []
        for key in sorted(set(candidate_rows) & set(baseline_rows)):
            candidate_stats = _question_stats(candidate_rows[key], k)
            baseline_stats = _question_stats(baseline_rows[key], k)
            recall_delta = candidate_stats["recall"] - baseline_stats["recall"]
            if recall_delta == 0.0 and candidate_stats["first_clue_rank"] == baseline_stats["first_clue_rank"]:
                continue
            changes.append(
                {
                    "question_id": key[0],
                    "question": key[1],
                    "recall_delta": recall_delta,
                    "candidate": candidate_stats,
                    "baseline": baseline_stats,
                }
            )
        gains = sorted(
            (change for change in changes if change["recall_delta"] > 0.0),
            key=lambda item: (item["recall_delta"], -(item["candidate"]["first_clue_rank"] or 10**9)),
            reverse=True,
        )[:limit]
        losses = sorted(
            (change for change in changes if change["recall_delta"] < 0.0),
            key=lambda item: (item["recall_delta"], item["candidate"]["first_clue_rank"] or 10**9),
        )[:limit]
        datasets.append(
            {
                "task_name": task_name,
                "candidate_run_dir": candidate_entry["run_dir"],
                "baseline_run_dir": baseline_entry["run_dir"],
                "metric_delta": _metric_delta(candidate_entry, baseline_entry, k),
                "question_changes": {"gains": gains, "losses": losses},
            }
        )

    result = {
        "candidate_suite": str(candidate_dir),
        "baseline_suite": str(baseline_dir),
        "candidate_commit": candidate.get("git_commit", "unknown"),
        "baseline_commit": baseline.get("git_commit", "unknown"),
        "k": k,
        "datasets": datasets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "retrieval_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    _write_markdown(output_dir / "retrieval_comparison.md", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clue-round retrieval results from two retrieval suites.")
    parser.add_argument("--candidate-suite", required=True, type=Path)
    parser.add_argument("--baseline-suite", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--per-task-limit", type=int, default=10)
    args = parser.parse_args()
    if args.k <= 0 or args.per_task_limit <= 0:
        parser.error("--k and --per-task-limit must be positive")
    output_dir = args.output_dir or args.candidate_suite / f"comparison_vs_{args.baseline_suite.name}"
    compare_suites(args.candidate_suite, args.baseline_suite, output_dir, args.k, args.per_task_limit)
    print(f"[INFO] Saved comparison report: {output_dir}")


if __name__ == "__main__":
    main()