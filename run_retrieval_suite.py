"""Run retrieval-only evaluation across multiple task configs and aggregate the results."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark.common import REPO_ROOT, SCRIPT_DIR, get_git_commit, resolve_config_path, write_json
from benchmark.retrieval_eval import DEFAULT_K_VALUES, run_modular_retrieval_benchmark, summarize_retrievals


def parse_k_values(raw: str) -> List[int]:
    try:
        values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks must be a comma-separated list of positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--ks must contain at least one positive integer")
    return values


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _macro_over_datasets(task_payloads: List[Dict[str, Any]], ks: Iterable[int]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for k in ks:
        entries = [payload["summary"]["by_k"][str(k)] for payload in task_payloads]
        keys = [
            "clue_round_recall_micro",
            "clue_round_recall_macro",
            "clue_round_precision_micro",
            "clue_round_precision_macro",
            "hit_rate",
            "full_clue_coverage_rate",
            "mrr",
        ]
        output[str(k)] = {
            key: sum(float(entry.get(key, 0.0)) for entry in entries) / len(entries) if entries else 0.0
            for key in keys
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate retrieval-only evaluations over multiple datasets.")
    parser.add_argument("--task-config", action="append", required=True, help="Repeat once per task config.")
    parser.add_argument("--model-config", default="config/models/gpt_4_1_nano.yaml")
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--ks", type=parse_k_values, default=list(DEFAULT_K_VALUES))
    args = parser.parse_args()

    task_payloads: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for task_config in args.task_config:
        payload = run_modular_retrieval_benchmark(
            task_config_path=task_config,
            model_config_path=args.model_config,
            method_config_path=args.method_config,
            output_root=args.output_root,
            max_questions=args.max_questions,
            k_values=args.ks,
        )
        run_dir = Path(payload["run_dir"])
        rows = _load_jsonl(run_dir / "retrievals.jsonl")
        task_payloads.append(payload)
        for row in rows:
            all_rows.append({"task_name": payload["task_name"], **row})

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root
    model_name = resolve_config_path(args.model_config).stem
    method_name = resolve_config_path(args.method_config).stem
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = output_root.resolve() / "retrieval_suites" / f"{timestamp}_{model_name}_{method_name}"
    suite_dir.mkdir(parents=True, exist_ok=False)

    suite_payload = {
        "task_configs": args.task_config,
        "model_config": args.model_config,
        "method_config": args.method_config,
        "k_values": args.ks,
        "git_commit": get_git_commit(REPO_ROOT),
        "task_runs": [
            {
                "task_name": payload["task_name"],
                "run_dir": payload["run_dir"],
                "summary": payload["summary"],
            }
            for payload in task_payloads
        ],
        "pooled_all_questions": summarize_retrievals(all_rows, args.ks),
        "macro_over_datasets": _macro_over_datasets(task_payloads, args.ks),
    }
    write_json(suite_dir / "suite_metrics.json", suite_payload)
    write_json(suite_dir / "suite_runs.json", {"runs": suite_payload["task_runs"]})
    print(f"[INFO] Saved retrieval suite summary: {suite_dir}")


if __name__ == "__main__":
    main()