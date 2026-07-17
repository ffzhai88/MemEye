"""Run end-to-end QA evaluation across multiple task configs."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List

from benchmark.common import REPO_ROOT, SCRIPT_DIR, get_git_commit, resolve_config_path, write_json
from benchmark.runner import run_modular_benchmark


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _numeric_macro(task_payloads: List[Dict[str, Any]]) -> Dict[str, float]:
    common: Dict[str, List[float]] = {}
    for payload in task_payloads:
        for key, value in payload.get("summary", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                common.setdefault(str(key), []).append(float(value))
    task_count = len(task_payloads)
    return {
        key: sum(values) / len(values)
        for key, values in sorted(common.items())
        if len(values) == task_count
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and aggregate end-to-end QA evaluations over multiple datasets."
    )
    parser.add_argument("--task-config", action="append", required=True)
    parser.add_argument("--model-config", default="config/models/gpt_4_1_nano.yaml")
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--mode", choices=["open", "mcq", "both"], default="open")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--enable-bert-score", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root
    model_name = resolve_config_path(args.model_config).stem
    method_name = resolve_config_path(args.method_config).stem
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = output_root.resolve() / "QA" / f"{timestamp}_{model_name}_{method_name}"
    suite_dir.mkdir(parents=True, exist_ok=False)

    suite_config = {
        "task_configs": args.task_config,
        "model_config": args.model_config,
        "method_config": args.method_config,
        "mode": args.mode,
        "max_questions": args.max_questions,
        "enable_bert_score": args.enable_bert_score,
        "git_commit": get_git_commit(REPO_ROOT),
        "suite_dir": str(suite_dir),
    }
    write_json(suite_dir / "suite_config.json", suite_config)

    task_payloads: List[Dict[str, Any]] = []
    for task_config in args.task_config:
        task_dir = suite_dir / resolve_config_path(task_config).stem
        task_dir.mkdir(parents=True, exist_ok=False)
        log_path = task_dir / "qa_run.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            tee_out = Tee(sys.stdout, log_file)
            tee_err = Tee(sys.stderr, log_file)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                print(f"[QA-SUITE] task={task_config}")
                payload = run_modular_benchmark(
                    task_config_path=task_config,
                    model_config_path=args.model_config,
                    method_config_path=args.method_config,
                    output_root=str(output_root),
                    mode=args.mode,
                    max_questions=args.max_questions,
                    enable_bert_score=args.enable_bert_score,
                    run_dir=task_dir,
                )
        task_payloads.append(payload)
        write_json(
            suite_dir / "suite_runs.json",
            {
                "runs": [
                    {
                        "task_name": item["task_name"],
                        "run_dir": item["run_dir"],
                        "summary": item["summary"],
                    }
                    for item in task_payloads
                ]
            },
        )

    suite_metrics = {
        **suite_config,
        "num_tasks": len(task_payloads),
        "num_qas_run": sum(int(item.get("num_qas_run", 0)) for item in task_payloads),
        "macro_over_datasets": _numeric_macro(task_payloads),
        "task_runs": [
            {
                "task_name": item["task_name"],
                "run_dir": item["run_dir"],
                "num_qas_run": item.get("num_qas_run", 0),
                "summary": item["summary"],
            }
            for item in task_payloads
        ],
    }
    write_json(suite_dir / "suite_metrics.json", suite_metrics)
    print(f"[INFO] Saved QA suite summary: {suite_dir}")


if __name__ == "__main__":
    main()