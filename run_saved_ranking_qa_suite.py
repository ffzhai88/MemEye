"""Run end-to-end MemEye QA from rankings saved by an offline analyzer."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from benchmark.common import (
    REPO_ROOT,
    SCRIPT_DIR,
    get_git_commit,
    resolve_config_path,
    write_json,
)
from benchmark.retrieval import clear_retriever_cache
from benchmark.runner import run_modular_benchmark
from run_qa_suite import Tee, _numeric_macro


DEFAULT_TASK_CONFIGS = [
    "config/tasks_external/brand_memory_test.yaml",
    "config/tasks_external/card_playlog_test.yaml",
    "config/tasks_external/cartoon_entertainment_companion.yaml",
    "config/tasks_external/home_renovation_interior_design.yaml",
    "config/tasks_external/multi_scene_visual_case_archive_assistant.yaml",
    "config/tasks_external/outdoor_navigation_route_memory_assistant.yaml",
    "config/tasks_external/personal_health_dashboard_assistant.yaml",
    "config/tasks_external/social_chat_memory_test.yaml",
]


def _acquire_run_lock(
    output_root: Path,
    source_rankings: Path,
    model_config: str,
    strategy: str,
    top_k: int,
):
    """Prevent concurrent duplicate QA suites with the same frozen inputs."""

    identity = json.dumps(
        {
            "source_rankings": str(source_rankings),
            "model_config": str(resolve_config_path(model_config)),
            "strategy": strategy,
            "top_k": top_k,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    lock_dir = output_root / "QA" / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"saved_ranking_{digest}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "owner metadata unavailable"
        handle.close()
        raise RuntimeError(
            "An identical saved-ranking QA suite is already running: "
            f"{owner}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps({
            "pid": os.getpid(),
            "started_at": dt.datetime.now().isoformat(),
            "identity": json.loads(identity),
        }, ensure_ascii=False)
    )
    handle.flush()
    return handle


def _method_config(
    path: Path,
    source_rankings: Path,
    dataset: str,
    strategy: str,
    top_k: int,
) -> None:
    payload = {
        "method": "saved_ranking_replay",
        "name": f"saved_ranking_replay_{strategy}",
        "modality": "multimodal",
        "source_rankings_jsonl": str(source_rankings),
        "source_benchmark": "memeye",
        "source_dataset": dataset,
        "ranking_strategy": strategy,
        "context_top_k": top_k,
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _validate_source(
    source_rankings: Path,
    task_configs: List[str],
    strategy: str,
    top_k: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with source_rankings.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("benchmark", "")) != "memeye":
                continue
            dataset = str(row.get("dataset", ""))
            ranking = (row.get("rankings") or {}).get(strategy)
            if not isinstance(ranking, list) or len(dict.fromkeys(ranking)) < top_k:
                raise ValueError(
                    f"Invalid {strategy} ranking for "
                    f"{dataset}/{row.get('question_id')}"
                )
            counts[dataset] = counts.get(dataset, 0) + 1
    missing = [
        resolve_config_path(task).stem
        for task in task_configs
        if resolve_config_path(task).stem not in counts
    ]
    if missing:
        raise ValueError(f"Saved rankings are missing datasets: {missing}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rankings", required=True)
    parser.add_argument(
        "--model-config",
        default="config/models/qwen3_vl_8B_ali.yaml",
    )
    parser.add_argument("--task-config", action="append")
    parser.add_argument("--ranking-strategy", default="selective_vlm")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["open", "mcq", "both"], default="mcq")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    source_rankings = Path(args.source_rankings).expanduser().resolve()
    if not source_rankings.is_file():
        parser.error(f"Saved rankings not found: {source_rankings}")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    task_configs = args.task_config or list(DEFAULT_TASK_CONFIGS)
    source_counts = _validate_source(
        source_rankings,
        task_configs,
        args.ranking_strategy,
        args.top_k,
    )

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root
    output_root = output_root.resolve()
    run_lock = _acquire_run_lock(
        output_root,
        source_rankings,
        args.model_config,
        args.ranking_strategy,
        args.top_k,
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = (
        output_root
        / "QA"
        / (
            f"{timestamp}_{resolve_config_path(args.model_config).stem}_"
            f"saved_{args.ranking_strategy}_top{args.top_k}"
        )
    )
    suite_dir.mkdir(parents=True, exist_ok=False)
    suite_config = {
        "task_configs": task_configs,
        "model_config": args.model_config,
        "method": "saved_ranking_replay",
        "source_rankings": str(source_rankings),
        "source_counts": source_counts,
        "ranking_strategy": args.ranking_strategy,
        "top_k": args.top_k,
        "mode": args.mode,
        "max_questions": args.max_questions,
        "git_commit": get_git_commit(REPO_ROOT),
        "suite_dir": str(suite_dir),
        "prepare_only": args.prepare_only,
    }
    write_json(suite_dir / "suite_config.json", suite_config)

    task_payloads: List[Dict[str, Any]] = []
    for task_config in task_configs:
        task_name = resolve_config_path(task_config).stem
        task_dir = suite_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=False)
        method_config = task_dir / "method_config.yaml"
        _method_config(
            method_config,
            source_rankings,
            task_name,
            args.ranking_strategy,
            args.top_k,
        )
        if args.prepare_only:
            continue
        log_path = task_dir / "qa_run.log"
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                tee_out = Tee(sys.stdout, log_file)
                tee_err = Tee(sys.stderr, log_file)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    print(
                        f"[SAVED-RANKING-QA] task={task_name} "
                        f"strategy={args.ranking_strategy} top_k={args.top_k}"
                    )
                    payload = run_modular_benchmark(
                        task_config_path=task_config,
                        model_config_path=args.model_config,
                        method_config_path=str(method_config),
                        output_root=str(output_root),
                        mode=args.mode,
                        max_questions=args.max_questions,
                        run_dir=task_dir,
                    )
        finally:
            clear_retriever_cache()
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

    if args.prepare_only:
        print(f"[SAVED-RANKING-QA] prepared: {suite_dir}")
        return
    suite_metrics = {
        **suite_config,
        "num_tasks": len(task_payloads),
        "num_qas_run": sum(
            int(item.get("num_qas_run", 0)) for item in task_payloads
        ),
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
    print(f"[SAVED-RANKING-QA] complete: {suite_dir}")
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
    run_lock.close()


if __name__ == "__main__":
    main()
