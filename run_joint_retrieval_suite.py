"""Run MemEye and MEMLENS retrieval experiments under one experiment directory."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, TextIO

from benchmark.common import REPO_ROOT, SCRIPT_DIR, get_git_commit, resolve_config_path, write_json
from benchmark.memlens.retrieval_suite import run_memlens_retrieval_suite
from benchmark.retrieval import clear_retriever_cache
from benchmark.retrieval_eval import (
    DEFAULT_K_VALUES,
    run_modular_retrieval_benchmark,
    summarize_component_retrievals,
    summarize_retrievals,
)


DEFAULT_MEMEYE_TASKS = [
    "config/tasks_external/brand_memory_test.yaml",
    "config/tasks_external/card_playlog_test.yaml",
    "config/tasks_external/cartoon_entertainment_companion.yaml",
    "config/tasks_external/home_renovation_interior_design.yaml",
    "config/tasks_external/multi_scene_visual_case_archive_assistant.yaml",
    "config/tasks_external/outdoor_navigation_route_memory_assistant.yaml",
    "config/tasks_external/personal_health_dashboard_assistant.yaml",
    "config/tasks_external/social_chat_memory_test.yaml",
]


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _parse_ks(raw: str) -> List[int]:
    try:
        values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks requires comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--ks requires comma-separated positive integers")
    return values


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _macro_over_tasks(payloads: List[Dict[str, Any]], ks: Iterable[int]) -> Dict[str, Any]:
    keys = (
        "clue_round_recall_micro", "clue_round_recall_macro",
        "clue_round_precision_micro", "clue_round_precision_macro",
        "hit_rate", "full_clue_coverage_rate", "mrr",
    )
    result: Dict[str, Any] = {}
    for k in ks:
        rows = [payload["summary"]["by_k"][str(k)] for payload in payloads]
        result[str(k)] = {
            key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0
            for key in keys
        }
    return result


def _run(args: argparse.Namespace, experiment_dir: Path) -> None:
    memeye_dir = experiment_dir / "memeye"
    memlens_dir = experiment_dir / "memlens"
    memeye_dir.mkdir(parents=True, exist_ok=False)
    memlens_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "experiment_type": "joint_memeye_memlens_retrieval",
        "created_at": dt.datetime.now().isoformat(),
        "git_commit": get_git_commit(REPO_ROOT),
        "model_config": str(resolve_config_path(args.model_config)),
        "method_config": str(resolve_config_path(args.method_config)),
        "k_values": args.ks,
        "memeye_task_configs": [str(resolve_config_path(path)) for path in args.task_config],
        "memeye_max_questions_per_task": args.memeye_max_questions,
        "memlens_manifest": str(Path(args.memlens_manifest).resolve()),
        "memlens_image_root": str(Path(args.memlens_image_root).resolve()),
        "memlens_max_questions": args.memlens_max_questions,
        "experiment_dir": str(experiment_dir),
    }
    write_json(experiment_dir / "joint_config.json", config)
    write_json(experiment_dir / "joint_status.json", {"status": "running", "completed": []})

    task_payloads: List[Dict[str, Any]] = []
    all_memeye_rows: List[Dict[str, Any]] = []
    completed: List[str] = []
    try:
        for index, task_config in enumerate(args.task_config, start=1):
            task_name = resolve_config_path(task_config).stem
            print(f"[JOINT][MEMEYE][{index}/{len(args.task_config)}] task={task_name}")
            payload = run_modular_retrieval_benchmark(
                task_config_path=task_config,
                model_config_path=args.model_config,
                method_config_path=args.method_config,
                output_root=str(experiment_dir),
                max_questions=args.memeye_max_questions,
                k_values=args.ks,
                run_dir=memeye_dir / task_name,
            )
            task_payloads.append(payload)
            for row in _load_jsonl(Path(payload["run_dir"]) / "retrievals.jsonl"):
                all_memeye_rows.append({"task_name": payload["task_name"], **row})
            completed.append(f"memeye/{task_name}")
            write_json(experiment_dir / "joint_status.json", {"status": "running", "completed": completed})
            clear_retriever_cache(keep_embedding_models=True)
            gc.collect()

        memeye_metrics = {
            "task_runs": [
                {"task_name": p["task_name"], "run_dir": p["run_dir"], "summary": p["summary"]}
                for p in task_payloads
            ],
            "pooled_all_questions": summarize_retrievals(all_memeye_rows, args.ks),
            "component_diagnostics": summarize_component_retrievals(all_memeye_rows, args.ks),
            "macro_over_tasks": _macro_over_tasks(task_payloads, args.ks),
        }
        write_json(memeye_dir / "suite_metrics.json", memeye_metrics)

        clear_retriever_cache(keep_embedding_models=True)
        gc.collect()
        print(f"[JOINT][MEMLENS] manifest={args.memlens_manifest}")
        run_memlens_retrieval_suite(
            manifest_path=Path(args.memlens_manifest),
            image_root=Path(args.memlens_image_root),
            model_config=args.model_config,
            method_config=args.method_config,
            output_root=experiment_dir,
            max_questions=args.memlens_max_questions,
            k_values=args.ks,
            run_dir=memlens_dir,
            clear_cache_every=args.clear_cache_every,
            fail_fast=args.fail_fast,
        )
        completed.append("memlens")
        memlens_metrics = json.loads((memlens_dir / "retrieval_metrics.json").read_text(encoding="utf-8"))
        joint_metrics = {
            "aggregation_policy": "side_by_side; metrics are not pooled across benchmarks with different annotation semantics",
            "memeye": memeye_metrics,
            "memlens": memlens_metrics,
        }
        write_json(experiment_dir / "joint_metrics.json", joint_metrics)
        write_json(experiment_dir / "joint_status.json", {
            "status": "complete", "completed": completed,
            "completed_at": dt.datetime.now().isoformat(),
        })
        print(f"[JOINT] complete experiment_dir={experiment_dir}")
    except BaseException as exc:
        write_json(experiment_dir / "joint_status.json", {
            "status": "failed", "completed": completed, "error": str(exc),
            "traceback": traceback.format_exc(), "failed_at": dt.datetime.now().isoformat(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", action="append", dest="task_configs")
    parser.add_argument("--model-config", default="config/models/qwen3_vl_8b_openrouter.yaml")
    parser.add_argument("--method-config", default="config/methods/evi_retrieval_multifacet_multimodal.yaml")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--memeye-max-questions", type=int, default=0)
    parser.add_argument("--memlens-manifest", default="data/memlens/converted_32k_agent195/manifest.json")
    parser.add_argument("--memlens-image-root", default="data/memlens")
    parser.add_argument("--memlens-max-questions", type=int, default=0)
    parser.add_argument("--ks", type=_parse_ks, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--clear-cache-every", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    args.task_config = args.task_configs or list(DEFAULT_MEMEYE_TASKS)

    if args.run_dir:
        experiment_dir = Path(args.run_dir).resolve()
    else:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = SCRIPT_DIR / output_root
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = output_root.resolve() / "JOINT-retrieval" / (
            f"{timestamp}_{resolve_config_path(args.model_config).stem}_{resolve_config_path(args.method_config).stem}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    with (experiment_dir / "joint_run.log").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)), contextlib.redirect_stderr(_Tee(sys.stderr, log_file)):
            _run(args, experiment_dir)


if __name__ == "__main__":
    main()
