"""Replay saved EVI contexts under controlled clue-oracle QA diagnostics."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from benchmark.common import REPO_ROOT, SCRIPT_DIR, get_git_commit, resolve_config_path, write_json


DEFAULT_VARIANTS = [
    "evi_replay_control",
    "retrieved_clues_only",
    "oracle_complete_top10",
    "oracle_clue_only",
]


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        round_id = str(value).strip()
        if round_id and round_id not in seen:
            out.append(round_id)
            seen.add(round_id)
    return out


def _score(row: Dict[str, Any]) -> float:
    return float(row.get("debiased_em", row.get("em", 0.0)) or 0.0)


def _extract_cases(
    semantic_rows: List[Dict[str, Any]],
    evi_rows: List[Dict[str, Any]],
    dataset: str,
) -> List[Dict[str, Any]]:
    semantic_by_idx = {int(row["idx"]): row for row in semantic_rows}
    cases: List[Dict[str, Any]] = []
    for evi in evi_rows:
        source_idx = int(evi["idx"])
        semantic = semantic_by_idx.get(source_idx)
        if semantic is None:
            continue
        clues = _unique(evi.get("clue_rounds", []))
        clue_set = set(clues)
        semantic_context = _unique(semantic.get("context_round_ids", []))
        evi_context = _unique(evi.get("context_round_ids", []))
        semantic_hits = [round_id for round_id in clues if round_id in set(semantic_context)]
        evi_hits = [round_id for round_id in clues if round_id in set(evi_context)]
        semantic_em = _score(semantic)
        evi_em = _score(evi)
        if len(evi_hits) < len(semantic_hits) or evi_em >= semantic_em:
            continue
        cases.append(
            {
                "dataset": dataset,
                "source_idx": source_idx,
                "question": str(evi.get("question", "")),
                "point": evi.get("point"),
                "semantic_em": semantic_em,
                "evi_em": evi_em,
                "semantic_context_round_ids": semantic_context,
                "evi_context_round_ids": evi_context,
                "clue_round_ids": clues,
                "semantic_clue_hits": semantic_hits,
                "evi_clue_hits": evi_hits,
                "semantic_full_coverage": bool(clues and len(semantic_hits) == len(clues)),
                "evi_full_coverage": bool(clues and len(evi_hits) == len(clues)),
                "context_overlap": len(set(semantic_context) & set(evi_context)),
                "evi_non_clue_round_ids": [
                    round_id for round_id in evi_context if round_id not in clue_set
                ],
            }
        )
    return cases


def _variant_config(
    path: Path,
    variant: str,
    source_predictions: Path,
    top_k: int,
) -> None:
    payload = {
        "method": "saved_context_replay",
        "name": f"saved_context_replay_{variant}",
        "modality": "multimodal",
        "source_predictions_jsonl": str(source_predictions.resolve()),
        "context_variant": variant,
        "context_top_k": top_k,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _aggregate_variant(variant: str, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = [
        result
        for payload in payloads
        for result in list(payload.get("results", []) or [])
    ]
    return {
        "variant": variant,
        "num_tasks": len(payloads),
        "num_questions": len(results),
        "question_macro_em": (
            sum(_score(result) for result in results) / len(results) if results else None
        ),
        "task_metrics": [
            {
                "task_name": payload.get("task_name"),
                "num_questions": len(payload.get("results", []) or []),
                "mcq_em": (
                    payload.get("summary", {})
                    .get("mcq_overall", {})
                    .get("em")
                ),
                "evidence_context": payload.get("summary", {}).get("evidence_context", {}),
            }
            for payload in payloads
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract cases where EVI coverage is no worse but QA is worse, then replay "
            "controlled context variants without rerunning retrieval."
        )
    )
    parser.add_argument("--semantic-suite", required=True)
    parser.add_argument("--evi-suite", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--variant", action="append", choices=DEFAULT_VARIANTS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    semantic_suite = Path(args.semantic_suite).expanduser().resolve()
    evi_suite = Path(args.evi_suite).expanduser().resolve()
    semantic_config = _read_json(semantic_suite / "suite_config.json")
    evi_config = _read_json(evi_suite / "suite_config.json")
    task_configs = list(semantic_config.get("task_configs", []) or [])
    if not task_configs:
        raise ValueError("Semantic suite contains no task configs")

    model_config = args.model_config or str(evi_config.get("model_config", ""))
    if not model_config:
        raise ValueError("A model config is required")
    variants = args.variant or list(DEFAULT_VARIANTS)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = (
        output_root.resolve()
        / "QA"
        / "context_diagnostics"
        / f"{timestamp}_{resolve_config_path(model_config).stem}"
    )
    suite_dir.mkdir(parents=True, exist_ok=False)

    all_cases: List[Dict[str, Any]] = []
    task_cases: Dict[str, List[Dict[str, Any]]] = {}
    for task_config in task_configs:
        task_name = resolve_config_path(task_config).stem
        semantic_predictions = semantic_suite / task_name / "predictions.jsonl"
        evi_predictions = evi_suite / task_name / "predictions.jsonl"
        if not semantic_predictions.is_file() or not evi_predictions.is_file():
            raise FileNotFoundError(f"Missing source predictions for task: {task_name}")
        cases = _extract_cases(
            _read_jsonl(semantic_predictions),
            _read_jsonl(evi_predictions),
            task_name,
        )
        task_cases[task_name] = cases
        all_cases.extend(cases)

    _write_jsonl(suite_dir / "cases.jsonl", all_cases)
    write_json(
        suite_dir / "diagnostic_config.json",
        {
            "semantic_suite": str(semantic_suite),
            "evi_suite": str(evi_suite),
            "model_config": model_config,
            "task_configs": task_configs,
            "variants": variants,
            "top_k": args.top_k,
            "num_cases": len(all_cases),
            "git_commit": get_git_commit(REPO_ROOT),
        },
    )
    print(f"[CONTEXT-DIAGNOSTIC] extracted cases={len(all_cases)}")
    if args.prepare_only:
        print(f"[CONTEXT-DIAGNOSTIC] prepared only: {suite_dir}")
        return

    from benchmark.retrieval import clear_retriever_cache
    from benchmark.runner import run_modular_benchmark

    variant_summaries: List[Dict[str, Any]] = []
    variant_case_results: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        payloads: List[Dict[str, Any]] = []
        for task_config in task_configs:
            task_name = resolve_config_path(task_config).stem
            cases = task_cases.get(task_name, [])
            if not cases:
                continue
            source_predictions = evi_suite / task_name / "predictions.jsonl"
            task_dir = suite_dir / variant / task_name
            task_dir.mkdir(parents=True, exist_ok=False)
            method_config = task_dir / "method_config.yaml"
            _variant_config(method_config, variant, source_predictions, args.top_k)
            qa_indices = [int(case["source_idx"]) for case in cases]
            log_path = task_dir / "qa_run.log"
            try:
                with log_path.open("w", encoding="utf-8") as log_file:
                    tee_out = Tee(sys.stdout, log_file)
                    tee_err = Tee(sys.stderr, log_file)
                    with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                        print(
                            f"[CONTEXT-DIAGNOSTIC] variant={variant} "
                            f"task={task_name} questions={len(qa_indices)}"
                        )
                        payload = run_modular_benchmark(
                            task_config_path=task_config,
                            model_config_path=model_config,
                            method_config_path=str(method_config),
                            output_root=str(output_root),
                            mode="mcq",
                            max_questions=0,
                            run_dir=task_dir,
                            qa_indices=qa_indices,
                        )
            finally:
                clear_retriever_cache()
            payloads.append(payload)
            for result in list(payload.get("results", []) or []):
                runtime = dict(result.get("method_runtime", {}) or {})
                source_idx = int(runtime.get("source_prediction_idx", 0) or 0)
                case_key = f"{task_name}:{source_idx}"
                variant_case_results.setdefault(case_key, {})[variant] = {
                    "em": _score(result),
                    "pred": result.get("pred"),
                    "choice": result.get("choice"),
                    "context_round_ids": result.get("context_round_ids", []),
                    "context_round_count": result.get("context_round_count", 0),
                    "added_round_ids": runtime.get("added_round_ids", []),
                    "removed_round_ids": runtime.get("removed_round_ids", []),
                    "clue_capacity_exceeded": runtime.get("clue_capacity_exceeded", False),
                }
        summary = _aggregate_variant(variant, payloads)
        variant_summaries.append(summary)
        write_json(suite_dir / variant / "variant_metrics.json", summary)

    enriched_cases = []
    for case in all_cases:
        case_key = f"{case['dataset']}:{case['source_idx']}"
        enriched = dict(case)
        enriched["variant_results"] = variant_case_results.get(case_key, {})
        enriched_cases.append(enriched)
    _write_jsonl(suite_dir / "case_results.jsonl", enriched_cases)

    write_json(
        suite_dir / "diagnostic_metrics.json",
        {
            "num_cases": len(all_cases),
            "source_semantic_question_macro_em": (
                sum(float(case["semantic_em"]) for case in all_cases) / len(all_cases)
                if all_cases else None
            ),
            "source_evi_question_macro_em": (
                sum(float(case["evi_em"]) for case in all_cases) / len(all_cases)
                if all_cases else None
            ),
            "variants": variant_summaries,
        },
    )
    print(f"[CONTEXT-DIAGNOSTIC] saved: {suite_dir}")


if __name__ == "__main__":
    main()