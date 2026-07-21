"""End-to-end MEMLENS evaluation with MemEye methods and the official judge."""

from __future__ import annotations

import contextlib
import datetime as dt
import gc
import inspect
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO

from benchmark.common import REPO_ROOT, get_git_commit, load_json, load_yaml, resolve_config_path, write_json
from benchmark.dataset import MemoryBenchmarkDataset
from benchmark.evaluator import bleu_score, f1_score, score_open
from benchmark.methods import get_method
from benchmark.retrieval import clear_retriever_cache
from benchmark.runner import instantiate_router, load_sys_prompt


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def format_memlens_question(qa: Dict[str, Any]) -> str:
    """The date is part of the task query, not merely evaluation metadata."""
    question = str(qa.get("question", "")).strip()
    question_date = str(qa.get("question_date", "")).strip()
    return f"Question date: {question_date}\n\nQuestion:\n{question}" if question_date else question


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            question_id = str(row.get("question_id", "")).strip()
            if question_id:
                rows[question_id] = row
    return rows


def selected_round_ids(runtime: Dict[str, Any], history: Iterable[Dict[str, Any]]) -> List[str]:
    for key in ("final_context_round_ids", "selected_round_ids", "retrieved_round_ids"):
        values = runtime.get(key)
        if isinstance(values, list):
            return list(dict.fromkeys(str(v) for v in values if str(v).strip()))
    return list(dict.fromkeys(str(m.get("round_id")) for m in history if m.get("round_id")))


def _retrieval_diagnostics(dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], round_ids: List[str]) -> Dict[str, Any]:
    clues = {str(value) for value in qa.get("clue", []) if str(value).strip()}
    selected = set(round_ids)
    answer_sessions = {str(value) for value in qa.get("session_id", []) if str(value).strip()}
    selected_sessions = {
        str(dataset.rounds[rid].get("session_id", ""))
        for rid in selected
        if rid in dataset.rounds
    }
    clue_hits = len(clues & selected)
    session_hits = len(answer_sessions & selected_sessions)
    return {
        "context_round_ids": round_ids,
        "context_session_ids": sorted(value for value in selected_sessions if value),
        "clue_hits": clue_hits,
        "clue_count": len(clues),
        "clue_recall": clue_hits / len(clues) if clues else None,
        "answer_session_hits": session_hits,
        "answer_session_count": len(answer_sessions),
        "answer_session_recall": session_hits / len(answer_sessions) if answer_sessions else None,
    }


def aggregate_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = ("exact_match", "contains_gt", "f1", "bleu", "clue_recall", "answer_session_recall")

    def summarize(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"count": len(group)}
        for key in metric_keys:
            values = [float(row[key]) for row in group if isinstance(row.get(key), (int, float, bool))]
            if values:
                result[key] = sum(values) / len(values)
        result["latency_ms_total"] = sum(int(row.get("latency_ms", 0) or 0) for row in group)
        return result

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "question_type": defaultdict(list),
        "question_subtype": defaultdict(list),
    }
    for row in rows:
        for field in grouped:
            grouped[field][str(row.get(field, "unknown") or "unknown")].append(row)
    return {
        "summary": summarize(rows),
        "by_question_type": {key: summarize(value) for key, value in sorted(grouped["question_type"].items())},
        "by_question_subtype": {key: summarize(value) for key, value in sorted(grouped["question_subtype"].items())},
    }


def official_judge_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "question_type": row.get("question_type", ""),
            "question_subtype": row.get("question_subtype", ""),
            "reference_answer": row["gt"],
            "prediction": row["pred"],
            "parsed_output": row["pred"],
            "output_len": int(row.get("usage", {}).get("completion_tokens", 0) or len(str(row["pred"]).split())),
        }
        for row in rows
    ]


def run_official_judge(
    *, official_dir: Path, input_path: Path, output_dir: Path, log_path: Path,
    model: str, base_url: str, key_env: str, workers: int, questions_file: Optional[Path],
) -> None:
    script = official_dir / "llm_judge.py"
    if not script.exists():
        raise FileNotFoundError(f"Official judge not found: {script}")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise ValueError(f"Judge API key environment variable is empty: {key_env}")
    command = [
        sys.executable, str(script), "--input_file", str(input_path), "--output_dir", str(output_dir),
        "--api_model", model, "--api_base_url", base_url, "--num_workers", str(workers),
    ]
    if questions_file and questions_file.exists():
        command.extend(["--questions_file", str(questions_file)])
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        print(f"[OFFICIAL-JUDGE] model={model} base_url={base_url} workers={workers}")
        process = subprocess.Popen(
            command, cwd=official_dir, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"Official MEMLENS judge exited with code {code}; see {log_path}")


def run_suite(args: Any) -> Path:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    image_root = Path(args.image_root or manifest.get("runtime_image_root", "")).resolve()
    model_cfg = load_yaml(resolve_config_path(args.model_config))
    method_cfg = load_yaml(resolve_config_path(args.method_config))
    method_name = str(method_cfg.get("method") or method_cfg.get("name") or "method")
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_root).resolve() / "MEMLENS" / f"{timestamp}_{resolve_config_path(args.model_config).stem}_{resolve_config_path(args.method_config).stem}"
        run_dir.mkdir(parents=True, exist_ok=False)

    items = list(manifest.get("items", []))
    if args.max_questions > 0:
        items = items[: args.max_questions]
    predictions_path = run_dir / "predictions.jsonl"
    existing = load_jsonl_by_id(predictions_path)
    suite_config = {
        "manifest": str(manifest_path), "image_root": str(image_root),
        "model_config": str(resolve_config_path(args.model_config)),
        "method_config": str(resolve_config_path(args.method_config)),
        "max_questions": args.max_questions, "selected_questions": len(items),
        "git_commit": get_git_commit(REPO_ROOT), "run_dir": str(run_dir),
        "question_date_in_query": True,
        "keep_embedding_models": not bool(getattr(args, "unload_embedding_models", False)),
        "cleanup_policy": "drop per-item methods and retrievers; retain shared embedding backends",
        "judge": {"enabled": not args.skip_judge, "model": args.judge_model,
                  "base_url": args.judge_base_url, "key_env": args.judge_key_env,
                  "key_available": bool(os.environ.get(args.judge_key_env)), "workers": args.judge_workers},
    }
    write_json(run_dir / "suite_config.json", suite_config)
    log_mode = "a" if (run_dir / "memlens_run.log").exists() else "w"
    with (run_dir / "memlens_run.log").open(log_mode, encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"[MEMLENS] run_dir={run_dir} selected={len(items)} resumed={len(existing)}")
            agent_probe = get_method(method_name, config={**method_cfg, "_model_cfg": model_cfg})
            is_agentic = callable(getattr(agent_probe, "answer", None))
            del agent_probe
            router = None if is_agentic else instantiate_router(model_cfg, load_sys_prompt("open", method_cfg))
            errors = 0
            for index, item in enumerate(items, 1):
                question_id = str(item.get("question_id", ""))
                if question_id in existing:
                    print(f"[MEMLENS][{index}/{len(items)}] skip completed id={question_id}")
                    continue
                method = None
                try:
                    item_path = manifest_path.parent / str(item["path"])
                    dataset = MemoryBenchmarkDataset(item_path, image_root=image_root)
                    if len(dataset.qas) != 1:
                        raise ValueError(f"Expected exactly one QA in {item_path}, found {len(dataset.qas)}")
                    qa = dataset.qas[0]
                    question_id = str(qa.get("question_id") or question_id)
                    question = format_memlens_question(qa)
                    runtime_cfg = dict(method_cfg)
                    runtime_cfg.update({"_model_cfg": model_cfg, "_runtime_paths": {
                        "output_root": str(run_dir.parent), "output_json": "", "run_dir": str(run_dir)},
                        "_eval_cfg": {"mode": "open"}})
                    method = get_method(method_name, config=runtime_cfg)
                    question_images = dataset.resolve_question_images(qa)
                    history: List[Dict[str, Any]] = []
                    started = dt.datetime.now()
                    print(f"[MEMLENS][{index}/{len(items)}] start id={question_id} type={qa.get('question_type')} subtype={qa.get('question_subtype')}")
                    if is_agentic:
                        kwargs: Dict[str, Any] = {}
                        try:
                            signature = inspect.signature(method.answer)
                            if "question_images" in signature.parameters:
                                kwargs["question_images"] = question_images
                        except (TypeError, ValueError):
                            pass
                        pred = method.answer(dataset, qa, question, **kwargs)
                    else:
                        history = method.build_history(dataset, qa)
                        pred = router.answer(history, question, question_images=question_images)
                    latency_ms = int((dt.datetime.now() - started).total_seconds() * 1000)
                    gt = str(qa.get("answer", ""))
                    exact, contains = score_open(str(pred), gt)
                    runtime = dict(getattr(method, "runtime_info", {}) or {})
                    diagnostics = _retrieval_diagnostics(dataset, qa, selected_round_ids(runtime, history))
                    usage = dict(getattr(router, "last_usage", {}) or {}) if router is not None else dict(runtime.get("usage", {}) or {})
                    row = {
                        "idx": index, "question_id": question_id, "question": str(qa.get("question", "")),
                        "question_date": qa.get("question_date", ""), "formatted_question": question,
                        "question_type": qa.get("question_type", ""), "question_subtype": qa.get("question_subtype", ""),
                        "gt": gt, "pred": str(pred), "exact_match": exact, "contains_gt": contains,
                        "f1": f1_score(str(pred), gt), "bleu": bleu_score(str(pred), gt),
                        "latency_ms": latency_ms, "usage": usage, "method_name": method.name,
                        "method_modality": getattr(method, "modality", method_cfg.get("modality", "")),
                        "source_sessions": qa.get("session_id", []), "clue_rounds": qa.get("clue", []),
                        "method_runtime": runtime, **diagnostics,
                    }
                    append_jsonl(predictions_path, row)
                    existing[question_id] = row
                    print(f"[MEMLENS][{index}/{len(items)}] done id={question_id} f1={row['f1']:.4f} latency_ms={latency_ms} clue_recall={row['clue_recall']}")
                except Exception as exc:
                    errors += 1
                    append_jsonl(run_dir / "errors.jsonl", {"idx": index, "question_id": question_id,
                                 "error": str(exc), "traceback": traceback.format_exc(), "time": dt.datetime.now().isoformat()})
                    print(f"[MEMLENS][{index}/{len(items)}] ERROR id={question_id}: {exc}", file=sys.stderr)
                    if args.fail_fast:
                        raise
                finally:
                    method = None
                    if args.clear_cache_every > 0 and index % args.clear_cache_every == 0:
                        clear_retriever_cache(
                            keep_embedding_models=not bool(
                                getattr(args, "unload_embedding_models", False)
                            )
                        )
                        gc.collect()
            ordered = [existing[str(item.get("question_id", ""))] for item in items if str(item.get("question_id", "")) in existing]
            metrics = aggregate_predictions(ordered)
            metrics["requested_count"] = len(items)
            metrics["completed_count"] = len(ordered)
            metrics["errors_this_invocation"] = errors
            write_json(run_dir / "metrics.json", metrics)
            judge_input = run_dir / "official_judge_input.json"
            write_json(judge_input, {"data": official_judge_rows(ordered), "meta": {"run_dir": str(run_dir)}})
            print(f"[MEMLENS] generation complete completed={len(ordered)}/{len(items)} errors={errors}")
            if not args.skip_judge:
                if not args.judge_model:
                    raise ValueError("--judge-model is required unless --skip-judge is used")
                run_official_judge(
                    official_dir=Path(args.official_dir).resolve(), input_path=judge_input,
                    output_dir=run_dir / "official_judge", log_path=run_dir / "official_judge.log",
                    model=args.judge_model, base_url=args.judge_base_url,
                    key_env=args.judge_key_env, workers=args.judge_workers,
                    questions_file=Path(args.questions_file).resolve() if args.questions_file else None,
                )
                judge_metrics_path = run_dir / "official_judge" / "judge_metrics.json"
                if judge_metrics_path.exists():
                    metrics["official_judge"] = load_json(judge_metrics_path)
                    write_json(run_dir / "metrics.json", metrics)
                print("[MEMLENS] official judge complete")
    return run_dir
