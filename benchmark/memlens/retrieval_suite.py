"""Retrieval-only evaluation over independently converted MEMLENS items."""

from __future__ import annotations

import copy
import datetime as dt
import gc
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from benchmark.common import REPO_ROOT, get_git_commit, load_json, load_yaml, resolve_config_path, write_json
from benchmark.dataset import MemoryBenchmarkDataset
from benchmark.memlens.suite import append_jsonl, format_memlens_question, load_jsonl_by_id
from benchmark.retrieval import clear_retriever_cache, select_round_ids_for_qa
from benchmark.retrieval_eval import (
    _build_retriever,
    _clue_round_ids,
    _parse_k_values,
    _rank_details,
    _retrieval_components,
    summarize_component_retrievals,
    summarize_retrievals,
)


def _ranked_session_ids(dataset: MemoryBenchmarkDataset, ranked_round_ids: Iterable[str]) -> List[str]:
    sessions: List[str] = []
    seen = set()
    for round_id in ranked_round_ids:
        payload = dataset.rounds.get(str(round_id), {})
        session_id = str(payload.get("session_id", "")).strip()
        if session_id and session_id not in seen:
            sessions.append(session_id)
            seen.add(session_id)
    return sessions


def summarize_answer_sessions(rows: List[Dict[str, Any]], k_values: Iterable[int]) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("answer_session_ids")]
    by_k: Dict[str, Any] = {}
    for k in _parse_k_values(k_values):
        recalls: List[float] = []
        hit_values: List[float] = []
        full_values: List[float] = []
        reciprocal_ranks: List[float] = []
        total = hits = 0
        for row in eligible:
            expected = list(dict.fromkeys(str(v) for v in row["answer_session_ids"] if str(v)))
            ranked = list(row.get("ranked_session_ids", []))[:k]
            ranked_set = set(ranked)
            hit_count = sum(value in ranked_set for value in expected)
            total += len(expected)
            hits += hit_count
            recalls.append(hit_count / len(expected))
            hit_values.append(float(hit_count > 0))
            full_values.append(float(hit_count == len(expected)))
            first = next((idx for idx, value in enumerate(ranked, 1) if value in set(expected)), None)
            reciprocal_ranks.append(1.0 / first if first else 0.0)
        count = len(eligible)
        by_k[str(k)] = {
            "num_questions": count,
            "answer_session_recall_micro": hits / total if total else 0.0,
            "answer_session_recall_macro": sum(recalls) / count if count else 0.0,
            "answer_session_hit_rate": sum(hit_values) / count if count else 0.0,
            "answer_session_full_coverage_rate": sum(full_values) / count if count else 0.0,
            "answer_session_mrr": sum(reciprocal_ranks) / count if count else 0.0,
            "total_answer_sessions": total,
            "total_answer_session_hits": hits,
        }
    return {"num_questions_with_answer_sessions": len(eligible), "by_k": by_k}


def _group_summaries(rows: List[Dict[str, Any]], ks: List[int], field: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown") or "unknown")].append(row)
    output: Dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        output[name] = {
            "clue_rounds": summarize_retrievals(group, ks),
            "answer_sessions": summarize_answer_sessions(group, ks),
        }
    return output


def run_memlens_retrieval_suite(
    *, manifest_path: Path, image_root: Path, model_config: str,
    method_config: str, output_root: Path, max_questions: int = 0,
    k_values: Iterable[int] = (1, 3, 5, 10, 20), run_dir: Optional[Path] = None,
    clear_cache_every: int = 1, fail_fast: bool = False,
) -> Path:
    manifest_path = manifest_path.resolve()
    image_root = image_root.resolve()
    manifest = load_json(manifest_path)
    model_cfg = load_yaml(resolve_config_path(model_config))
    method_cfg = load_yaml(resolve_config_path(method_config))
    ks = _parse_k_values(k_values)
    max_k = max(ks)
    items = list(manifest.get("items", []))
    if max_questions > 0:
        items = items[:max_questions]
    if run_dir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root.resolve() / "MEMLENS_retrieval" / (
            f"{timestamp}_{resolve_config_path(model_config).stem}_{resolve_config_path(method_config).stem}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "manifest": str(manifest_path), "image_root": str(image_root),
        "model_config": str(resolve_config_path(model_config)),
        "method_config": str(resolve_config_path(method_config)),
        "k_values": ks, "max_questions": max_questions,
        "selected_questions": len(items), "question_date_in_query": True,
        "git_commit": get_git_commit(REPO_ROOT), "run_dir": str(run_dir),
        "answer_refusal_policy": "retained in rows; excluded from clue/session recall when annotations are empty",
    }
    write_json(run_dir / "suite_config.json", config)
    output_path = run_dir / "retrievals.jsonl"
    existing = load_jsonl_by_id(output_path)
    log_path = run_dir / "retrieval_run.log"
    with log_path.open("a" if log_path.exists() else "w", encoding="utf-8") as log:
        def emit(message: str) -> None:
            print(message)
            log.write(message + "\n")
            log.flush()

        emit(f"[MEMLENS-RETRIEVAL] run_dir={run_dir} selected={len(items)} resumed={len(existing)} ks={ks}")
        errors = 0
        for index, item in enumerate(items, 1):
            question_id = str(item.get("question_id", ""))
            if question_id in existing:
                emit(f"[MEMLENS-RETRIEVAL][{index}/{len(items)}] skip id={question_id}")
                continue
            retriever = None
            try:
                dataset = MemoryBenchmarkDataset(manifest_path.parent / str(item["path"]), image_root)
                if len(dataset.qas) != 1:
                    raise ValueError(f"Expected one QA, found {len(dataset.qas)}")
                original_qa = dataset.qas[0]
                qa = copy.deepcopy(original_qa)
                qa["question"] = format_memlens_question(original_qa)
                question_id = str(qa.get("question_id") or question_id)
                effective = copy.deepcopy(method_cfg)
                effective["_model_cfg"] = copy.deepcopy(model_cfg)
                effective["_runtime_paths"] = {"run_dir": str(run_dir)}
                started = time.perf_counter()
                method_kind, retriever, effective, build_seconds = _build_retriever(
                    dataset, effective, run_dir, max_k
                )
                if method_kind == "evi_faceted":
                    ranked, trace = retriever.retrieve_faceted_rounds(qa, dataset=dataset, limit=max_k)
                else:
                    trace = {}
                    ranked = select_round_ids_for_qa(dataset, qa, effective, runtime_info=trace)
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                clues = _clue_round_ids(original_qa)
                ranking = _rank_details(ranked, clues)
                sessions = _ranked_session_ids(dataset, ranked)
                answer_sessions = [str(v) for v in original_qa.get("session_id", []) if str(v)]
                row = {
                    "idx": index, "question_id": question_id,
                    "question": str(original_qa.get("question", "")),
                    "question_date": original_qa.get("question_date", ""),
                    "formatted_question": qa["question"],
                    "question_type": original_qa.get("question_type", ""),
                    "question_subtype": original_qa.get("question_subtype", ""),
                    "clue_round_ids": clues, "answer_session_ids": answer_sessions,
                    "ranked_round_ids": ranked, "ranked_round_count": len(ranked),
                    "ranked_session_ids": sessions, "ranked_session_count": len(sessions),
                    "latency_ms": latency_ms, "index_build_seconds": build_seconds,
                    "retrieval_method_kind": method_kind, "retrieval_trace": trace,
                    "retrieval_components": _retrieval_components(trace), **ranking,
                }
                append_jsonl(output_path, row)
                existing[question_id] = row
                emit(
                    f"[MEMLENS-RETRIEVAL][{index}/{len(items)}] done id={question_id} "
                    f"clues={len(clues)} hits@{max_k}={len(set(clues) & set(ranked[:max_k]))} "
                    f"first={ranking['first_clue_rank']} latency_ms={latency_ms}"
                )
            except Exception as exc:
                errors += 1
                append_jsonl(run_dir / "errors.jsonl", {
                    "idx": index, "question_id": question_id, "error": str(exc),
                    "traceback": traceback.format_exc(), "time": dt.datetime.now().isoformat(),
                })
                emit(f"[MEMLENS-RETRIEVAL][{index}/{len(items)}] ERROR id={question_id}: {exc}")
                if fail_fast:
                    raise
            finally:
                retriever = None
                if clear_cache_every > 0 and index % clear_cache_every == 0:
                    clear_retriever_cache()
                    gc.collect()

    ordered = [existing[str(item.get("question_id", ""))] for item in items if str(item.get("question_id", "")) in existing]
    clue_summary = summarize_retrievals(ordered, ks)
    component_summary = summarize_component_retrievals(ordered, ks)
    metrics = {
        "requested_count": len(items), "completed_count": len(ordered),
        "errors_this_invocation": errors,
        "num_answer_refusal_without_clues": sum(
            not row.get("clue_round_ids") for row in ordered
            if row.get("question_type") == "answer_refusal"
        ),
        "clue_rounds": clue_summary,
        "answer_sessions": summarize_answer_sessions(ordered, ks),
        "component_diagnostics": component_summary,
        "by_question_type": _group_summaries(ordered, ks, "question_type"),
        "by_question_subtype": _group_summaries(ordered, ks, "question_subtype"),
    }
    write_json(run_dir / "retrieval_metrics.json", metrics)
    print(f"[MEMLENS-RETRIEVAL] complete {len(ordered)}/{len(items)} errors={errors} run_dir={run_dir}")
    return run_dir
