"""Standalone retrieval evaluation for multimodal memory methods."""
from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common import (
    REPO_ROOT,
    SCRIPT_DIR,
    get_git_commit,
    load_yaml,
    resolve_config_path,
    resolve_dataset_path,
    write_json,
    write_jsonl,
)
from .dataset import MemoryBenchmarkDataset
from .retrieval import select_round_ids_for_qa


log = logging.getLogger(__name__)
DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


def _compose_retrieval_config(
    task_config_path: str,
    model_config_path: str,
    method_config_path: str,
    output_root: str,
    max_questions: int,
) -> Tuple[Dict[str, Any], Path]:
    task_path = resolve_config_path(task_config_path)
    cfg = {
        "task": load_yaml(task_path),
        "model": load_yaml(resolve_config_path(model_config_path)),
        "method": load_yaml(resolve_config_path(method_config_path)),
        "run": {"output_root": output_root} if output_root else {},
    }
    cfg["dataset"] = dict(cfg["task"].get("dataset", {}))
    cfg["eval"] = dict(cfg["task"].get("eval", {}))
    if max_questions:
        cfg["eval"]["max_questions"] = max_questions
    return cfg, task_path.parent


def _resolve_retrieval_paths(cfg: Dict[str, Any], config_dir: Path) -> Dict[str, Path]:
    dataset_cfg = cfg.get("dataset", {})
    dialog_json = resolve_dataset_path(str(dataset_cfg["dialog_json"]), config_dir)
    image_root_raw = str(dataset_cfg.get("image_root", "")).strip()
    image_root = resolve_dataset_path(image_root_raw, config_dir) if image_root_raw else None
    output_root_raw = str(cfg.get("run", {}).get("output_root", "")).strip()
    if output_root_raw:
        candidate = Path(output_root_raw)
        output_root = candidate if candidate.is_absolute() else (SCRIPT_DIR / candidate)
    else:
        output_root = SCRIPT_DIR / "runs"
    return {
        "dialog_json": dialog_json,
        "image_root": image_root,
        "output_root": output_root.resolve(),
    }

def _parse_k_values(values: Iterable[int]) -> List[int]:
    ks = sorted({int(value) for value in values if int(value) > 0})
    if not ks:
        raise ValueError("At least one positive retrieval K is required")
    return ks


def _clue_round_ids(qa: Dict[str, Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in qa.get("clue", []) or []:
        round_id = str(value or "").strip()
        if round_id and round_id not in seen:
            out.append(round_id)
            seen.add(round_id)
    return out


def _rank_details(ranked_round_ids: List[str], clue_round_ids: List[str]) -> Dict[str, Any]:
    ranks = {round_id: index for index, round_id in enumerate(ranked_round_ids, start=1)}
    clue_ranks = {round_id: ranks[round_id] for round_id in clue_round_ids if round_id in ranks}
    return {
        "clue_round_ids": clue_round_ids,
        "clue_ranks": clue_ranks,
        "first_clue_rank": min(clue_ranks.values()) if clue_ranks else None,
    }


def _retrieval_components(trace: Dict[str, Any]) -> Dict[str, List[str]]:
    image_fusion = trace.get("image_fusion", {}) or {}
    raw_multimodal_fusion = trace.get("raw_multimodal_candidate_fusion", {}) or {}
    anchor_ids = list(
        trace.get("anchor_ranked_round_ids", [])
        or image_fusion.get("anchor_ranked_round_ids", [])
        or []
    )
    image_ids = list(
        image_fusion.get("image_ranked_round_ids", [])
        or [item.get("round_id") for item in trace.get("image_ranking", []) or []]
    )
    episode_trace = trace.get("episode_set_retrieval", {}) or {}
    episode_ids = list(episode_trace.get("episode_ranked_round_ids", []) or [])
    direct_episode_ids = list(
        episode_trace.get("direct_episode_fused_round_ids", [])
        or trace.get("pre_image_ranked_round_ids", [])
        or []
    )
    return {
        "anchor_ranked_round_ids": [str(value) for value in anchor_ids if value],
        "image_ranked_round_ids": [str(value) for value in image_ids if value],
        "episode_ranked_round_ids": [str(value) for value in episode_ids if value],
        "direct_episode_fused_round_ids": [
            str(value) for value in direct_episode_ids if value
        ],
        "raw_multimodal_ranked_round_ids": [
            str(value)
            for value in raw_multimodal_fusion.get(
                "raw_multimodal_ranked_round_ids", []
            )
            if value
        ],
        "evi_raw_multimodal_fused_round_ids": [
            str(value)
            for value in raw_multimodal_fusion.get("fused_ranked_round_ids", [])
            if value
        ],
    }


def summarize_component_retrievals(
    rows: List[Dict[str, Any]], k_values: Iterable[int]
) -> Dict[str, Any]:
    eligible = [
        row for row in rows
        if row.get("clue_round_ids")
        and row.get("retrieval_components", {}).get("anchor_ranked_round_ids")
        and row.get("retrieval_components", {}).get("image_ranked_round_ids")
    ]
    by_k: Dict[str, Any] = {}
    for k in _parse_k_values(k_values):
        anchor_hits = image_hits = image_unique_hits = oracle_hits = clue_total = 0
        overlap_total = 0
        for row in eligible:
            clues = set(row["clue_round_ids"])
            components = row["retrieval_components"]
            anchor = set(components["anchor_ranked_round_ids"][:k])
            image = set(components["image_ranked_round_ids"][:k])
            clue_total += len(clues)
            anchor_hits += len(clues & anchor)
            image_hits += len(clues & image)
            image_unique_hits += len((clues & image) - anchor)
            oracle_hits += len(clues & (anchor | image))
            overlap_total += len(anchor & image)
        by_k[str(k)] = {
            "num_questions": len(eligible),
            "anchor_clue_round_recall_micro": anchor_hits / clue_total if clue_total else 0.0,
            "image_clue_round_recall_micro": image_hits / clue_total if clue_total else 0.0,
            "image_unique_clue_hits": image_unique_hits,
            "image_unique_clue_hit_rate": image_unique_hits / clue_total if clue_total else 0.0,
            "oracle_union_clue_round_recall_micro": oracle_hits / clue_total if clue_total else 0.0,
            "ranking_overlap_count_mean": overlap_total / len(eligible) if eligible else 0.0,
            "ranking_overlap_rate_mean": overlap_total / (len(eligible) * k) if eligible else 0.0,
        }
    episode_eligible = [
        row for row in rows
        if row.get("clue_round_ids")
        and row.get("retrieval_components", {}).get("anchor_ranked_round_ids")
        and row.get("retrieval_components", {}).get("episode_ranked_round_ids")
    ]
    episode_by_k: Dict[str, Any] = {}
    for k in _parse_k_values(k_values):
        direct_hits = episode_hits = episode_unique_hits = oracle_hits = fused_hits = 0
        overlap_total = clue_total = 0
        for row in episode_eligible:
            clues = set(row["clue_round_ids"])
            components = row["retrieval_components"]
            direct = set(components["anchor_ranked_round_ids"][:k])
            episode = set(components["episode_ranked_round_ids"][:k])
            fused = set(components.get("direct_episode_fused_round_ids", [])[:k])
            clue_total += len(clues)
            direct_hits += len(clues & direct)
            episode_hits += len(clues & episode)
            episode_unique_hits += len((clues & episode) - direct)
            oracle_hits += len(clues & (direct | episode))
            fused_hits += len(clues & fused)
            overlap_total += len(direct & episode)
        count = len(episode_eligible)
        episode_by_k[str(k)] = {
            "num_questions": count,
            "direct_clue_round_recall_micro": direct_hits / clue_total if clue_total else 0.0,
            "episode_clue_round_recall_micro": episode_hits / clue_total if clue_total else 0.0,
            "episode_unique_clue_hits": episode_unique_hits,
            "episode_unique_clue_hit_rate": episode_unique_hits / clue_total if clue_total else 0.0,
            "direct_episode_oracle_recall_micro": oracle_hits / clue_total if clue_total else 0.0,
            "direct_episode_fused_recall_micro": fused_hits / clue_total if clue_total else 0.0,
            "ranking_overlap_count_mean": overlap_total / count if count else 0.0,
            "ranking_overlap_rate_mean": overlap_total / (count * k) if count else 0.0,
        }
    return {
        "num_questions": len(eligible),
        "by_k": by_k,
        "episode_num_questions": len(episode_eligible),
        "episode_by_k": episode_by_k,
    }


def summarize_retrievals(rows: List[Dict[str, Any]], k_values: Iterable[int]) -> Dict[str, Any]:
    ks = _parse_k_values(k_values)
    with_clues = [row for row in rows if row.get("clue_round_ids")]
    summary: Dict[str, Any] = {
        "num_questions": len(rows),
        "num_questions_with_clues": len(with_clues),
        "k_values": ks,
        "by_k": {},
    }
    for k in ks:
        macro_recall: List[float] = []
        macro_precision: List[float] = []
        reciprocal_ranks: List[float] = []
        hit_values: List[float] = []
        full_values: List[float] = []
        retrieved_total = 0
        clue_total = 0
        hit_total = 0
        for row in with_clues:
            ranked = list(row.get("ranked_round_ids", []) or [])[:k]
            clues = list(row["clue_round_ids"])
            hit_count = sum(1 for round_id in clues if round_id in set(ranked))
            retrieved_count = len(ranked)
            macro_recall.append(hit_count / len(clues))
            macro_precision.append(hit_count / retrieved_count if retrieved_count else 0.0)
            hit_values.append(1.0 if hit_count else 0.0)
            full_values.append(1.0 if hit_count == len(clues) else 0.0)
            clue_total += len(clues)
            retrieved_total += retrieved_count
            hit_total += hit_count
            first_rank = next((index for index, round_id in enumerate(ranked, start=1) if round_id in set(clues)), None)
            reciprocal_ranks.append(1.0 / first_rank if first_rank is not None else 0.0)
        count = len(with_clues)
        summary["by_k"][str(k)] = {
            "clue_round_recall_micro": hit_total / clue_total if clue_total else 0.0,
            "clue_round_recall_macro": sum(macro_recall) / count if count else 0.0,
            "clue_round_precision_micro": hit_total / retrieved_total if retrieved_total else 0.0,
            "clue_round_precision_macro": sum(macro_precision) / count if count else 0.0,
            "hit_rate": sum(hit_values) / count if count else 0.0,
            "full_clue_coverage_rate": sum(full_values) / count if count else 0.0,
            "mrr": sum(reciprocal_ranks) / count if count else 0.0,
            "total_clue_rounds": clue_total,
            "total_clue_round_hits": hit_total,
            "total_retrieved_rounds": retrieved_total,
        }
    return summary


def _build_retriever(
    dataset: MemoryBenchmarkDataset,
    method_cfg: Dict[str, Any],
    run_dir: Path,
    max_k: int,
) -> Tuple[str, Any, Dict[str, Any], float]:
    config = copy.deepcopy(method_cfg)
    method_name = str(config.get("method", config.get("name", ""))).strip().lower()
    config["top_k"] = max_k
    if int(config.get("neighbor_window", 0) or 0) != 0:
        raise ValueError("Retrieval evaluation requires neighbor_window: 0 so ranks remain well-defined")

    started = time.perf_counter()
    if method_name == "evi":
        from .evi import EVISystem

        image_pool_k = (
            int(config.get("evi_image_round_search_k", 0) or 0)
            if config.get("evi_use_raw_image_retrieval")
            else 0
        )
        raw_multimodal_pool_k = (
            int(config.get("evi_raw_multimodal_candidate_k", 0) or 0)
            if config.get("evi_apply_raw_multimodal_candidate_rank_fusion")
            else 0
        )
        config["max_candidates"] = max(
            max_k, image_pool_k, raw_multimodal_pool_k
        )
        config["evi_retrieval_only"] = True
        config["_runtime_paths"] = {"run_dir": str(run_dir)}
        system = EVISystem(config)
        system.process_all_sessions(dataset)
        build_seconds = time.perf_counter() - started
        return "evi_faceted", system, config, build_seconds

    if method_name.startswith("semantic_rag"):
        # The shared retriever is lazily constructed on the first query.
        return "semantic_rag", None, config, 0.0

    raise ValueError(
        "Unsupported retrieval method. Use config/methods/evi.yaml or a semantic_rag_* method config."
    )


def run_retrieval_benchmark(
    cfg: Dict[str, Any],
    config_dir: Path,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ks = _parse_k_values(k_values)
    max_k = max(ks)
    paths = _resolve_retrieval_paths(cfg, config_dir)
    task_name = str(cfg.get("task", {}).get("name", "task")).strip() or "task"
    model_name = str(cfg.get("model", {}).get("name", "model")).strip() or "model"
    method_name = str(cfg.get("method", {}).get("name", "method")).strip() or "method"
    if run_dir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = paths["output_root"] / "retrieval" / f"{timestamp}_{model_name}_{method_name}" / task_name
    else:
        run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    file_handler = logging.FileHandler(run_dir / "retrieval_debug.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(file_handler)
    log.setLevel(logging.INFO)
    try:
        dataset = MemoryBenchmarkDataset(paths["dialog_json"], paths["image_root"])
        method_cfg = copy.deepcopy(cfg.get("method", {}))
        method_cfg["_model_cfg"] = copy.deepcopy(cfg.get("model", {}))
        method_kind, retriever, effective_method_cfg, build_seconds = _build_retriever(
            dataset, method_cfg, run_dir, max_k
        )
        max_questions = int(cfg.get("eval", {}).get("max_questions", 0) or 0)
        qas = dataset.iter_qas(limit=max_questions)
        rows: List[Dict[str, Any]] = []
        log.info(
            "Retrieval evaluation start task=%s method=%s questions=%d ks=%s build_seconds=%.3f",
            task_name, method_name, len(qas), ks, build_seconds,
        )

        for index, qa in enumerate(qas, start=1):
            started = time.perf_counter()
            if method_kind == "evi_faceted":
                ranked_round_ids, trace = retriever.retrieve_faceted_rounds(qa, dataset=dataset, limit=max_k)
            else:
                runtime_info: Dict[str, Any] = {}
                ranked_round_ids = select_round_ids_for_qa(dataset, qa, effective_method_cfg, runtime_info=runtime_info)
                trace = runtime_info
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            clues = _clue_round_ids(qa)
            ranking = _rank_details(ranked_round_ids, clues)
            components = _retrieval_components(trace)
            row = {
                "idx": index,
                "question_id": qa.get("id") or qa.get("question_id") or "",
                "point": qa.get("point"),
                "question": str(qa.get("question", "")),
                "clue_round_ids": clues,
                "ranked_round_ids": ranked_round_ids,
                "ranked_round_count": len(ranked_round_ids),
                "latency_ms": latency_ms,
                "retrieval_trace": trace,
                "retrieval_components": components,
                **ranking,
            }
            rows.append(row)
            log.info(
                "QA %d/%d question=%s rank_count=%d clue_hits=%s first_clue_rank=%s ranked=%s",
                index,
                len(qas),
                row["question"],
                len(ranked_round_ids),
                ranking["clue_ranks"],
                ranking["first_clue_rank"],
                ranked_round_ids,
            )
            log.info(
                "QA %d retrieval trace:\n%s",
                index,
                json.dumps(trace, ensure_ascii=False, indent=2, default=str),
            )

        summary = summarize_retrievals(rows, ks)
        component_summary = summarize_component_retrievals(rows, ks)
        if component_summary:
            summary["component_diagnostics"] = component_summary
        query_latency = [float(row["latency_ms"]) for row in rows]
        summary["latency"] = {
            "index_build_seconds": build_seconds,
            "query_latency_ms_mean": sum(query_latency) / len(query_latency) if query_latency else 0.0,
            "query_latency_ms_total": sum(query_latency),
        }
        payload = {
            "task_name": task_name,
            "method_name": method_name,
            "retrieval_method_kind": method_kind,
            "model_name": model_name,
            "dialog_json": str(paths["dialog_json"]),
            "image_root": str(paths["image_root"] or ""),
            "run_dir": str(run_dir),
            "git_commit": get_git_commit(REPO_ROOT),
            "summary": summary,
        }
        config_to_write = copy.deepcopy(cfg)
        config_to_write["retrieval_eval"] = {
            "k_values": ks,
            "max_k": max_k,
            "method_kind": method_kind,
            "retrieval_method_config": effective_method_cfg,
        }
        write_json(run_dir / "config.json", config_to_write)
        write_json(run_dir / "retrieval_metrics.json", payload)
        write_jsonl(run_dir / "retrievals.jsonl", rows)
        log.info("Retrieval evaluation complete run_dir=%s", run_dir)
        print(f"[INFO] Saved retrieval artifacts: {run_dir}")
        return payload
    finally:
        log.removeHandler(file_handler)
        file_handler.close()


def run_modular_retrieval_benchmark(
    task_config_path: str,
    model_config_path: str,
    method_config_path: str,
    output_root: str = "",
    max_questions: int = 0,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg, config_dir = _compose_retrieval_config(
        task_config_path=task_config_path,
        model_config_path=model_config_path,
        method_config_path=method_config_path,
        output_root=output_root,
        max_questions=max_questions,
    )
    return run_retrieval_benchmark(cfg, config_dir, k_values=k_values, run_dir=run_dir)
