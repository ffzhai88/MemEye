"""Selectively verify contested retrieval candidates with a VLM.

The base ranking is saved Anchor-only plus Raw-MM mean-rank. The VLM checks
one raw round at a time and never receives benchmark names, question types,
clue labels, or answers. Only a parseable, high-confidence ``not_useful``
verdict can demote a candidate; every other outcome preserves the base order.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from analyze_candidate_rescoring import (
    _memeye_dataset,
    _read_json,
    discover_memeye_task_dirs,
)
from benchmark.dataset import MemoryBenchmarkDataset
from benchmark.evi.selective_verifier import (
    PROMPT_VERSION,
    SelectiveEvidenceVerifier,
    build_verification_prompt,
    conservative_rerank,
    verification_candidate_ids,
)
from benchmark.evi.vlm import make_openai_vlm


log = logging.getLogger("selective_vlm_verification")
STRATEGIES = (
    "abstract_evi",
    "raw_multimodal_fixed",
    "evi_raw_mean_rank",
    "selective_vlm",
)


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "selective_verification.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        log.addHandler(handler)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def _row_key(row: Mapping[str, Any]) -> str:
    return "::".join(
        str(row.get(field, ""))
        for field in ("benchmark", "dataset", "question_id")
    )


def _load_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {_row_key(row): row for row in _iter_jsonl(path)}


def _selected_rows(
    path: Path, benchmark_filter: str, max_questions: int
) -> Iterable[Dict[str, Any]]:
    """Stream selected rows; provenance diagnostics can be very large."""
    counts: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        benchmark = str(row.get("benchmark", ""))
        if benchmark_filter != "all" and benchmark != benchmark_filter:
            continue
        dataset_key = f"{benchmark}/{row.get('dataset', '')}"
        if max_questions > 0 and counts[dataset_key] >= max_questions:
            continue
        counts[dataset_key] += 1
        yield row


def _question_id(row: Mapping[str, Any]) -> str:
    return str(row.get("question_id", row.get("idx", "")))


def _row_map(path: Path) -> Dict[str, Dict[str, Any]]:
    return {_question_id(row): row for row in _iter_jsonl(path)}


def _qa_map(dataset: MemoryBenchmarkDataset) -> Dict[str, Dict[str, Any]]:
    return {
        str(qa.get("question_id", qa.get("id", index))): qa
        for index, qa in enumerate(dataset.qas)
    }


class DatasetResolver:
    """Resolve raw evidence without exposing annotations to the verifier."""

    def __init__(
        self,
        input_dir: Path,
        data_root: Path,
        manifest_path: Path,
        memlens_image_root: Path,
    ) -> None:
        self.data_root = data_root
        self.manifest_path = manifest_path
        self.memlens_image_root = memlens_image_root
        self.memeye_dirs = {
            path.name: path
            for path in discover_memeye_task_dirs(input_dir / "memeye")
        }
        if not self.memeye_dirs:
            raise FileNotFoundError(
                f"No completed MemEye tasks under {input_dir / 'memeye'}"
            )
        self._memeye: Dict[str, Tuple[Any, Any, Any]] = {}
        manifest = _read_json(manifest_path)
        self.memlens_items = {
            str(item.get("question_id", "")): item
            for item in manifest.get("items", [])
        }
        self.memlens_rows = _row_map(
            input_dir / "memlens" / "retrievals.jsonl"
        )

    def resolve(
        self, benchmark: str, dataset_name: str, question_id: str
    ) -> Tuple[MemoryBenchmarkDataset, Dict[str, Any], Dict[str, Any]]:
        if benchmark == "memeye":
            if dataset_name not in self.memeye_dirs:
                raise KeyError(f"Unknown MemEye task directory: {dataset_name}")
            if dataset_name not in self._memeye:
                task_dir = self.memeye_dirs[dataset_name]
                dataset, _ = _memeye_dataset(task_dir, self.data_root)
                self._memeye[dataset_name] = (
                    dataset,
                    _row_map(task_dir / "retrievals.jsonl"),
                    _qa_map(dataset),
                )
            dataset, rows, qas = self._memeye[dataset_name]
            if question_id not in rows:
                raise KeyError(
                    f"No saved retrieval row for {dataset_name}/{question_id}"
                )
            return dataset, rows[question_id], qas.get(question_id, {})

        if benchmark != "memlens":
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        item = self.memlens_items.get(question_id)
        if item is None:
            raise KeyError(f"MEMLENS manifest has no item for {question_id}")
        original_row = self.memlens_rows.get(question_id)
        if original_row is None:
            raise KeyError(
                f"No saved MEMLENS retrieval row for {question_id}"
            )
        dataset = MemoryBenchmarkDataset(
            self.manifest_path.parent / str(item["path"]),
            self.memlens_image_root,
        )
        return dataset, original_row, _qa_map(dataset).get(question_id, {})


def _trace_rounds(row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rounds = list(
        (row.get("retrieval_trace") or {}).get("rounds") or []
    )
    return {
        str(item.get("round_id", "")): dict(item) for item in rounds
    }


def _session_dates(dataset: MemoryBenchmarkDataset) -> Dict[str, str]:
    return {
        str(session.get("session_id", "")): str(session.get("date", ""))
        for session in dataset.sessions
    }


def _anchor_summary(
    anchors: Sequence[Any], limit: int = 5
) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for anchor in list(anchors)[:limit]:
        if isinstance(anchor, str):
            output.append({"text": anchor[:1000]})
        elif isinstance(anchor, Mapping):
            output.append({
                "type": str(
                    anchor.get("source")
                    or anchor.get("type")
                    or anchor.get("kind")
                    or ""
                ),
                "facet": str(anchor.get("facet") or "")[:500],
                "text": str(
                    anchor.get("text")
                    or anchor.get("anchor")
                    or anchor.get("description")
                    or ""
                )[:1000],
            })
    return output


def _ranking_payload(
    row: Mapping[str, Any], candidate_k: int
) -> Dict[str, List[str]]:
    by_k = dict(row.get("candidate_k_results") or {})
    result = dict(by_k.get(str(candidate_k)) or {})
    if not result:
        raise KeyError(
            f"Question {_question_id(row)} lacks candidate K={candidate_k}; "
            f"available={sorted(by_k)}"
        )
    rankings = dict(result.get("rankings") or {})
    required = (
        "abstract_evi",
        "raw_multimodal_fixed",
        "evi_raw_mean_rank",
    )
    missing = [name for name in required if name not in rankings]
    if missing:
        raise KeyError(
            f"Question {_question_id(row)} lacks rankings: {missing}"
        )
    return {
        name: [str(value) for value in rankings[name]]
        for name in required
    }


def _rank_positions(ranking: Sequence[str]) -> Dict[str, int]:
    return {
        str(round_id): index
        for index, round_id in enumerate(ranking, 1)
    }


def _verify_one(
    verifier: SelectiveEvidenceVerifier,
    dataset: MemoryBenchmarkDataset,
    trace_by_round: Mapping[str, Mapping[str, Any]],
    session_dates: Mapping[str, str],
    question: str,
    question_date: str,
    facets: Sequence[str],
    round_id: str,
    ranks: Mapping[str, int],
    max_images: int,
) -> Dict[str, Any]:
    payload = dict(dataset.rounds.get(round_id) or {})
    if not payload:
        raise KeyError(f"Raw dataset has no round {round_id}")
    session_id = str(payload.get("session_id", ""))
    user_text = str(payload.get("user", ""))
    assistant_text = str(payload.get("assistant", ""))
    dialogue_text = "\n".join(
        value for value in (user_text, assistant_text) if value
    )
    images = [
        str(path)
        for path in list(payload.get("images") or [])[:max_images]
    ]
    trace = dict(trace_by_round.get(round_id) or {})
    anchors = list(trace.get("top_anchors") or [])
    prompt = build_verification_prompt(
        question=question,
        question_date=question_date,
        facets=facets,
        session_id=session_id,
        round_id=round_id,
        session_date=session_dates.get(session_id, ""),
        user_text=user_text,
        assistant_text=assistant_text,
        anchors=anchors,
        image_count=len(images),
    )
    result = verifier.verify(prompt, images, dialogue_text)
    return {
        "round_id": round_id,
        "session_id": session_id,
        "session_date": session_dates.get(session_id, ""),
        "ranks": dict(ranks),
        "image_count": len(images),
        "image_paths": images,
        "anchor_hints": _anchor_summary(anchors),
        "prompt_chars": len(prompt),
        **result,
    }


def _process_question(
    row: Mapping[str, Any],
    resolver: DatasetResolver,
    verifier: SelectiveEvidenceVerifier,
    candidate_k: int,
    eval_k: int,
    verification_top_k: int,
    max_images: int,
    executor: ThreadPoolExecutor,
) -> Dict[str, Any]:
    benchmark = str(row.get("benchmark", ""))
    dataset_name = str(row.get("dataset", ""))
    question_id = _question_id(row)
    dataset, original_row, qa = resolver.resolve(
        benchmark, dataset_name, question_id
    )
    rankings = _ranking_payload(row, candidate_k)
    initial_selected = verification_candidate_ids(
        rankings["abstract_evi"],
        rankings["raw_multimodal_fixed"],
        top_k=verification_top_k,
        order_ranking=rankings["evi_raw_mean_rank"],
    )
    rank_maps = {
        name: _rank_positions(values) for name, values in rankings.items()
    }
    question = str(
        row.get("question")
        or original_row.get("question")
        or qa.get("question")
        or ""
    )
    question_date = str(
        original_row.get("question_date")
        or qa.get("question_date")
        or ""
    )
    facets = [
        str(value)
        for value in list(row.get("facets") or [])
        if str(value).strip()
    ]
    trace_by_round = _trace_rounds(original_row)
    dates = _session_dates(dataset)

    results_by_id: Dict[str, Dict[str, Any]] = {}
    verified_order: List[str] = []

    def verify_batch(round_ids: Sequence[str]) -> None:
        futures = {}
        for round_id in round_ids:
            if round_id in results_by_id:
                continue
            ranks = {
                name: mapping.get(round_id, 0)
                for name, mapping in rank_maps.items()
            }
            future = executor.submit(
                _verify_one,
                verifier,
                dataset,
                trace_by_round,
                dates,
                question,
                question_date,
                facets,
                round_id,
                ranks,
                max_images,
            )
            futures[future] = round_id
        for future in as_completed(futures):
            round_id = futures[future]
            result = future.result()
            if result.get("error"):
                raise RuntimeError(
                    f"Verifier API failed for candidate {round_id}: "
                    f"{result['error']}"
                )
            results_by_id[round_id] = result
        verified_order.extend(
            rid
            for rid in round_ids
            if rid in results_by_id and rid not in verified_order
        )

    verify_batch(initial_selected)
    lazy_selected: List[str] = []
    while True:
        verdicts = {
            rid: dict(item.get("verdict") or {})
            for rid, item in results_by_id.items()
        }
        final_ranking, rerank_trace = conservative_rerank(
            rankings["evi_raw_mean_rank"],
            rankings["abstract_evi"],
            rankings["raw_multimodal_fixed"],
            verdicts,
            top_k=verification_top_k,
        )
        pending = str(rerank_trace.get("pending_round_id") or "")
        if not pending:
            break
        lazy_selected.append(pending)
        verify_batch([pending])

    verification_results = [
        results_by_id[round_id] for round_id in verified_order
    ]
    rankings["selective_vlm"] = final_ranking

    clues = [
        str(value) for value in list(row.get("clue_round_ids") or [])
    ]
    clue_ranks = {
        strategy: {
            clue: (
                ranking.index(clue) + 1 if clue in ranking else None
            )
            for clue in clues
        }
        for strategy, ranking in rankings.items()
    }
    return {
        "benchmark": benchmark,
        "dataset": dataset_name,
        "question_id": question_id,
        "question": question,
        "question_type": str(row.get("question_type") or ""),
        "question_subtype": str(row.get("question_subtype") or ""),
        "clue_round_ids": clues,
        "facets": facets,
        "candidate_k": candidate_k,
        "evaluation_k": eval_k,
        "verification_top_k": verification_top_k,
        "selection_policy": "mean_top_k_disagreement_then_lazy_backfill",
        "initial_selected_round_ids": initial_selected,
        "lazy_selected_round_ids": lazy_selected,
        "selected_round_ids": verified_order,
        "rankings": rankings,
        "clue_ranks": clue_ranks,
        "verification_results": verification_results,
        "rerank_trace": rerank_trace,
    }


def _retrieval_metrics(
    rows: Sequence[Mapping[str, Any]], strategy: str, eval_k: int
) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("clue_round_ids")]
    hits = clues = full = hit_questions = wins = losses = 0
    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    for row in eligible:
        expected = set(map(str, row["clue_round_ids"]))
        ranking = list(map(str, row["rankings"][strategy]))
        baseline = list(
            map(str, row["rankings"]["evi_raw_mean_rank"])
        )
        found = len(expected & set(ranking[:eval_k]))
        base_found = len(expected & set(baseline[:eval_k]))
        hits += found
        clues += len(expected)
        recalls.append(found / len(expected))
        full += int(found == len(expected))
        hit_questions += int(found > 0)
        first = min(
            (
                ranking.index(clue) + 1
                for clue in expected
                if clue in ranking
            ),
            default=0,
        )
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        wins += int(found > base_found)
        losses += int(found < base_found)
    count = len(eligible)
    return {
        "num_questions": count,
        "total_clues": clues,
        "clue_round_recall_micro": hits / clues if clues else 0.0,
        "clue_round_recall_macro": sum(recalls) / count if count else 0.0,
        "hit_at_k": hit_questions / count if count else 0.0,
        "full_clue_coverage_at_k": full / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
        "wins_vs_mean_rank": wins,
        "ties_vs_mean_rank": count - wins - losses,
        "losses_vs_mean_rank": losses,
    }


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], field: str, eval_k: int
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown") or "unknown")].append(row)
    return {
        key: {
            strategy: _retrieval_metrics(group, strategy, eval_k)
            for strategy in STRATEGIES
        }
        for key, group in sorted(groups.items())
    }


def _verifier_diagnostics(
    rows: Sequence[Mapping[str, Any]], eval_k: int
) -> Dict[str, Any]:
    utility: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    grounding: Counter[str] = Counter()
    checked = parsed = errors = demoted = demoted_clues = changed = 0
    stable_intersection_drops = initial_checked = lazy_checked = 0
    consensus_moderate_selected = 0
    for row in rows:
        clues = set(map(str, row.get("clue_round_ids", [])))
        before = list(
            row["rankings"]["evi_raw_mean_rank"][:eval_k]
        )
        after = list(row["rankings"]["selective_vlm"][:eval_k])
        changed += int(before != after)
        abstract_top = set(row["rankings"]["abstract_evi"][:eval_k])
        raw_top = set(row["rankings"]["raw_multimodal_fixed"][:eval_k])
        stable_intersection_drops += len(
            ((abstract_top & raw_top) & set(before)) - set(after)
        )
        consensus_moderate_selected += len(
            set(after) - (abstract_top | raw_top)
        )
        initial_checked += len(row.get("initial_selected_round_ids", []))
        lazy_checked += len(row.get("lazy_selected_round_ids", []))
        demoted_ids = set(
            row.get("rerank_trace", {}).get("demoted_round_ids", [])
        )
        demoted += len(demoted_ids)
        demoted_clues += len(clues & demoted_ids)
        for item in row.get("verification_results", []):
            checked += 1
            verdict = dict(item.get("verdict") or {})
            parsed += int(bool(verdict.get("parse_valid")))
            errors += int(bool(item.get("error")))
            utility[str(verdict.get("evidence_utility", "unknown"))] += 1
            confidence[str(verdict.get("confidence", "unknown"))] += 1
            grounding[str(verdict.get("anchor_grounding", "unknown"))] += 1
    return {
        "checked_candidates": checked,
        "parse_valid_candidates": parsed,
        "parse_valid_rate": parsed / checked if checked else 0.0,
        "api_error_candidates": errors,
        "evidence_utility_counts": dict(sorted(utility.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "anchor_grounding_counts": dict(sorted(grounding.items())),
        "demoted_candidates": demoted,
        "demoted_annotated_clues": demoted_clues,
        "questions_with_changed_top_k": changed,
        "stable_mean_top_k_drops": stable_intersection_drops,
        "initial_checked_candidates": initial_checked,
        "lazy_backfill_checked_candidates": lazy_checked,
        "consensus_moderate_selected_candidates": consensus_moderate_selected,
    }


def _write_metrics(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    verifier: SelectiveEvidenceVerifier,
) -> None:
    by_benchmark: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
        key = f"{row['benchmark']}/{row['dataset']}"
        by_dataset[key].append(row)
    metrics = {
        "input": str(Path(args.input).resolve()),
        "candidate_k": args.candidate_k,
        "evaluation_k": args.eval_k,
        "verification_top_k": args.verification_top_k or args.eval_k,
        "strategies": list(STRATEGIES),
        "label_policy": (
            "question type, subtype, clues, and answers never enter "
            "VLM prompts or reranking"
        ),
        "selection_policy": (
            "initially verify mean-rank Top-K candidates in the Anchor/Raw-MM "
            "Top-K symmetric difference; verify lower disputed candidates "
            "only when needed for backfill"
        ),
        "decision_policy": (
            "preserve mean-rank order for all agreement cases and skip only "
            "parse-valid high-confidence not_useful disputed candidates"
        ),
        "verifier": {
            "prompt_version": PROMPT_VERSION,
            "model": verifier.model_namespace,
            "cache_dir": str(verifier.cache_dir),
            "cache_hits_this_process": verifier.cache_hits,
            "cache_misses_this_process": verifier.cache_misses,
            **_verifier_diagnostics(rows, args.eval_k),
        },
        "by_benchmark": {
            key: {
                strategy: _retrieval_metrics(group, strategy, args.eval_k)
                for strategy in STRATEGIES
            }
            for key, group in sorted(by_benchmark.items())
        },
        "by_dataset": {
            key: {
                strategy: _retrieval_metrics(group, strategy, args.eval_k)
                for strategy in STRATEGIES
            }
            for key, group in sorted(by_dataset.items())
        },
        "memlens_by_question_subtype": _group_metrics(
            by_benchmark.get("memlens", []),
            "question_subtype",
            args.eval_k,
        ),
    }
    _write_json(
        output_dir / "selective_verification_metrics.json", metrics
    )

    lines = [
        "# Selective VLM evidence verification",
        "",
        (
            "Base: Anchor-only plus Raw-MM mean-rank. The VLM sees raw "
            "evidence first for disputed candidates already in mean Top-K, "
            "then lazily for disputed backfill candidates."
        ),
        (
            "Candidates with agreement retain mean-rank behavior, including "
            "consistent moderate candidates. All annotations are evaluation-only."
        ),
        "",
        "| Benchmark | Strategy | Micro R | Macro R | Hit@K | Full@K | W/T/L vs mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark, strategies in sorted(
        metrics["by_benchmark"].items()
    ):
        for strategy in STRATEGIES:
            item = strategies[strategy]
            lines.append(
                f"| {benchmark} | {strategy} | "
                f"{item['clue_round_recall_micro']:.4f} | "
                f"{item['clue_round_recall_macro']:.4f} | "
                f"{item['hit_at_k']:.4f} | "
                f"{item['full_clue_coverage_at_k']:.4f} | "
                f"{item['wins_vs_mean_rank']}/"
                f"{item['ties_vs_mean_rank']}/"
                f"{item['losses_vs_mean_rank']} |"
            )
    diagnostics = metrics["verifier"]
    lines.extend([
        "",
        "## Verifier diagnostics",
        "",
        f"- Checked candidates: {diagnostics['checked_candidates']}",
        f"- Parse-valid rate: {diagnostics['parse_valid_rate']:.4f}",
        f"- Demoted candidates: {diagnostics['demoted_candidates']}",
        f"- Demoted annotated clues: {diagnostics['demoted_annotated_clues']}",
        f"- Stable mean-Top-K drops: {diagnostics['stable_mean_top_k_drops']}",
        f"- Initial checked candidates: {diagnostics['initial_checked_candidates']}",
        f"- Lazy backfill checks: {diagnostics['lazy_backfill_checked_candidates']}",
        f"- Consensus-moderate final selections: {diagnostics['consensus_moderate_selected_candidates']}",
        (
            f"- Questions with a changed Top-{args.eval_k}: "
            f"{diagnostics['questions_with_changed_top_k']}"
        ),
        "",
    ])
    (output_dir / "selective_verification_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _model_config(
    path: Path, args: argparse.Namespace
) -> Tuple[Dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if str(config.get("provider", "openai_api")) != "openai_api":
        raise ValueError(
            "Selective verifier currently requires provider=openai_api"
        )
    model = str(config.get("model", ""))
    if not model:
        raise ValueError(f"Model config has no model: {path}")
    api_key = str(config.get("api_key", ""))
    api_key_env = str(
        config.get("api_key_env", "OPENAI_API_KEY")
    )
    if not api_key:
        api_key = os.environ.get(api_key_env, "")
    base_url = str(
        config.get("base_url", "https://api.openai.com/v1")
    )
    max_tokens = args.max_new_tokens or int(
        config.get("max_new_tokens", 512) or 512
    )
    timeout = args.timeout or int(config.get("timeout", 90) or 90)
    namespace_payload = {
        "provider": "openai_api",
        "model": model,
        "base_url": base_url,
        "max_new_tokens": max_tokens,
        "timeout": timeout,
        "prompt_version": PROMPT_VERSION,
        "dry_run": bool(args.dry_run),
    }
    namespace = json.dumps(namespace_payload, sort_keys=True)
    if args.dry_run:
        def dry_vlm(
            _system: str, _user: str, _images: List[str]
        ) -> str:
            return json.dumps({
                "anchor_grounding": "uncertain",
                "evidence_utility": "possibly_useful",
                "confidence": "low",
                "supported_facet_indices": [],
                "dialogue_grounding": "",
                "visual_grounding": "",
                "reason": "Dry run preserves the base ranking.",
            })
        return {"vlm": dry_vlm, **namespace_payload}, namespace
    vlm = make_openai_vlm(
        api_key, base_url, model, max_tokens, timeout
    )
    return {"vlm": vlm, **namespace_payload}, namespace


def _run_config(
    args: argparse.Namespace,
    input_dir: Path,
    rescoring_path: Path,
    model_path: Path,
    namespace: str,
) -> Dict[str, Any]:
    return {
        "input": str(input_dir),
        "rescoring_questions": str(rescoring_path),
        "verifier_model_config": str(model_path),
        "verifier_model_namespace": namespace,
        "prompt_version": PROMPT_VERSION,
        "candidate_k": args.candidate_k,
        "evaluation_k": args.eval_k,
        "verification_top_k": args.verification_top_k or args.eval_k,
        "max_images_per_round": args.max_images_per_round,
        "benchmark": args.benchmark,
        "max_questions_per_dataset": args.max_questions,
        "dry_run": args.dry_run,
        "selection_policy": "mean_top_k_disagreement_then_lazy_backfill",
        "decision_policy": "mean_rank_preserving_lazy_disagreement_verification",
        "label_isolation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="Completed abstract-candidate joint run",
    )
    parser.add_argument("--rescoring-questions", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--verifier-model-config",
        default="config/models/qwen3_vl_8b_openrouter.yaml",
    )
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument(
        "--verification-top-k",
        type=int,
        default=0,
        help="Top-K inclusion boundary; 0 follows --eval-k (effective default: 10)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument(
        "--max-images-per-round", type=int, default=4
    )
    parser.add_argument(
        "--max-questions", type=int, default=0,
        help="Limit per dataset",
    )
    parser.add_argument(
        "--benchmark",
        choices=("all", "memeye", "memlens"),
        default="all",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--memlens-manifest",
        default=(
            "data/memlens/converted_32k_agent195/manifest.json"
        ),
    )
    parser.add_argument(
        "--memlens-image-root", default="data/memlens"
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    positive = (
        "candidate_k", "eval_k", "workers",
        "max_images_per_round", "max_new_tokens",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be positive"
            )
    if args.verification_top_k < 0:
        parser.error(
            "--verification-top-k must be non-negative"
        )
    if args.verification_top_k not in (0, args.eval_k):
        parser.error(
            "--verification-top-k must equal --eval-k because verification "
            "resolves that exact Top-K inclusion boundary"
        )

    input_dir = Path(args.input).resolve()
    rescoring_path = (
        Path(args.rescoring_questions).resolve()
        if args.rescoring_questions
        else input_dir
        / "provenance_verification"
        / "provenance_verification_questions.jsonl"
    )
    if not rescoring_path.exists():
        raise FileNotFoundError(
            f"Missing saved Raw-MM rescoring output: {rescoring_path}. "
            "Run analyze_provenance_verification.py first."
        )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / "selective_verification"
    )
    _configure_logging(output_dir)
    model_path = Path(args.verifier_model_config).resolve()
    model_payload, namespace = _model_config(model_path, args)
    config = _run_config(
        args, input_dir, rescoring_path, model_path, namespace
    )
    config_path = output_dir / "selective_verification_config.json"
    if config_path.exists() and _read_json(config_path) != config:
        raise RuntimeError(
            "Existing output has a different configuration: "
            f"{config_path}. Choose another --output-dir."
        )
    _write_json(config_path, config)

    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else output_dir / "vlm_cache"
    )
    verifier = SelectiveEvidenceVerifier(
        model_payload["vlm"],
        cache_dir=cache_dir,
        model_namespace=namespace,
    )
    resolver = DatasetResolver(
        input_dir=input_dir,
        data_root=Path(args.data_root).resolve(),
        manifest_path=Path(args.memlens_manifest).resolve(),
        memlens_image_root=Path(args.memlens_image_root).resolve(),
    )

    output_jsonl = (
        output_dir / "selective_verification_questions.jsonl"
    )
    completed = (
        {} if args.no_resume else _load_completed(output_jsonl)
    )
    selected_count = sum(
        1
        for _ in _selected_rows(
            rescoring_path, args.benchmark, args.max_questions
        )
    )
    resumed_count = sum(
        _row_key(row) in completed
        for row in _selected_rows(
            rescoring_path, args.benchmark, args.max_questions
        )
    )

    verification_top_k = args.verification_top_k or args.eval_k
    log.info(
        "Verifier model=%s candidate_k=%d eval_k=%d "
        "verify_top_k=%d workers=%d dry_run=%s",
        model_payload["model"],
        args.candidate_k,
        args.eval_k,
        verification_top_k,
        args.workers,
        args.dry_run,
    )
    log.info(
        "Selected questions=%d resumed=%d cache=%s",
        selected_count,
        resumed_count,
        cache_dir,
    )
    mode = "w" if args.no_resume else "a"
    processed = 0
    failed_keys: List[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        with output_jsonl.open(mode, encoding="utf-8") as handle:
            rows = _selected_rows(
                rescoring_path, args.benchmark, args.max_questions
            )
            for index, row in enumerate(rows, 1):
                key = _row_key(row)
                if key in completed:
                    continue
                try:
                    result = _process_question(
                        row,
                        resolver,
                        verifier,
                        candidate_k=args.candidate_k,
                        eval_k=args.eval_k,
                        verification_top_k=verification_top_k,
                        max_images=args.max_images_per_round,
                        executor=executor,
                    )
                except Exception:
                    failed_keys.append(key)
                    log.exception(
                        "[%d/%d] %s failed; cached successful "
                        "candidate calls will be reused on retry",
                        index, selected_count, key,
                    )
                    continue
                handle.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                handle.flush()
                completed[key] = result
                processed += 1
                parse_valid = sum(
                    bool(
                        item.get("verdict", {}).get("parse_valid")
                    )
                    for item in result["verification_results"]
                )
                log.info(
                    "[%d/%d] %s candidates=%d initial=%d lazy=%d "
                    "parse_valid=%d demoted=%d cache_hits=%d",
                    index,
                    selected_count,
                    key,
                    len(result["verification_results"]),
                    len(result["initial_selected_round_ids"]),
                    len(result["lazy_selected_round_ids"]),
                    parse_valid,
                    result["rerank_trace"]["demoted_count"],
                    verifier.cache_hits,
                )
                _write_json(
                    output_dir / "selective_verification_status.json",
                    {
                        "status": "running",
                        "selected_questions": selected_count,
                        "completed_questions": len(completed),
                        "processed_this_run": processed,
                        "failed_this_run": len(failed_keys),
                        "cache_hits_this_process": verifier.cache_hits,
                        "cache_misses_this_process": verifier.cache_misses,
                    },
                )

    final_rows = list(_load_completed(output_jsonl).values())
    _write_metrics(output_dir, final_rows, args, verifier)
    _write_json(
        output_dir / "selective_verification_status.json",
        {
            "status": (
                "complete" if not failed_keys else "incomplete"
            ),
            "selected_questions": selected_count,
            "completed_questions": len(final_rows),
            "processed_this_run": processed,
            "failed_this_run": len(failed_keys),
            "failed_question_keys": failed_keys,
            "cache_hits_this_process": verifier.cache_hits,
            "cache_misses_this_process": verifier.cache_misses,
        },
    )
    log.info(
        "Finished: questions=%d failed=%d output=%s",
        len(final_rows), len(failed_keys), output_dir,
    )
    if failed_keys:
        raise RuntimeError(
            f"{len(failed_keys)} question(s) remain incomplete; rerun "
            "the same command to retry only missing VLM calls"
        )


if __name__ == "__main__":
    main()
