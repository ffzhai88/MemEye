"""Replay provenance-conditioned raw-evidence verification on abstract EVI traces.

The input must be a joint retrieval run produced by the anchor-only abstract
candidate configuration. Evaluation labels are read only after every ranking
has been produced.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from analyze_candidate_rescoring import (
    PersistentTextVectorCache,
    _diagnostic_method_config,
    _memeye_dataset,
    _read_json,
    _read_jsonl,
    discover_memeye_task_dirs,
)
from benchmark.dataset import MemoryBenchmarkDataset, build_round_retrieval_text
from benchmark.embeddings import TextEmbedder
from benchmark.evi.facet_multimodal import empirical_midrank_percentiles
from benchmark.evi.raw_multimodal import mean_rank_fuse
from benchmark.image_retrieval import RawImageRoundIndex


log = logging.getLogger("provenance_verification")

STRATEGIES = (
    "abstract_evi",
    "raw_multimodal_fixed",
    "evi_raw_mean_rank",
    "provenance_raw_only",
    "provenance_min",
    "provenance_product",
)


def _rank(
    scores: Mapping[str, float], primary_ranks: Mapping[str, int]
) -> List[str]:
    return sorted(
        scores,
        key=lambda rid: (
            -float(scores[rid]),
            int(primary_ranks.get(rid, 10**9)),
            str(rid),
        ),
    )


def _facet_aggregate(
    candidates: Sequence[str],
    facet_values: Sequence[Mapping[str, float]],
    primary_ranks: Mapping[str, int],
) -> Tuple[List[str], Dict[str, Any]]:
    facet_ranks: List[Dict[str, int]] = []
    for values in facet_values:
        ordered = _rank(values, primary_ranks)
        facet_ranks.append({rid: rank for rank, rid in enumerate(ordered, 1)})
    scores: Dict[str, float] = {}
    rows: Dict[str, Any] = {}
    for rid in candidates:
        available = [
            (index, float(values[rid]))
            for index, values in enumerate(facet_values)
            if rid in values
        ]
        max_score = max((value for _, value in available), default=0.0)
        consensus = sum(
            1.0 / facet_ranks[index][rid] for index, _ in available
        )
        scores[rid] = max_score * consensus
        rows[rid] = {
            "max_verified_facet_score": max_score,
            "facet_rank_consensus": consensus,
            "score": scores[rid],
            "verified_facet_count": len(available),
            "facet_ranks": {
                f"f{index}": facet_ranks[index][rid]
                for index, _ in available
            },
        }
    return _rank(scores, primary_ranks), rows


def score_provenance_strategies(
    candidate_ids: Sequence[str],
    facets: Sequence[str],
    round_trace_by_id: Mapping[str, Mapping[str, Any]],
    raw_dialogue_by_facet: Sequence[Mapping[str, float]],
    raw_image_by_facet: Sequence[Mapping[str, float]],
    image_round_ids: set[str],
    raw_full_dialogue: Mapping[str, float] | None = None,
    raw_full_image: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Rank fixed candidates with source-conditioned conjunctive verification."""
    candidates = list(dict.fromkeys(str(value) for value in candidate_ids if str(value)))
    primary_ranks = {rid: rank for rank, rid in enumerate(candidates, 1)}
    product_facets: List[Dict[str, float]] = []
    minimum_facets: List[Dict[str, float]] = []
    raw_only_facets: List[Dict[str, float]] = []
    path_rows: List[Dict[str, Any]] = []

    for facet_index, facet in enumerate(facets):
        source_memory_raw: Dict[str, Dict[str, float]] = {
            "dialogue": {}, "visual": {},
        }
        for rid in candidates:
            trace_row = dict(round_trace_by_id.get(rid, {}))
            source_scores = dict(trace_row.get("source_facet_scores", {}))
            for source in ("dialogue", "visual"):
                key = f"f{facet_index}:{source}"
                if key in source_scores:
                    source_memory_raw[source][rid] = float(source_scores[key])

        source_product: Dict[str, Dict[str, float]] = {}
        source_minimum: Dict[str, Dict[str, float]] = {}
        source_raw_only: Dict[str, Dict[str, float]] = {}
        for source in ("dialogue", "visual"):
            memory_raw = source_memory_raw[source]
            eligible_ids = [
                rid for rid in memory_raw
                if source == "dialogue" or rid in image_round_ids
            ]
            raw_source = (
                raw_dialogue_by_facet[facet_index]
                if source == "dialogue"
                else raw_image_by_facet[facet_index]
            )
            evidence_raw = {
                rid: float(raw_source.get(rid, 0.0)) for rid in eligible_ids
            }
            memory_cal = empirical_midrank_percentiles(
                {rid: memory_raw[rid] for rid in eligible_ids}
            )
            evidence_cal = empirical_midrank_percentiles(evidence_raw)
            source_product[source] = {
                rid: memory_cal[rid] * evidence_cal[rid] for rid in eligible_ids
            }
            source_minimum[source] = {
                rid: min(memory_cal[rid], evidence_cal[rid]) for rid in eligible_ids
            }
            source_raw_only[source] = dict(evidence_cal)
            for rid in eligible_ids:
                path_rows.append({
                    "round_id": rid,
                    "facet_index": facet_index,
                    "facet": facet,
                    "source": source,
                    "memory_score_raw": memory_raw[rid],
                    "memory_score_percentile": memory_cal[rid],
                    "raw_support_score": evidence_raw[rid],
                    "raw_support_percentile": evidence_cal[rid],
                    "product_score": source_product[source][rid],
                    "minimum_score": source_minimum[source][rid],
                })

        def best_source(
            values: Mapping[str, Mapping[str, float]]
        ) -> Dict[str, float]:
            return {
                rid: max(
                    source_values[rid]
                    for source_values in values.values()
                    if rid in source_values
                )
                for rid in candidates
                if any(rid in source_values for source_values in values.values())
            }

        product_facets.append(best_source(source_product))
        minimum_facets.append(best_source(source_minimum))
        raw_only_facets.append(best_source(source_raw_only))

    product_ranking, product_rows = _facet_aggregate(
        candidates, product_facets, primary_ranks
    )
    minimum_ranking, minimum_rows = _facet_aggregate(
        candidates, minimum_facets, primary_ranks
    )
    raw_only_ranking, raw_only_rows = _facet_aggregate(
        candidates, raw_only_facets, primary_ranks
    )
    full_text = (
        raw_full_dialogue
        if raw_full_dialogue is not None
        else (raw_dialogue_by_facet[0] if raw_dialogue_by_facet else {})
    )
    full_image = (
        raw_full_image
        if raw_full_image is not None
        else (raw_image_by_facet[0] if raw_image_by_facet else {})
    )
    raw_fixed_scores = {
        rid: 0.5 * float(full_text.get(rid, 0.0))
        + 0.5 * float(full_image.get(rid, 0.0))
        for rid in candidates
    }
    raw_fixed_ranking = _rank(raw_fixed_scores, primary_ranks)
    mean_ranked, mean_rank_rows = mean_rank_fuse(candidates, raw_fixed_ranking)
    return {
        "rankings": {
            "abstract_evi": candidates,
            "raw_multimodal_fixed": raw_fixed_ranking,
            "evi_raw_mean_rank": mean_ranked,
            "provenance_raw_only": raw_only_ranking,
            "provenance_min": minimum_ranking,
            "provenance_product": product_ranking,
        },
        "path_rows": path_rows,
        "strategy_rows": {
            "provenance_product": product_rows,
            "provenance_min": minimum_rows,
            "provenance_raw_only": raw_only_rows,
            "evi_raw_mean_rank": mean_rank_rows,
        },
    }


def _score_rows(
    rows: Sequence[Dict[str, Any]],
    dataset: MemoryBenchmarkDataset,
    dataset_name: str,
    benchmark: str,
    method_cfg: Mapping[str, Any],
    text_cache: PersistentTextVectorCache,
    candidate_ks: Sequence[int],
) -> List[Dict[str, Any]]:
    image_index = RawImageRoundIndex(dataset, dict(method_cfg))
    output: List[Dict[str, Any]] = []
    maximum_k = max(candidate_ks)
    for index, row in enumerate(rows, 1):
        trace = dict(row.get("retrieval_trace") or {})
        scorer = str(trace.get("facet_round_scorer", "anchor") or "anchor")
        if scorer != "anchor":
            raise ValueError(
                f"Question {row.get('question_id', row.get('idx'))} uses "
                f"facet_round_scorer={scorer!r}; provenance verification requires "
                "an anchor-only abstract candidate run"
            )
        round_trace = list(trace.get("rounds") or [])[:maximum_k]
        candidates = [str(item.get("round_id", "")) for item in round_trace]
        candidates = [rid for rid in candidates if rid]
        if not candidates:
            continue
        facets = [str(item.get("text", "")) for item in trace.get("facets", [])]
        facets = [facet for facet in facets if facet]
        if not facets:
            facets = [str(trace.get("question_stem") or row.get("question") or "")]
        texts = [
            build_round_retrieval_text(dataset.rounds.get(rid, {}), "multimodal")
            for rid in candidates
        ]
        document_vectors = text_cache.embed(texts, role="document")
        full_question = str(
            trace.get("question_stem") or row.get("question") or facets[0]
        )
        query_vectors = text_cache.embed([*facets, full_question], role="query")
        facet_query_vectors = query_vectors[:len(facets)]
        full_query_vector = query_vectors[-1]
        raw_dialogue: List[Dict[str, float]] = []
        raw_image: List[Dict[str, float]] = []
        for facet, query_vector in zip(facets, facet_query_vectors):
            raw_dialogue.append({
                rid: _cosine(query_vector, vector)
                for rid, vector in zip(candidates, document_vectors)
            })
            image_hits, _ = image_index.search(facet, len(dataset.rounds))
            image_scores = {
                str(item["round_id"]): float(item["score"]) for item in image_hits
            }
            raw_image.append({rid: image_scores.get(rid, 0.0) for rid in candidates})
        raw_full_dialogue = {
            rid: _cosine(full_query_vector, vector)
            for rid, vector in zip(candidates, document_vectors)
        }
        full_image_hits, _ = image_index.search(full_question, len(dataset.rounds))
        full_image_scores = {
            str(item["round_id"]): float(item["score"])
            for item in full_image_hits
        }
        raw_full_image = {
            rid: full_image_scores.get(rid, 0.0) for rid in candidates
        }
        image_round_ids = {
            rid for rid in candidates
            if list(dataset.rounds.get(rid, {}).get("images", []) or [])
        }
        trace_by_id = {str(item.get("round_id")): item for item in round_trace}
        by_k: Dict[str, Any] = {}
        for candidate_k in candidate_ks:
            selected = candidates[:candidate_k]
            scored = score_provenance_strategies(
                selected,
                facets,
                trace_by_id,
                [{rid: values[rid] for rid in selected} for values in raw_dialogue],
                [{rid: values[rid] for rid in selected} for values in raw_image],
                image_round_ids & set(selected),
                {rid: raw_full_dialogue[rid] for rid in selected},
                {rid: raw_full_image[rid] for rid in selected},
            )
            by_k[str(candidate_k)] = {
                "candidate_count": len(selected),
                **scored,
            }
        output.append({
            "benchmark": benchmark,
            "dataset": dataset_name,
            "question_id": row.get("question_id", row.get("idx", "")),
            "question": row.get("question", ""),
            "question_type": row.get("question_type", ""),
            "question_subtype": row.get("question_subtype", ""),
            "clue_round_ids": list(map(str, row.get("clue_round_ids", []))),
            "facets": facets,
            "image_candidate_count": len(image_round_ids),
            "candidate_k_results": by_k,
        })
        if index % 25 == 0 or index == len(rows):
            log.info("[%s/%s] verified %d/%d", benchmark, dataset_name, index, len(rows))
    return output


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(value) ** 2 for value in left) ** 0.5
    right_norm = sum(float(value) ** 2 for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _metrics(
    rows: Sequence[Dict[str, Any]], strategy: str, candidate_k: int, eval_k: int
) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("clue_round_ids")]
    hits = clues = wins = losses = 0
    recalls: List[float] = []
    for row in eligible:
        expected = set(row["clue_round_ids"])
        rankings = row["candidate_k_results"][str(candidate_k)]["rankings"]
        found = len(expected & set(rankings[strategy][:eval_k]))
        baseline = len(expected & set(rankings["abstract_evi"][:eval_k]))
        hits += found
        clues += len(expected)
        recalls.append(found / len(expected))
        wins += int(found > baseline)
        losses += int(found < baseline)
    count = len(eligible)
    return {
        "num_questions": count,
        "total_clues": clues,
        "clue_round_recall_micro": hits / clues if clues else 0.0,
        "clue_round_recall_macro": sum(recalls) / count if count else 0.0,
        "wins_vs_abstract_evi": wins,
        "ties_vs_abstract_evi": count - wins - losses,
        "losses_vs_abstract_evi": losses,
    }


def _group_metrics(
    rows: Sequence[Dict[str, Any]], strategy: str, candidate_k: int,
    eval_k: int, field: str,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown") or "unknown")].append(row)
    return {
        key: _metrics(group, strategy, candidate_k, eval_k)
        for key, group in sorted(groups.items())
    }


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "provenance_verification.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        log.addHandler(handler)


def _memlens_method_config(
    memlens_dir: Path,
    shared_method_config: Mapping[str, Any],
    cache_root: Path,
) -> Dict[str, Any]:
    """Load a merged legacy config or reuse the joint run's shared method config."""
    merged_config = memlens_dir / "config.json"
    if merged_config.exists():
        return _diagnostic_method_config(_read_json(merged_config), cache_root)
    suite_config = memlens_dir / "suite_config.json"
    if not suite_config.exists():
        raise FileNotFoundError(
            f"MEMLENS run has neither {merged_config.name} nor "
            f"{suite_config.name}: {memlens_dir}"
        )
    # Current MEMLENS retrieval suites save only run metadata in suite_config;
    # the joint runner uses the same method YAML for MemEye and MEMLENS.
    log.info(
        "[memlens] %s contains run metadata only; reusing the joint "
        "MemEye method configuration",
        suite_config,
    )
    return dict(shared_method_config)


def _write_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    text_cache: PersistentTextVectorCache,
) -> None:
    with (output_dir / "provenance_verification_questions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    benchmarks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    datasets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        benchmarks[row["benchmark"]].append(row)
        datasets[f"{row['benchmark']}/{row['dataset']}"].append(row)
    metrics: Dict[str, Any] = {
        "input": str(Path(args.input).resolve()),
        "label_policy": "clues and metadata are used only after ranking",
        "candidate_ks": args.candidate_ks,
        "primary_candidate_k": args.primary_candidate_k,
        "evaluation_k": args.eval_k,
        "strategies": list(STRATEGIES),
        "verification": {
            "path": "facet_to_anchor_source_AND_facet_to_raw_provenance",
            "calibration": "per_facet_per_source_empirical_midrank_percentile",
            "primary_operator": "product",
            "source_selection": "best_available_verified_provenance_path",
            "missing_modality": "path_absent_not_zero_candidate_penalty",
        },
        "text_embedding_model": text_cache.model_name,
        "text_embedding_cache": {
            "path": str(text_cache.root),
            "hits": text_cache.hits,
            "misses": text_cache.misses,
        },
        "by_candidate_k": {},
    }
    for candidate_k in args.candidate_ks:
        key = str(candidate_k)
        metrics["by_candidate_k"][key] = {
            "by_benchmark": {
                benchmark: {
                    strategy: _metrics(group, strategy, candidate_k, args.eval_k)
                    for strategy in STRATEGIES
                }
                for benchmark, group in sorted(benchmarks.items())
            },
            "by_dataset": {
                dataset: {
                    strategy: _metrics(group, strategy, candidate_k, args.eval_k)
                    for strategy in STRATEGIES
                }
                for dataset, group in sorted(datasets.items())
            },
            "memlens_by_question_subtype": {
                strategy: _group_metrics(
                    benchmarks.get("memlens", []), strategy, candidate_k,
                    args.eval_k, "question_subtype",
                )
                for strategy in STRATEGIES
            },
        }
    (output_dir / "provenance_verification_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Provenance-conditioned evidence verification", "",
        "Raw evidence verifies an existing facet/source path; it cannot create a path.",
        "Evaluation annotations never enter scoring.", "",
    ]
    for candidate_k in args.candidate_ks:
        lines.extend([
            f"## Candidate K={candidate_k}", "",
            "| Benchmark | Strategy | Micro R | Macro R | W/T/L |",
            "|---|---|---:|---:|---:|",
        ])
        payload = metrics["by_candidate_k"][str(candidate_k)]["by_benchmark"]
        for benchmark, strategies in sorted(payload.items()):
            for strategy in STRATEGIES:
                item = strategies[strategy]
                lines.append(
                    f"| {benchmark} | {strategy} | "
                    f"{item['clue_round_recall_micro']:.4f} | "
                    f"{item['clue_round_recall_macro']:.4f} | "
                    f"{item['wins_vs_abstract_evi']}/"
                    f"{item['ties_vs_abstract_evi']}/"
                    f"{item['losses_vs_abstract_evi']} |"
                )
        lines.append("")
    (output_dir / "provenance_verification_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _parse_ks(value: str) -> List[int]:
    output = sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    if not output or any(item <= 0 for item in output):
        raise argparse.ArgumentTypeError("candidate ks must be positive integers")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Completed abstract-EVI joint run")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--candidate-ks", type=_parse_ks, default=_parse_ks("20,30,50"))
    parser.add_argument("--primary-candidate-k", type=int, default=30)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--memlens-manifest",
        default="data/memlens/converted_32k_agent195/manifest.json",
    )
    parser.add_argument("--memlens-image-root", default="data/memlens")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()
    if args.primary_candidate_k not in args.candidate_ks:
        parser.error("--primary-candidate-k must be included in --candidate-ks")

    input_dir = Path(args.input).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / "provenance_verification"
    )
    _configure_logging(output_dir)
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / "embedding_cache"
    data_root = Path(args.data_root).resolve()
    task_dirs = discover_memeye_task_dirs(input_dir / "memeye")
    if not task_dirs:
        raise FileNotFoundError(f"No completed MemEye tasks under {input_dir / 'memeye'}")
    first_config = _read_json(task_dirs[0] / "config.json")
    first_method = _diagnostic_method_config(first_config, cache_root)
    text_model = str(first_method.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL))
    text_cache = PersistentTextVectorCache(
        cache_root / "text", text_model,
        dict(first_method.get("text_embedding_kwargs") or {}),
    )
    output_rows: List[Dict[str, Any]] = []
    for task_dir in task_dirs:
        rows = _read_jsonl(task_dir / "retrievals.jsonl")
        if args.max_questions > 0:
            rows = rows[:args.max_questions]
        dataset, config = _memeye_dataset(task_dir, data_root)
        output_rows.extend(_score_rows(
            rows, dataset, task_dir.name, "memeye",
            _diagnostic_method_config(config, cache_root), text_cache,
            args.candidate_ks,
        ))

    manifest_path = Path(args.memlens_manifest).resolve()
    manifest = _read_json(manifest_path)
    item_by_id = {
        str(item.get("question_id", "")): item for item in manifest.get("items", [])
    }
    memlens_rows = _read_jsonl(input_dir / "memlens" / "retrievals.jsonl")
    if args.max_questions > 0:
        memlens_rows = memlens_rows[:args.max_questions]
    memlens_method = _memlens_method_config(
        input_dir / "memlens", first_method, cache_root
    )
    image_root = Path(args.memlens_image_root).resolve()
    for index, row in enumerate(memlens_rows, 1):
        question_id = str(row.get("question_id", ""))
        item = item_by_id.get(question_id)
        if item is None:
            raise KeyError(f"MEMLENS manifest has no item for {question_id}")
        dataset = MemoryBenchmarkDataset(
            manifest_path.parent / str(item["path"]), image_root
        )
        output_rows.extend(_score_rows(
            [row], dataset, "memlens", "memlens", memlens_method,
            text_cache, args.candidate_ks,
        ))
        if index % 25 == 0 or index == len(memlens_rows):
            log.info("[memlens] loaded and verified %d/%d", index, len(memlens_rows))
    _write_outputs(output_dir, output_rows, args, text_cache)
    log.info("Complete: rows=%d output=%s", len(output_rows), output_dir)


if __name__ == "__main__":
    main()
