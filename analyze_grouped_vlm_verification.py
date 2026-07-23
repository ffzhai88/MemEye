"""Run rank-batch or consensus-neighborhood multimodal evidence verification.

The fixed Anchor Top-30 and Anchor/Raw-MM mean-rank remain the candidate source
and global prior. The VLM sees small natural-language/image groups and can only
make explicit within-group drop-to-keep replacements. Benchmark names, question
types, answers, and clue annotations never enter grouping, prompts, or ranking.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from analyze_consensus_evidence_neighborhoods import (
    _trace_features,
    build_relation_graph,
)
from analyze_selective_vlm_verification import (
    DatasetResolver,
    _load_completed,
    _question_id,
    _ranking_payload,
    _retrieval_metrics,
    _row_key,
    _selected_rows,
    _session_dates,
)
from benchmark.evi.grouped_verifier import (
    PROMPT_VERSION,
    GroupedEvidenceVerifier,
    build_group_plan,
    build_group_prompt,
    local_replace_rerank,
)
from benchmark.evi.vlm import make_openai_vlm

log = logging.getLogger("grouped_vlm_verification")
BASE_STRATEGIES = (
    "abstract_evi",
    "raw_multimodal_fixed",
    "evi_raw_mean_rank",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(
            output_dir / "grouped_verification.log", encoding="utf-8"
        ),
    ):
        handler.setFormatter(formatter)
        log.addHandler(handler)


def _model(
    path: Path, args: argparse.Namespace
) -> tuple[Dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if str(config.get("provider", "openai_api")) != "openai_api":
        raise ValueError("Grouped verifier requires provider=openai_api")
    model = str(config.get("model", ""))
    if not model:
        raise ValueError(f"Model config has no model: {path}")
    base_url = str(
        config.get("base_url", "https://api.openai.com/v1")
    )
    max_tokens = args.max_new_tokens or int(
        config.get("max_new_tokens", 1024) or 1024
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
            _system: str, prompt: str, _images: List[str]
        ) -> str:
            ids = list(dict.fromkeys(re.findall(
                r"Round ([^ ]+) from session", prompt
            )))
            return json.dumps({
                "members": [
                    {
                        "round_id": rid,
                        "state": "uncertain",
                        "confidence": "low",
                        "reason": "Dry run preserves mean-rank.",
                    }
                    for rid in ids
                ],
                "joint_groups": [],
            })
        return {"vlm": dry_vlm, **namespace_payload}, namespace
    api_key = str(config.get("api_key", ""))
    if not api_key:
        api_key = os.environ.get(
            str(config.get("api_key_env", "OPENAI_API_KEY")), ""
        )
    return {
        "vlm": make_openai_vlm(
            api_key, base_url, model, max_tokens, timeout
        ),
        **namespace_payload,
    }, namespace


def _members(
    dataset: Any,
    round_ids: Sequence[str],
    dates: Mapping[str, str],
    max_images_per_round: int,
    max_images_per_batch: int,
) -> tuple[List[Dict[str, Any]], List[str], int]:
    output: List[Dict[str, Any]] = []
    images: List[str] = []
    omitted = 0
    for round_id in round_ids:
        payload = dict(dataset.rounds.get(round_id) or {})
        if not payload:
            raise KeyError(f"Raw dataset has no round {round_id}")
        candidates = [
            str(path)
            for path in list(payload.get("images") or [])
        ][:max_images_per_round]
        available = max(0, max_images_per_batch - len(images))
        chosen = candidates[:available]
        omitted += len(candidates) - len(chosen)
        images.extend(chosen)
        session_id = str(payload.get("session_id", ""))
        output.append({
            "round_id": round_id,
            "session_id": session_id,
            "session_date": dates.get(session_id, ""),
            "user_text": str(payload.get("user", "")),
            "assistant_text": str(payload.get("assistant", "")),
            "images": chosen,
        })
    return output, images, omitted


def _process_question(
    row: Mapping[str, Any],
    resolver: DatasetResolver,
    verifier: GroupedEvidenceVerifier,
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
) -> Dict[str, Any]:
    benchmark = str(row.get("benchmark", ""))
    dataset_name = str(row.get("dataset", ""))
    question_id = _question_id(row)
    dataset, original, qa = resolver.resolve(
        benchmark, dataset_name, question_id
    )
    rankings = _ranking_payload(row, args.candidate_k)
    rankings = {
        key: list(values[:args.candidate_k])
        for key, values in rankings.items()
    }
    pool = rankings["evi_raw_mean_rank"]
    features = _trace_features(original, pool)
    edges = build_relation_graph(features)
    plan = build_group_plan(
        pool,
        rankings["abstract_evi"],
        rankings["raw_multimodal_fixed"],
        edges,
        top_k=args.eval_k,
        max_rounds=args.max_rounds_per_batch,
        mode=args.grouping_mode,
    )
    question = str(
        row.get("question")
        or original.get("question")
        or qa.get("question")
        or ""
    )
    question_date = str(
        original.get("question_date")
        or qa.get("question_date")
        or ""
    )
    dates = _session_dates(dataset)

    futures = {}
    payloads = {}
    for index, round_ids in enumerate(plan["batches"]):
        members, images, omitted = _members(
            dataset,
            round_ids,
            dates,
            args.max_images_per_round,
            args.max_images_per_batch,
        )
        prompt = build_group_prompt(
            question=question,
            question_date=question_date,
            members=members,
        )
        raw_image_bytes = sum(
            Path(path).stat().st_size
            for path in images
            if Path(path).is_file()
        )
        estimated_base64_bytes = sum(
            4 * ((Path(path).stat().st_size + 2) // 3)
            for path in images
            if Path(path).is_file()
        )
        payloads[index] = {
            "batch_index": index,
            "round_ids": list(round_ids),
            "members": members,
            "image_paths": images,
            "image_count": len(images),
            "raw_image_bytes": raw_image_bytes,
            "estimated_base64_image_bytes": estimated_base64_bytes,
            "estimated_request_bytes": (
                estimated_base64_bytes
                + len(prompt.encode("utf-8"))
                + 4096
            ),
            "omitted_image_count": omitted,
            "prompt_chars": len(prompt),
        }
        log.info(
            "[REQUEST] %s/%s/%s batch=%d rounds=%s images=%d "
            "raw_image_mb=%.2f estimated_request_mb=%.2f",
            benchmark,
            dataset_name,
            question_id,
            index,
            ",".join(map(str, round_ids)),
            len(images),
            raw_image_bytes / (1024 * 1024),
            payloads[index]["estimated_request_bytes"]
            / (1024 * 1024),
        )
        futures[executor.submit(
            verifier.verify, prompt, images, round_ids
        )] = index

    results_by_index: Dict[int, Dict[str, Any]] = {}
    for future in as_completed(futures):
        index = futures[future]
        result = future.result()
        if result.get("error"):
            raise RuntimeError(
                f"Grouped verifier failed for batch {index}: "
                f"{result['error']}"
            )
        results_by_index[index] = {
            **payloads[index],
            **result,
        }
    group_results = [
        results_by_index[index]
        for index in range(len(plan["batches"]))
    ]
    final_ranking, rerank_trace = local_replace_rerank(
        pool,
        plan["batches"],
        [item["verdict"] for item in group_results],
        top_k=args.eval_k,
    )
    rankings["grouped_vlm"] = final_ranking
    clues = list(map(str, row.get("clue_round_ids") or []))
    return {
        "benchmark": benchmark,
        "dataset": dataset_name,
        "question_id": question_id,
        "question": question,
        "question_type": str(row.get("question_type", "")),
        "question_subtype": str(row.get("question_subtype", "")),
        "clue_round_ids": clues,
        "candidate_k": args.candidate_k,
        "evaluation_k": args.eval_k,
        "grouping_mode": args.grouping_mode,
        "group_plan": plan,
        "group_results": group_results,
        "rankings": rankings,
        "clue_ranks": {
            name: {
                clue: (
                    ranking.index(clue) + 1
                    if clue in ranking else None
                )
                for clue in clues
            }
            for name, ranking in rankings.items()
        },
        "rerank_trace": rerank_trace,
    }


def _diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    calls = parsed = members = replacements = dropped_clues = 0
    images = omitted_images = changed = 0
    states: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    for row in rows:
        clues = set(map(str, row.get("clue_round_ids") or []))
        before = row["rankings"]["evi_raw_mean_rank"][
            :row["evaluation_k"]
        ]
        after = row["rankings"]["grouped_vlm"][
            :row["evaluation_k"]
        ]
        changed += int(before != after)
        for item in row.get("group_results") or []:
            calls += 1
            images += int(item.get("image_count", 0))
            omitted_images += int(item.get("omitted_image_count", 0))
            verdict = dict(item.get("verdict") or {})
            parsed += int(bool(verdict.get("parse_valid")))
            for rid, member in dict(
                verdict.get("members") or {}
            ).items():
                members += 1
                states[str(member.get("state", "unknown"))] += 1
                confidence[str(
                    member.get("confidence", "unknown")
                )] += 1
        pairs = list(
            row.get("rerank_trace", {}).get("replacements") or []
        )
        replacements += len(pairs)
        dropped_clues += sum(
            str(pair.get("dropped_round_id", "")) in clues
            for pair in pairs
        )
    return {
        "questions": len(rows),
        "vlm_calls": calls,
        "calls_per_question": calls / len(rows) if rows else 0.0,
        "parse_valid_batches": parsed,
        "parse_valid_batch_rate": parsed / calls if calls else 0.0,
        "member_decisions": members,
        "state_counts": dict(sorted(states.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "replacement_pairs": replacements,
        "replaced_annotated_clues": dropped_clues,
        "questions_with_changed_top_k": changed,
        "images_sent": images,
        "images_omitted_by_batch_cap": omitted_images,
    }


def _write_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    verifier: GroupedEvidenceVerifier,
    args: argparse.Namespace,
) -> None:
    strategies = (*BASE_STRATEGIES, "grouped_vlm")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["benchmark"]].append(row)
    diagnostics = _diagnostics(rows)
    metrics = {
        "input": str(Path(args.input).resolve()),
        "grouping_mode": args.grouping_mode,
        "candidate_k": args.candidate_k,
        "evaluation_k": args.eval_k,
        "max_rounds_per_batch": args.max_rounds_per_batch,
        "max_images_per_round": args.max_images_per_round,
        "max_images_per_batch": args.max_images_per_batch,
        "label_isolation": True,
        "ranking_policy": (
            "mean-rank plus explicit within-group high-confidence "
            "drop-to-keep local replacements"
        ),
        "cache_hits_this_process": verifier.cache_hits,
        "cache_misses_this_process": verifier.cache_misses,
        "diagnostics": diagnostics,
        "by_benchmark": {
            benchmark: {
                strategy: _retrieval_metrics(
                    group, strategy, args.eval_k
                )
                for strategy in strategies
            }
            for benchmark, group in sorted(groups.items())
        },
        "memlens_by_question_subtype": {
            subtype: {
                strategy: _retrieval_metrics(
                    subtype_rows, strategy, args.eval_k
                )
                for strategy in strategies
            }
            for subtype, subtype_rows in sorted(_group_by(
                groups.get("memlens", []), "question_subtype"
            ).items())
        },
    }
    _write_json(
        output_dir / "grouped_verification_metrics.json", metrics
    )
    lines = [
        f"# Grouped VLM verification: {args.grouping_mode}",
        "",
        "| Benchmark | Strategy | Micro R | Macro R | Hit@K | Full@K | W/T/L |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark, values in metrics["by_benchmark"].items():
        for strategy in strategies:
            item = values[strategy]
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
    lines.extend([
        "",
        f"- VLM calls: {diagnostics['vlm_calls']}",
        (
            "- Calls/question: "
            f"{diagnostics['calls_per_question']:.2f}"
        ),
        (
            "- Parse-valid batch rate: "
            f"{diagnostics['parse_valid_batch_rate']:.4f}"
        ),
        f"- Local replacements: {diagnostics['replacement_pairs']}",
        (
            "- Replaced annotated clues: "
            f"{diagnostics['replaced_annotated_clues']}"
        ),
        f"- Images sent: {diagnostics['images_sent']}",
        (
            "- Images omitted by batch cap: "
            f"{diagnostics['images_omitted_by_batch_cap']}"
        ),
        "",
    ])
    (output_dir / "grouped_verification_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _group_by(
    rows: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, List[Mapping[str, Any]]]:
    output: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row.get(field, "unknown") or "unknown")].append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--rescoring-questions", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--grouping-mode",
        choices=("rank_batch", "neighborhood"),
        default="neighborhood",
    )
    parser.add_argument(
        "--verifier-model-config",
        default="config/models/qwen3_vl_8b_openrouter.yaml",
    )
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--max-rounds-per-batch", type=int, default=6)
    parser.add_argument("--max-images-per-round", type=int, default=4)
    parser.add_argument("--max-images-per-batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument(
        "--benchmark",
        choices=("all", "memeye", "memlens"),
        default="all",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--memlens-manifest",
        default="data/memlens/converted_32k_agent195/manifest.json",
    )
    parser.add_argument(
        "--memlens-image-root", default="data/memlens"
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "candidate_k", "eval_k", "max_rounds_per_batch",
        "max_images_per_round", "max_images_per_batch",
        "workers", "max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be positive"
            )
    if args.eval_k > args.candidate_k:
        parser.error("--eval-k cannot exceed --candidate-k")

    input_dir = Path(args.input).resolve()
    rescoring_path = (
        Path(args.rescoring_questions).resolve()
        if args.rescoring_questions
        else input_dir / "provenance_verification"
        / "provenance_verification_questions.jsonl"
    )
    if not rescoring_path.is_file():
        raise FileNotFoundError(rescoring_path)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / f"grouped_verification_{args.grouping_mode}"
    )
    _configure_logging(output_dir)
    model_payload, namespace = _model(
        Path(args.verifier_model_config).resolve(), args
    )
    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir else output_dir / "vlm_cache"
    )
    verifier = GroupedEvidenceVerifier(
        model_payload["vlm"],
        cache_dir,
        namespace,
        request_log_dir=output_dir / "request_manifests",
    )
    resolver = DatasetResolver(
        input_dir,
        Path(args.data_root).resolve(),
        Path(args.memlens_manifest).resolve(),
        Path(args.memlens_image_root).resolve(),
    )
    config = {
        "input": str(input_dir),
        "rescoring_questions": str(rescoring_path),
        "grouping_mode": args.grouping_mode,
        "model_namespace": namespace,
        "prompt_version": PROMPT_VERSION,
        "candidate_k": args.candidate_k,
        "eval_k": args.eval_k,
        "max_rounds_per_batch": args.max_rounds_per_batch,
        "max_images_per_round": args.max_images_per_round,
        "max_images_per_batch": args.max_images_per_batch,
        "cache_dir": str(cache_dir),
        "label_isolation": True,
    }
    config_path = output_dir / "grouped_verification_config.json"
    if config_path.exists() and json.loads(
        config_path.read_text(encoding="utf-8")
    ) != config:
        raise RuntimeError(
            f"Existing output has different config: {config_path}"
        )
    _write_json(config_path, config)

    output_jsonl = output_dir / "grouped_verification_questions.jsonl"
    completed = (
        {} if args.no_resume else _load_completed(output_jsonl)
    )
    rows = list(_selected_rows(
        rescoring_path, args.benchmark, args.max_questions
    ))
    mode = "w" if args.no_resume else "a"
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        with output_jsonl.open(mode, encoding="utf-8") as handle:
            for index, row in enumerate(rows, 1):
                key = _row_key(row)
                if key in completed:
                    continue
                try:
                    result = _process_question(
                        row, resolver, verifier, args, executor
                    )
                except Exception:
                    failed.append(key)
                    log.exception("[%d/%d] failed %s", index, len(rows), key)
                    continue
                handle.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                handle.flush()
                completed[key] = result
                log.info(
                    "[%d/%d] %s groups=%d replacements=%d cache_hits=%d",
                    index, len(rows), key,
                    len(result["group_results"]),
                    len(result["rerank_trace"]["replacements"]),
                    verifier.cache_hits,
                )
    selected_keys = {_row_key(row) for row in rows}
    final_rows = [
        completed[key] for key in sorted(completed)
        if key in selected_keys
    ]
    if len(final_rows) != len(rows):
        raise RuntimeError(
            f"Completed {len(final_rows)}/{len(rows)}; failed={failed}"
        )
    _write_outputs(output_dir, final_rows, verifier, args)
    _write_json(output_dir / "grouped_verification_status.json", {
        "selected_questions": len(rows),
        "completed_questions": len(final_rows),
        "failed_questions": failed,
        "cache_hits_this_process": verifier.cache_hits,
        "cache_misses_this_process": verifier.cache_misses,
    })


if __name__ == "__main__":
    main()
