"""Replay parameter-free multifacet fusion from saved EVI retrieval traces."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


FIELDS = ("dialogue_calibrated", "visual_anchor_calibrated")


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("retrievals.jsonl"))
    rows: List[Dict[str, Any]] = []
    for file in files:
        dataset = file.parent.name
        benchmark = (
            "memlens"
            if any("memlens" in part.lower() for part in file.parts)
            else "memeye"
        )
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append({
                        "_benchmark": benchmark,
                        "_dataset": dataset,
                        **json.loads(line),
                    })
    return rows


def _ranks(values: Mapping[str, float]) -> Dict[str, int]:
    ordered = sorted(values, key=lambda rid: (-float(values[rid]), str(rid)))
    return {rid: rank for rank, rid in enumerate(ordered, start=1)}


def _ranks_from_midrank_percentiles(
    values: Mapping[str, float], total_count: int,
) -> Dict[str, float]:
    """Recover descending midranks while retaining omitted image-only gaps."""
    total = max(int(total_count), len(values), 1)
    return {
        rid: max(1.0, total * (1.0 - float(percentile)) + 0.5)
        for rid, percentile in values.items()
    }


def replay_visual_corroborated_best_source(
    trace: Dict[str, Any], source_top_k: int = 30,
) -> List[str]:
    facets = list((trace.get("facet_multimodal") or {}).get("facets") or [])
    consensus: Dict[str, float] = defaultdict(float)
    max_strength: Dict[str, float] = defaultdict(float)
    for facet in facets:
        rounds = list(facet.get("rounds") or [])
        scores = {str(row["round_id"]): dict(row.get("multimodal_scores") or {}) for row in rounds}
        dialogue = {
            rid: float(item["dialogue_calibrated"])
            for rid, item in scores.items() if item.get("dialogue_calibrated") is not None
        }
        visual = {
            rid: float(item["visual_anchor_calibrated"])
            for rid, item in scores.items() if item.get("visual_anchor_calibrated") is not None
        }
        raw_image = {
            rid: float(item["raw_image_calibrated"])
            for rid, item in scores.items() if item.get("raw_image_calibrated") is not None
        }
        dialogue_ranks = _ranks(dialogue)
        visual_ranks = _ranks(visual)
        raw_image_ranks = _ranks_from_midrank_percentiles(
            raw_image, int(facet.get("num_image_rounds", len(raw_image)))
        )
        visual_rrf = {
            rid: 1.0 / visual_ranks[rid]
            + (1.0 / raw_image_ranks[rid] if rid in raw_image_ranks else 0.0)
            for rid in visual_ranks
        }
        corroborated_visual_ranks = _ranks(visual_rrf)
        dialogue_candidates = {
            rid for rid, rank in dialogue_ranks.items() if rank <= source_top_k
        }
        visual_candidates = {
            rid for rid, rank in corroborated_visual_ranks.items() if rank <= source_top_k
        }
        for rid in dialogue_candidates | visual_candidates:
            source_ranks = []
            if rid in dialogue_candidates:
                source_ranks.append(dialogue_ranks[rid])
            if rid in visual_candidates:
                source_ranks.append(corroborated_visual_ranks[rid])
            consensus[rid] += 1.0 / min(source_ranks)
            max_strength[rid] = max(
                max_strength[rid], dialogue.get(rid, 0.0), visual.get(rid, 0.0)
            )
    return sorted(
        consensus,
        key=lambda rid: (-(consensus[rid] * max_strength[rid]), str(rid)),
    )


def _summary(rows: Iterable[Dict[str, Any]], ks: List[int], rank_key: str) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("clue_round_ids")]
    by_k: Dict[str, Any] = {}
    for k in ks:
        total_clues = total_hits = 0
        recalls: List[float] = []
        hit_count = full_count = 0
        reciprocal_ranks: List[float] = []
        for row in eligible:
            clues = set(map(str, row["clue_round_ids"]))
            ranking = list(map(str, row.get(rank_key, [])))
            hits = len(clues & set(ranking[:k]))
            total_clues += len(clues)
            total_hits += hits
            recalls.append(hits / len(clues))
            hit_count += int(hits > 0)
            full_count += int(hits == len(clues))
            first = next((rank for rank, rid in enumerate(ranking[:k], 1) if rid in clues), None)
            reciprocal_ranks.append(1.0 / first if first else 0.0)
        n = len(eligible)
        by_k[str(k)] = {
            "num_questions": n,
            "clue_round_recall_micro": total_hits / total_clues if total_clues else 0.0,
            "clue_round_recall_macro": sum(recalls) / n if n else 0.0,
            "hit_rate": hit_count / n if n else 0.0,
            "full_clue_coverage_rate": full_count / n if n else 0.0,
            "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        }
    return {"num_questions_with_clues": len(eligible), "by_k": by_k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Joint run, suite, task, or retrievals.jsonl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-top-k", type=int, default=30)
    parser.add_argument("--ks", default="1,3,5,10,20")
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            input_path / "fusion_replay"
            if input_path.is_dir()
            else input_path.parent / f"{input_path.stem}_fusion_replay"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted({int(value) for value in args.ks.split(",") if value.strip()})
    rows = _load_rows(input_path)
    usable: List[Dict[str, Any]] = []
    for row in rows:
        replayed = replay_visual_corroborated_best_source(
            dict(row.get("retrieval_trace") or {}), args.source_top_k
        )
        if replayed:
            usable.append({**row, "replayed_ranked_round_ids": replayed})

    datasets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    benchmarks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in usable:
        datasets[str(row["_dataset"])].append(row)
        benchmarks[str(row["_benchmark"])].append(row)
    report = {
        "input": str(input_path),
        "policy": "visual_corroborated_best_source",
        "source_top_k": args.source_top_k,
        "k_values": ks,
        "num_loaded_rows": len(rows),
        "num_replayed_rows": len(usable),
        "aggregation_policy": "side_by_side_by_benchmark",
        "by_benchmark": {
            name: {
                "saved_ranking": _summary(group, ks, "ranked_round_ids"),
                "replayed_ranking": _summary(group, ks, "replayed_ranked_round_ids"),
            }
            for name, group in sorted(benchmarks.items())
        },
        "by_dataset": {
            name: {
                "saved_ranking": _summary(group, ks, "ranked_round_ids"),
                "replayed_ranking": _summary(group, ks, "replayed_ranked_round_ids"),
            }
            for name, group in sorted(datasets.items())
        },
    }
    (output_dir / "fusion_replay_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "fusion_replay_questions.jsonl").open("w", encoding="utf-8") as handle:
        for row in usable:
            handle.write(json.dumps({
                "dataset": row["_dataset"],
                "benchmark": row["_benchmark"],
                "question_id": row.get("question_id", row.get("qa_index")),
                "question": row.get("question", ""),
                "clue_round_ids": row.get("clue_round_ids", []),
                "saved_ranked_round_ids": row.get("ranked_round_ids", []),
                "replayed_ranked_round_ids": row["replayed_ranked_round_ids"],
            }, ensure_ascii=False) + "\n")
    report_lines = [
        "# Multifacet fusion replay",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Source Top-K: `{args.source_top_k}`",
        f"- Replayed rows: `{len(usable)}/{len(rows)}`",
        "- Aggregation: benchmarks are reported side by side and never pooled.",
        "",
        "| Benchmark | Saved Macro R@10 | Replayed Macro R@10 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name, payload in sorted(report["by_benchmark"].items()):
        if "10" not in payload["saved_ranking"]["by_k"]:
            continue
        saved = payload["saved_ranking"]["by_k"]["10"]["clue_round_recall_macro"]
        replayed = payload["replayed_ranking"]["by_k"]["10"]["clue_round_recall_macro"]
        report_lines.append(
            f"| {name} | {saved:.4f} | {replayed:.4f} | {replayed - saved:+.4f} |"
        )
    (output_dir / "fusion_replay_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(f"[FUSION-REPLAY] rows={len(usable)}/{len(rows)} output={output_dir}")


if __name__ == "__main__":
    main()
