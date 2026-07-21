"""Diagnose raw-evidence scoring on a fixed saved EVI candidate pool.

This is an offline diagnostic, not a retrieval method.  It never uses clue,
question-type, or dataset labels while scoring.  Saved EVI traces define the
candidate pool; raw dialogue and raw images only reorder those candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from benchmark.common import REPO_ROOT
from benchmark.dataset import MemoryBenchmarkDataset, build_round_retrieval_text
from benchmark.embeddings import TextEmbedder
from benchmark.image_retrieval import RawImageRoundIndex


log = logging.getLogger("candidate_rescoring")

STRATEGIES = (
    "saved_evi",
    "raw_dialogue",
    "raw_image",
    "raw_multimodal_fixed",
    "raw_multimodal_available",
    "raw_rank_fusion",
    "facet_raw_multimodal_fixed",
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(v) * float(v) for v in left))
    right_norm = math.sqrt(sum(float(v) * float(v) for v in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _normalize_facet(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)


def dedupe_facets(values: Iterable[str]) -> List[str]:
    """Deduplicate full-question/facet variants after whitespace normalization."""
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = _normalize_facet(text)
        if text and key and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def candidate_round_ids(row: Mapping[str, Any], candidate_k: int) -> List[str]:
    trace_rounds = list((row.get("retrieval_trace") or {}).get("rounds") or [])
    values = [str(item.get("round_id", "")) for item in trace_rounds]
    if not any(values):
        values = [str(value) for value in row.get("ranked_round_ids", [])]
    return [value for value in values if value][: max(0, int(candidate_k))]


def row_facets(row: Mapping[str, Any]) -> List[str]:
    trace = dict(row.get("retrieval_trace") or {})
    values = [str(item.get("text", "")) for item in trace.get("facets", [])]
    full_question = str(
        row.get("formatted_question") or trace.get("question_stem") or row.get("question") or ""
    )
    return dedupe_facets([full_question, *values]) or [full_question]


def _rank(scores: Mapping[str, float]) -> List[str]:
    return sorted(scores, key=lambda rid: (-float(scores[rid]), str(rid)))


def _reciprocal_ranks(scores: Mapping[str, float], eligible: Optional[set[str]] = None) -> Dict[str, float]:
    ranking = [rid for rid in _rank(scores) if eligible is None or rid in eligible]
    return {rid: 1.0 / rank for rank, rid in enumerate(ranking, 1)}


def score_candidate_strategies(
    candidate_ids: Sequence[str],
    facet_text_scores: Sequence[Mapping[str, float]],
    facet_image_scores: Sequence[Mapping[str, float]],
    image_round_ids: set[str],
) -> Dict[str, List[str]]:
    """Return label-free rankings over exactly ``candidate_ids``.

    The first facet is the normalized full question.  Fixed multimodal scoring
    reproduces Semantic RAG's missing-image behavior (image score zero), while
    available scoring treats an absent image as unavailable rather than negative.
    """
    candidates = list(dict.fromkeys(map(str, candidate_ids)))
    if not candidates:
        return {name: [] for name in STRATEGIES if name != "saved_evi"}
    if not facet_text_scores or not facet_image_scores:
        raise ValueError("At least one text and image facet score mapping is required")

    text = facet_text_scores[0]
    image = facet_image_scores[0]
    text_scores = {rid: float(text.get(rid, 0.0)) for rid in candidates}
    image_scores = {rid: float(image.get(rid, 0.0)) for rid in candidates}
    fixed_scores = {
        rid: 0.5 * text_scores[rid] + 0.5 * image_scores[rid] for rid in candidates
    }
    available_scores = {
        rid: (
            0.5 * text_scores[rid] + 0.5 * image_scores[rid]
            if rid in image_round_ids
            else text_scores[rid]
        )
        for rid in candidates
    }
    text_rr = _reciprocal_ranks(text_scores)
    image_rr = _reciprocal_ranks(image_scores, eligible=image_round_ids)
    rank_fusion_scores = {
        rid: text_rr.get(rid, 0.0) + image_rr.get(rid, 0.0) for rid in candidates
    }

    facet_consensus: Dict[str, float] = defaultdict(float)
    facet_max: Dict[str, float] = defaultdict(float)
    for text_by_round, image_by_round in zip(facet_text_scores, facet_image_scores):
        scores = {
            rid: 0.5 * float(text_by_round.get(rid, 0.0))
            + 0.5 * float(image_by_round.get(rid, 0.0))
            for rid in candidates
        }
        ranks = {rid: rank for rank, rid in enumerate(_rank(scores), 1)}
        for rid in candidates:
            facet_consensus[rid] += 1.0 / ranks[rid]
            facet_max[rid] = max(facet_max[rid], scores[rid])
    facet_scores = {
        rid: facet_max[rid] * facet_consensus[rid] for rid in candidates
    }

    return {
        "raw_dialogue": _rank(text_scores),
        "raw_image": _rank(image_scores),
        "raw_multimodal_fixed": _rank(fixed_scores),
        "raw_multimodal_available": _rank(available_scores),
        "raw_rank_fusion": _rank(rank_fusion_scores),
        "facet_raw_multimodal_fixed": _rank(facet_scores),
    }


class PersistentTextVectorCache:
    """Small exact-input cache used only by this offline analyzer."""

    VERSION = "candidate_rescoring_text_v1"

    def __init__(self, root: Path, model_name: str, model_kwargs: Mapping[str, Any]) -> None:
        self.root = root
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs)
        self.embedder = TextEmbedder(model_name, **self.model_kwargs)
        self.hits = 0
        self.misses = 0

    def _path(self, text: str, role: str) -> Path:
        payload = json.dumps(
            {"version": self.VERSION, "model": self.model_name, "kwargs": self.model_kwargs,
             "role": role, "text": text},
            sort_keys=True, ensure_ascii=False,
        )
        return self.root / f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}.json"

    @staticmethod
    def _load(path: Path) -> Optional[List[float]]:
        try:
            values = json.loads(path.read_text(encoding="utf-8")).get("vector")
            return [float(value) for value in values] if isinstance(values, list) and values else None
        except Exception:
            return None

    @staticmethod
    def _write(path: Path, vector: Sequence[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"vector": list(vector)}), encoding="utf-8")
        temporary.replace(path)

    def embed(self, texts: Sequence[str], role: str) -> List[List[float]]:
        output: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        for index, text in enumerate(texts):
            path = self._path(text, role)
            vector = self._load(path) if path.exists() else None
            if vector:
                self.hits += 1
                output[index] = vector
            else:
                missing_indices.append(index)
        if missing_indices:
            missing_texts = [texts[index] for index in missing_indices]
            vectors = (
                [self.embedder.embed_query(text) for text in missing_texts]
                if role == "query"
                else self.embedder.embed_batch(missing_texts)
            )
            for index, vector in zip(missing_indices, vectors):
                self.misses += 1
                output[index] = vector
                self._write(self._path(texts[index], role), vector)
        return [list(vector or []) for vector in output]


def _existing_or_relocated(path_text: str, fallback: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.exists() else fallback.resolve()


def _memeye_dataset(task_dir: Path, data_root: Path) -> Tuple[MemoryBenchmarkDataset, Dict[str, Any]]:
    config = _read_json(task_dir / "config.json")
    dataset_cfg = dict(config.get("dataset") or config.get("task", {}).get("dataset") or {})
    saved_dialog = str(dataset_cfg.get("dialog_json", ""))
    dialog_path = _existing_or_relocated(
        saved_dialog, data_root / "dialog" / Path(saved_dialog).name
    )
    saved_image_root = str(dataset_cfg.get("image_root", ""))
    image_root = _existing_or_relocated(saved_image_root, data_root / "image")
    if not dialog_path.exists():
        raise FileNotFoundError(f"Could not resolve MemEye dialogue JSON: {dialog_path}")
    return MemoryBenchmarkDataset(dialog_path, image_root), config


def _method_config_from_saved(config: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(config.get("method") or config.get("retrieval_eval", {}).get("retrieval_method_config") or {})


def _diagnostic_method_config(
    config: Mapping[str, Any], cache_root: Path,
) -> Dict[str, Any]:
    output = _method_config_from_saved(config)
    output["use_image_embedding_cache"] = True
    output["image_embedding_cache_dir"] = str(cache_root / "image")
    return output


def _score_dataset_rows(
    rows: Sequence[Dict[str, Any]],
    dataset: MemoryBenchmarkDataset,
    dataset_name: str,
    benchmark: str,
    method_cfg: Mapping[str, Any],
    text_cache: PersistentTextVectorCache,
    candidate_k: int,
) -> List[Dict[str, Any]]:
    image_index = RawImageRoundIndex(dataset, dict(method_cfg))
    output: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        candidates = candidate_round_ids(row, candidate_k)
        facets = row_facets(row)
        round_texts = [
            build_round_retrieval_text(dataset.rounds.get(rid, {}), modality="multimodal")
            for rid in candidates
        ]
        round_vectors = text_cache.embed(round_texts, role="document")
        query_vectors = text_cache.embed(facets, role="query")
        facet_text_scores: List[Dict[str, float]] = []
        facet_image_scores: List[Dict[str, float]] = []
        image_round_ids = {
            rid for rid in candidates if list(dataset.rounds.get(rid, {}).get("images", []) or [])
        }
        for facet, query_vector in zip(facets, query_vectors):
            facet_text_scores.append({
                rid: _cosine(query_vector, vector)
                for rid, vector in zip(candidates, round_vectors)
            })
            image_hits, _ = image_index.search(facet, len(dataset.rounds))
            image_by_round = {str(item["round_id"]): float(item["score"]) for item in image_hits}
            facet_image_scores.append({rid: image_by_round.get(rid, 0.0) for rid in candidates})
        rankings = score_candidate_strategies(
            candidates, facet_text_scores, facet_image_scores, image_round_ids
        )
        rankings = {"saved_evi": candidates, **rankings}
        clues = list(map(str, row.get("clue_round_ids", [])))
        clue_image_count = sum(rid in image_round_ids for rid in clues)
        clue_count = len(clues)
        if clue_count == 0:
            clue_count_bucket = "0"
            clue_image_coverage = "no_clues"
        else:
            clue_count_bucket = (
                "1" if clue_count == 1 else
                "2-3" if clue_count <= 3 else
                "4-5" if clue_count <= 5 else "6+"
            )
            clue_image_coverage = (
                "none" if clue_image_count == 0 else
                "all" if clue_image_count == clue_count else "partial"
            )
        clue_ranks = {
            strategy: {
                clue: (ranking.index(clue) + 1 if clue in ranking else None)
                for clue in clues
            }
            for strategy, ranking in rankings.items()
        }
        output.append({
            "benchmark": benchmark,
            "dataset": dataset_name,
            "question_id": row.get("question_id", row.get("qa_index", "")),
            "question": row.get("question", ""),
            "question_type": row.get("question_type", ""),
            "question_subtype": row.get("question_subtype", ""),
            "clue_round_ids": clues,
            "clue_count": clue_count,
            "clue_image_count": clue_image_count,
            "clue_count_bucket": clue_count_bucket,
            "clue_image_coverage": clue_image_coverage,
            "candidate_round_ids": candidates,
            "candidate_count": len(candidates),
            "facets": facets,
            "rankings": rankings,
            "clue_ranks": clue_ranks,
            "candidate_text_scores": facet_text_scores[0] if facet_text_scores else {},
            "candidate_image_scores": facet_image_scores[0] if facet_image_scores else {},
        })
        if index % 25 == 0 or index == len(rows):
            log.info("[%s/%s] rescored %d/%d", benchmark, dataset_name, index, len(rows))
    return output


def _metrics(rows: Sequence[Mapping[str, Any]], strategy: str, k: int) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("clue_round_ids")]
    total_clues = total_hits = wins = losses = 0
    recalls: List[float] = []
    for row in eligible:
        clues = set(map(str, row["clue_round_ids"]))
        ranking = list(map(str, row["rankings"][strategy]))[:k]
        hits = len(clues & set(ranking))
        total_clues += len(clues)
        total_hits += hits
        recalls.append(hits / len(clues))
        if strategy != "saved_evi":
            saved_hits = len(clues & set(row["rankings"]["saved_evi"][:k]))
            wins += int(hits > saved_hits)
            losses += int(hits < saved_hits)
    count = len(eligible)
    return {
        "num_questions": count,
        "total_clues": total_clues,
        "clue_round_recall_micro": total_hits / total_clues if total_clues else 0.0,
        "clue_round_recall_macro": sum(recalls) / count if count else 0.0,
        "wins_vs_saved_evi": wins,
        "ties_vs_saved_evi": count - wins - losses if strategy != "saved_evi" else count,
        "losses_vs_saved_evi": losses,
    }


def _group_metrics(
    rows: Sequence[Dict[str, Any]], strategy: str, k: int, field: str
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown") or "unknown")].append(row)
    return {name: _metrics(group, strategy, k) for name, group in sorted(groups.items())}


def _write_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    text_cache: PersistentTextVectorCache,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "candidate_rescoring_questions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    benchmarks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    datasets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        benchmarks[row["benchmark"]].append(row)
        datasets[f"{row['benchmark']}/{row['dataset']}"].append(row)
    metrics = {
        "input": str(Path(args.input).resolve()),
        "candidate_policy": "saved_evi_trace_top_k_fixed_before_all_rescoring",
        "candidate_k": args.candidate_k,
        "evaluation_k": args.eval_k,
        "scoring_label_policy": "clues and question metadata are used only after ranking",
        "strategies": list(STRATEGIES),
        "text_embedding_model": text_cache.model_name,
        "text_embedding_cache": {
            "path": str(text_cache.root), "hits": text_cache.hits, "misses": text_cache.misses,
        },
        "candidate_pool_recall": {
            benchmark: _metrics(group, "saved_evi", args.candidate_k)
            for benchmark, group in sorted(benchmarks.items())
        },
        "by_benchmark": {
            benchmark: {
                strategy: _metrics(group, strategy, args.eval_k) for strategy in STRATEGIES
            }
            for benchmark, group in sorted(benchmarks.items())
        },
        "by_dataset": {
            dataset: {
                strategy: _metrics(group, strategy, args.eval_k) for strategy in STRATEGIES
            }
            for dataset, group in sorted(datasets.items())
        },
        "memlens_by_question_subtype": {
            strategy: _group_metrics(benchmarks.get("memlens", []), strategy, args.eval_k, "question_subtype")
            for strategy in STRATEGIES
        },
        "by_clue_count": {
            benchmark: {
                strategy: _group_metrics(group, strategy, args.eval_k, "clue_count_bucket")
                for strategy in STRATEGIES
            }
            for benchmark, group in sorted(benchmarks.items())
        },
        "by_clue_image_coverage": {
            benchmark: {
                strategy: _group_metrics(group, strategy, args.eval_k, "clue_image_coverage")
                for strategy in STRATEGIES
            }
            for benchmark, group in sorted(benchmarks.items())
        },
    }
    (output_dir / "candidate_rescoring_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Fixed-candidate raw-evidence rescoring",
        "",
        f"- Candidate pool: saved EVI Top-{args.candidate_k}",
        f"- Evaluation: Recall@{args.eval_k}",
        "- Clues, question types, and dataset labels never enter a scoring function.",
        "",
        "| Benchmark | Strategy | Micro R | Macro R | Delta Macro | W/T/L |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for benchmark, payload in sorted(metrics["by_benchmark"].items()):
        baseline = payload["saved_evi"]["clue_round_recall_macro"]
        for strategy in STRATEGIES:
            item = payload[strategy]
            lines.append(
                f"| {benchmark} | {strategy} | {item['clue_round_recall_micro']:.4f} | "
                f"{item['clue_round_recall_macro']:.4f} | "
                f"{item['clue_round_recall_macro'] - baseline:+.4f} | "
                f"{item['wins_vs_saved_evi']}/{item['ties_vs_saved_evi']}/{item['losses_vs_saved_evi']} |"
            )
    lines.extend([
        "", "## MEMLENS knowledge_update", "",
        "| Strategy | Micro R | Macro R | W/T/L |",
        "|---|---:|---:|---:|",
    ])
    for strategy in STRATEGIES:
        item = metrics["memlens_by_question_subtype"].get(strategy, {}).get("knowledge_update")
        if item:
            lines.append(
                f"| {strategy} | {item['clue_round_recall_micro']:.4f} | "
                f"{item['clue_round_recall_macro']:.4f} | "
                f"{item['wins_vs_saved_evi']}/{item['ties_vs_saved_evi']}/{item['losses_vs_saved_evi']} |"
            )
    (output_dir / "candidate_rescoring_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "candidate_rescoring.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    log.addHandler(stream)
    log.addHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Completed joint EVI retrieval run")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--memlens-manifest", default="data/memlens/converted_32k_agent195/manifest.json")
    parser.add_argument("--memlens-image-root", default="data/memlens")
    parser.add_argument(
        "--cache-dir", default="",
        help="Persistent text/image embedding cache (default: <output-dir>/embedding_cache)",
    )
    parser.add_argument("--max-questions", type=int, default=0, help="Per dataset smoke-test limit; 0 means all")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "candidate_rescoring"
    _configure_logging(output_dir)
    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / "embedding_cache"

    task_dirs = sorted((input_dir / "memeye").glob("*/"))
    if not task_dirs:
        raise FileNotFoundError(f"No MemEye task directories found under {input_dir / 'memeye'}")
    first_config = _read_json(task_dirs[0] / "config.json")
    method_cfg = _diagnostic_method_config(first_config, cache_root)
    text_model = str(method_cfg.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL))
    text_kwargs = dict(method_cfg.get("text_embedding_kwargs") or {})
    text_cache = PersistentTextVectorCache(
        cache_root / "text", text_model, text_kwargs
    )

    output_rows: List[Dict[str, Any]] = []
    for task_dir in task_dirs:
        rows = _read_jsonl(task_dir / "retrievals.jsonl")
        if args.max_questions > 0:
            rows = rows[: args.max_questions]
        dataset, config = _memeye_dataset(task_dir, data_root)
        output_rows.extend(_score_dataset_rows(
            rows, dataset, task_dir.name, "memeye", _diagnostic_method_config(config, cache_root),
            text_cache, args.candidate_k,
        ))

    manifest_path = Path(args.memlens_manifest).resolve()
    manifest = _read_json(manifest_path)
    item_by_id = {str(item.get("question_id", "")): item for item in manifest.get("items", [])}
    memlens_rows = _read_jsonl(input_dir / "memlens" / "retrievals.jsonl")
    if args.max_questions > 0:
        memlens_rows = memlens_rows[: args.max_questions]
    memlens_image_root = Path(args.memlens_image_root).resolve()
    for index, row in enumerate(memlens_rows, 1):
        question_id = str(row.get("question_id", ""))
        item = item_by_id.get(question_id)
        if not item:
            raise KeyError(f"MEMLENS manifest has no item for {question_id}")
        dataset = MemoryBenchmarkDataset(manifest_path.parent / str(item["path"]), memlens_image_root)
        output_rows.extend(_score_dataset_rows(
            [row], dataset, "memlens", "memlens", method_cfg, text_cache, args.candidate_k,
        ))
        if index % 25 == 0 or index == len(memlens_rows):
            log.info("[memlens] loaded and rescored %d/%d items", index, len(memlens_rows))

    _write_outputs(output_dir, output_rows, args, text_cache)
    log.info("Complete: rows=%d output=%s", len(output_rows), output_dir)


if __name__ == "__main__":
    main()
