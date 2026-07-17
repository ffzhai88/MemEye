"""Re-embed saved session-card fields and replay retrieval without VLM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from analyze_episode_directory_v2 import (
    _aggregate_round_metrics,
    _aggregate_session_metrics,
    _load_json,
    _load_jsonl,
    _ordered_unique,
    _packet_ranking,
    _replay,
    _rrf,
    _session_question_metrics,
)
from analyze_episode_oracle import _fallback_session_id, _question_stats, _resolve_task_run_dir
from benchmark.common import write_json, write_jsonl, write_text
from benchmark.embeddings import TextEmbedder
from benchmark.evi.indexes import cosine


REPRESENTATIONS = ("card_identity", "card_identity_evidence")


def _between(text: str, start: str, end: str | None = None) -> str:
    lower = text.casefold()
    begin = lower.find(start.casefold())
    if begin < 0:
        return ""
    begin += len(start)
    finish = lower.find(end.casefold(), begin) if end else len(text)
    if finish < 0:
        finish = len(text)
    return text[begin:finish].strip()


def split_card(text: str) -> Dict[str, str]:
    identity = _between(text, "Episode identity:", "Ordered progression:")
    evidence = _between(text, "Distinctive evidence:")
    identity_text = f"Episode identity: {identity}" if identity else ""
    identity_evidence = identity_text
    if evidence:
        identity_evidence = (
            identity_text + "\n\nDistinctive evidence:\n" + evidence
        ).strip()
    return {
        "card_identity": identity_text,
        "card_identity_evidence": identity_evidence,
    }


class EmbeddingCache:
    def __init__(
        self,
        root: Path,
        model_name: str,
        model_kwargs: Dict[str, Any],
    ) -> None:
        namespace = json.dumps(
            {"model": model_name, "kwargs": model_kwargs},
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        self.root = root / digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = TextEmbedder(model_name, **model_kwargs)
        self.last_cache_hits = 0
        self.last_cache_misses = 0

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]
        return self.root / f"{digest}.npy"

    def embed_many(self, texts: Iterable[str]) -> Dict[str, List[float]]:
        unique = list(dict.fromkeys(str(text).strip() for text in texts if str(text).strip()))
        vectors: Dict[str, List[float]] = {}
        missing: List[str] = []
        for text in unique:
            path = self._path(text)
            if path.exists():
                try:
                    vectors[text] = np.load(path).astype(float).tolist()
                    continue
                except Exception:
                    pass
            missing.append(text)
        self.last_cache_hits = len(unique) - len(missing)
        self.last_cache_misses = len(missing)
        if missing:
            encoded = self.embedder.embed_batch(missing)
            if len(encoded) != len(missing):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(missing)}, got {len(encoded)}"
                )
            for text, vector in zip(missing, encoded):
                values = [float(value) for value in vector]
                vectors[text] = values
                np.save(self._path(text), np.asarray(values, dtype=np.float32))
        return vectors


def _rank_sessions(
    query_vector: List[float],
    session_text: Dict[str, str],
    vectors: Dict[str, List[float]],
) -> List[str]:
    return sorted(
        session_text,
        key=lambda session_id: (
            -cosine(query_vector, vectors[session_text[session_id]]),
            session_id,
        ),
    )


def _extract_task_cards(source: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    trace = source.get("retrieval_trace", {}) or {}
    rows = (
        trace.get("episode_directory_v3", {}).get("ranked_sessions", []) or []
    )
    variants = {name: {} for name in REPRESENTATIONS}
    full_cards: Dict[str, str] = {}
    for row in rows:
        session_id = str(row.get("session_id", ""))
        card = str(row.get("card_text", "") or "").strip()
        if not session_id or not card:
            continue
        full_cards[session_id] = card
        parts = split_card(card)
        for name in REPRESENTATIONS:
            text = parts[name]
            if not text:
                raise ValueError(f"Card {session_id} has no usable {name} section")
            variants[name][session_id] = text
    return variants, full_cards


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    session_strategies = sorted(set.intersection(*(set(row["session_metrics"]) for row in rows)))
    round_strategies = sorted(set.intersection(*(set(row["round_metrics"]) for row in rows)))
    return {
        "num_questions": len(rows),
        "session_metrics": {
            name: _aggregate_session_metrics(rows, name) for name in session_strategies
        },
        "round_replay_metrics": {
            name: _aggregate_round_metrics(rows, name) for name in round_strategies
        },
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    pooled = payload["pooled"]
    lines = [
        "# Session Card Representation Ablation",
        "",
        f"Suite: `{payload['suite_dir']}`",
        f"Embedding model: `{payload['embedding_model']}`",
        f"Evaluation K: {payload['k']}",
        "",
        "This diagnostic re-embeds saved query-independent card fields. It does not call a VLM.",
        "Clue annotations are used only for evaluation.",
        "",
        "## Pooled Session Ranking",
        "",
        "| Strategy | Recall@M micro | MAP | nDCG |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in pooled["session_metrics"].items():
        lines.append(
            f"| {name} | {metrics['recall_at_m_micro']:.4f} | "
            f"{metrics['map']:.4f} | {metrics['ndcg']:.4f} |"
        )
    lines.extend([
        "",
        "## Pooled Fixed-Pipeline Replay",
        "",
        "| Strategy | Clue Recall micro | Full coverage |",
        "| --- | ---: | ---: |",
    ])
    for name, metrics in pooled["round_replay_metrics"].items():
        lines.append(
            f"| {name} | {metrics['clue_round_recall_micro']:.4f} | "
            f"{metrics['full_clue_coverage_rate']:.4f} |"
        )
    lines.extend([
        "",
        "## Dataset Replay",
        "",
        "| Dataset | Current | Current + packet | Current + full card | Current + identity | Current + identity/evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    names = [
        "current",
        "fusion_current_packet_max",
        "fusion_current_card_full",
        "fusion_current_card_identity",
        "fusion_current_card_identity_evidence",
    ]
    for dataset in payload["datasets"]:
        metrics = dataset["round_replay_metrics"]
        values = [metrics[name]["clue_round_recall_micro"] for name in names]
        lines.append(
            f"| {dataset['task_name']} | "
            + " | ".join(f"{value:.4f}" for value in values)
            + " |"
        )
    write_text(path, "\n".join(lines) + "\n")


def analyze_suite(
    suite_dir: Path,
    k: int,
    embedding_model: str | None = None,
    embedding_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    suite = _load_json(suite_dir / "suite_metrics.json")
    task_entries = list(suite.get("task_runs", []) or [])
    if not task_entries:
        raise ValueError("suite_metrics.json has no task_runs")

    first_dir = _resolve_task_run_dir(task_entries[0], suite_dir)
    config = _load_json(first_dir / "config.json")
    method = config.get("method", {}) or {}
    model = embedding_model or str(method.get("text_embedding_model") or TextEmbedder.DEFAULT_MODEL)
    kwargs = dict(embedding_kwargs or method.get("text_embedding_kwargs") or {})
    cache = EmbeddingCache(suite_dir / "session_card_ablation_embedding_cache", model, kwargs)

    all_rows: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    for entry in task_entries:
        task_name = str(entry["task_name"])
        task_dir = _resolve_task_run_dir(entry, suite_dir)
        sources = _load_jsonl(task_dir / "retrievals.jsonl")
        if not sources:
            continue
        variants, _ = _extract_task_cards(sources[0])
        questions = [str((source.get("retrieval_trace", {}) or {}).get("question_stem") or source.get("question", "")).strip() for source in sources]
        texts = questions + [text for values in variants.values() for text in values.values()]
        vectors = cache.embed_many(texts)
        print(
            f"[CARD-ABLATION] task={task_name} sessions={len(next(iter(variants.values())))} "
            f"questions={len(sources)} unique_texts={len(vectors)} "
            f"cache_hits={cache.last_cache_hits} cache_misses={cache.last_cache_misses}"
        )

        rows: List[Dict[str, Any]] = []
        for source, question in zip(sources, questions):
            trace = source.get("retrieval_trace", {}) or {}
            episode_trace = trace.get("episode_set_retrieval", {}) or {}
            packet_rows = list(trace.get("episode_directory_v2", {}).get("ranked_sessions", []) or [])
            full_card_rows = list(trace.get("episode_directory_v3", {}).get("ranked_sessions", []) or [])
            current = [str(item.get("session_id", "")) for item in episode_trace.get("episodes", []) or [] if str(item.get("session_id", ""))]
            packet = _packet_ranking(packet_rows, "packet_max")
            full_card = [str(item.get("session_id", "")) for item in full_card_rows if str(item.get("session_id", ""))]
            rankings = {
                "current": current,
                "packet_max": packet,
                "card_full": full_card,
                "fusion_current_packet_max": _rrf([current, packet]),
                "fusion_current_card_full": _rrf([current, full_card]),
            }
            query_vector = vectors[question]
            for name in REPRESENTATIONS:
                ranking = _rank_sessions(query_vector, variants[name], vectors)
                rankings[name] = ranking
                rankings[f"fusion_current_{name}"] = _rrf([current, ranking])
                rankings[f"fusion_current_packet_{name}"] = _rrf([current, packet, ranking])

            clue_round_ids = [str(value) for value in source.get("clue_round_ids", []) or []]
            clue_sessions = _ordered_unique(_fallback_session_id(value) for value in clue_round_ids)
            session_metrics = {name: _session_question_metrics(ranking, clue_sessions) for name, ranking in rankings.items()}
            round_metrics = {
                "current": _question_stats([str(value) for value in source.get("ranked_round_ids", []) or []], clue_round_ids, k)
            }
            for name, ranking in rankings.items():
                if name == "current":
                    continue
                final_ids, _, _ = _replay(ranking, trace, packet_rows)
                round_metrics[name] = _question_stats(final_ids, clue_round_ids, k)
            rows.append({
                "task_name": task_name,
                "idx": source.get("idx"),
                "question": source.get("question", ""),
                "clue_round_ids": clue_round_ids,
                "session_rankings": rankings,
                "session_metrics": session_metrics,
                "round_metrics": round_metrics,
            })
        datasets.append({"task_name": task_name, **_summarize(rows)})
        all_rows.extend(rows)
        print(f"[CARD-ABLATION] completed task={task_name}")

    payload = {
        "suite_dir": str(suite_dir),
        "suite_git_commit": suite.get("git_commit", "unknown"),
        "embedding_model": model,
        "embedding_kwargs": kwargs,
        "k": k,
        "representations": {
            "card_full": "saved full-card ranking",
            "card_identity": "Episode identity section only",
            "card_identity_evidence": "Episode identity plus Distinctive evidence",
        },
        "datasets": datasets,
        "pooled": _summarize(all_rows),
    }
    write_json(suite_dir / "session_card_ablation_metrics.json", payload)
    write_jsonl(suite_dir / "session_card_ablation_questions.jsonl", all_rows)
    _write_report(suite_dir / "session_card_ablation_report.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline session-card field ablation without VLM calls.")
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-kwargs", default="{}", help="JSON object passed to TextEmbedder")
    args = parser.parse_args()
    kwargs = json.loads(args.embedding_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--embedding-kwargs must decode to a JSON object")
    payload = analyze_suite(args.suite.resolve(), args.k, args.embedding_model, kwargs)
    print(f"[INFO] Saved card ablation: {Path(payload['suite_dir']) / 'session_card_ablation_report.md'}")


if __name__ == "__main__":
    main()
