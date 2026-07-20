"""Compatibility fixes for MEMLENS queries and retrieval session metrics.

Kept isolated so experiment entry points can apply the fixes before constructing
methods. This module can be removed once the same changes are folded into the
core EVI and MEMLENS retrieval modules.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List


def _dated_question(qa: Dict[str, Any]) -> str:
    question = str(qa.get("question", "")).strip()
    date = str(qa.get("question_date", "")).strip()
    if date and not question.startswith("Question date:"):
        return f"Question date: {date}\n\nQuestion:\n{question}"
    return question


def install_evi_question_date_fix() -> None:
    """Ensure agentic EVI uses question_date in its internal retrieval query."""
    from benchmark.methods import EVIMethod

    if getattr(EVIMethod.answer, "_memlens_date_fixed", False):
        return
    original = EVIMethod.answer

    def answer(self: Any, dataset: Any, qa: Dict[str, Any], question: str, question_images: Any = None) -> str:
        effective_qa = copy.deepcopy(qa)
        effective_qa["question"] = _dated_question(qa)
        return original(self, dataset, effective_qa, question, question_images=question_images)

    answer._memlens_date_fixed = True  # type: ignore[attr-defined]
    EVIMethod.answer = answer  # type: ignore[assignment]


def corrected_answer_session_summary(rows: List[Dict[str, Any]], k_values: Iterable[int]) -> Dict[str, Any]:
    """Report context coverage at round K and retain unique-session routing at K."""
    from benchmark.retrieval_eval import _parse_k_values

    eligible = [row for row in rows if row.get("answer_session_ids")]
    by_k: Dict[str, Any] = {}
    for k in _parse_k_values(k_values):
        context_recalls: List[float] = []
        context_hits: List[float] = []
        context_full: List[float] = []
        context_rr: List[float] = []
        route_recalls: List[float] = []
        route_hits: List[float] = []
        route_full: List[float] = []
        route_rr: List[float] = []
        total = context_total_hits = route_total_hits = 0
        for row in eligible:
            expected = list(dict.fromkeys(str(v) for v in row["answer_session_ids"] if str(v)))
            expected_set = set(expected)
            round_ids = list(row.get("ranked_round_ids", []))
            round_sessions = [str(value).rsplit(":R", 1)[0] for value in round_ids]
            context_sessions = list(dict.fromkeys(round_sessions[:k]))
            route_sessions = list(dict.fromkeys(round_sessions))[:k]

            def values(ranked: List[str]) -> tuple[int, float]:
                count = sum(value in set(ranked) for value in expected)
                first = next((idx for idx, value in enumerate(ranked, 1) if value in expected_set), None)
                return count, 1.0 / first if first else 0.0

            context_count, context_reciprocal = values(context_sessions)
            route_count, route_reciprocal = values(route_sessions)
            total += len(expected)
            context_total_hits += context_count
            route_total_hits += route_count
            context_recalls.append(context_count / len(expected))
            context_hits.append(float(context_count > 0))
            context_full.append(float(context_count == len(expected)))
            context_rr.append(context_reciprocal)
            route_recalls.append(route_count / len(expected))
            route_hits.append(float(route_count > 0))
            route_full.append(float(route_count == len(expected)))
            route_rr.append(route_reciprocal)
        count = len(eligible)
        by_k[str(k)] = {
            "num_questions": count,
            "answer_session_recall_micro": context_total_hits / total if total else 0.0,
            "answer_session_recall_macro": sum(context_recalls) / count if count else 0.0,
            "answer_session_hit_rate": sum(context_hits) / count if count else 0.0,
            "answer_session_full_coverage_rate": sum(context_full) / count if count else 0.0,
            "answer_session_mrr": sum(context_rr) / count if count else 0.0,
            "total_answer_sessions": total,
            "total_answer_session_hits": context_total_hits,
            "metric_scope": "answer sessions covered by the top-k ranked rounds",
            "deduplicated_session_recall_micro": route_total_hits / total if total else 0.0,
            "deduplicated_session_recall_macro": sum(route_recalls) / count if count else 0.0,
            "deduplicated_session_hit_rate": sum(route_hits) / count if count else 0.0,
            "deduplicated_session_full_coverage_rate": sum(route_full) / count if count else 0.0,
            "deduplicated_session_mrr": sum(route_rr) / count if count else 0.0,
            "total_deduplicated_session_hits": route_total_hits,
        }
    return {"num_questions_with_answer_sessions": len(eligible), "by_k": by_k}


def install_retrieval_session_metric_fix() -> None:
    from benchmark.memlens import retrieval_suite

    retrieval_suite.summarize_answer_sessions = corrected_answer_session_summary
