from benchmark.memlens.suite import (
    aggregate_predictions,
    effective_memlens_method_config,
    format_memlens_question,
    official_judge_rows,
)


def test_question_date_is_part_of_model_query():
    text = format_memlens_question({"question_date": "2024-01-03", "question": "What changed?"})
    assert text == "Question date: 2024-01-03\n\nQuestion:\nWhat changed?"


def test_memlens_answer_context_enables_session_markers_by_default():
    original = {"name": "semantic_rag_multimodal", "top_k": 10}
    effective = effective_memlens_method_config(original)
    assert effective["include_session_markers"] is True
    assert "include_session_markers" not in original


def test_memlens_answer_context_allows_explicit_marker_ablation():
    effective = effective_memlens_method_config({
        "name": "semantic_rag_multimodal",
        "include_session_markers": False,
    })
    assert effective["include_session_markers"] is False


def test_aggregate_and_official_schema():
    row = {
        "question_id": "q1", "question": "Q", "question_type": "KU",
        "question_subtype": "single-session-user", "gt": "gold", "pred": "gold",
        "exact_match": True, "contains_gt": True, "f1": 1.0, "bleu": 1.0,
        "clue_recall": 0.5, "answer_session_recall": 1.0, "latency_ms": 12,
        "usage": {"completion_tokens": 3},
    }
    metrics = aggregate_predictions([row])
    assert metrics["summary"]["count"] == 1
    assert metrics["by_question_type"]["KU"]["clue_recall"] == 0.5
    official = official_judge_rows([row])[0]
    assert official["reference_answer"] == "gold"
    assert official["prediction"] == "gold"
    assert official["output_len"] == 3
