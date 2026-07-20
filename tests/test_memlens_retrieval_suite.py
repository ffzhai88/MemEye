from benchmark.memlens.retrieval_suite import summarize_answer_sessions


def test_answer_session_metrics_exclude_unannotated_rows():
    rows = [
        {"answer_session_ids": ["s1", "s2"], "ranked_session_ids": ["s2", "s3", "s1"]},
        {"answer_session_ids": [], "ranked_session_ids": ["s9"]},
    ]
    metrics = summarize_answer_sessions(rows, [1, 3])
    assert metrics["num_questions_with_answer_sessions"] == 1
    assert metrics["by_k"]["1"]["answer_session_recall_macro"] == 0.5
    assert metrics["by_k"]["3"]["answer_session_full_coverage_rate"] == 1.0
