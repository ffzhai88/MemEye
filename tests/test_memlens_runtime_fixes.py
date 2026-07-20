from benchmark.memlens.runtime_fixes import corrected_answer_session_summary


def test_round_k_coverage_and_unique_session_routing_are_separate():
    rows = [{
        "answer_session_ids": ["s1", "s2"],
        "ranked_round_ids": ["s1:R1", "x:R1", "x:R2", "s2:R1"],
    }]
    metrics = corrected_answer_session_summary(rows, [3])["by_k"]["3"]
    assert metrics["answer_session_recall_macro"] == 0.5
    assert metrics["deduplicated_session_recall_macro"] == 1.0
