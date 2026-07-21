from analyze_candidate_rescoring import (
    candidate_round_ids,
    dedupe_facets,
    score_candidate_strategies,
)


def test_dedupe_facets_normalizes_full_question_whitespace_and_punctuation():
    assert dedupe_facets([
        "Question date: today\n\nWhat do I prefer now?",
        "Question date: today What do I prefer now?",
        "curved monitor preference",
    ]) == [
        "Question date: today\n\nWhat do I prefer now?",
        "curved monitor preference",
    ]


def test_candidate_round_ids_are_fixed_to_saved_trace_order():
    row = {
        "ranked_round_ids": ["fallback"],
        "retrieval_trace": {"rounds": [
            {"round_id": "r2"}, {"round_id": "r1"}, {"round_id": "r3"},
        ]},
    }
    assert candidate_round_ids(row, 2) == ["r2", "r1"]


def test_fixed_multimodal_rewards_same_round_agreement():
    candidates = ["text-only", "balanced", "image-only"]
    rankings = score_candidate_strategies(
        candidates,
        [{"text-only": 0.9, "balanced": 0.7, "image-only": 0.1}],
        [{"text-only": 0.0, "balanced": 0.7, "image-only": 0.9}],
        {"balanced", "image-only"},
    )
    assert rankings["raw_multimodal_fixed"][0] == "balanced"
    assert rankings["raw_dialogue"][0] == "text-only"
    assert rankings["raw_image"][0] == "image-only"


def test_available_mean_is_separate_from_fixed_missing_image_penalty():
    candidates = ["text-only", "balanced"]
    rankings = score_candidate_strategies(
        candidates,
        [{"text-only": 0.9, "balanced": 0.6}],
        [{"text-only": 0.0, "balanced": 0.6}],
        {"balanced"},
    )
    assert rankings["raw_multimodal_fixed"][0] == "balanced"
    assert rankings["raw_multimodal_available"][0] == "text-only"


def test_facet_strategy_uses_distinct_facets_without_labels():
    candidates = ["one-facet", "both-facets", "second-facet"]
    rankings = score_candidate_strategies(
        candidates,
        [
            {"one-facet": 0.8, "both-facets": 0.9, "second-facet": 0.1},
            {"one-facet": 0.1, "both-facets": 0.9, "second-facet": 0.8},
        ],
        [
            {"one-facet": 0.8, "both-facets": 0.9, "second-facet": 0.1},
            {"one-facet": 0.1, "both-facets": 0.9, "second-facet": 0.8},
        ],
        {"one-facet", "both-facets", "second-facet"},
    )
    assert rankings["facet_raw_multimodal_fixed"][0] == "both-facets"
