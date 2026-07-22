from analyze_provenance_verification import score_provenance_strategies


def test_product_verification_prefers_joint_memory_and_raw_support():
    candidates = ["memory_only", "joint", "raw_only"]
    traces = {
        "memory_only": {"source_facet_scores": {"f0:dialogue": 0.9}},
        "joint": {"source_facet_scores": {"f0:dialogue": 0.6}},
        "raw_only": {"source_facet_scores": {"f0:dialogue": 0.1}},
    }
    scored = score_provenance_strategies(
        candidates,
        ["target facet"],
        traces,
        [{"memory_only": 0.1, "joint": 0.6, "raw_only": 0.9}],
        [{}],
        set(),
    )
    assert scored["rankings"]["provenance_product"][0] == "joint"
    assert scored["rankings"]["provenance_min"][0] == "joint"
    assert scored["rankings"]["provenance_raw_only"][0] == "raw_only"


def test_missing_image_removes_only_visual_path():
    traces = {
        "text_round": {"source_facet_scores": {"f0:dialogue": 0.4, "f0:visual": 0.99}},
        "image_round": {"source_facet_scores": {"f0:dialogue": 0.3, "f0:visual": 0.7}},
    }
    scored = score_provenance_strategies(
        ["text_round", "image_round"],
        ["target facet"],
        traces,
        [{"text_round": 0.8, "image_round": 0.2}],
        [{"text_round": 1.0, "image_round": 0.8}],
        {"image_round"},
    )
    paths = {(row["round_id"], row["source"]) for row in scored["path_rows"]}
    assert ("text_round", "dialogue") in paths
    assert ("text_round", "visual") not in paths
    assert ("image_round", "visual") in paths
