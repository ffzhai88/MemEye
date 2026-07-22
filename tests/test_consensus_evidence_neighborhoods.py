from analyze_consensus_evidence_neighborhoods import (
    build_neighborhood_plan,
    build_relation_graph,
    classify_round,
    partition_candidates,
)


def test_three_way_partition_is_rank_defined():
    anchor = ["both", "anchor", "a3"]
    raw = ["both", "raw", "r3"]
    pool = ["both", "anchor", "raw", "low"]

    assert classify_round("both", set(anchor[:2]), set(raw[:2])) == "consensus_high"
    assert classify_round("anchor", set(anchor[:2]), set(raw[:2])) == "disputed"
    assert classify_round("raw", set(anchor[:2]), set(raw[:2])) == "disputed"
    assert classify_round("low", set(anchor[:2]), set(raw[:2])) == "consensus_low"
    assert partition_candidates(anchor, raw, pool, 2) == {
        "consensus_high": ["both"],
        "disputed": ["anchor", "raw"],
        "consensus_low": ["low"],
    }


def test_relation_graph_preserves_typed_structural_edges():
    features = {
        "S1:R1": {
            "session_id": "S1",
            "round_number": 1,
            "dominant_specific_facet": "1",
            "anchor_tokens": {"green", "chair"},
            "facet_profile": [0.8, 0.9],
        },
        "S1:R2": {
            "session_id": "S1",
            "round_number": 2,
            "dominant_specific_facet": "1",
            "anchor_tokens": {"green", "seat"},
            "facet_profile": [0.7, 0.8],
        },
        "S2:R1": {
            "session_id": "S2",
            "round_number": 1,
            "dominant_specific_facet": "2",
            "anchor_tokens": {"unrelated"},
            "facet_profile": [0.1, 0.0],
        },
    }
    edges = build_relation_graph(features)
    relation_types = edges[("S1:R1", "S1:R2")]

    assert "same_session" in relation_types
    assert "round_adjacent" in relation_types
    assert "same_dominant_specific_facet" in relation_types
    assert "anchor_text_mutual_neighbor" in relation_types
    assert "facet_profile_mutual_neighbor" in relation_types


def test_neighborhood_plan_attaches_active_candidates_and_keeps_residual():
    pool = ["seed", "linked", "low", "residual"]
    partitions = {
        "consensus_high": ["seed"],
        "disputed": ["linked"],
        "consensus_low": ["low", "residual"],
    }
    edges = {
        ("linked", "seed"): {"same_session"},
        ("low", "seed"): {"facet_profile_mutual_neighbor"},
    }
    plan = build_neighborhood_plan(
        pool,
        partitions,
        edges,
        active_ids=set(pool),
        mean_ranks={rid: index for index, rid in enumerate(pool, 1)},
        max_rounds=3,
        max_seed_rounds=1,
    )

    assert plan["estimated_seeded_calls"] == 1
    assert plan["neighborhoods"][0]["seed_round_ids"] == ["seed"]
    assert plan["neighborhoods"][0]["candidate_round_ids"] == ["linked", "low"]
    assert plan["residual_round_ids"] == ["residual"]
    assert plan["estimated_calls_with_residual"] == 2
