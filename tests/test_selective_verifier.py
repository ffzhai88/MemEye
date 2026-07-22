import json

from benchmark.evi.selective_verifier import (
    SelectiveEvidenceVerifier,
    build_verification_prompt,
    conservative_rerank,
    parse_verdict,
    verification_candidate_ids,
)


def test_prompt_uses_narrative_evidence_context():
    prompt = build_verification_prompt(
        question="Which chair do I prefer now?",
        question_date="2026-07-22",
        facets=["current chair preference", "preference update"],
        session_id="S2",
        round_id="S2:R3",
        session_date="2026-06-01",
        user_text="I replaced the blue chair with the green one.",
        assistant_text="The green chair is your new choice.",
        anchors=[{"type": "visual", "text": "a green chair"}],
        image_count=1,
    )

    assert "While searching" in prompt
    assert "stored memory descriptions included" in prompt
    assert "Now inspect the original memory" in prompt
    assert "clue_round_ids" not in prompt
    assert "question_subtype" not in prompt
    assert "Do not require it to answer the entire question" in prompt


def test_parse_verdict_validates_dialogue_quote():
    response = json.dumps({
        "anchor_grounding": "partial",
        "evidence_utility": "useful",
        "confidence": "high",
        "supported_facet_indices": [2, 2, "1"],
        "dialogue_grounding": "green chair",
        "visual_grounding": "A green chair is visible.",
        "reason": "It records the later preference.",
    })
    verdict = parse_verdict(
        response, dialogue_text="The green chair is my new choice."
    )
    assert verdict.parse_valid
    assert verdict.dialogue_grounding == "green chair"
    assert verdict.supported_facet_indices == (2, 1)

    fabricated = parse_verdict(
        response, dialogue_text="Only a blue chair was discussed."
    )
    assert fabricated.dialogue_grounding == ""


def test_invalid_response_fails_closed():
    verdict = parse_verdict("not json")
    assert not verdict.parse_valid
    assert verdict.evidence_utility == "possibly_useful"
    assert verdict.confidence == "low"


def test_candidate_selection_is_top_k_symmetric_difference():
    result = verification_candidate_ids(
        ["a", "b", "c"],
        ["d", "b", "e"],
        top_k=2,
        order_ranking=["b", "d", "a", "f"],
    )
    assert result == ["d", "a"]


def test_conservative_rerank_never_promotes_consensus_excluded_round():
    ranking, trace = conservative_rerank(
        ["stable", "anchor", "outside", "raw", "tail"],
        ["stable", "anchor", "outside"],
        ["stable", "raw", "outside"],
        {
            "anchor": {
                "parse_valid": True,
                "evidence_utility": "not_useful",
                "confidence": "high",
            },
            "raw": {
                "parse_valid": True,
                "evidence_utility": "useful",
                "confidence": "high",
            },
        },
        top_k=2,
    )
    assert ranking[:2] == ["stable", "raw"]
    assert "outside" not in ranking[:2]
    assert trace["demoted_round_ids"] == ["anchor"]
    assert trace["promoted_round_ids"] == ["raw"]


def test_conservative_rerank_falls_back_within_union_when_all_contested_rejected():
    rejected = {
        rid: {
            "parse_valid": True,
            "evidence_utility": "not_useful",
            "confidence": "high",
        }
        for rid in ("anchor", "raw")
    }
    ranking, trace = conservative_rerank(
        ["stable", "anchor", "outside", "raw"],
        ["stable", "anchor"],
        ["stable", "raw"],
        rejected,
        top_k=2,
    )
    assert ranking[:2] == ["stable", "anchor"]
    assert "outside" not in ranking[:2]
    assert trace["demoted_count"] == 0


def test_verifier_uses_exact_cache(tmp_path):
    calls = []

    def fake_vlm(system, user, images):
        calls.append((system, user, images))
        return json.dumps({
            "anchor_grounding": "supported",
            "evidence_utility": "useful",
            "confidence": "high",
            "supported_facet_indices": [1],
            "dialogue_grounding": "a fact",
            "visual_grounding": "",
            "reason": "The fact is directly stated.",
        })

    verifier = SelectiveEvidenceVerifier(
        fake_vlm, tmp_path, model_namespace="fake-model"
    )
    first = verifier.verify("prompt", [], "a fact appears here")
    second = verifier.verify("prompt", [], "a fact appears here")

    assert len(calls) == 1
    assert not first["cache_hit"]
    assert second["cache_hit"]
    cached = json.loads(
        next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    )
    assert cached["user_prompt"] == "prompt"


def test_empty_api_response_is_retryable(tmp_path):
    verifier = SelectiveEvidenceVerifier(
        lambda _system, _user, _images: "",
        tmp_path,
        model_namespace="empty-model",
    )
    result = verifier.verify("prompt", [], "dialogue")
    assert result["error"] == "empty_response"
    assert not result["verdict"]["parse_valid"]
    assert not list(tmp_path.glob("*.json"))
