from benchmark.evi.facet_multimodal import (
    empirical_midrank_percentiles,
    score_multimodal_facet_rounds,
)
from benchmark.evi.schemas import EvidenceAnchor
from benchmark.evi.indexes import EvidenceIndex


def _anchor(round_id: str, source: str = "dialogue") -> EvidenceAnchor:
    return EvidenceAnchor(
        id=f"{round_id}:{source}", session_id="session-1", round_id=round_id,
        date="2026-01-01", evidence_type=source, text=f"{source} evidence",
        vector=[1.0, 0.0], image_path="image.jpg" if source == "visual" else "",
    )


def _row(round_id: str, dialogue=None, visual=None):
    scores, anchors = {}, {}
    if dialogue is not None:
        scores["dialogue"] = dialogue
        anchors["dialogue"] = _anchor(round_id, "dialogue")
    if visual is not None:
        scores["visual"] = visual
        anchors["visual"] = _anchor(round_id, "visual")
    return {"round_id": round_id, "session_id": "session-1", "date": "2026-01-01",
            "source_scores": scores, "source_anchors": anchors}


def test_empirical_midrank_percentiles_are_tie_stable():
    assert empirical_midrank_percentiles({"a": 1.0, "b": 1.0}) == {"a": 0.5, "b": 0.5}
    assert empirical_midrank_percentiles({"low": 0.0, "high": 1.0}) == {"low": 0.25, "high": 0.75}


def test_missing_visual_evidence_does_not_penalize_text_only_round():
    ranked, _ = score_multimodal_facet_rounds(
        [_row("strong", dialogue=0.9), _row("weak", dialogue=0.1)], [], top_k=10)
    assert [item["round_id"] for item in ranked] == ["strong", "weak"]
    assert ranked[0]["score"] == 0.75
    assert ranked[0]["has_visual"] is False


def test_visual_anchor_and_raw_image_are_one_visual_branch():
    ranked, _ = score_multimodal_facet_rounds(
        [_row("r1", dialogue=0.9, visual=0.8), _row("r2", dialogue=0.1, visual=0.2)],
        [{"round_id": "r1", "score": 0.7, "best_image_path": "r1.jpg"},
         {"round_id": "r2", "score": 0.3, "best_image_path": "r2.jpg"}], top_k=10)
    assert ranked[0]["round_id"] == "r1"
    assert ranked[0]["multimodal_scores"]["visual_combined"] == 0.75
    assert ranked[0]["multimodal_scores"]["multimodal"] == 0.75


def test_raw_image_cannot_introduce_round_without_evi_provenance():
    ranked, trace = score_multimodal_facet_rounds(
        [_row("known", dialogue=0.5)], [{"round_id": "image-only", "score": 1.0}], top_k=10)
    assert [item["round_id"] for item in ranked] == ["known"]
    assert trace["num_image_rounds"] == 1
    assert trace["num_scored_rounds"] == 1


def test_evidence_index_keeps_best_anchor_per_round_and_source():
    index = EvidenceIndex()
    weak = _anchor("r1", "dialogue")
    weak.vector = [0.0, 1.0]
    strong = _anchor("r1", "dialogue")
    strong.id = "r1:dialogue:strong"
    strong.vector = [1.0, 0.0]
    visual = _anchor("r1", "visual")
    visual.vector = [0.5, 0.5]
    index.extend([weak, strong, visual])

    rows = index.score_rounds_by_source([1.0, 0.0])
    assert len(rows) == 1
    assert rows[0]["source_anchors"]["dialogue"].id == strong.id
    assert rows[0]["source_scores"]["dialogue"] == 1.0
    assert rows[0]["source_scores"]["visual"] > 0.0
