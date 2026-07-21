from analyze_multifacet_fusion import replay_visual_corroborated_best_source


def _round(round_id, dialogue=None, visual=None, image=None):
    return {
        "round_id": round_id,
        "multimodal_scores": {
            "dialogue_calibrated": dialogue,
            "visual_anchor_calibrated": visual,
            "raw_image_calibrated": image,
        },
    }


def test_replay_uses_visual_image_agreement_without_question_labels():
    trace = {
        "facet_multimodal": {
            "facets": [
                {
                    "rounds": [
                        _round("visual-agreement", visual=0.8, image=0.9),
                        _round("visual-only", visual=0.9, image=0.1),
                        _round("visual-distractor", visual=0.1, image=0.8),
                        _round("dialogue", dialogue=0.95),
                    ]
                }
            ]
        }
    }
    ranking = replay_visual_corroborated_best_source(trace, source_top_k=3)
    assert set(ranking) == {
        "visual-agreement", "visual-only", "visual-distractor", "dialogue"
    }
    assert ranking.index("visual-agreement") < ranking.index("visual-only")


def test_replay_does_not_admit_raw_image_without_visual_anchor():
    trace = {
        "facet_multimodal": {
            "facets": [
                {
                    "rounds": [
                        _round("grounded", visual=0.5, image=0.5),
                        _round("image-only", image=1.0),
                    ]
                }
            ]
        }
    }
    assert replay_visual_corroborated_best_source(trace, 30) == ["grounded"]
