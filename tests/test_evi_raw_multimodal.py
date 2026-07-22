from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark.evi.raw_multimodal import (
    RawMultimodalCandidateReranker,
    mean_rank_fuse,
)
from benchmark.evi.system import EVISystem
from benchmark.retrieval_eval import _retrieval_components


class _Dataset:
    rounds = {
        "alpha": {"user": "alpha", "assistant": "", "images": []},
        "balanced": {"user": "balanced", "assistant": "", "images": ["b.jpg"]},
        "visual": {"user": "visual", "assistant": "", "images": ["v.jpg"]},
    }


class _Embedder:
    vectors = {
        "alpha": [1.0, 0.0],
        "balanced": [0.7, 0.7],
        "visual": [0.0, 1.0],
        "question": [1.0, 0.0],
    }

    def embed_query(self, text):
        return self.vectors[text]

    def embed_batch(self, texts):
        return [self.vectors[text] for text in texts]


class _ImageIndex:
    def search(self, question, top_k):
        assert question == "question"
        return [
            {"round_id": "visual", "score": 0.9, "best_image_path": "v.jpg"},
            {"round_id": "balanced", "score": 0.7, "best_image_path": "b.jpg"},
        ], {"image_embedding_backend": "fake", "image_embedding_model": "fake-mm"}


def test_mean_rank_fusion_uses_primary_rank_as_deterministic_tie_break():
    fused, rows = mean_rank_fuse(
        ["a", "b", "c", "d", "e"],
        ["c", "d", "e", "a", "b"],
    )
    assert fused == ["c", "a", "d", "b", "e"]
    assert rows[0] == {
        "round_id": "c",
        "evi_rank": 3,
        "raw_multimodal_rank": 1,
        "mean_rank": 2.0,
        "final_rank": 1,
    }


def test_raw_multimodal_reranker_matches_fixed_missing_image_policy_and_caches():
    with TemporaryDirectory() as directory:
        config = {
            "evi_raw_multimodal_text_weight": 0.5,
            "evi_raw_multimodal_image_weight": 0.5,
            "raw_multimodal_text_embedding_cache_dir": directory,
        }
        reranker = RawMultimodalCandidateReranker(
            _Dataset(), _Embedder(), _ImageIndex(), config, "fake-text"
        )
        fused, trace = reranker.rerank(
            "question", ["alpha", "visual", "balanced"]
        )
        assert trace["raw_multimodal_ranked_round_ids"] == [
            "balanced", "alpha", "visual"
        ]
        assert fused == ["alpha", "balanced", "visual"]
        by_id = {row["round_id"]: row for row in trace["rows"]}
        assert by_id["alpha"]["has_image"] is False
        assert by_id["alpha"]["image_score"] == 0.0
        assert trace["missing_image_policy"] == "zero_image_score"

        cached = RawMultimodalCandidateReranker(
            _Dataset(), _Embedder(), _ImageIndex(), config, "fake-text"
        )
        cached.rerank("question", ["alpha", "visual", "balanced"])
        assert cached.cache_hits == 4
        assert Path(directory, "document").is_dir()
        assert Path(directory, "query").is_dir()


def test_raw_multimodal_candidate_fusion_requires_raw_image_retrieval():
    try:
        EVISystem({
            "evi_apply_raw_multimodal_candidate_rank_fusion": True,
            "evi_use_raw_image_retrieval": False,
        })
    except ValueError as exc:
        assert "requires evi_use_raw_image_retrieval=true" in str(exc)
    else:
        raise AssertionError("Expected invalid raw multimodal configuration to fail")


def test_retrieval_components_expose_raw_multimodal_rankings():
    components = _retrieval_components({
        "raw_multimodal_candidate_fusion": {
            "raw_multimodal_ranked_round_ids": ["r2", "r1"],
            "fused_ranked_round_ids": ["r1", "r2"],
        }
    })
    assert components["raw_multimodal_ranked_round_ids"] == ["r2", "r1"]
    assert components["evi_raw_multimodal_fused_round_ids"] == ["r1", "r2"]
