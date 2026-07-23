import base64
import io
import json
from pathlib import Path

from benchmark.evi.vlm import _encode_image
from benchmark.evi.grouped_verifier import (
    GroupedEvidenceVerifier,
    build_group_plan,
    build_group_prompt,
    local_replace_rerank,
    parse_group_verdict,
)


def test_neighborhood_uses_only_same_session_or_dual_mnn():
    pool = ["seed", "same", "dual", "anchor_only", "residual"]
    edges = {
        ("same", "seed"): {"same_session"},
        ("dual", "seed"): {
            "anchor_text_mutual_neighbor",
            "facet_profile_mutual_neighbor",
        },
        ("anchor_only", "seed"): {"anchor_text_mutual_neighbor"},
    }
    plan = build_group_plan(
        pool,
        ["seed", "a"],
        ["seed", "b"],
        edges,
        top_k=2,
        max_rounds=4,
        mode="neighborhood",
    )
    assert plan["batches"][0] == ["seed", "same", "dual"]
    assert "anchor_only" in plan["residual_round_ids"]
    assert "residual" in plan["residual_round_ids"]


def test_seed_components_merge_only_by_session():
    edges = {
        ("s1", "s2"): {
            "anchor_text_mutual_neighbor",
            "facet_profile_mutual_neighbor",
        },
        ("c", "s1"): {"same_session"},
    }
    plan = build_group_plan(
        ["s1", "s2", "c"],
        ["s1", "s2"],
        ["s1", "s2"],
        edges,
        top_k=2,
        max_rounds=3,
    )
    assert plan["seed_components"] == [["s1"], ["s2"]]


def test_rank_batch_is_equal_size_control():
    plan = build_group_plan(
        list("abcdefg"), [], [], {},
        max_rounds=3, mode="rank_batch",
    )
    assert plan["batches"] == [
        ["a", "b", "c"], ["d", "e", "f"], ["g"]
    ]


def test_group_prompt_is_narrative_and_mentions_joint_evidence():
    prompt = build_group_prompt(
        question="Which chair is preferred now?",
        question_date="2026-07-23",
        members=[
            {
                "round_id": "S1:R1",
                "session_id": "S1",
                "session_date": "2026-06-01",
                "user_text": "I liked the blue chair.",
                "assistant_text": "Noted.",
                "images": ["one.jpg"],
            },
            {
                "round_id": "S2:R1",
                "session_id": "S2",
                "session_date": "2026-07-01",
                "user_text": "I replaced it with green.",
                "assistant_text": "Green is current.",
                "images": [],
            },
        ],
    )
    assert "Consider them together" in prompt
    assert "User: I liked the blue chair." in prompt
    assert "clue_round_ids" not in prompt
    assert "question_subtype" not in prompt


def test_parse_group_verdict_fails_closed_per_member():
    parsed = parse_group_verdict(json.dumps({
        "members": [
            {
                "round_id": "a",
                "state": "keep_joint",
                "confidence": "high",
                "reason": "Earlier endpoint.",
            }
        ],
        "joint_groups": [["a", "b"]],
    }), ["a", "b"])
    assert parsed["members"]["a"]["parse_valid"]
    assert not parsed["members"]["b"]["parse_valid"]
    assert not parsed["parse_valid"]
    assert parsed["joint_groups"] == [["a", "b"]]


def test_local_rerank_requires_within_batch_drop_to_keep_pair():
    verdict = {
        "members": {
            "high": {
                "state": "drop",
                "confidence": "high",
                "parse_valid": True,
            },
            "low": {
                "state": "keep_joint",
                "confidence": "medium",
                "parse_valid": True,
            },
        }
    }
    ranking, trace = local_replace_rerank(
        ["stable", "high", "other", "low"],
        [["high", "low"]],
        [verdict],
        top_k=3,
    )
    assert ranking[:3] == ["stable", "low", "other"]
    assert trace["replacements"] == [{
        "dropped_round_id": "high",
        "promoted_round_id": "low",
    }]


def test_local_rerank_does_not_promote_without_explicit_drop():
    verdict = {
        "members": {
            "high": {
                "state": "uncertain",
                "confidence": "low",
                "parse_valid": True,
            },
            "low": {
                "state": "keep_direct",
                "confidence": "high",
                "parse_valid": True,
            },
        }
    }
    ranking, trace = local_replace_rerank(
        ["high", "other", "low"],
        [["high", "low"]],
        [verdict],
        top_k=2,
    )
    assert ranking[:2] == ["high", "other"]
    assert not trace["replacements"]
def test_overlapping_verdicts_keep_blocks_drop():
    drop = {
        "members": {
            "high": {
                "state": "drop",
                "confidence": "high",
                "parse_valid": True,
            },
            "low": {
                "state": "keep_direct",
                "confidence": "high",
                "parse_valid": True,
            },
        }
    }
    keep = {
        "members": {
            "high": {
                "state": "keep_joint",
                "confidence": "medium",
                "parse_valid": True,
            }
        }
    }
    ranking, trace = local_replace_rerank(
        ["high", "other", "low"],
        [["high", "low"], ["high"]],
        [drop, keep],
        top_k=2,
    )
    assert ranking[:2] == ["high", "other"]
    assert trace["aggregate_member_verdicts"]["high"]["state"] == "keep_joint"

def test_failed_request_still_writes_exact_manifest(tmp_path):
    image = tmp_path / "large.jpg"
    image.write_bytes(b"1234567")

    def fail(_system, _prompt, _images):
        raise RuntimeError("request too large")

    verifier = GroupedEvidenceVerifier(
        fail,
        tmp_path / "cache",
        "fake-model",
        request_log_dir=tmp_path / "requests",
    )
    result = verifier.verify(
        "full user prompt",
        [str(image)],
        ["S1:R1"],
    )

    assert result["error"] == "RuntimeError: request too large"
    manifest_path = tmp_path / "requests" / (
        result["cache_key"] + ".request.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["round_ids"] == ["S1:R1"]
    assert manifest["user_prompt"] == "full user prompt"
    assert manifest["images"][0]["file_bytes"] == 7
    assert manifest["images"][0]["estimated_base64_bytes"] == 12
    assert manifest["result"] == "failed"

def test_srag_compatible_image_preprocessing(tmp_path):
    from PIL import Image

    image = tmp_path / "large.png"
    Image.new("RGB", (2000, 1200), color=(120, 30, 20)).save(image)
    raw_url = _encode_image(str(image), max_long_edge=0)
    resized_url = _encode_image(str(image), max_long_edge=768)

    assert resized_url.startswith("data:image/jpeg;base64,")
    assert len(resized_url) < len(raw_url)
    processed = Image.open(io.BytesIO(base64.b64decode(
        resized_url.split(",", 1)[1]
    )))
    assert max(processed.size) == 768


def test_manifest_records_processed_payload_size(tmp_path):
    from PIL import Image

    image = tmp_path / "large.png"
    Image.new("RGB", (2000, 1200), color=(120, 30, 20)).save(image)
    verifier = GroupedEvidenceVerifier(
        lambda _system, _prompt, _images: json.dumps({
            "members": [{
                "round_id": "S1:R1",
                "state": "uncertain",
                "confidence": "low",
                "reason": "test",
            }],
            "joint_groups": [],
        }),
        tmp_path / "cache",
        "fake-model-img768",
        request_log_dir=tmp_path / "requests",
        image_max_long_edge=768,
    )
    result = verifier.verify("prompt", [str(image)], ["S1:R1"])
    manifest = json.loads(Path(
        result["request_manifest_path"]
    ).read_text(encoding="utf-8"))
    item = manifest["images"][0]
    assert manifest["image_preprocessing"]["max_long_edge"] == 768
    assert item["processed_data_url_bytes"] < item["estimated_data_url_bytes"]
    assert manifest["estimated_encoded_image_bytes"] == item["processed_data_url_bytes"]
