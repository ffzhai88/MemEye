from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.memlens.prepare import convert_record, prepare_memlens_records, validate_record


def _record() -> dict:
    return {
        "question_id": "q_test",
        "question_type": "multi_session_reasoning",
        "question": "What changed? Answer briefly.",
        "answer": "The red item became blue.",
        "question_date": "2024/02/01",
        "haystack_dates": ["2024/01/01", "2024/01/02"],
        "haystack_session_ids": ["S1", "S2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "It was red.", "images": [], "has_answer": True},
                {"role": "assistant", "content": "Noted.", "images": [], "has_answer": False},
            ],
            [
                {"role": "user", "content": "It changed.", "images": [], "has_answer": False},
                {
                    "role": "assistant",
                    "content": "It is blue now.",
                    "images": [
                        {
                            "file": "needle_images/blue.jpg",
                            "blip_caption": "A blue object",
                            "image_url": "https://example.invalid/blue.jpg",
                        }
                    ],
                    "has_answer": True,
                },
            ],
        ],
        "answer_session_ids": ["S1", "S2"],
    }


class MemlensPrepareTests(unittest.TestCase):
    def test_convert_preserves_assistant_image_role_and_clues(self) -> None:
        converted = convert_record(_record())
        qa = converted["human_annotated_qas"][0]
        self.assertEqual(qa["clue"], ["S1:R0001", "S2:R0001"])
        second_round = converted["multi_session_dialogues"][1]["dialogues"][0]
        self.assertEqual(second_round["memlens_image_roles"], ["assistant"])
        self.assertEqual(second_round["input_image"], ["release_images/needle_images/blue.jpg"])

    def test_validation_reports_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            errors, stats = validate_record(_record(), Path(tmp), check_images=True)
        self.assertIn("missing_image_files:1", errors)
        self.assertEqual(stats["num_images"], 1)

    def test_prepare_filters_by_official_order_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "needle_images" / "blue.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"test")
            dataset = root / "dataset.json"
            subset = root / "subset.json"
            dataset.write_text(json.dumps([_record()]), encoding="utf-8")
            subset.write_text(json.dumps(["q_test"]), encoding="utf-8")
            output = root / "converted"
            manifest = prepare_memlens_records(
                dataset, subset, root / "images", output, expected_subset_size=1
            )
            self.assertEqual(manifest["converted_items"], 1)
            self.assertTrue((output / "items" / "q_test.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
