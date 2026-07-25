import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.methods import SavedRankingReplayMethod


class _Dataset:
    rounds = {
        "S1:R1": {"round": "S1:R1"},
        "S2:R1": {"round": "S2:R1"},
    }

    def session_order(self):
        return ["S1", "S2"]

    def get_session(self, session_id):
        return {"session_id": session_id, "dialogues": []}


class SavedRankingReplayTest(unittest.TestCase):
    def test_selects_requested_strategy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "rankings.jsonl"
            source.write_text(
                json.dumps({
                    "benchmark": "memeye",
                    "dataset": "example",
                    "question_id": "Q1",
                    "rankings": {
                        "selective_vlm": ["S2:R1", "S1:R1"],
                        "evi_raw_mean_rank": ["S1:R1", "S2:R1"],
                    },
                }) + "\n",
                encoding="utf-8",
            )
            captured = []

            def fake_history(session, rounds, allowed, modality):
                captured.append((session["session_id"], set(allowed), modality))
                return []

            with patch(
                "benchmark.methods.history_from_round_ids",
                side_effect=fake_history,
            ):
                method = SavedRankingReplayMethod({
                    "source_rankings_jsonl": str(source),
                    "source_dataset": "example",
                    "ranking_strategy": "selective_vlm",
                    "context_top_k": 2,
                })
                self.assertEqual(
                    method.build_history(_Dataset(), {"question_id": "Q1"}),
                    [],
                )
            self.assertEqual(captured, [
                ("S1", {"S1:R1", "S2:R1"}, "multimodal"),
                ("S2", {"S1:R1", "S2:R1"}, "multimodal"),
            ])
            self.assertEqual(
                method.runtime_info["selected_round_ids"],
                ["S2:R1", "S1:R1"],
            )


if __name__ == "__main__":
    unittest.main()
