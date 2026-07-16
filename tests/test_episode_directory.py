from __future__ import annotations

import unittest

from analyze_episode_oracle import (
    _available_session_ranking_fields,
    _build_balanced_episode_path,
    _paired_session_bootstrap,
)
from benchmark.evi.episode_directory import (
    EpisodeDirectoryEntry,
    EpisodeDirectoryIndex,
    build_episode_directory_entry,
)
from benchmark.evi.schemas import EvidenceAnchor


def _anchor(
    anchor_id: str,
    session_id: str,
    round_id: str,
    text: str,
    vector: list[float],
    image_path: str | None = None,
) -> EvidenceAnchor:
    return EvidenceAnchor(
        id=anchor_id,
        session_id=session_id,
        round_id=round_id,
        date="2026-01-01",
        evidence_type="entity",
        text=text,
        vector=vector,
        image_path=image_path,
    )


class EpisodeDirectoryTests(unittest.TestCase):
    def test_directory_document_preserves_round_order_and_deduplicates_anchors(self) -> None:
        captured: list[str] = []

        def embed(text: str) -> list[float]:
            captured.append(text)
            return [1.0, 0.0]

        entry = build_episode_directory_entry(
            session_id="S1",
            date="2026-01-01",
            round_ids=["S1:R1", "S1:R2"],
            round_text={
                "S1:R1": "User: first\nAssistant: noted",
                "S1:R2": "User: second\nAssistant: noted",
            },
            anchors_by_round={
                "S1:R1": [
                    _anchor("a1", "S1", "S1:R1", "Red sign", [1.0, 0.0]),
                    _anchor("a2", "S1", "S1:R1", "  red   SIGN  ", [1.0, 0.0]),
                ],
                "S1:R2": [
                    _anchor(
                        "a3", "S1", "S1:R2", "Blue vehicle", [0.0, 1.0], "image.jpg"
                    )
                ],
            },
            embed=embed,
        )

        self.assertEqual(entry.anchor_count, 2)
        self.assertLess(entry.text.index("Round S1:R1"), entry.text.index("Round S1:R2"))
        self.assertEqual(entry.text.count("Red sign"), 1)
        self.assertIn("[visual/entity] Blue vehicle", entry.text)
        self.assertEqual(captured, [entry.text])

    def test_directory_score_is_holistic_and_witness_is_diagnostic_only(self) -> None:
        index = EpisodeDirectoryIndex()
        index.add(
            EpisodeDirectoryEntry("S1", "d", ["S1:R1"], "one", [1.0, 0.0], 1)
        )
        index.add(
            EpisodeDirectoryEntry("S2", "d", ["S2:R1"], "two", [0.0, 1.0], 1)
        )
        hits = index.search(
            [1.0, 0.0],
            {
                "S1:R1": [_anchor("a1", "S1", "S1:R1", "weak", [0.0, 1.0])],
                "S2:R1": [_anchor("a2", "S2", "S2:R1", "strong", [1.0, 0.0])],
            },
        )

        self.assertEqual([hit["session_id"] for hit in hits], ["S1", "S2"])
        self.assertEqual(hits[0]["witness_round_id"], "S1:R1")
        self.assertEqual(hits[1]["witness_round_id"], "S2:R1")

    def test_balanced_expansion_round_robins_under_the_same_budget(self) -> None:
        episodes = [
            {"session_id": "S1", "member_round_ids": ["S1:R1", "S1:R2", "S1:R3"]},
            {"session_id": "S2", "member_round_ids": ["S2:R1", "S2:R2", "S2:R3"]},
        ]
        ranked, _ = _build_balanced_episode_path(
            episodes,
            ["S1:R2", "S2:R2", "S1:R1", "S2:R1"],
            limit=4,
        )
        self.assertEqual(ranked, ["S1:R2", "S2:R2", "S1:R1", "S2:R1"])

    def test_analyzer_discovers_and_compares_directory_ranking(self) -> None:
        rows = [
            {
                "clue_session_ids": ["S1"],
                "current_ranked_session_ids": ["S2", "S1"],
                "directory_ranked_session_ids": ["S1", "S2"],
                "episode_directory_enabled": True,
                "episode_directory_complete": True,
            }
        ]
        fields = _available_session_ranking_fields(rows)
        bootstrap = _paired_session_bootstrap(
            rows,
            "directory_ranked_session_ids",
            "current_ranked_session_ids",
            samples=100,
        )

        self.assertEqual(fields["holistic_directory"], "directory_ranked_session_ids")
        self.assertEqual(bootstrap["recall_at_m_delta_mean"], 1.0)
        self.assertEqual(bootstrap["map_delta_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
