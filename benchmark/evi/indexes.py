"""
EVI: Three flat in-memory indexes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from .schemas import GistEntry, TagEntry


class GistIndex:
    """Map[round_id -> GistEntry]. Supports embedding similarity and attr filter."""

    def __init__(self) -> None:
        self._data: Dict[str, GistEntry] = {}

    def put(self, round_id: str, entry: GistEntry) -> None:
        self._data[round_id] = entry

    def get(self, round_id: str) -> Optional[GistEntry]:
        return self._data.get(round_id)

    @property
    def all_rounds(self) -> List[str]:
        return list(self._data.keys())

    def filter_by_scene_attr(self, attr: str, value: str) -> List[str]:
        """Return round_ids where scene_attributes[attr] == value (case-insensitive)."""
        return [
            rid for rid, e in self._data.items()
            if e.scene_attributes.get(attr, "").lower() == value.lower()
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data.items())


class TagIndex:
    """Inverted index: tag_noun -> list of TagEntry.
    Also supports filtering by color/position.
    """

    def __init__(self) -> None:
        self._data: Dict[str, List[TagEntry]] = defaultdict(list)

    def add(self, noun: str, entry: TagEntry) -> None:
        self._data[noun.lower()].append(entry)

    def get(self, noun: str) -> List[TagEntry]:
        return self._data.get(noun.lower(), [])

    def round_ids_for_noun(self, noun: str) -> Set[str]:
        return {e.round_id for e in self._data.get(noun.lower(), [])}

    def filter_by_color(self, round_ids: Set[str], color: str) -> Set[str]:
        return {
            e.round_id for entries in self._data.values()
            for e in entries
            if e.round_id in round_ids and e.color.lower() == color.lower()
        }

    def filter_by_position(self, round_ids: Set[str], pos: str) -> Set[str]:
        return {
            e.round_id for entries in self._data.values()
            for e in entries
            if e.round_id in round_ids and pos.lower() in e.position.lower()
        }

    @property
    def all_nouns(self) -> List[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._data.values())

    def __iter__(self):
        return iter(self._data.items())


class AnchorIndex:
    """Inverted index: normalized keyword -> set[round_id]."""

    def __init__(self) -> None:
        self._data: Dict[str, Set[str]] = defaultdict(set)

    def add(self, keyword: str, round_id: str) -> None:
        """Normalize and add. Keyword should already be lowercased."""
        self._data[keyword].add(round_id)

    def get(self, keyword: str) -> Set[str]:
        return self._data.get(keyword, set())

    def match(self, raw_text: str) -> Set[str]:
        """Match a raw query string against all keys (bidirectional substring)."""
        import re
        q = re.sub(r'[^a-z0-9 ]', '', raw_text.lower())
        result: Set[str] = set()
        for key, rids in self._data.items():
            k = key.replace('_', ' ')
            if k in q or q in k:
                result |= rids
        return result

    @property
    def all_keys(self) -> List[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data.items())
