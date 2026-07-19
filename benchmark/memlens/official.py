"""Adapter for the role/evidence conventions used by the official release."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .prepare import (
    _write_json,
    convert_record,
    load_records,
    load_subset_ids,
    validate_record,
)


def _canonical_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role == "user":
        return "user"
    if role in {"assistant", "ai assistant", "ai_assistant"}:
        return "assistant"
    return role


def canonicalize_official_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize roles and make generic needle flags question-specific.

    MEMLENS ``has_answer`` marks needle-bearing turns, including distractor
    needles belonging to other questions.  ``answer_session_ids`` is the
    authoritative question-specific evidence scope.
    """
    answer_sessions = {str(value) for value in record.get("answer_session_ids", [])}
    for sid_raw, session in zip(
        record.get("haystack_session_ids", []), record.get("haystack_sessions", [])
    ):
        sid = str(sid_raw)
        for turn in session:
            turn["role"] = _canonical_role(turn.get("role"))
            if sid not in answer_sessions:
                turn["has_answer"] = False
    return record


def prepare_official_memlens_records(
    dataset_path: Path,
    subset_path: Path,
    image_root: Path,
    output_dir: Path,
    *,
    check_images: bool = True,
    expected_subset_size: int = 195,
) -> Dict[str, Any]:
    records = load_records(dataset_path)
    subset_ids = load_subset_ids(subset_path)
    if expected_subset_size and len(subset_ids) != expected_subset_size:
        raise ValueError(
            f"Expected {expected_subset_size} subset IDs, found {len(subset_ids)}"
        )
    by_id = {str(record.get("question_id", "")): record for record in records}
    missing = [question_id for question_id in subset_ids if question_id not in by_id]
    if missing:
        raise ValueError(f"Official subset IDs absent from dataset: {missing[:10]}")

    item_dir = output_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    manifest_items: List[Dict[str, Any]] = []
    invalid_items: List[Dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for question_id in subset_ids:
        record = canonicalize_official_record(by_id[question_id])
        errors, stats = validate_record(record, image_root, check_images=check_images)
        if errors:
            invalid_items.append({**stats, "errors": errors})
            continue
        target = item_dir / f"{question_id}.json"
        _write_json(target, convert_record(record))
        manifest_items.append(
            {**stats, "path": str(target.relative_to(output_dir)).replace("\\", "/")}
        )
        totals[stats["question_type"]] += 1

    manifest = {
        "source_dataset": str(dataset_path.resolve()),
        "source_subset": str(subset_path.resolve()),
        "image_root_for_validation": str(image_root.resolve()),
        "runtime_image_root": str(image_root.parent.resolve()),
        "context_length": "32k",
        "requested_ids": len(subset_ids),
        "converted_items": len(manifest_items),
        "invalid_items": len(invalid_items),
        "question_type_counts": dict(sorted(totals.items())),
        "items": manifest_items,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "invalid_items.json", invalid_items)
    return manifest
