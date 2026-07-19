"""Preserve official fine-grained question subtype metadata in converted records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def add_question_subtypes(dataset_path: Path, converted_dir: Path) -> Dict[str, Any]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    subtype_by_id = {
        str(record["question_id"]): str(record.get("question_subtype", ""))
        for record in records
    }
    manifest_path = converted_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    updated = 0
    missing = []
    subtype_counts: Dict[str, int] = {}
    for item in manifest.get("items", []):
        question_id = str(item["question_id"])
        target = converted_dir / item["path"]
        if question_id not in subtype_by_id:
            missing.append(question_id)
            continue
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        subtype = subtype_by_id[question_id]
        payload["human_annotated_qas"][0]["question_subtype"] = subtype
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        item["question_subtype"] = subtype
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        updated += 1

    manifest["question_subtype_counts"] = dict(sorted(subtype_counts.items()))
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return {"updated": updated, "missing": missing, "subtype_counts": subtype_counts}
