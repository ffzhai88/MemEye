from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


QUESTION_TYPES = {
    "information_extraction",
    "knowledge_update",
    "temporal_reasoning",
    "multi_session_reasoning",
    "answer_refusal",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_subset_ids(path: Path) -> List[str]:
    """Load the official indexing file, tolerating its list/dict variants."""
    payload = _read_json(path)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = None
        for key in ("question_ids", "ids", "agent_subset", "subset_ids"):
            if isinstance(payload.get(key), list):
                values = payload[key]
                break
        if values is None:
            raise ValueError(f"No question-id list found in {path}")
    else:
        raise ValueError(f"Unsupported subset index format in {path}")
    ids = [str(value).strip() for value in values if str(value).strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    return ids


def load_records(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    else:
        raise ValueError(f"Expected a JSON list of MEMLENS records in {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Non-object record found in {path}")
    return records


def _image_path(image_root: Path, image: Dict[str, Any]) -> Path:
    relative = str(image.get("file", "")).strip().replace("\\", "/")
    if not relative:
        return image_root / "__missing_image_file__"
    return image_root / Path(relative)


def validate_record(
    record: Dict[str, Any], image_root: Path, *, check_images: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    question_id = str(record.get("question_id", "")).strip()
    question_type = str(record.get("question_type", "")).strip()
    if not question_id:
        errors.append("missing_question_id")
    if question_type not in QUESTION_TYPES:
        errors.append(f"invalid_question_type:{question_type or '<empty>'}")
    if not str(record.get("question", "")).strip():
        errors.append("missing_question")
    if record.get("answer") is None or not str(record.get("answer", "")).strip():
        errors.append("missing_answer")

    session_ids = record.get("haystack_session_ids")
    sessions = record.get("haystack_sessions")
    dates = record.get("haystack_dates", [])
    if not isinstance(session_ids, list) or not isinstance(sessions, list):
        errors.append("invalid_sessions_container")
        session_ids, sessions = [], []
    if len(session_ids) != len(sessions):
        errors.append("session_id_count_mismatch")
    if dates and (not isinstance(dates, list) or len(dates) != len(sessions)):
        errors.append("session_date_count_mismatch")
    normalized_ids = [str(value).strip() for value in session_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        errors.append("duplicate_session_id")

    answer_sessions = record.get("answer_session_ids", [])
    if not isinstance(answer_sessions, list):
        errors.append("invalid_answer_session_ids")
        answer_sessions = []
    missing_answer_sessions = sorted(set(map(str, answer_sessions)) - set(normalized_ids))
    if missing_answer_sessions:
        errors.append("unknown_answer_sessions:" + ",".join(missing_answer_sessions))

    num_turns = 0
    num_images = 0
    num_answer_turns = 0
    missing_images: List[str] = []
    answer_turn_sessions: set[str] = set()
    for session_index, session in enumerate(sessions):
        sid = normalized_ids[session_index] if session_index < len(normalized_ids) else ""
        if not isinstance(session, list) or not session:
            errors.append(f"invalid_or_empty_session:{sid or session_index}")
            continue
        for turn_index, turn in enumerate(session):
            num_turns += 1
            if not isinstance(turn, dict):
                errors.append(f"invalid_turn:{sid}:{turn_index}")
                continue
            role = str(turn.get("role", "")).strip()
            if role not in {"user", "assistant"}:
                errors.append(f"invalid_role:{sid}:{turn_index}:{role or '<empty>'}")
            if bool(turn.get("has_answer")):
                num_answer_turns += 1
                answer_turn_sessions.add(sid)
            images = turn.get("images", []) or []
            if not isinstance(images, list):
                errors.append(f"invalid_images:{sid}:{turn_index}")
                continue
            for image in images:
                num_images += 1
                if not isinstance(image, dict) or not str(image.get("file", "")).strip():
                    errors.append(f"invalid_image_ref:{sid}:{turn_index}")
                    continue
                path = _image_path(image_root, image)
                if check_images and not path.is_file():
                    missing_images.append(str(image.get("file")))

    if question_type != "answer_refusal" and not answer_sessions:
        errors.append("missing_answer_session")
    if question_type != "answer_refusal" and num_answer_turns == 0:
        errors.append("missing_answer_turn")
    if answer_turn_sessions and not answer_turn_sessions.issubset(set(map(str, answer_sessions))):
        errors.append("answer_turn_outside_answer_sessions")
    if missing_images:
        errors.append(f"missing_image_files:{len(set(missing_images))}")

    stats = {
        "question_id": question_id,
        "question_type": question_type,
        "num_sessions": len(sessions),
        "num_turns": num_turns,
        "num_images": num_images,
        "num_answer_turns": num_answer_turns,
        "missing_images": sorted(set(missing_images)),
    }
    return errors, stats


def _flush_round(
    rounds: List[Dict[str, Any]], sid: str, ordinal: int, turns: List[Dict[str, Any]]
) -> None:
    if not turns:
        return
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    image_paths: List[str] = []
    captions: List[str] = []
    image_roles: List[str] = []
    source_turn_indices: List[int] = []
    has_answer = False
    for source_index, turn in turns:
        role = str(turn.get("role", ""))
        content = str(turn.get("content", "")).strip()
        if content:
            (user_parts if role == "user" else assistant_parts).append(content)
        source_turn_indices.append(source_index)
        has_answer = has_answer or bool(turn.get("has_answer"))
        for image in turn.get("images", []) or []:
            image_paths.append("release_images/" + str(image["file"]).replace("\\", "/"))
            captions.append(str(image.get("blip_caption", "")).strip())
            image_roles.append(role)
    round_id = f"{sid}:R{ordinal:04d}"
    rounds.append(
        {
            "round": round_id,
            "user": "\n".join(user_parts),
            "assistant": "\n".join(assistant_parts),
            "input_image": image_paths,
            "image_caption": captions,
            "image_id": [Path(path).stem for path in image_paths],
            "memlens_image_roles": image_roles,
            "memlens_source_turn_indices": source_turn_indices,
            "memlens_has_answer": has_answer,
        }
    )


def _convert_session(sid: str, date: str, turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair user/assistant turns while retaining provenance and assistant-side images."""
    rounds: List[Dict[str, Any]] = []
    pending: List[Tuple[int, Dict[str, Any]]] = []
    ordinal = 1
    for source_index, turn in enumerate(turns):
        role = str(turn.get("role", ""))
        if role == "user" and pending:
            _flush_round(rounds, sid, ordinal, pending)
            ordinal += 1
            pending = []
        pending.append((source_index, turn))
        if role == "assistant":
            _flush_round(rounds, sid, ordinal, pending)
            ordinal += 1
            pending = []
    if pending:
        _flush_round(rounds, sid, ordinal, pending)
    return {"session_id": sid, "date": date, "dialogues": rounds}


def convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
    sessions: List[Dict[str, Any]] = []
    clue_rounds: List[str] = []
    dates = record.get("haystack_dates", []) or []
    for index, (sid_raw, turns) in enumerate(
        zip(record["haystack_session_ids"], record["haystack_sessions"])
    ):
        sid = str(sid_raw)
        date = str(dates[index]) if index < len(dates) else ""
        converted = _convert_session(sid, date, turns)
        sessions.append(converted)
        clue_rounds.extend(
            dialogue["round"]
            for dialogue in converted["dialogues"]
            if dialogue.get("memlens_has_answer")
        )

    qa = {
        "question_id": str(record["question_id"]),
        "question": str(record["question"]),
        "answer": str(record["answer"]),
        "question_type": str(record["question_type"]),
        "original_question_type": str(record["question_type"]),
        "question_date": str(record.get("question_date", "")),
        "session_id": [str(value) for value in record.get("answer_session_ids", [])],
        "clue": clue_rounds,
        "answer_session_ids": [str(value) for value in record.get("answer_session_ids", [])],
        "source_dataset": "memlens",
        "context_length": "32k",
        "evidence_semantics": "answer-bearing rounds derived from MEMLENS has_answer",
    }
    return {
        "source_dataset": "memlens",
        "source_question_id": str(record["question_id"]),
        "context_length": "32k",
        "multi_session_dialogues": sessions,
        "human_annotated_qas": [qa],
    }


def prepare_memlens_records(
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
            f"Expected {expected_subset_size} subset IDs, found {len(subset_ids)} in {subset_path}"
        )
    by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_ids: List[str] = []
    for record in records:
        question_id = str(record.get("question_id", "")).strip()
        if question_id in by_id:
            duplicate_ids.append(question_id)
        by_id[question_id] = record
    if duplicate_ids:
        raise ValueError("Duplicate question IDs in dataset: " + ", ".join(sorted(set(duplicate_ids))))
    missing_ids = [question_id for question_id in subset_ids if question_id not in by_id]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} official subset IDs are absent from {dataset_path}: {missing_ids[:10]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    item_dir = output_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    manifest_items: List[Dict[str, Any]] = []
    invalid_items: List[Dict[str, Any]] = []
    totals = Counter()
    for question_id in subset_ids:
        record = by_id[question_id]
        errors, stats = validate_record(record, image_root, check_images=check_images)
        if errors:
            invalid_items.append({**stats, "errors": errors})
            continue
        target = item_dir / f"{question_id}.json"
        _write_json(target, convert_record(record))
        manifest_items.append({**stats, "path": str(target.relative_to(output_dir)).replace("\\", "/")})
        totals[stats["question_type"]] += 1

    manifest = {
        "source_dataset": str(dataset_path.resolve()),
        "source_subset": str(subset_path.resolve()),
        "image_root": str(image_root.resolve()),
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
