"""Small-set multimodal evidence verification and local rank correction."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple

from ._utils import extract_json

PROMPT_VERSION = "grouped-evidence-utility-v1"
SYSTEM_PROMPT = """You inspect a small set of retrieved memory rounds.

Do not answer the user's question and do not rank the whole memory. Decide only
which supplied rounds can contribute evidence. A round may be useful jointly
even when it cannot answer the question alone: it can be one item in a count,
one endpoint of a comparison, an earlier or later state, a negative fact, or a
boundary needed with another round. Use drop only for a clearly irrelevant
round. When unsure, use uncertain. Judge original dialogue and images, not the
retrieval hints."""
KEEP_STATES = {"keep_direct", "keep_joint"}


def formal_relation(relations: Sequence[str]) -> bool:
    values = set(map(str, relations))
    return "same_session" in values or {
        "anchor_text_mutual_neighbor", "facet_profile_mutual_neighbor"
    }.issubset(values)


def build_group_plan(
    base_ranking: Sequence[str],
    abstract_ranking: Sequence[str],
    raw_ranking: Sequence[str],
    edges: Mapping[Tuple[str, str], Set[str]],
    *,
    top_k: int = 10,
    max_rounds: int = 6,
    mode: str = "neighborhood",
) -> Dict[str, Any]:
    """Build deterministic rank batches or consensus-seeded neighborhoods."""
    pool = list(dict.fromkeys(map(str, base_ranking)))
    max_rounds = max(2, int(max_rounds))
    if mode == "rank_batch":
        return {
            "mode": mode,
            "seed_round_ids": [],
            "batches": [
                pool[i:i + max_rounds]
                for i in range(0, len(pool), max_rounds)
            ],
            "residual_round_ids": [],
        }
    if mode != "neighborhood":
        raise ValueError(f"Unsupported grouping mode: {mode}")

    abstract_top = set(map(str, abstract_ranking[:top_k]))
    raw_top = set(map(str, raw_ranking[:top_k]))
    seeds = [rid for rid in pool if rid in abstract_top & raw_top]
    seed_set = set(seeds)

    # Seed components merge only through natural-session provenance.
    components: List[List[str]] = []
    remaining = set(seeds)
    while remaining:
        first = min(remaining, key=pool.index)
        stack, component = [first], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            for other in seeds:
                rel = edges.get(tuple(sorted((current, other))), set())
                if "same_session" in rel and other not in component:
                    stack.append(other)
        remaining -= component
        components.append(sorted(component, key=pool.index))

    attached: Dict[int, List[str]] = {
        index: [] for index in range(len(components))
    }
    links: Dict[str, List[int]] = {}
    for rid in pool:
        if rid in seed_set:
            continue
        matched = []
        for index, component in enumerate(components):
            if any(
                formal_relation(
                    edges.get(tuple(sorted((rid, seed))), set())
                )
                for seed in component
            ):
                matched.append(index)
                attached[index].append(rid)
        if matched:
            links[rid] = matched

    batches: List[List[str]] = []
    included: Set[str] = set()
    for index, component in enumerate(components):
        candidates = attached[index]
        if not candidates:
            continue
        seed_refs = component[:min(2, max_rounds - 1)]
        capacity = max_rounds - len(seed_refs)
        for offset in range(0, len(candidates), capacity):
            members = seed_refs + candidates[offset:offset + capacity]
            members = list(dict.fromkeys(members))
            batches.append(members)
            included.update(members)

    residual = [
        rid for rid in pool
        if rid not in included and rid not in seed_set
    ]
    batches.extend(
        residual[i:i + max_rounds]
        for i in range(0, len(residual), max_rounds)
    )
    return {
        "mode": mode,
        "seed_round_ids": seeds,
        "seed_components": components,
        "candidate_component_links": links,
        "batches": batches,
        "residual_round_ids": residual,
    }


def build_group_prompt(
    *,
    question: str,
    question_date: str,
    members: Sequence[Mapping[str, Any]],
) -> str:
    chunks = [
        "We are retrieving past evidence for this question:",
        question.strip(),
        f"The question date is {question_date or 'not provided'}.",
        (
            "Below are potentially related memory rounds. Consider them "
            "together. A round can be necessary only in combination with "
            "another round."
        ),
    ]
    image_number = 0
    for member in members:
        image_refs = []
        for _ in member.get("images") or []:
            image_number += 1
            image_refs.append(str(image_number))
        chunks.extend([
            (
                f"Round {member['round_id']} from session "
                f"{member.get('session_id', '')}, dated "
                f"{member.get('session_date') or 'not provided'}:"
            ),
            f"User: {member.get('user_text', '')}",
            f"Assistant: {member.get('assistant_text', '')}",
            (
                "Original images supplied after the text: "
                + ", ".join(image_refs) + "."
                if image_refs else "This round has no image."
            ),
        ])
    ids = [str(member["round_id"]) for member in members]
    example = {
        "members": [
            {
                "round_id": rid,
                "state": "uncertain",
                "confidence": "low",
                "reason": "short reason",
            }
            for rid in ids
        ],
        "joint_groups": [ids[:2]] if len(ids) > 1 else [],
    }
    chunks.extend([
        (
            "For every supplied round, return one state: keep_direct, "
            "keep_joint, uncertain, or drop. keep_joint means it contributes "
            "only when combined with another supplied round. Use drop only "
            "when clearly irrelevant."
        ),
        "Return only JSON in this form:",
        json.dumps(example, ensure_ascii=False),
    ])
    return "\n\n".join(chunks)


def parse_group_verdict(
    raw_response: str, round_ids: Sequence[str]
) -> Dict[str, Any]:
    parsed = extract_json(raw_response or "")
    valid_ids = set(map(str, round_ids))
    members: Dict[str, Dict[str, Any]] = {
        rid: {
            "state": "uncertain",
            "confidence": "low",
            "reason": "Missing or invalid verdict.",
            "parse_valid": False,
        }
        for rid in valid_ids
    }
    if (
        not isinstance(parsed, Mapping)
        or not isinstance(parsed.get("members"), list)
    ):
        return {
            "members": members,
            "joint_groups": [],
            "parse_valid": False,
        }
    allowed_states = KEEP_STATES | {"uncertain", "drop"}
    allowed_confidence = {"high", "medium", "low"}
    seen = set()
    for item in parsed["members"]:
        if not isinstance(item, Mapping):
            continue
        rid = str(item.get("round_id", ""))
        state = str(item.get("state", "")).strip().lower()
        confidence = str(item.get("confidence", "")).strip().lower()
        if (
            rid not in valid_ids
            or state not in allowed_states
            or confidence not in allowed_confidence
        ):
            continue
        seen.add(rid)
        members[rid] = {
            "state": state,
            "confidence": confidence,
            "reason": str(item.get("reason", ""))[:800],
            "parse_valid": True,
        }
    groups = []
    for group in parsed.get("joint_groups") or []:
        if not isinstance(group, list):
            continue
        cleaned = list(dict.fromkeys(
            str(value) for value in group
            if str(value) in valid_ids
        ))
        if len(cleaned) >= 2:
            groups.append(cleaned)
    joint_ids = {
        round_id for group in groups for round_id in group
    }
    for rid, member in members.items():
        if member["state"] == "keep_joint" and rid not in joint_ids:
            member.update({
                "state": "uncertain",
                "confidence": "low",
                "reason": "keep_joint lacked a valid joint group.",
                "parse_valid": False,
            })
    return {
        "members": members,
        "joint_groups": groups,
        "parse_valid": (
            seen == valid_ids
            and all(item["parse_valid"] for item in members.values())
        ),
    }


def local_replace_rerank(
    base_ranking: Sequence[str],
    batches: Sequence[Sequence[str]],
    batch_verdicts: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> Tuple[List[str], Dict[str, Any]]:
    """Apply only explicit, within-batch drop-to-keep replacements."""
    base = list(dict.fromkeys(map(str, base_ranking)))
    selected = base[:top_k]
    selected_set = set(selected)
    replacements = []

    # A round can occur in overlapping neighborhoods. Aggregate conservatively:
    # any keep blocks a drop, and only unanimous high-confidence drops act.
    appearances: Dict[str, List[Mapping[str, Any]]] = {}
    for verdict in batch_verdicts:
        for rid, member in dict(verdict.get("members") or {}).items():
            appearances.setdefault(str(rid), []).append(dict(member))
    aggregate: Dict[str, Dict[str, Any]] = {}
    for rid, values in appearances.items():
        valid = [item for item in values if item.get("parse_valid")]
        states = {str(item.get("state", "")) for item in valid}
        if states & KEEP_STATES:
            state = (
                "keep_direct"
                if "keep_direct" in states else "keep_joint"
            )
            confidence = "high" if any(
                item.get("state") in KEEP_STATES
                and item.get("confidence") == "high"
                for item in valid
            ) else "medium"
        elif valid and all(
            item.get("state") == "drop"
            and item.get("confidence") == "high"
            for item in valid
        ):
            state, confidence = "drop", "high"
        else:
            state, confidence = "uncertain", "low"
        aggregate[rid] = {
            "state": state,
            "confidence": confidence,
            "parse_valid": bool(valid),
            "appearance_count": len(values),
        }

    for batch, verdict in zip(batches, batch_verdicts):
        member_map = aggregate
        drops = [
            rid for rid in batch
            if rid in selected_set
            and member_map.get(rid, {}).get("state") == "drop"
            and member_map.get(rid, {}).get("confidence") == "high"
            and member_map.get(rid, {}).get("parse_valid")
        ]
        keeps = [
            rid for rid in batch
            if rid not in selected_set
            and member_map.get(rid, {}).get("state") in KEEP_STATES
            and member_map.get(rid, {}).get("confidence")
            in {"high", "medium"}
            and member_map.get(rid, {}).get("parse_valid")
        ]
        for dropped, promoted in zip(drops, keeps):
            position = selected.index(dropped)
            selected[position] = promoted
            selected_set.remove(dropped)
            selected_set.add(promoted)
            replacements.append({
                "dropped_round_id": dropped,
                "promoted_round_id": promoted,
            })
    ranking = selected + [rid for rid in base if rid not in selected_set]
    return ranking, {
        "policy": "within_group_drop_to_keep_local_replacement",
        "replacements": replacements,
        "final_top_k_round_ids": selected,
        "aggregate_member_verdicts": aggregate,
    }


class GroupedEvidenceVerifier:
    """Exact-cache wrapper for one small-set VLM decision."""

    def __init__(
        self,
        vlm: Callable[[str, str, List[str]], str],
        cache_dir: Path,
        model_namespace: str,
    ) -> None:
        self.vlm = vlm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_namespace = model_namespace
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def verify(
        self,
        prompt: str,
        images: Sequence[str],
        round_ids: Sequence[str],
    ) -> Dict[str, Any]:
        image_state = []
        for value in images:
            path = Path(value)
            try:
                stat = path.stat()
                image_state.append(
                    (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
                )
            except OSError:
                image_state.append((str(path), "missing"))
        payload = {
            "version": PROMPT_VERSION,
            "model": self.model_namespace,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "images": image_state,
        }
        key = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        raw_response, cache_hit, error = "", False, ""
        with self._lock:
            if cache_path.exists():
                try:
                    raw_response = str(json.loads(
                        cache_path.read_text(encoding="utf-8")
                    ).get("raw_response") or "")
                    cache_hit = True
                    self.cache_hits += 1
                except (OSError, json.JSONDecodeError):
                    pass
        if not cache_hit:
            try:
                raw_response = self.vlm(
                    SYSTEM_PROMPT, prompt, list(images)
                ) or ""
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if not raw_response and not error:
                error = "empty_response"
            with self._lock:
                self.cache_misses += 1
                if raw_response:
                    cache_path.write_text(
                        json.dumps(
                            {
                                "raw_response": raw_response,
                                "prompt": prompt,
                                "images": list(images),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        return {
            "verdict": parse_group_verdict(raw_response, round_ids),
            "raw_response": raw_response,
            "cache_hit": cache_hit,
            "cache_key": key,
            "error": error,
        }
