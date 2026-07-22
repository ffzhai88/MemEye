"""Offline analysis of consensus-seeded evidence neighborhoods.

This diagnostic reads a completed abstract-candidate JOINT retrieval run and
its selective-verification JSONL.  It never calls an embedding model or a VLM.
Question labels and clue annotations are used only after candidate partitions
and relation graphs have been constructed, for evaluation.

The analysis distinguishes three rank-defined partitions:

* ``consensus_high``: in both Anchor Top-K and Raw-MM Top-K;
* ``disputed``: in exactly one of the two Top-K lists;
* ``consensus_low``: in neither Top-K, but still in the fixed Top-N pool.

Relations are derived from saved, label-free retrieval traces.  The output is
a diagnostic neighborhood plan, not an online retrieval method.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple


PARTITIONS = ("consensus_high", "disputed", "consensus_low")
BASE_RELATIONS = (
    "same_session",
    "round_adjacent",
    "same_dominant_specific_facet",
    "anchor_text_mutual_neighbor",
    "facet_profile_mutual_neighbor",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _question_id(row: Mapping[str, Any]) -> str:
    return str(row.get("question_id", row.get("idx", "")))


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("benchmark", "")),
        str(row.get("dataset", "")),
        _question_id(row),
    )


def discover_selective_file(input_dir: Path, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            path = path / "selective_verification_questions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates = sorted(
        input_dir.glob("selective_verification*/selective_verification_questions.jsonl")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one selective_verification*/"
            f"selective_verification_questions.jsonl under {input_dir}; found {candidates}"
        )
    return candidates[0].resolve()


def load_original_rows(input_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    output: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    memeye_root = input_dir / "memeye"
    if memeye_root.is_dir():
        for task_dir in sorted(memeye_root.iterdir()):
            path = task_dir / "retrievals.jsonl"
            if not task_dir.is_dir() or not path.is_file():
                continue
            for row in _read_jsonl(path):
                output[("memeye", task_dir.name, _question_id(row))] = row
    memlens_path = input_dir / "memlens" / "retrievals.jsonl"
    if memlens_path.is_file():
        for row in _read_jsonl(memlens_path):
            output[("memlens", "memlens", _question_id(row))] = row
    return output


def classify_round(
    round_id: str, anchor_top: Set[str], raw_top: Set[str]
) -> str:
    in_anchor = round_id in anchor_top
    in_raw = round_id in raw_top
    if in_anchor and in_raw:
        return "consensus_high"
    if in_anchor != in_raw:
        return "disputed"
    return "consensus_low"


def partition_candidates(
    anchor_ranking: Sequence[str],
    raw_ranking: Sequence[str],
    pool: Sequence[str],
    top_k: int,
) -> Dict[str, List[str]]:
    anchor_top = set(map(str, anchor_ranking[:top_k]))
    raw_top = set(map(str, raw_ranking[:top_k]))
    output = {name: [] for name in PARTITIONS}
    for round_id in dict.fromkeys(map(str, pool)):
        output[classify_round(round_id, anchor_top, raw_top)].append(round_id)
    return output


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _tokens(value: Any) -> Set[str]:
    values = re.findall(r"[\w]+", _normalize_text(value), flags=re.UNICODE)
    ignored = {
        "round", "user", "assistant", "session", "the", "and", "that",
        "this", "with", "from", "for", "was", "were", "are", "on",
    }
    return {token for token in values if len(token) >= 3 and token not in ignored}


def _jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return _safe_div(len(left & right), len(union)) if union else 0.0


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    ln = math.sqrt(sum(float(a) * float(a) for a in left))
    rn = math.sqrt(sum(float(b) * float(b) for b in right))
    return dot / (ln * rn) if ln and rn else 0.0


def _round_number(round_id: str) -> int | None:
    match = re.search(r":R(?:0*)(\d+)$", str(round_id), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _trace_features(
    original_row: Mapping[str, Any], pool: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    trace = dict(original_row.get("retrieval_trace") or {})
    facet_rows = list(trace.get("facets") or [])
    facets = [str(item.get("text", "")) for item in facet_rows]
    by_id = {
        str(item.get("round_id", "")): dict(item)
        for item in list(trace.get("rounds") or [])
        if str(item.get("round_id", ""))
    }
    output: Dict[str, Dict[str, Any]] = {}
    for round_id in pool:
        row = by_id.get(round_id, {})
        facet_scores = dict(row.get("facet_scores") or {})
        profile = [float(facet_scores.get(facet, 0.0) or 0.0) for facet in facets]
        specific = profile[1:]
        dominant = ""
        if specific and max(specific) > 0.0:
            dominant = str(1 + max(range(len(specific)), key=specific.__getitem__))
        anchor_text = " ".join(
            str(item.get("text", ""))
            for item in list(row.get("top_anchors") or [])
            if isinstance(item, Mapping)
        )
        output[round_id] = {
            "session_id": str(row.get("session_id", "")),
            "round_number": _round_number(round_id),
            "dominant_specific_facet": dominant,
            "anchor_tokens": _tokens(anchor_text),
            "facet_profile": profile,
        }
    return output


def _add_edge(
    edges: MutableMapping[Tuple[str, str], Set[str]],
    left: str,
    right: str,
    relation: str,
) -> None:
    if not left or not right or left == right:
        return
    key = tuple(sorted((left, right)))
    edges[key].add(relation)


def _mutual_neighbor_edges(
    ids: Sequence[str],
    similarity,
) -> Set[Tuple[str, str]]:
    best: Dict[str, Set[str]] = {}
    for left in ids:
        scores = {
            right: float(similarity(left, right))
            for right in ids
            if right != left
        }
        maximum = max(scores.values(), default=0.0)
        best[left] = {
            right for right, score in scores.items()
            if maximum > 0.0 and abs(score - maximum) <= 1e-12
        }
    return {
        tuple(sorted((left, right)))
        for left in ids
        for right in best.get(left, set())
        if left in best.get(right, set())
    }


def build_relation_graph(
    features: Mapping[str, Mapping[str, Any]],
) -> Dict[Tuple[str, str], Set[str]]:
    ids = list(features)
    edges: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for index, left in enumerate(ids):
        lf = features[left]
        for right in ids[index + 1:]:
            rf = features[right]
            same_session = bool(lf.get("session_id")) and (
                lf.get("session_id") == rf.get("session_id")
            )
            if same_session:
                _add_edge(edges, left, right, "same_session")
                ln, rn = lf.get("round_number"), rf.get("round_number")
                if ln is not None and rn is not None and abs(int(ln) - int(rn)) == 1:
                    _add_edge(edges, left, right, "round_adjacent")
            dominant = str(lf.get("dominant_specific_facet", ""))
            if dominant and dominant == str(rf.get("dominant_specific_facet", "")):
                _add_edge(edges, left, right, "same_dominant_specific_facet")

    for left, right in _mutual_neighbor_edges(
        ids,
        lambda a, b: _jaccard(
            set(features[a].get("anchor_tokens") or []),
            set(features[b].get("anchor_tokens") or []),
        ),
    ):
        _add_edge(edges, left, right, "anchor_text_mutual_neighbor")
    for left, right in _mutual_neighbor_edges(
        ids,
        lambda a, b: _cosine(
            list(features[a].get("facet_profile") or []),
            list(features[b].get("facet_profile") or []),
        ),
    ):
        _add_edge(edges, left, right, "facet_profile_mutual_neighbor")
    return dict(edges)


def _neighbors(
    ids: Sequence[str], edges: Mapping[Tuple[str, str], Set[str]]
) -> Dict[str, Set[str]]:
    output = {round_id: set() for round_id in ids}
    for (left, right), relations in edges.items():
        if relations:
            output.setdefault(left, set()).add(right)
            output.setdefault(right, set()).add(left)
    return output


def _relation_seed_links(
    candidate: str,
    seeds: Set[str],
    edges: Mapping[Tuple[str, str], Set[str]],
) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for seed in seeds:
        relations = edges.get(tuple(sorted((candidate, seed))), set())
        if relations:
            output[seed] = sorted(relations)
    return output


def _seed_components(
    seeds: Sequence[str], edges: Mapping[Tuple[str, str], Set[str]]
) -> List[List[str]]:
    remaining = set(seeds)
    components: List[List[str]] = []
    seed_set = set(seeds)
    graph = _neighbors(seeds, {
        edge: rel for edge, rel in edges.items()
        if edge[0] in seed_set and edge[1] in seed_set
    })
    while remaining:
        first = min(remaining)
        stack = [first]
        component: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(graph.get(current, set()) - component)
        remaining -= component
        components.append(sorted(component))
    return components


def build_neighborhood_plan(
    pool: Sequence[str],
    partitions: Mapping[str, Sequence[str]],
    edges: Mapping[Tuple[str, str], Set[str]],
    active_ids: Set[str],
    mean_ranks: Mapping[str, int],
    max_rounds: int,
    max_seed_rounds: int,
) -> Dict[str, Any]:
    seeds = list(partitions["consensus_high"])
    seed_set = set(seeds)
    components = _seed_components(seeds, edges)
    links = {
        candidate: _relation_seed_links(candidate, seed_set, edges)
        for candidate in pool
        if candidate not in seed_set
    }
    assigned: Dict[int, List[str]] = defaultdict(list)
    multi_component_links: Dict[str, List[int]] = {}
    for candidate in pool:
        if candidate not in active_ids or candidate in seed_set:
            continue
        candidate_links = links.get(candidate, {})
        matches: List[Tuple[int, int, int]] = []
        for index, component in enumerate(components):
            relation_types = {
                relation
                for seed in component
                for relation in candidate_links.get(seed, [])
            }
            if relation_types:
                best_seed_rank = min(mean_ranks.get(seed, 10**9) for seed in component)
                matches.append((-len(relation_types), best_seed_rank, index))
        if not matches:
            continue
        matches.sort()
        chosen = matches[0][2]
        assigned[chosen].append(candidate)
        multi_component_links[candidate] = [item[2] for item in matches]

    neighborhoods: List[Dict[str, Any]] = []
    included: Set[str] = set()
    max_rounds = max(2, int(max_rounds))
    max_seed_rounds = max(1, min(int(max_seed_rounds), max_rounds - 1))
    for component_index, component in enumerate(components):
        candidates = sorted(
            assigned.get(component_index, []), key=lambda rid: mean_ranks.get(rid, 10**9)
        )
        if not candidates:
            continue
        seed_refs = sorted(component, key=lambda rid: mean_ranks.get(rid, 10**9))[:max_seed_rounds]
        capacity = max(1, max_rounds - len(seed_refs))
        for offset in range(0, len(candidates), capacity):
            members = candidates[offset:offset + capacity]
            included.update(members)
            neighborhoods.append({
                "component_index": component_index,
                "seed_round_ids": seed_refs,
                "all_component_seed_round_ids": component,
                "candidate_round_ids": members,
                "disputed_round_ids": [
                    rid for rid in members if rid in set(partitions["disputed"])
                ],
                "consensus_low_round_ids": [
                    rid for rid in members if rid in set(partitions["consensus_low"])
                ],
                "candidate_seed_relations": {
                    rid: links.get(rid, {}) for rid in members
                },
            })

    residual = sorted(
        active_ids - seed_set - included, key=lambda rid: mean_ranks.get(rid, 10**9)
    )
    residual_capacity = max_rounds
    residual_batches = [
        residual[offset:offset + residual_capacity]
        for offset in range(0, len(residual), residual_capacity)
    ]
    return {
        "seed_components": components,
        "candidate_component_links": multi_component_links,
        "neighborhoods": neighborhoods,
        "residual_round_ids": residual,
        "residual_batches": residual_batches,
        "estimated_seeded_calls": len(neighborhoods),
        "estimated_calls_with_residual": len(neighborhoods) + len(residual_batches),
    }


def _partition_stats_template() -> Dict[str, Counter]:
    return {name: Counter() for name in PARTITIONS}


def _finalize_partition_stats(
    stats: Mapping[str, Counter], total_clues: int
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for name in PARTITIONS:
        values = stats[name]
        output[name] = {
            **dict(values),
            "clue_density": _safe_div(values["clue_rounds"], values["rounds"]),
            "clue_coverage": _safe_div(values["clue_rounds"], total_clues),
            "question_clue_hit_rate": _safe_div(
                values["questions_with_clue"], values["questions"]
            ),
        }
    return output


def analyze(
    input_dir: Path,
    selective_file: Path,
    candidate_k: int,
    top_k: int,
    diagnostic_ks: Sequence[int],
    max_neighborhood_rounds: int,
    max_seed_rounds: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    selective_rows = _read_jsonl(selective_file)
    original_rows = load_original_rows(input_dir)
    if not selective_rows:
        raise ValueError(f"No rows in {selective_file}")

    by_k: Dict[int, Dict[str, Dict[str, Counter]]] = {
        k: defaultdict(_partition_stats_template) for k in diagnostic_ks
    }
    total_clues_by_group: Counter = Counter()
    total_questions_by_group: Counter = Counter()
    relation_stats: Dict[str, Dict[str, Dict[str, Counter]]] = defaultdict(
        lambda: defaultdict(_partition_stats_template)
    )
    transition_stats: Dict[str, Dict[str, Counter]] = defaultdict(_partition_stats_template)
    verdicts: Dict[str, Dict[str, Counter]] = defaultdict(_partition_stats_template)
    cost_stats: Dict[str, Counter] = defaultdict(Counter)
    question_outputs: List[Dict[str, Any]] = []

    for selective in selective_rows:
        benchmark, dataset, question_id = _row_key(selective)
        group_names = ("all", benchmark)
        original = original_rows.get((benchmark, dataset, question_id))
        if original is None and benchmark == "memlens":
            original = original_rows.get((benchmark, "memlens", question_id))
        if original is None:
            raise KeyError(f"No original retrieval row for {(benchmark, dataset, question_id)}")

        rankings = dict(selective.get("rankings") or {})
        anchor = list(map(str, rankings.get("abstract_evi") or []))
        raw = list(map(str, rankings.get("raw_multimodal_fixed") or []))
        mean = list(map(str, rankings.get("evi_raw_mean_rank") or []))
        final = list(map(str, rankings.get("selective_vlm") or []))
        pool = list(dict.fromkeys(mean[:candidate_k]))
        clues = set(map(str, selective.get("clue_round_ids") or []))
        mean_ranks = {rid: rank for rank, rid in enumerate(mean, 1)}

        for group in group_names:
            total_clues_by_group[group] += len(clues)
            total_questions_by_group[group] += 1
        for diagnostic_k in diagnostic_ks:
            parts_k = partition_candidates(anchor, raw, pool, diagnostic_k)
            for group in group_names:
                for name, ids in parts_k.items():
                    values = by_k[diagnostic_k][group][name]
                    values["questions"] += 1
                    values["rounds"] += len(ids)
                    hits = len(set(ids) & clues)
                    values["clue_rounds"] += hits
                    values["questions_with_clue"] += int(hits > 0)

        partitions = partition_candidates(anchor, raw, pool, top_k)
        anchor_top, raw_top = set(anchor[:top_k]), set(raw[:top_k])
        features = _trace_features(original, pool)
        edges = build_relation_graph(features)
        graph = _neighbors(pool, edges)
        seeds = set(partitions["consensus_high"])
        direct = {
            rid for rid in pool if rid not in seeds and bool(graph.get(rid, set()) & seeds)
        }
        one_hop = seeds | direct
        two_hop = direct | {
            rid for rid in pool
            if rid not in seeds and bool(graph.get(rid, set()) & one_hop)
        }

        checked_ids = {
            str(item.get("round_id", ""))
            for item in list(selective.get("verification_results") or [])
        }
        active_ids = (
            set(mean[:top_k]) | set(final[:top_k]) | checked_ids
        ) & set(pool)
        plan = build_neighborhood_plan(
            pool,
            partitions,
            edges,
            active_ids,
            mean_ranks,
            max_neighborhood_rounds,
            max_seed_rounds,
        )

        for group in group_names:
            cost = cost_stats[group]
            cost["questions"] += 1
            cost["current_verifier_calls"] += len(checked_ids)
            cost["active_non_seed_rounds"] += len(active_ids - seeds)
            cost["estimated_seeded_calls"] += plan["estimated_seeded_calls"]
            cost["estimated_calls_with_residual"] += plan["estimated_calls_with_residual"]
            cost["seed_components"] += len(plan["seed_components"])
            cost["residual_active_rounds"] += len(plan["residual_round_ids"])

        for relation in (*BASE_RELATIONS, "any_direct_relation", "any_two_hop_relation"):
            if relation == "any_direct_relation":
                connected = direct
            elif relation == "any_two_hop_relation":
                connected = two_hop
            else:
                connected = {
                    rid for rid in pool if rid not in seeds and any(
                        relation in edges.get(tuple(sorted((rid, seed))), set())
                        for seed in seeds
                    )
                }
            for group in group_names:
                for name in ("disputed", "consensus_low"):
                    ids = set(partitions[name])
                    values = relation_stats[relation][group][name]
                    values["questions"] += 1
                    values["rounds"] += len(ids)
                    values["clue_rounds"] += len(ids & clues)
                    linked = ids & connected
                    values["connected_rounds"] += len(linked)
                    values["connected_clues"] += len(linked & clues)
                    values["connected_non_clues"] += len(linked - clues)

        mean_top, final_top = set(mean[:top_k]), set(final[:top_k])
        for group in group_names:
            for rid in pool:
                name = classify_round(rid, anchor_top, raw_top)
                values = transition_stats[group][name]
                values["pool_rounds"] += 1
                values["pool_clues"] += int(rid in clues)
                values["mean_top_rounds"] += int(rid in mean_top)
                values["mean_top_clues"] += int(rid in mean_top and rid in clues)
                values["final_top_rounds"] += int(rid in final_top)
                values["final_top_clues"] += int(rid in final_top and rid in clues)
                values["dropped_rounds"] += int(rid in mean_top and rid not in final_top)
                values["dropped_clues"] += int(rid in mean_top and rid not in final_top and rid in clues)
                values["added_rounds"] += int(rid in final_top and rid not in mean_top)
                values["added_clues"] += int(rid in final_top and rid not in mean_top and rid in clues)
            for item in list(selective.get("verification_results") or []):
                rid = str(item.get("round_id", ""))
                name = classify_round(rid, anchor_top, raw_top)
                verdict = str((item.get("verdict") or {}).get("evidence_utility", "unknown"))
                verdicts[group][name]["checked"] += 1
                verdicts[group][name][verdict] += 1

        relation_rows = [
            {
                "left_round_id": left,
                "right_round_id": right,
                "relation_types": sorted(relations),
            }
            for (left, right), relations in sorted(edges.items())
        ]
        question_outputs.append({
            "benchmark": benchmark,
            "dataset": dataset,
            "question_id": question_id,
            "question": str(selective.get("question", "")),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "clue_round_ids": sorted(clues),
            "partitions": partitions,
            "direct_seed_connected_round_ids": sorted(direct, key=lambda rid: mean_ranks.get(rid, 10**9)),
            "two_hop_seed_connected_round_ids": sorted(two_hop, key=lambda rid: mean_ranks.get(rid, 10**9)),
            "relations": relation_rows,
            "active_round_ids": sorted(active_ids, key=lambda rid: mean_ranks.get(rid, 10**9)),
            "neighborhood_plan": plan,
            "evaluation": {
                "partition_clue_counts": {
                    name: len(set(ids) & clues) for name, ids in partitions.items()
                },
                "direct_connected_clue_counts": {
                    name: len(set(ids) & direct & clues) for name, ids in partitions.items()
                },
                "two_hop_connected_clue_counts": {
                    name: len(set(ids) & two_hop & clues) for name, ids in partitions.items()
                },
            },
        })

    finalized_by_k: Dict[str, Any] = {}
    for diagnostic_k, groups in by_k.items():
        finalized_by_k[str(diagnostic_k)] = {
            group: _finalize_partition_stats(
                group_stats, total_clues_by_group[group]
            )
            for group, group_stats in groups.items()
        }

    finalized_relations: Dict[str, Any] = {}
    for relation, groups in relation_stats.items():
        finalized_relations[relation] = {}
        for group, partitions_by_name in groups.items():
            finalized_relations[relation][group] = {}
            for name in ("disputed", "consensus_low"):
                values = partitions_by_name[name]
                finalized_relations[relation][group][name] = {
                    **dict(values),
                    "candidate_connection_rate": _safe_div(
                        values["connected_rounds"], values["rounds"]
                    ),
                    "clue_connection_recall": _safe_div(
                        values["connected_clues"], values["clue_rounds"]
                    ),
                    "connected_clue_density": _safe_div(
                        values["connected_clues"], values["connected_rounds"]
                    ),
                }

    finalized_transition: Dict[str, Any] = {}
    for group, group_stats in transition_stats.items():
        finalized_transition[group] = {}
        for name in PARTITIONS:
            values = group_stats[name]
            finalized_transition[group][name] = {
                **dict(values),
                "pool_clue_density": _safe_div(values["pool_clues"], values["pool_rounds"]),
                "mean_top_clue_density": _safe_div(values["mean_top_clues"], values["mean_top_rounds"]),
                "final_top_clue_density": _safe_div(values["final_top_clues"], values["final_top_rounds"]),
                "verifier_verdicts": dict(verdicts[group][name]),
            }

    finalized_costs: Dict[str, Any] = {}
    for group, values in cost_stats.items():
        questions = values["questions"]
        finalized_costs[group] = {
            **dict(values),
            "current_calls_per_question": _safe_div(values["current_verifier_calls"], questions),
            "estimated_seeded_calls_per_question": _safe_div(values["estimated_seeded_calls"], questions),
            "estimated_calls_with_residual_per_question": _safe_div(values["estimated_calls_with_residual"], questions),
            "seed_components_per_question": _safe_div(values["seed_components"], questions),
            "residual_active_rounds_per_question": _safe_div(values["residual_active_rounds"], questions),
        }

    metrics = {
        "analysis_type": "consensus_seeded_evidence_neighborhoods_offline",
        "input_dir": str(input_dir),
        "selective_file": str(selective_file),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "diagnostic_ks": list(diagnostic_ks),
        "max_neighborhood_rounds": max_neighborhood_rounds,
        "max_seed_rounds": max_seed_rounds,
        "num_questions": len(selective_rows),
        "total_clues": dict(total_clues_by_group),
        "partition_quality_by_k": finalized_by_k,
        "relation_quality": finalized_relations,
        "current_verifier_transition": finalized_transition,
        "neighborhood_cost_estimate": finalized_costs,
        "limitations": [
            "Clue annotations are evaluation-only and may not exhaust all useful evidence.",
            "Saved traces do not contain candidate-to-candidate raw image embeddings.",
            "Anchor-text and facet-profile mutual-neighbor edges are trace-derived proxies, not new embeddings.",
            "Cost estimates count requests, not image tokens or repeated-seed visual cost.",
        ],
    }
    return metrics, question_outputs


def render_report(metrics: Mapping[str, Any]) -> str:
    top_k = str(metrics["top_k"])
    partition = metrics["partition_quality_by_k"][top_k]
    lines = [
        "# Consensus-seeded evidence neighborhood diagnostic",
        "",
        "This is an offline, evaluation-only diagnostic. It calls no embedding model or VLM.",
        "",
        f"- Questions: {metrics['num_questions']}",
        f"- Candidate pool: Top-{metrics['candidate_k']}",
        f"- Consensus boundary: Top-{metrics['top_k']}",
        "",
    ]
    for group in ("all", "memeye", "memlens"):
        if group not in partition:
            continue
        lines.extend([
            f"## Partition quality: {group}",
            "",
            "| Partition | Rounds | Clues | Clue density | Clue coverage |",
            "|---|---:|---:|---:|---:|",
        ])
        for name in PARTITIONS:
            row = partition[group][name]
            lines.append(
                f"| {name} | {row.get('rounds', 0)} | {row.get('clue_rounds', 0)} | "
                f"{row['clue_density']:.4f} | {row['clue_coverage']:.4f} |"
            )
        lines.append("")

    lines.extend([
        "## Relations to consensus-high seeds",
        "",
        "| Relation | Partition | Connected rounds | Candidate rate | Connected clues | Clue recall | Connected density |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for relation, groups in metrics["relation_quality"].items():
        if "all" not in groups:
            continue
        for name in ("disputed", "consensus_low"):
            row = groups["all"][name]
            lines.append(
                f"| {relation} | {name} | {row.get('connected_rounds', 0)} | "
                f"{row['candidate_connection_rate']:.4f} | {row.get('connected_clues', 0)} | "
                f"{row['clue_connection_recall']:.4f} | {row['connected_clue_density']:.4f} |"
            )

    lines.extend([
        "",
        "## Current verifier transition",
        "",
        "| Benchmark | Partition | Mean slots/clues | Final slots/clues | Mean density | Final density | Added clues | Dropped clues |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in ("all", "memeye", "memlens"):
        values = metrics["current_verifier_transition"].get(group, {})
        for name in PARTITIONS:
            if name not in values:
                continue
            row = values[name]
            lines.append(
                f"| {group} | {name} | {row.get('mean_top_rounds', 0)}/{row.get('mean_top_clues', 0)} | "
                f"{row.get('final_top_rounds', 0)}/{row.get('final_top_clues', 0)} | "
                f"{row['mean_top_clue_density']:.4f} | {row['final_top_clue_density']:.4f} | "
                f"{row.get('added_clues', 0)} | {row.get('dropped_clues', 0)} |"
            )

    lines.extend([
        "",
        "## Request-count estimate",
        "",
        "| Benchmark | Current calls/q | Seeded calls/q | With residual/q | Seed components/q | Residual active rounds/q |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for group in ("all", "memeye", "memlens"):
        row = metrics["neighborhood_cost_estimate"].get(group)
        if not row:
            continue
        lines.append(
            f"| {group} | {row['current_calls_per_question']:.3f} | "
            f"{row['estimated_seeded_calls_per_question']:.3f} | "
            f"{row['estimated_calls_with_residual_per_question']:.3f} | "
            f"{row['seed_components_per_question']:.3f} | "
            f"{row['residual_active_rounds_per_question']:.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation limits",
        "",
    ])
    lines.extend(f"- {item}" for item in metrics.get("limitations", []))
    lines.append("")
    return "\n".join(lines)


def _parse_ks(value: str) -> List[int]:
    output = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not output or min(output) <= 0:
        raise argparse.ArgumentTypeError("K values must be positive integers")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Abstract-candidate JOINT run directory")
    parser.add_argument("--selective-results", default="", help="Selective directory or questions JSONL")
    parser.add_argument("--output-dir", default="", help="Default: <input>/consensus_neighborhood_analysis")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--diagnostic-ks", type=_parse_ks, default=_parse_ks("5,10,20"))
    parser.add_argument("--max-neighborhood-rounds", type=int, default=6)
    parser.add_argument("--max-seed-rounds", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    selective_file = discover_selective_file(input_dir, args.selective_results)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / "consensus_neighborhood_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, questions = analyze(
        input_dir=input_dir,
        selective_file=selective_file,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        diagnostic_ks=args.diagnostic_ks,
        max_neighborhood_rounds=args.max_neighborhood_rounds,
        max_seed_rounds=args.max_seed_rounds,
    )
    _write_json(output_dir / "consensus_neighborhood_metrics.json", metrics)
    _write_jsonl(output_dir / "consensus_neighborhood_questions.jsonl", questions)
    (output_dir / "consensus_neighborhood_report.md").write_text(
        render_report(metrics), encoding="utf-8"
    )
    _write_json(output_dir / "consensus_neighborhood_config.json", {
        "input": str(input_dir),
        "selective_results": str(selective_file),
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "diagnostic_ks": args.diagnostic_ks,
        "max_neighborhood_rounds": args.max_neighborhood_rounds,
        "max_seed_rounds": args.max_seed_rounds,
    })
    print(f"[CONSENSUS-NEIGHBORHOODS] questions={len(questions)} output={output_dir}")


if __name__ == "__main__":
    main()
