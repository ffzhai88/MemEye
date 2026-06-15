"""
EVI: Phase 2 — Track A (aggregate) and Track B (retrieve + route + ground).
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Set

if TYPE_CHECKING:
    from .vlm import VLMCallable

from .indexes import AnchorIndex, GistIndex, TagIndex
from .schemas import AggregateResult, RichDirectory, RichDirectoryEntry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Track A — aggregate query on flat indexes
# ---------------------------------------------------------------------------

_COUNT_KEYWORDS = [
    "how many", "count", "total", "how much",
    "number of", "across all", "of the",
]


def _has_count_keyword(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _COUNT_KEYWORDS)


def _extract_filters(
    question: str,
    anchor_index: AnchorIndex,
) -> dict:
    """Extract simple filter conditions from question text.
    Returns dict with optional keys: brand, bg, color.
    """
    filters: dict = {}
    q_clean = re.sub(r'[^a-z0-9 ]', '', question.lower())

    # Brand / anchor match
    for key in anchor_index.all_keys:
        k_clean = key.replace('_', ' ')
        if k_clean in q_clean or q_clean in k_clean:
            filters["brand"] = key
            break

    # Color + background
    COLORS = [
        "red", "blue", "green", "yellow", "white", "black",
        "purple", "orange", "pink", "brown", "gray", "teal",
    ]
    has_bg = "background" in question.lower() or "backdrop" in question.lower()
    for color in COLORS:
        if color in question.lower():
            if has_bg:
                filters["bg"] = color
            else:
                filters["color"] = color
            break

    return filters


def run_track_a(
    question: str,
    gist_index: GistIndex,
    tag_index: TagIndex,
    anchor_index: AnchorIndex,
) -> AggregateResult:
    """Try to answer via aggregation. Cost ~zero (no LLM, no images)."""
    if not _has_count_keyword(question):
        return AggregateResult(answer="", confidence=0.0, candidates=[])

    filters = _extract_filters(question, anchor_index)
    if not filters:
        return AggregateResult(answer="", confidence=0.0, candidates=[])

    candidates: Set[str] = set(gist_index.all_rounds)

    for k, v in filters.items():
        if k in ("bg", "background"):
            candidates = {
                rid for rid in candidates
                if v.lower() == (gist_index.get(rid).scene_attributes.get("background_color", "") if gist_index.get(rid) else "").lower()
            }
            # Re-implement concisely:
            filtered: Set[str] = set()
            for rid in candidates:
                entry = gist_index.get(rid)
                if entry and entry.scene_attributes.get("background_color", "").lower() == v.lower():
                    filtered.add(rid)
            candidates = filtered

        elif k == "brand":
            key = v.lower().replace(" ", "_")
            candidates &= anchor_index.get(key)

        elif k == "color":
            filtered = set()
            for noun in tag_index.all_nouns:
                for te in tag_index.get(noun):
                    if te.round_id in candidates and te.color.lower() == v.lower():
                        filtered.add(te.round_id)
            candidates = filtered

    count = len(candidates)

    # Confidence
    confidence = 1.0
    if count == 0:
        confidence = 0.0
    if len(filters) == 1 and "color" in filters and count > 20:
        confidence = 0.3  # too many — likely noise

    return AggregateResult(
        answer=str(count),
        confidence=confidence,
        candidates=sorted(candidates),
    )


# ---------------------------------------------------------------------------
# Track B — retrieval pipeline
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _embed_text(text: str, embedder: Any) -> List[float]:
    """Embed text with a callable embedder."""
    if embedder is None:
        return []
    try:
        return embedder.embed_query(text)
    except AttributeError:
        try:
            return embedder(text)
        except Exception:
            return []


def step1_coarse_retrieval(
    question: str,
    gist_index: GistIndex,
    tag_index: TagIndex,
    anchor_index: AnchorIndex,
    text_embedder: Any = None,
    top_k: int = 15,
) -> RichDirectory:
    """BM25-like on tags + embedding on gist + anchor match. No images."""
    candidates: Set[str] = set()

    # 1. Anchor match
    candidates |= anchor_index.match(question)

    # 2. BM25 on tag nouns
    q_terms = set(re.findall(r'[a-z]+', question.lower()))
    for noun in tag_index.all_nouns:
        if noun in q_terms:
            candidates |= tag_index.round_ids_for_noun(noun)

    # 3. Embedding on gist free_text
    embed_scores: dict = {}
    if text_embedder is not None:
        q_vec = _embed_text(question, text_embedder)
        if q_vec:
            for rid, gist_entry in gist_index:
                # lazy compute + cache
                if gist_entry._embedding is None:
                    gist_entry._embedding = _embed_text(gist_entry.free_text, text_embedder)
                if gist_entry._embedding:
                    score = _cosine(q_vec, gist_entry._embedding)
                    if score > 0.3:
                        embed_scores[rid] = score

    # Fusion rank
    combined: dict = {}
    for rid in candidates:
        combined[rid] = 1.0
    for rid, score in embed_scores.items():
        combined[rid] = combined.get(rid, 0) + score * 2.0

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    top_ids = [rid for rid, _ in ranked[:top_k]]

    # Assemble Rich Directory
    entries: List[RichDirectoryEntry] = []
    for rid in top_ids:
        gist = gist_index.get(rid)
        if gist is None:
            continue
        tags_for_round: List[str] = []
        for noun in tag_index.all_nouns:
            for te in tag_index.get(noun):
                if te.round_id == rid:
                    tags_for_round.append(f"{noun}({te.color},{te.position})")
        anchors_for_round = anchor_index.match(rid)

        entries.append(RichDirectoryEntry(
            round_id=rid,
            free_text=gist.free_text,
            scene_attributes=dict(gist.scene_attributes),
            tags=tags_for_round[:10],
            anchors=list(anchors_for_round)[:5],
        ))

    return RichDirectory(entries=entries)


def step2_llm_routing(
    question: str,
    directory: RichDirectory,
    text_callable: Callable[[str], str],
    top_n: int = 3,
) -> List[str]:
    """LLM selects the top-N most relevant round IDs from the rich directory.

    Args:
        question: The user's question.
        directory: Rich directory of candidate rounds.
        text_callable: A callable(prompt_text) -> response_text.
        top_n: Number of rounds to select.

    Returns:
        List of selected round IDs.
    """
    if not directory.entries:
        return []

    candidate_text = ""
    for entry in directory.entries:
        candidate_text += f"""
Round {entry.round_id}:
  Scene: {entry.free_text}
  Tags: {', '.join(entry.tags[:10])}
  Anchors: {', '.join(entry.anchors[:5])}
"""

    prompt = f"""You are a memory scheduler for a multimodal conversation system.
Given a question and candidate conversation rounds, select the TOP {top_n} most relevant round IDs.

Question: {question}

Candidates:
{candidate_text}

Return ONLY a JSON array of round IDs, e.g. ["S1:R1", "S5:R3", "S8:R2"]."""

    try:
        response = text_callable(prompt)
        # Try JSON parse; some models return within ```json``` blocks
        import re
        json_match = re.search(r'(\[.*?\])', response.strip(), re.DOTALL)
        raw = json.loads(json_match.group(1) if json_match else response)
        if isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
        selected = [str(r) for r in raw if isinstance(r, str)]
        return selected[:top_n]
    except Exception as exc:
        log.warning("step2_llm_routing failed: %s", exc)
        return [e.round_id for e in directory.entries[:top_n]]


def step3_vlm_grounding(
    question: str,
    round_ids: List[str],
    gist_index: GistIndex,
    image_index: dict,
    vlm_callable: "VLMCallable",
) -> str:
    """VLM answers the question by looking at original images + gist evidence.

    Args:
        question: The user's question.
        round_ids: Selected round IDs to examine.
        gist_index: GistIndex for scene descriptions.
        image_index: Dict mapping round_id -> image_path.
        vlm_callable: A callable(system_prompt, user_text, [image_paths]) -> str.

    Returns:
        Answer text from VLM.
    """
    evidence_lines: List[str] = []
    image_paths: List[str] = []

    for rid in round_ids:
        gist = gist_index.get(rid)
        if gist:
            evidence_lines.append(f"Round {rid}: {gist.free_text}")
        img_path = image_index.get(rid)
        if img_path:
            image_paths.append(img_path)

    evidence = "\n".join(evidence_lines)

    user_text = f"""Here are {len(image_paths)} historical image(s) from the conversation.

{evidence}

Question: {question}
Answer concisely based on the images and descriptions."""

    return vlm_callable("", user_text, image_paths)
