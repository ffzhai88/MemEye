"""
EVI v2: Phase 1 — Multi-dimensional image description + micro knowledge graph.

Three VLM calls per image:
  1. Image-only:      generates multiple free-form descriptions (unchanged)
  2. Context-aware:   generates a semantic NAME for the image + role-in-conversation description
  3. Fact extraction: extracts structured triples centered around the image name

All three results merged into a single JSON and cached (v2 format).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .vlm import VLMCallable

from .schemas import Fact, ImageNode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR: Optional[str] = None
_CACHE_VERSION = 2


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_descriptions"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _cache_key(image_path: str) -> str:
    return hashlib.sha256(image_path.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Prompt 1: image-only, free-form multiple descriptions (unchanged)
# ---------------------------------------------------------------------------

IMAGE_ONLY_PROMPT = """You are a visual analyst. Look at this image and generate multiple short, self-contained descriptions from different angles.

Output a JSON object with a single field "descriptions": a list of strings.
Each string is a 2-3 sentence description covering ONE specific aspect of the image.

Cover as many distinct aspects as are relevant. Possible aspects include but are not limited to:
- Overall scene composition and layout
- Background color, lighting, atmosphere
- Main objects/characters/people, their colors and attributes
- Text, labels, numbers, or branding visible
- Spatial arrangement (what is where)
- Style, mood, or visual treatment
- Any distinctive details or anomalies

IMPORTANT: Let the image content determine how many descriptions you write.
A simple image might need 2-3, a complex one might need 6-8.
Each description must be self-contained (searchable in isolation) and focus on ONE angle.

Example output for a product ad image:
{
  "descriptions": [
    "The background is solid red with no other visual elements, drawing full attention to the centered product.",
    "A Coca-Cola bottle is positioned in the center of the frame, featuring the classic red label and contoured glass shape.",
    "The lighting is bright and even, giving the product a clean, polished look against the red backdrop.",
    "Small white text at the bottom reads 'Share a Coke with...' suggesting a personalization campaign."
  ]
}"""


# ---------------------------------------------------------------------------
# Prompt 2: context-aware — image name + role description (JSON output)
#
# image_name captures what this image IS (its identity/type), not what is in it.
# context_description captures the image's role in the broader conversation.
# ---------------------------------------------------------------------------

CONTEXT_PROMPT = """You are a visual analyst. Given an image and its conversation context, identify what this image is and describe its role in the conversation.

Output a JSON object with exactly two fields:
1. "image_name": a short name (1-8 words) that captures what this image IS — its identity, type, or subject.
   - This is the IMAGE'S NAME, not a description of what's in it.
   - Think of it as a photo album label: concise, distinctive within the session.
   - Examples: "Red burger on a river", "Vintage car advertisement poster", "User's profile photo"
   - Do NOT list elements inside the image — name the image itself.

2. "context_description": 2-4 sentences about the image's role in the conversation.
   - Why did the user share this image at this point?
   - What aspect of the image is being discussed or compared?
   - How does this image relate to earlier images or topics in the session?

Example output:
{
  "image_name": "Red burger on a river",
  "context_description": "The user shared this image to ask about the composition and positioning of the burger in relation to the river behind it. The burger is the main subject and its bright red color contrasts with the green river bank."
}

Return ONLY valid JSON — no markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# Prompt 3: fact extraction — build a micro knowledge graph centered on image_name
# ---------------------------------------------------------------------------

FACT_PROMPT = """You are a knowledge graph builder. Given an image, its name, and conversation context, extract structured factual triples (subject, predicate, object) that form a micro knowledge graph.

The image's name is: {image_name}

Output a JSON object with a single field "facts": a list of objects, each with "subject", "predicate", and "object" strings.

Extract facts across three levels:

1. Image-level facts — use the image name as the subject:
   (image_name, depicts, "what the image shows overall")
   (image_name, background, "scene / background description")
   (image_name, overall_color, "dominant color(s)")
   (image_name, scene_type, "indoor / outdoor / abstract / ...")
   (image_name, has_object, "object name")

2. Entity-level facts — use entity names (short nouns) as the subject:
   (entity, color, "value")
   (entity, position, "top / center / left / ...")
   (entity, size, "large / small / ...")
   (entity, count, "number")

3. Relation-level facts — connect entities via relations:
   (entity1, is_on_top_of, entity2)
   (entity1, next_to, entity2)
   (entity1, wearing, entity2)
   (entity1, holding, entity2)

Requirements:
- Be concrete and visually grounded. Only extract what you can see.
- Each triple must be atomic — one fact per triple.
- subject and object should be short noun phrases.
- Use the exact image_name as-is for image-level facts.
- Extract 3-10 facts depending on complexity. Fewer is fine for simple images.
- If an entity appears in multiple facts, reuse the same entity name for consistency.

Return ONLY valid JSON — no markdown fences, no extra text.

Example output for a burger image named "Red burger on a river":
{{
  "facts": [
    {{"subject": "Red burger on a river", "predicate": "depicts", "object": "a burger on a wooden board above a river"}},
    {{"subject": "Red burger on a river", "predicate": "background", "object": "a winding river with green banks"}},
    {{"subject": "burger", "predicate": "color", "object": "red"}},
    {{"subject": "burger", "predicate": "position", "object": "center of frame"}},
    {{"subject": "burger", "predicate": "is_on_top_of", "object": "river"}},
    {{"subject": "river", "predicate": "appearance", "object": "narrow and winding"}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from VLM response, handling markdown fences and surrounding text."""
    import re
    # Try ```json ... ``` block first
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try raw JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try { ... } block (greedy, last resort)
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def _build_context_text(prior_rounds_text: str, user_text: str) -> str:
    """Build the conversation context block for VLM prompts."""
    parts = []
    if prior_rounds_text:
        parts.append(f"--- Prior conversation ---\n{prior_rounds_text}")
    parts.append(f"--- Current round ---\n{user_text}")
    return "\n\n".join(parts)


def _is_valid_cache(data: dict) -> bool:
    """Check if cached data is in v2 format (with image_name and facts)."""
    if not isinstance(data, dict):
        return False
    if data.get("version") != _CACHE_VERSION:
        return False
    # Must have all fields from the three calls
    if "image_descriptions" not in data:
        return False
    if "image_name" not in data or "context_description" not in data:
        return False
    if "facts" not in data:
        return False
    return True


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------


def describe_image(
    image_path: str,
    user_text: str,
    prior_rounds_text: str,  # concatenated text of all prior rounds in this session
    vlm_callable: "VLMCallable",
    use_cache: bool = True,
) -> Optional[ImageNode]:
    """Three-phase image description extraction with micro-KG.

    Call 1 — Image-only:      VLM looks at the image alone, generates N free-form descriptions.
    Call 2 — Context-aware:    VLM reads conversation context, outputs image_name + context_description.
    Call 3 — Fact extraction:  VLM reads image + context + image_name, outputs structured triples.

    All three results merged into one cache entry (v2 format with version field).
    Cache files without version or with version < 2 are regenerated.

    Args:
        image_path: Absolute path to the image.
        user_text: User utterance in the current round.
        prior_rounds_text: All prior rounds in this session, concatenated.
        vlm_callable: (system_prompt, user_text, [image_paths]) -> str.
        use_cache: Whether to check/save disk cache.

    Returns:
        ImageNode with image_name, image_descs, context_desc, facts; or None on total failure.
    """
    # ---- Try cache ----
    ck = _cache_key(image_path)
    cache_file = Path(_cache_dir()) / f"{ck}.json"
    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if _is_valid_cache(data):
                log.info("  [DESCRIBE] Cache HIT for %s", image_path)
                return ImageNode(
                    round_id="",
                    session_id="",
                    image_path=image_path,
                    image_name=data.get("image_name", ""),
                    image_descs=data.get("image_descriptions", []),
                    context_desc=data.get("context_description", ""),
                    facts=[Fact.from_dict(f) for f in data.get("facts", [])],
                )
        except Exception:
            pass

    # Build shared context text for Call 2 and Call 3
    context_text = _build_context_text(prior_rounds_text, user_text)

    # ============================================================
    # Call 1: Image-only — generate multiple free-form descriptions
    # ============================================================
    log.info("  [DESCRIBE] Call 1 (image-only) for %s ...", image_path)
    raw1 = vlm_callable(IMAGE_ONLY_PROMPT, "Describe this image.", [image_path])
    image_descs: List[str] = []
    if raw1:
        try:
            data1 = _extract_json(raw1) or {}
            image_descs = data1.get("descriptions", [])
            if isinstance(image_descs, str):
                image_descs = [image_descs]
        except Exception as exc:
            log.warning("  [DESCRIBE] Call 1 JSON parse failed: %s", exc)

    if not image_descs:
        log.warning("  [DESCRIBE] Call 1 returned no descriptions for %s", image_path)

    log.info("  [DESCRIBE]   -> %d descriptions generated", len(image_descs))

    # ============================================================
    # Call 2: Context-aware — generate image_name + context_description
    # ============================================================
    log.info("  [DESCRIBE] Call 2 (context-aware) for %s ...", image_path)
    raw2 = vlm_callable(CONTEXT_PROMPT, context_text, [image_path])

    image_name = ""
    context_desc = ""
    if raw2:
        try:
            data2 = _extract_json(raw2) or {}
            log.info("  [DESCRIBE] Call 2 raw response: %s", raw2)
            image_name = str(data2.get("image_name", "")).strip()
            context_desc = str(data2.get("context_description", "")).strip()
        except Exception as exc:
            log.warning("  [DESCRIBE] Call 2 JSON parse failed: %s", exc)

    if not image_name:
        log.warning("  [DESCRIBE] Call 2 returned empty image_name for %s", image_path)
    if not context_desc:
        log.warning("  [DESCRIBE] Call 2 returned empty context_description for %s", image_path)

    log.info("  [DESCRIBE]   -> image_name=%s, context=%d chars", image_name, len(context_desc))

    # ============================================================
    # Call 3: Fact extraction — build micro knowledge graph
    # ============================================================
    facts: List[Fact] = []
    if image_name:
        fact_prompt = FACT_PROMPT.format(image_name=image_name)
        log.info("  [DESCRIBE] Call 3 (fact extraction) for %s ...", image_path)
        raw3 = vlm_callable(fact_prompt, context_text, [image_path])

        if raw3:
            try:
                data3 = _extract_json(raw3) or {}
                raw_facts = data3.get("facts", [])
                if isinstance(raw_facts, list):
                    for f in raw_facts:
                        if isinstance(f, dict) and "subject" in f and "predicate" in f and "object" in f:
                            facts.append(Fact.from_dict(f))
            except Exception as exc:
                log.warning("  [DESCRIBE] Call 3 JSON parse failed: %s", exc)

        log.info("  [DESCRIBE]   -> %d facts extracted", len(facts))
    else:
        log.warning("  [DESCRIBE] Skipping Call 3 because image_name is empty")

    # ============================================================
    # Save merged cache (v2 format)
    # ============================================================
    if use_cache:
        merged = {
            "version": _CACHE_VERSION,
            "image_path": image_path,
            "image_descriptions": image_descs,
            "image_name": image_name,
            "context_description": context_desc,
            "facts": [{"subject": f.subject, "predicate": f.predicate, "object": f.object} for f in facts],
        }
        try:
            cache_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("  [DESCRIBE] Cache write failed: %s", exc)

    return ImageNode(
        round_id="",
        session_id="",
        image_path=image_path,
        image_name=image_name,
        image_descs=image_descs,
        context_desc=context_desc,
        facts=facts,
    )
