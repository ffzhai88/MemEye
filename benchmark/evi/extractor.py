"""
EVI: Phase 1 — VLM-based three-level extraction.
One VLM call per image-bearing round.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .vlm import VLMCallable

from .schemas import (
    AnchorData,
    ExtractionResult,
    GistData,
    SceneAttributes,
    TagData,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (fixed, not task-adaptive)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a visual scene analyzer. Given an image and its conversation context, extract three levels of information.

LEVEL 1 — GIST (scene summary):
Write 2-3 sentences describing the scene naturally. Cover:
- What is the setting and overall composition?
- What is the background color and lighting like?
- What are the main visual elements?
This text will be used for semantic search, so make it descriptive but concise.

Also extract coarse scene attributes:
- "background_color": main background color (standard names: red, blue, green, yellow, white, black, purple, orange, teal, gray, brown, beige, pink, dark_blue, light_blue, dark_green)
- "lighting": bright | dim | dark | mixed | natural | artificial
- "setting": type of place (outdoor, indoor, cave, office, lab, game_ui, dining_table, street, museum_gallery, restoration_table, dashboard_ui, chat_ui, kitchen, etc.)
- "mood": warm | cold | dramatic | calm | playful | tense | professional | cozy

LEVEL 2 — TAGS (salient object labels):
For each VISUALLY SALIENT object, character, or element, output a flat tag.
A tag is a short list: [noun, color_or_key_feature, rough_position, optional_count]

Rules:
- Only list things that are clearly visible (not tiny background details)
- Use short, precise nouns: "card", "coffee_cup", "dinosaur", "logo", "kpi_panel", "ad", "person", "bottle", "can", "burger", "fries", "furniture", "window", "door", "table", "chair", "lamp", "plant", "vehicle", "building", "road_sign", "microtube", "label", "paint_swatch"
- Color should be a standard name. If not applicable, use the most distinctive feature.
- Position: "left", "right", "center", "top", "bottom", "top_left", "bottom_right", "background", "foreground"
- count: only include if there are multiple indistinguishable items of the same type (e.g. 3 cards, 5 eggs).
- Keep total tags under 15 per image. Do NOT include tiny details.

Examples:
["card", "yellow", "bottom", 3]
["logo", "red_and_white", "top_left"]
["coffee_cup", "blue", "center"]
["dinosaur", "purple", "left"]
["ad", "coca_cola", "center"]

LEVEL 3 — ANCHORS (from conversation text):
Extract explicit names, labels, or identifiers from the conversation that help identify this round. These are NOT from the image — they are from the user message and assistant response.

- "explicit_labels": list of proper names (brands: Coca-Cola, Pepsi, McDonald's, etc.; character names; player IDs like Player 0; room names; episode numbers; person names)
- "explicit_topic": what this round is about (short phrase, e.g. "McDonald's healthy positioning campaign")
- "user_intent": what the user is doing (e.g. comparing, documenting, tracking progress, asking opinion, flagging a detail)

Output JSON schema:
{
  "gist": {
    "free_text": str,
    "scene_attributes": {
      "background_color": str,
      "lighting": str,
      "setting": str,
      "mood": str
    }
  },
  "tags": [[noun, color, position, optional_count], ...],
  "anchors": {
    "explicit_labels": [str, ...],
    "explicit_topic": str,
    "user_intent": str
  }
}"""


# ---------------------------------------------------------------------------
# Extraction routines
# ---------------------------------------------------------------------------

def _parse_raw(raw: dict) -> ExtractionResult:
    """Convert raw VLM JSON output into typed ExtractionResult."""
    result = ExtractionResult(round_id="", image_path="")

    # Gist
    gist_raw = raw.get("gist", {})
    attrs_raw = gist_raw.get("scene_attributes", {})
    result.gist = GistData(
        free_text=gist_raw.get("free_text", ""),
        scene_attributes=SceneAttributes(
            background_color=attrs_raw.get("background_color", ""),
            lighting=attrs_raw.get("lighting", ""),
            setting=attrs_raw.get("setting", ""),
            mood=attrs_raw.get("mood", ""),
        ),
    )

    # Tags
    for tag_raw in raw.get("tags", []):
        if not isinstance(tag_raw, list) or len(tag_raw) < 2:
            continue
        tag = TagData(
            noun=str(tag_raw[0]),
            color=str(tag_raw[1]) if len(tag_raw) > 1 else "",
            position=str(tag_raw[2]) if len(tag_raw) > 2 else "",
            count=int(tag_raw[3]) if len(tag_raw) > 3 and tag_raw[3] is not None else None,
        )
        result.tags.append(tag)

    # Anchors
    anc_raw = raw.get("anchors", {})
    result.anchors = AnchorData(
        explicit_labels=[str(l) for l in anc_raw.get("explicit_labels", []) if l],
        explicit_topic=str(anc_raw.get("explicit_topic", "")),
        user_intent=str(anc_raw.get("user_intent", "")),
    )

    return result


def extract_with_vlm(
    image_path: str,
    user_text: str,
    assistant_text: str,
    vlm_callable: "VLMCallable",
) -> ExtractionResult:
    """Call VLM once for an image-bearing round.

    Args:
        image_path: Absolute path to the image file.
        user_text: The user's utterance in this round.
        assistant_text: The assistant's response in this round.
        vlm_callable: A callable(system_prompt, user_text, [image_paths]) -> str.

    Returns:
        ExtractionResult with parsed fields. On failure, returns empty result.
    """
    user_message = f"User: {user_text}\nAssistant: {assistant_text}"

    raw_text = vlm_callable(SYSTEM_PROMPT, user_message, [image_path])
    if not raw_text:
        return ExtractionResult(round_id="", image_path=image_path)

    try:
        import json
        raw = json.loads(raw_text)
    except Exception as exc:
        log.warning("extract_with_vlm: JSON parse failed for %s: %s", image_path, exc)
        return ExtractionResult(round_id="", image_path=image_path)

    result = _parse_raw(raw)
    result.round_id = ""  # caller sets this
    result.image_path = image_path
    return result
