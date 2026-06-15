"""
EVI v2: Phase 1 — Multi-dimensional image description generator.

Two separate VLM calls per image:
  1. Image-only: generates multiple free-form descriptions from different angles
     (background, objects, colors, layout, details — VLM decides count)
  2. Context-aware: generates one description based on the full conversation context
     (user+assistant of this round + all prior rounds in this session)

Both results merged into a single JSON and cached.
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

from .schemas import ImageNode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR: Optional[str] = None


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
# Prompt 1: image-only, free-form multiple descriptions
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
# Prompt 2: context-aware, single description
# ---------------------------------------------------------------------------

CONTEXT_PROMPT = """You are a visual analyst. Given an image and its conversation context, describe what this image means in the broader conversation.

Focus on:
- Why did the user share this image at this point in the conversation?
- What aspect of the image is being discussed or compared?
- How does this image relate to earlier images or topics in the session?

Write 2-4 sentences. The description should capture the IMAGE'S ROLE in the conversation, not just its visual content.
It will be used for semantic search, so include both conversational keywords and visual references.
Do NOT output JSON — just write the description directly."""


# ---------------------------------------------------------------------------
# Extraction
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


def describe_image(
    image_path: str,
    user_text: str,
    prior_rounds_text: str,  # concatenated text of all prior rounds in this session
    vlm_callable: "VLMCallable",
    use_cache: bool = True,
) -> Optional[ImageNode]:
    """Two-call image description extraction.

    Call 1 — Image-only: VLM looks at the image alone, generates N free-form descriptions.
    Call 2 — Context-aware: VLM reads the conversation context, generates one description.

    Both merged into one cache entry. Each description becomes a separate embedding vector.

    Args:
        image_path: Absolute path to the image.
        user_text: User utterance in the current round.
        prior_rounds_text: All prior rounds in this session, concatenated.
        vlm_callable: (system_prompt, user_text, [image_paths]) -> str.
        use_cache: Whether to check/save disk cache.

    Returns:
        ImageNode with image_descs (list) and context_desc (str), or None on failure.
    """
    # ---- Try cache ----
    ck = _cache_key(image_path)
    cache_file = Path(_cache_dir()) / f"{ck}.json"
    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            log.info("  [DESCRIBE] Cache HIT for %s", image_path)
            return ImageNode(
                round_id="",
                session_id="",
                image_path=image_path,
                image_descs=data.get("image_descriptions", []),
                context_desc=data.get("context_description", ""),
            )
        except Exception:
            pass

    # ============================================================
    # Call 1: Image-only — generate multiple free-form descriptions
    # ============================================================
    log.info("  [DESCRIBE1] Call 1 (image-only) for %s ...", image_path)
    raw1 = vlm_callable(IMAGE_ONLY_PROMPT, "Describe this image.", [image_path])
    image_descs: List[str] = []
    if raw1:
        try:
            data1 = _extract_json(raw1) or {}
            image_descs = data1.get("descriptions", [])
            if isinstance(image_descs, str):
                image_descs = [image_descs]
        except Exception as exc:
            log.warning("  [DESCRIBE1] Call 1 JSON parse failed: %s", exc)

    if not image_descs:
        log.warning("  [DESCRIBE1] Call 1 returned no descriptions for %s", image_path)

    log.info("  [DESCRIE1]   -> %d descriptions generated", len(image_descs))

    # ============================================================
    # Call 2: Context-aware — describe the image's role in conversation
    # ============================================================
    ctx_parts = []
    if prior_rounds_text:
        ctx_parts.append(f"--- Prior conversation ---\n{prior_rounds_text}")
    ctx_parts.append(f"--- Current round ---\n{user_text}")
    context_prompt = "\n\n".join(ctx_parts)

    log.info("  [DESCRIBE] Call 2 (context-aware) for %s ...", image_path)
    context_desc = vlm_callable(CONTEXT_PROMPT, context_prompt, [image_path]).strip()

    if not context_desc:
        log.warning("  [DESCRIBE] Call 2 returned empty for %s", image_path)

    log.info("  [DESCRIBE]   -> context=%d chars", len(context_desc))

    # ============================================================
    # Save merged cache
    # ============================================================
    if use_cache:
        merged = {
            "image_path": image_path,
            "image_descriptions": image_descs,
            "context_description": context_desc,
        }
        try:
            cache_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("  [DESCRIBE] Cache write failed: %s", exc)

    return ImageNode(
        round_id="",
        session_id="",
        image_path=image_path,
        image_descs=image_descs,
        context_desc=context_desc,
    )
