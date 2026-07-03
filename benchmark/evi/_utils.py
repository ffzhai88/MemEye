"""Shared utility functions for the evi module."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, List, Optional, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., str])


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """Parse a JSON object from raw LLM output (handles markdown fences)."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

def retry_vlm_call(
    fn: Callable[[], str],
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff: float = 2.0,
    label: str = "VLM call",
) -> str:
    """Call *fn* with exponential-backoff retry on failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            if result:
                return result
            # Empty string is treated as failure
        except Exception as exc:
            last_exc = exc
            log.warning("  [RETRY] %s attempt %d/%d failed: %s", label, attempt + 1, max_retries, exc)

        if attempt < max_retries:
            delay = base_delay * (backoff ** attempt)
            log.info("  [RETRY] %s retrying in %.1fs ...", label, delay)
            time.sleep(delay)

    log.error("  [RETRY] %s exhausted %d retries, giving up", label, max_retries)
    return ""


# ---------------------------------------------------------------------------
# MIME type helpers
# ---------------------------------------------------------------------------

_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def guess_mime(path: str) -> str:
    """Guess MIME type from file extension."""
    from pathlib import Path

    suffix = Path(path).suffix.lower()
    return _MIME_MAP.get(suffix, "image/jpeg")
