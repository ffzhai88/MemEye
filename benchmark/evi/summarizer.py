"""
EVI v2: Session-level summarizer for LLM directory retrieval.

Generates a comprehensive summary for each session (the "table of contents"
entry) so that the retriever can first narrow down candidate sessions via
coarse-grained LLM directory lookup, then perform fine-grained vector search
within the selected sessions.

Results are cached to a single JSON file (session_id -> {summary, date}),
so repeated runs do not re-invoke the VLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .vlm import VLMCallable

from .schemas import SessionNode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SESSION_SUMMARY_PROMPT = """You are a conversation analyst. Review the conversation session below and write a ONE-SENTENCE summary that captures its core theme and key entities.

Focus on: what was the main topic, what specific items/brands/people/places were discussed, what key facts were established, and what images showed.

Requirements:
- ONE sentence only (20-30 words ideal, never more than 40).
- Dense with concrete keywords — every word should help a later directory lookup.
- No filler, no greetings, no meta-commentary.
- If images were shared, mention what they depicted.
- Write as a factual statement, not a question."""


# ---------------------------------------------------------------------------
# Cache — single JSON file: {session_id: {"summary": str, "date": str}}
# ---------------------------------------------------------------------------

_CACHE: Optional[Dict[str, Dict[str, str]]] = None
_CACHE_PATH: Optional[Path] = None
_CACHE_DIRTY = False


def _cache_path() -> Path:
    global _CACHE_PATH
    if _CACHE_PATH is not None:
        return _CACHE_PATH
    d = Path.home() / ".cache" / "evi_session_summaries"
    d.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH = d / "summaries_cache.json"
    return _CACHE_PATH


def _load_cache() -> Dict[str, Dict[str, str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    p = _cache_path()
    if p.exists():
        try:
            _CACHE = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}
    else:
        _CACHE = {}
    return _CACHE


def _flush_cache() -> None:
    global _CACHE, _CACHE_DIRTY
    if not _CACHE_DIRTY:
        return
    p = _cache_path()
    try:
        p.write_text(json.dumps(_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
        _CACHE_DIRTY = False
    except Exception as exc:
        log.warning("  [SESSION SUMMARIZER] Cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_session(
    session_id: str,
    session_text: str,
    date: str,
    vlm_callable: "VLMCallable",
    use_cache: bool = True,
) -> Optional[SessionNode]:
    """Generate (or load from cache) a comprehensive summary for one session.

    The cache is a single JSON file keyed by session_id, so the directory
    retriever can later load *all* summaries in one shot without calling
    the VLM again.

    Args:
        session_id: Unique session identifier.
        session_text: All dialogue rounds in this session, concatenated.
        date: Session date string.
        vlm_callable: (system_prompt, user_text, [image_paths]) -> str.
        use_cache: Whether to check / persist the disk cache.

    Returns:
        SessionNode with the VLM-generated summary, or None on failure.
    """
    # ---- Cache lookup ----
    if use_cache:
        cache = _load_cache()
        cached = cache.get(session_id)
        if cached is not None and isinstance(cached, dict) and "summary" in cached:
            log.info("  [SESSION SUMMARIZER] Cache HIT for %s (%d chars)", session_id, len(cached["summary"]))
            return SessionNode(session_id=session_id, summary=cached["summary"], date=cached.get("date", date))

    log.info("  [SESSION SUMMARIZER] Generating summary for %s ...", session_id)

    # ---- VLM call ----
    truncated = session_text[:10000]
    summary = vlm_callable(SESSION_SUMMARY_PROMPT, truncated, []).strip()

    if not summary:
        log.warning("  [SESSION SUMMARIZER] VLM returned empty for %s", session_id)
        return None

    log.info("  [SESSION SUMMARIZER]   -> summary=%d chars", len(summary))

    # ---- Persist to cache (keyed by session_id) ----
    if use_cache:
        cache = _load_cache()
        cache[session_id] = {"summary": summary, "date": date}
        global _CACHE_DIRTY
        _CACHE_DIRTY = True
        _flush_cache()

    return SessionNode(session_id=session_id, summary=summary, date=date)


def load_all_summaries() -> Dict[str, Dict[str, str]]:
    """Load the full session-summary cache.

    Returns a dict of ``{session_id: {"summary": str, "date": str}}``,
    or an empty dict if the cache does not exist.

    This is used by the LLM directory retriever to build a "table of
    contents" prompt listing every session the LLM can choose from.
    """
    return _load_cache()
