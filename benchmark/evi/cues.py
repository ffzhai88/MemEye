from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from ._utils import extract_json, retry_vlm_call

if False:  # pragma: no cover
    from .vlm import VLMCallable

log = logging.getLogger(__name__)

_PROMPT_VERSION = "retrieval_cues_v1"
_CACHE_DIR: Optional[str] = None

RETRIEVAL_CUE_SYSTEM_PROMPT = """Extract retrieval cues from the question for searching episodic memory.

A retrieval cue is a short phrase from the question that names or describes something likely to appear in a memory episode.

Return ONLY this JSON object:
{
  "retrieval_cues": ["short phrase from the question", "..."]
}

Guidelines:
- Use the question's original words whenever possible.
- Keep meaningful phrases together, such as "boxy light-blue compact car" or "solid red background".
- Do not paraphrase into broader concepts unless the question phrase is too awkward to search.
- Do not include the whole question.
"""


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_RETRIEVAL_CUE_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_retrieval_cues"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _cache_key(question_stem: str, cache_namespace: str, max_cues: int) -> str:
    raw = json.dumps(
        {
            "version": _PROMPT_VERSION,
            "cache_namespace": cache_namespace,
            "question_stem": question_stem,
            "max_cues": max_cues,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _shorten(text: object, max_chars: int = 1000) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... <truncated {len(value) - max_chars} chars>"


def _as_cue_texts(value: object, question_stem: str) -> List[str]:
    if isinstance(value, dict):
        value = value.get("retrieval_cues", [])
    if not isinstance(value, list):
        return []
    question_key = " ".join(str(question_stem or "").lower().split())
    out: List[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text", item.get("cue", item.get("phrase", "")))).strip()
        else:
            text = str(item).strip()
        text = " ".join(text.split())
        if not text:
            continue
        key = text.lower()
        if key == question_key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def extract_retrieval_cues(
    question_stem: str,
    vlm_callable: "VLMCallable",
    *,
    max_cues: int = 4,
    use_cache: bool = True,
    cache_namespace: str = "default",
) -> List[str]:
    """Extract short memory-groundable retrieval cues from a question stem."""
    question_stem = str(question_stem or "").strip()
    max_cues = max(0, int(max_cues or 0))
    if not question_stem or max_cues <= 0:
        return []

    cache_file: Optional[Path] = None
    if use_cache:
        cache_file = Path(_cache_dir()) / f"{_cache_key(question_stem, cache_namespace, max_cues)}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                cues = _as_cue_texts(data, question_stem)[:max_cues]
                log.info("QDMO-EVI retrieval cue cache HIT cues=%s", cues)
                return cues
            except Exception as exc:
                log.warning("QDMO-EVI retrieval cue cache read failed: %s", exc)

    user_text = "\n".join(["Question:", question_stem])
    raw = retry_vlm_call(
        lambda: vlm_callable(RETRIEVAL_CUE_SYSTEM_PROMPT, user_text, []),
        label="retrieval-cue extraction",
    )
    if not raw:
        log.warning("QDMO-EVI retrieval cue extraction returned empty response")
        return []

    parsed = extract_json(raw) or {}
    cues = _as_cue_texts(parsed, question_stem)[:max_cues]
    if not cues:
        log.warning("QDMO-EVI retrieval cue extraction produced no cues raw=%s", _shorten(raw))
        return []

    if use_cache and cache_file is not None:
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "version": _PROMPT_VERSION,
                        "cache_namespace": cache_namespace,
                        "question_stem": question_stem,
                        "retrieval_cues": cues,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("QDMO-EVI retrieval cue cache write failed: %s", exc)

    log.info("QDMO-EVI retrieval cues extracted: %s", cues)
    return cues
