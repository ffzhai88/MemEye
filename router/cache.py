from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseRouter

log = logging.getLogger(__name__)
QA_CACHE_VERSION = "qa_answer_cache_v1"


def _image_fingerprint(raw_path: Any) -> Dict[str, Any]:
    path = Path(str(raw_path or "")).expanduser()
    fingerprint: Dict[str, Any] = {"path": str(path.resolve(strict=False))}
    try:
        stat = path.stat()
        fingerprint.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    except OSError:
        fingerprint["missing"] = True
    return fingerprint


def _cache_root() -> Path:
    root = Path(
        os.environ.get(
            "MEMEYE_QA_CACHE_DIR",
            Path.home() / ".cache" / "memeye_qa_answers",
        )
    ).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


class CachedAnswerRouter(BaseRouter):
    """Persistent exact-input cache for final multimodal QA calls."""

    def __init__(self, router: BaseRouter, enabled: bool = True) -> None:
        self.router = router
        self.enabled = bool(enabled)
        self.last_usage: Dict[str, int] = {}
        self.last_cache_hit = False
        self.last_cache_key = ""
        self.last_cached_usage: Dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.router, name)

    def _namespace(self) -> Dict[str, Any]:
        return {
            "router_class": type(self.router).__name__,
            "model": str(getattr(self.router, "model", "")),
            "model_path": str(getattr(self.router, "model_path", "")),
            "base_url": str(getattr(self.router, "base_url", "")),
            "max_new_tokens": int(getattr(self.router, "max_new_tokens", 0) or 0),
            "max_images": int(getattr(self.router, "max_images", 0) or 0),
            "system_prompt": str(getattr(self.router, "system_prompt", "")),
        }

    def _key_payload(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        question_images: Optional[List[str]],
    ) -> Dict[str, Any]:
        history = []
        for message in history_messages:
            history.append(
                {
                    "role": str(message.get("role", "user")),
                    "text": str(message.get("text", "")),
                    "round_id": str(message.get("round_id", "")),
                    "images": [
                        _image_fingerprint(path)
                        for path in message.get("images", []) or []
                    ],
                }
            )
        return {
            "version": QA_CACHE_VERSION,
            "namespace": self._namespace(),
            "history": history,
            "question": str(question),
            "question_images": [
                _image_fingerprint(path) for path in question_images or []
            ],
        }

    def answer(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        if not self.enabled:
            self.last_cache_hit = False
            answer = self.router.answer(history_messages, question, question_images)
            self.last_usage = dict(getattr(self.router, "last_usage", {}) or {})
            return answer

        payload = self._key_payload(history_messages, question, question_images)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        cache_file = _cache_root() / f"{key}.json"
        self.last_cache_key = key

        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                answer = str(cached["answer"])
                self.last_cached_usage = {
                    str(name): int(value)
                    for name, value in dict(cached.get("usage", {})).items()
                }
                self.last_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                self.last_cache_hit = True
                print(f"[QA-CACHE] hit key={key[:16]}")
                log.info("Final QA cache hit key=%s", key)
                return answer
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("Ignoring invalid final QA cache entry: %s", cache_file)

        answer = self.router.answer(history_messages, question, question_images)
        self.last_usage = dict(getattr(self.router, "last_usage", {}) or {})
        self.last_cached_usage = {}
        self.last_cache_hit = False
        cache_payload = {
            "version": QA_CACHE_VERSION,
            "key": key,
            "answer": str(answer),
            "usage": self.last_usage,
        }
        temporary = cache_file.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(cache_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, cache_file)
        print(f"[QA-CACHE] miss key={key[:16]} stored=true")
        log.info("Final QA cache miss key=%s stored=%s", key, cache_file)
        return str(answer)