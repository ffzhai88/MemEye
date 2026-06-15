"""
EVI v2: VLM callable adapter with disk cache.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

# Callable type: (system_prompt, user_text, [image_paths]) -> response_text
VLMCallable = Callable[[str, str, List[str]], str]

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR: Optional[str] = None


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = os.environ.get(
            "EVI_VLM_CACHE_DIR",
            str(Path.home() / ".cache" / "evi_vlm"),
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _vlm_cache_key(system_prompt: str, user_text: str, images: List[str]) -> str:
    raw = f"{system_prompt}::|::{user_text}::|::{','.join(images)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------

def with_disk_cache(fn: VLMCallable) -> VLMCallable:
    """Wrap a VLMCallable with disk cache. Cache key = prompt + text + image paths."""
    def _cached(system_prompt: str, user_text: str, images: List[str]) -> str:
        key = _vlm_cache_key(system_prompt, user_text, images)
        cache_file = Path(_cache_dir()) / f"{key}.json"

        # Check cache
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                log.info("  [VLM CACHE] HIT (%d chars, %d images)", len(user_text), len(images))
                return data.get("response", "")
            except Exception:
                pass

        log.info("  [VLM CALL] sending (%d chars, %d images) ...", len(user_text), len(images))
        result = fn(system_prompt, user_text, images)
        log.info("  [VLM CALL] done (%d chars response)", len(result))

        # Save to cache
        try:
            cache_file.write_text(
                json.dumps({
                    "image_paths": images,
                    "prompt": system_prompt[:200] + ("..." if len(system_prompt) > 200 else ""),
                    "user_text": user_text[:300] + ("..." if len(user_text) > 300 else ""),
                    "response": result,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        return result
    return _cached


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def make_openai_vlm(api_key: str, base_url: str, model: str) -> VLMCallable:
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _call(system_prompt: str, user_text: str, images: List[str]) -> str:
        content: list = [{"type": "text", "text": user_text}]
        for path in images:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })
            except Exception as exc:
                log.warning("VLM: cannot read %s: %s", path, exc)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            log.warning("VLM call failed: %s", exc)
            return ""
    return _call


# ---------------------------------------------------------------------------
# Factory (always wraps with disk cache)
# ---------------------------------------------------------------------------

def make_vlm_callable(model_cfg: dict) -> VLMCallable:
    provider = str(model_cfg.get("provider", "openai_api"))
    model = str(model_cfg.get("model", "gpt-4o"))

    raw_vlm: Optional[VLMCallable] = None

    if provider == "openai_api":
        api_key = str(model_cfg.get("api_key", ""))
        api_key_env = str(model_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            import os
            api_key = os.environ.get(api_key_env, "")
        base_url = str(model_cfg.get("base_url", "https://api.openai.com/v1"))
        log.info("VLM adapter: openai_api model=%s base_url=%s", model, base_url)
        raw_vlm = make_openai_vlm(api_key, base_url, model)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Wrap with disk cache
    cached = with_disk_cache(raw_vlm)
    log.info("VLM cache enabled: %s", _cache_dir())
    return cached
