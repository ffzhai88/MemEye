from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

from ._utils import guess_mime, retry_vlm_call

log = logging.getLogger(__name__)

VLMCallable = Callable[[str, str, List[str]], str]

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


def _vlm_cache_key(namespace: str, system_prompt: str, user_text: str, images: List[str]) -> str:
    raw = json.dumps(
        {
            "namespace": namespace,
            "system_prompt": system_prompt,
            "user_text": user_text,
            "images": images,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def with_disk_cache(fn: VLMCallable, namespace: str) -> VLMCallable:
    def _cached(system_prompt: str, user_text: str, images: List[str]) -> str:
        key = _vlm_cache_key(namespace, system_prompt, user_text, images)
        cache_file = Path(_cache_dir()) / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                response = str(data.get("response", ""))
                if response:
                    log.info("  [VLM CACHE] HIT (%d chars, %d images)", len(user_text), len(images))
                    return response
                log.info("  [VLM CACHE] empty entry ignored")
            except Exception:
                pass

        log.info("  [VLM CALL] sending (%d chars, %d images) ...", len(user_text), len(images))
        result = fn(system_prompt, user_text, images)
        log.info("  [VLM CALL] done (%d chars response)", len(result))
        if not result:
            log.warning("  [VLM CACHE] skip empty response")
            return result
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "namespace": namespace,
                        "image_paths": images,
                        "prompt": system_prompt[:500],
                        "user_text": user_text[:1000],
                        "response": result,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        return result

    return _cached


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = guess_mime(path)
    return f"data:{mime};base64,{b64}"


def make_openai_vlm(
    api_key: str,
    base_url: str,
    model: str,
    max_new_tokens: int,
    timeout: int,
) -> VLMCallable:
    import openai

    if not api_key:
        raise RuntimeError("EVI VLM requires an API key for openai_api provider")
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _call(system_prompt: str, user_text: str, images: List[str]) -> str:
        content: list = [{"type": "text", "text": user_text}]
        for path in images:
            try:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _encode_image(path), "detail": "high"},
                })
            except Exception as exc:
                log.warning("VLM: cannot read %s: %s", path, exc)
        token_param = (
            {"max_completion_tokens": max_new_tokens}
            if any(model.startswith(prefix) for prefix in ("gpt-5", "o3", "o4"))
            else {"max_tokens": max_new_tokens}
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a concise visual reasoning assistant."},
                {"role": "user", "content": content},
            ],
            **token_param,
        }
        if not any(model.startswith(prefix) for prefix in ("gpt-5", "o3", "o4")):
            payload["temperature"] = 0.0

        def _do_call() -> str:
            resp = client.chat.completions.create(**payload)
            return resp.choices[0].message.content or ""

        return retry_vlm_call(_do_call, label=f"vlm {model}")

    return _call


def make_vlm_callable(model_cfg: dict) -> VLMCallable:
    provider = str(model_cfg.get("provider", "openai_api"))
    model = str(model_cfg.get("model", "gpt-4o"))
    max_new_tokens = int(model_cfg.get("max_new_tokens", 1024) or 1024)
    timeout = int(model_cfg.get("timeout", 90) or 90)

    if provider != "openai_api":
        raise ValueError(f"EVI currently supports provider=openai_api, got {provider!r}")

    api_key = str(model_cfg.get("api_key", ""))
    api_key_env = str(model_cfg.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        api_key = os.environ.get(api_key_env, "")
    base_url = str(model_cfg.get("base_url", "https://api.openai.com/v1"))
    namespace = json.dumps(
        {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "max_new_tokens": max_new_tokens,
            "timeout": timeout,
            "cache_version": "vlm_v2",
        },
        sort_keys=True,
    )
    log.info("VLM adapter: openai_api model=%s base_url=%s", model, base_url)
    return with_disk_cache(
        make_openai_vlm(api_key, base_url, model, max_new_tokens, timeout),
        namespace=namespace,
    )
