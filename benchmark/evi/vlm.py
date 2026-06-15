"""
EVI: VLM callable adapters for different model backends.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)

# Type alias: a callable that takes (system_prompt, user_text, image_paths)
# and returns response text.
VLMCallable = Callable[[str, str, List[str]], str]


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible adapter
# ---------------------------------------------------------------------------

def _make_openai_vlm(api_key: str, base_url: str, model: str) -> VLMCallable:
    """Create a VLMCallable backed by an OpenAI-compatible API."""
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
                log.warning("VLM: cannot read image %s: %s", path, exc)

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
# Qwen Local adapter
# ---------------------------------------------------------------------------

def _make_qwen_local_vlm(
    model_path: str,
    max_new_tokens: int = 512,
    max_time: float = 90,
) -> VLMCallable:
    """Create a VLMCallable backed by local Qwen model via QwenLocalRouter."""
    import importlib

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    qwen_vl_utils = importlib.import_module("qwen_vl_utils")

    AutoConfig = transformers.AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    model_type = getattr(cfg, "model_type", "unknown")

    model_cls = None
    if "qwen2" in model_type or "qwen3" in model_type or "qwen" in model_type:
        for candidate in ["Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"]:
            cls = getattr(transformers, candidate, None)
            if cls is not None:
                model_cls = cls
                break
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if model_cls is None:
        raise RuntimeError(f"Unsupported model_type: {model_type}")

    use_cuda = torch.cuda.is_available()
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else "cpu",
    )
    model.eval()
    processor = transformers.AutoProcessor.from_pretrained(model_path, use_fast=False)
    process_vision_info = qwen_vl_utils.process_vision_info

    if hasattr(model, "generation_config") and model.generation_config is not None:
        for attr in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(model.generation_config, attr):
                setattr(model.generation_config, attr, None)

    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)

    def _call(system_prompt: str, user_text: str, images: List[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        content: list = [{"type": "text", "text": user_text}]
        for path in images:
            content.append({"type": "image", "image": f"file://{path}"})
        messages.append({"role": "user", "content": content})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        processor_kwargs = dict(text=[text], padding=True, return_tensors="pt")
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        inputs = processor(**processor_kwargs)

        if use_cuda:
            inputs = inputs.to("cuda")

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_time=max_time,
            )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return (out[0] if out else "").strip()

    return _call


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_vlm_callable(model_cfg: dict) -> VLMCallable:
    """Create the appropriate VLMCallable from a model config dict.

    Supports:
      - provider=openai_api: uses OpenAI-compatible API (works with vLLM, Ollama, etc.)
      - provider=qwen_local: loads Qwen model locally via transformers
    """
    provider = str(model_cfg.get("provider", "openai_api"))
    model = str(model_cfg.get("model", "gpt-4o"))

    if provider == "openai_api":
        api_key = str(model_cfg.get("api_key", ""))
        api_key_env = str(model_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            import os
            api_key = os.environ.get(api_key_env, "")
        base_url = str(model_cfg.get("base_url", "https://api.openai.com/v1"))
        log.info("VLM adapter: openai_api model=%s base_url=%s", model, base_url)
        return _make_openai_vlm(api_key, base_url, model)

    elif provider == "qwen_local":
        model_path = str(model_cfg.get("model_path", ""))
        max_new_tokens = int(model_cfg.get("max_new_tokens", 512))
        max_time = float(model_cfg.get("max_time", 90))
        log.info("VLM adapter: qwen_local path=%s", model_path)
        return _make_qwen_local_vlm(model_path, max_new_tokens, max_time)

    else:
        raise ValueError(f"Unsupported VLM provider: {provider}")
