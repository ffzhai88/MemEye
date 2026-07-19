"""
Embedding wrappers for M2A. Faithful to official agent/embeddings/.

TextEmbedder       : all-MiniLM-L6-v2 via sentence-transformers (local, 384-dim)
                     OR nvidia/NV-Embed-v2 via transformers + trust_remote_code (local, 4096-dim)
                     switching via model_name.
MultimodalEmbedder : siglip2-base-patch16-384 via transformers (local, 768-dim)
LocalCLIPEmbedder  : openai/clip-vit-base-patch32 via transformers (local fallback, 512-dim)
"""
from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union


@contextmanager
def _sanitized_hf_token_env():
    # NOTE: not thread-safe — modifies os.environ globally.
    original = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if original is not None and any(ord(ch) > 127 for ch in original):
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        try:
            yield
        finally:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = original
    else:
        yield


def _is_nv_embed(model_name: str) -> bool:
    """Check if model_name corresponds to NV-Embed-v2 family."""
    return "NV-Embed-v2" in model_name


class TextEmbedder:
    """
    Local text embedder supporting multiple backends selected by model_name.

    - ``all-MiniLM-L6-v2`` (default) → sentence-transformers, 384-dim
    - ``nvidia/NV-Embed-v2``          → transformers + trust_remote_code, 4096-dim

    Extra kwargs are forwarded to the NV-Embed-v2 constructor when that
    backend is selected, so callers can configure encoding parameters
    directly (e.g. ``max_length=16384, batch_size=8``).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    # ── NV-Embed-v2 defaults (mirroring nv_embed/NVEmbedV2.py) ──────────────
    NVEMBED_MAX_LENGTH = 32768
    NVEMBED_BATCH_SIZE = 16
    NVEMBED_INSTRUCTION = ""
    NVEMBED_DTYPE = "auto"
    NVEMBED_NUM_WORKERS = 32

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        # NV-Embed-v2 specific parameters (ignored by sentence-transformers backend)
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        instruction: str = "",
        dtype: str = "auto",
        num_workers: int = 32,
        **_kwargs,
    ) -> None:
        self._model_name = model_name
        self._model = None
        self._is_nvembed = _is_nv_embed(model_name)

        # Store NV-Embed-v2 encoding parameters
        self._nv_max_length = max_length or self.NVEMBED_MAX_LENGTH
        self._nv_batch_size = batch_size or self.NVEMBED_BATCH_SIZE
        self._nv_instruction = instruction or self.NVEMBED_INSTRUCTION
        self._nv_dtype = dtype or self.NVEMBED_DTYPE
        self._nv_num_workers = num_workers or self.NVEMBED_NUM_WORKERS

        if self._is_nvembed:
            print(
                f"[TextEmbedder] NV-Embed-v2 mode: model={model_name}, "
                f"max_length={self._nv_max_length}, batch_size={self._nv_batch_size}, "
                f"dtype={self._nv_dtype}"
            )

    def _load(self) -> None:
        if self._model is not None:
            return

        if self._is_nvembed:
            self._load_nvembed()
        else:
            self._load_sentence_transformers()

    def _load_sentence_transformers(self) -> None:
        """Load via sentence-transformers (all-MiniLM-L6-v2 or similar)."""
        from sentence_transformers import SentenceTransformer  # type: ignore

        cache_folder = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
        with _sanitized_hf_token_env():
            self._model = SentenceTransformer(self._model_name, cache_folder=cache_folder)

    def _load_nvembed(self) -> None:
        """Load via transformers + trust_remote_code (nvidia/NV-Embed-v2)."""
        import torch
        from transformers import AutoModel

        cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
        with _sanitized_hf_token_env():
            self._model = AutoModel.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=self._nv_dtype,
                device_map="auto",
            )
        self._embedding_dim = self._model.config.hidden_size
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        print(
            f"[TextEmbedder] Loaded NV-Embed-v2: {self._model_name} "
            f"({self._embedding_dim}-dim) on {self._device}"
        )

    @property
    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def _format_nv_instruction(self) -> str:
        """Format instruction as NV-Embed-v2 expects: ``Instruct: {msg}\nQuery: ``."""
        if self._nv_instruction:
            return f"Instruct: {self._nv_instruction}\nQuery: "
        return ""

    def embed_query(self, text: str) -> List[float]:
        self._load()
        if self._is_nvembed:
            return self._nv_embed_query(text)
        # sentence-transformers path
        return self._model.encode(text, normalize_embeddings=True).tolist()  # type: ignore

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load()
        if self._is_nvembed:
            return self._nv_embed_batch(texts)
        # sentence-transformers path
        return self._model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore

    # ── NV-Embed-v2 internal helpers ───────────────────────────────────────

    def _nv_embed_query(self, text: str) -> List[float]:
        result = self._model.encode(  # type: ignore
            prompts=[text],
            max_length=self._nv_max_length,
            instruction=self._format_nv_instruction(),
        )
        if hasattr(result, "cpu"):
            result = result.cpu().numpy()
        return result[0].tolist()  # type: ignore

    def _nv_embed_batch(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        import torch
        from tqdm import tqdm

        instruction = self._format_nv_instruction()
        batch_size = self._nv_batch_size

        if len(texts) <= batch_size:
            result = self._model.encode(  # type: ignore
                prompts=texts,
                max_length=self._nv_max_length,
                instruction=instruction,
            )
            if hasattr(result, "cpu"):
                result = result.cpu().numpy()
            return result.tolist()  # type: ignore

        # Batch mode: split into chunks and concatenate
        all_results = []
        pbar = tqdm(total=len(texts), desc="NVEmbed Batch Encoding")
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            chunk_result = self._model.encode(  # type: ignore
                prompts=chunk,
                max_length=self._nv_max_length,
                instruction=instruction,
            )
            if hasattr(chunk_result, "detach"):
                chunk_result = chunk_result.detach().cpu()
            all_results.append(chunk_result)
            pbar.update(len(chunk))
        pbar.close()

        if all_results and hasattr(all_results[0], "cpu"):
            result = torch.cat(all_results, dim=0).numpy()
        elif all_results:
            result = np.concatenate(all_results, axis=0)
        else:
            return []
        return result.tolist()  # type: ignore


class MultimodalEmbedder:
    """
    SigLIP2 embedder via local transformers. Faithful to official M2A.
    Model: google/siglip2-base-patch16-384 (768-dim)

    Uses get_text_features() and get_image_features() directly from the
    HuggingFace model, ensuring text and image embeddings are in the same
    aligned contrastive space. This avoids the vLLM DispatchPooler issue
    that broke cross-modal alignment.
    """

    DEFAULT_MODEL = "google/siglip2-base-patch16-384"
    MAX_TEXT_LENGTH = 64  # SigLIP2 text encoder max tokens

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor  # type: ignore

        cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
        with _sanitized_hf_token_env():
            self._processor = AutoProcessor.from_pretrained(
                self._model_name, cache_dir=cache_dir
            )
            self._model = AutoModel.from_pretrained(
                self._model_name, cache_dir=cache_dir
            )

        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._model = self._model.to(self._device)
        self._model.eval()
        print(f"[MultimodalEmbedder] Loaded {self._model_name} on {self._device}")

    @property
    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as e:
            warnings.warn(f"[MultimodalEmbedder] Not available: {e}")
            return False

    def embed_text(self, text: str) -> List[float]:
        """Embed text for cross-modal text→image search."""
        import torch

        self._load()
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.MAX_TEXT_LENGTH,
            truncation=True,
        )
        text_inputs = {
            k: v.to(self._device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }

        with torch.no_grad():
            features = self._model.get_text_features(**text_inputs)

            if hasattr(features, 'pooler_output'):
                features = features.pooler_output

            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten().tolist()

    def embed_image(self, image_path: str) -> List[float]:
        """Embed image for visual similarity / image→image search."""
        import torch
        from PIL import Image  # type: ignore

        self._load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        image_inputs = {
            k: v.to(self._device)
            for k, v in inputs.items()
            if k in ("pixel_values",)
        }

        with torch.no_grad():
            features = self._model.get_image_features(**image_inputs)

            if hasattr(features, 'pooler_output'):
                features = features.pooler_output

            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten().tolist()


class LocalCLIPEmbedder:
    """
    Local CLIP embedder via transformers. Fallback when SigLIP2 is unavailable.
    Model: openai/clip-vit-base-patch32 (512-dim)
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, *, local_files_only: bool = False
    ) -> None:
        self._model_name = model_name
        self._local_files_only = local_files_only
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
            with _sanitized_hf_token_env():
                self._processor = CLIPProcessor.from_pretrained(
                    self._model_name,
                    cache_dir=cache_dir,
                    local_files_only=self._local_files_only,
                )
                self._model = CLIPModel.from_pretrained(
                    self._model_name,
                    cache_dir=cache_dir,
                    local_files_only=self._local_files_only,
                )

            # Use GPU if available
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            self._model = self._model.to(self._device)
            self._model.eval()

    @property
    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as e:
            warnings.warn(f"LocalCLIPEmbedder not available: {e}")
            return False

    def embed_text(self, text: str) -> List[float]:
        """Embed text for cross-modal text→image search."""
        import torch

        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self._device) for k, v in inputs.items()
                       if k in ("input_ids", "attention_mask")}

        with torch.no_grad():
            text_features = self._model.get_text_features(**text_inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().flatten().tolist()

    def embed_image(self, image_path: str) -> List[float]:
        """Embed image for visual similarity / image→image search."""
        import torch
        from PIL import Image  # type: ignore

        self._load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self._device) for k, v in inputs.items()
                        if k in ("pixel_values",)}

        with torch.no_grad():
            image_features = self._model.get_image_features(**image_inputs)
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten().tolist()


def get_multimodal_embedder(
    vllm_model: str = "siglip2-base-patch16-384",
    clip_model: str = "openai/clip-vit-base-patch32",
    **_kwargs,
) -> Optional[Union[MultimodalEmbedder, LocalCLIPEmbedder]]:
    """
    Factory function: try local SigLIP first, fallback to local CLIP.
    Returns None if neither is available.

    Args:
        vllm_model: SigLIP model name (short or full HuggingFace ID).
            Kept as 'vllm_model' for backwards config compatibility.
        clip_model: CLIP model name for fallback.
    """
    SHORT_NAME_MAP = {
        "siglip2-base-patch16-384": "google/siglip2-base-patch16-384",
        "siglip-so400m-patch14-384": "google/siglip-so400m-patch14-384",
    }
    model_name = SHORT_NAME_MAP.get(vllm_model, vllm_model)

    # Try primary multimodal embedder (SigLIP2 or SigLIP v1)
    mm_embedder = MultimodalEmbedder(model_name)
    if mm_embedder.is_available:
        print(f"[Embeddings] Using local {model_name} for multimodal embeddings")
        return mm_embedder

    # Fallback to local CLIP
    warnings.warn(
        f"[Embeddings] {model_name} not available. "
        "Falling back to local CLIP for image embeddings."
    )
    clip_embedder = LocalCLIPEmbedder(clip_model)
    if clip_embedder.is_available:
        print("[Embeddings] Using local CLIP for multimodal embeddings")
        return clip_embedder

    # Neither available
    warnings.warn(
        "[Embeddings] No multimodal embedder available. "
        "Image retrieval will be DISABLED."
    )
    return None
