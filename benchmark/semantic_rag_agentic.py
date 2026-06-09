from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from router import GeminiAPIRouter, OpenAIAPIRouter, QwenLocalRouter

from .dataset import MemoryBenchmarkDataset, history_from_round_ids
from .methods import HistoryMethod
from .retrieval import select_round_ids_for_qa


class SemanticRAGAgenticSystem:
    """Agentic memory system for semantic multimodal retrieval.

    这个类负责：
    1. 把整个数据集作为“记忆库”进行初始化；
    2. 每次问题到来时从记忆库中检索相关轮次；
    3. 通过统一 router 把检索结果送入模型生成答案。
    """

    def __init__(self, method_config: Dict[str, Any], model_config: Dict[str, Any]) -> None:
        self.config = method_config or {}
        self.model_config = model_config or {}
        self._dataset: Optional[MemoryBenchmarkDataset] = None
        self._dataset_key: Optional[int] = None
        self._routers: Dict[str, Any] = {}
        self._memory_round_ids: List[str] = []
        self.runtime_info: Dict[str, Any] = {}

    def _load_sys_prompt(self, mode: str = "open") -> str:
        prompt_dir = Path(__file__).parent / "prompt"
        if mode not in {"open", "mcq"}:
            mode = "open"
        mode_file = prompt_dir / f"sys_prompt_{mode}.txt"
        if mode_file.exists():
            return mode_file.read_text(encoding="utf-8").strip()
        return (prompt_dir / "sys_prompt.txt").read_text(encoding="utf-8").strip()

    def _instantiate_router(self, model_cfg: Dict[str, Any], system_prompt: str = ""):
        provider = str(model_cfg.get("provider", "qwen_local")).strip()
        if provider == "qwen_local":
            return QwenLocalRouter(
                model_path=str(model_cfg["model_path"]),
                max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
                system_prompt=system_prompt,
                max_time=model_cfg.get("max_time", 25),
            )
        if provider == "openai_api":
            return OpenAIAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "OPENAI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://api.openai.com/v1")),
                max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
        if provider == "gemini_api":
            return GeminiAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "GEMINI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")),
                max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def _get_router(self, mode: str) -> Any:
        if mode in self._routers:
            return self._routers[mode]
        model_cfg = dict(self.model_config or {})
        system_prompt = self._load_sys_prompt(mode)
        router = self._instantiate_router(model_cfg, system_prompt=system_prompt)
        self._routers[mode] = router
        return router

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._dataset is not None and self._dataset_key == dataset_id:
            return

        self._dataset = dataset
        self._dataset_key = dataset_id
        self._memory_round_ids = []

        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if round_id and round_id in dataset.rounds:
                    self._memory_round_ids.append(round_id)

        self.runtime_info["memory_size"] = len(self._memory_round_ids)

    def _resolve_question_images(self, qa: Dict[str, Any]) -> List[str]:
        question_images = qa.get("question_images") or qa.get("question_image") or []
        if question_images is None:
            return []
        if isinstance(question_images, list):
            return [str(path).strip() for path in question_images if str(path).strip()]
        return [str(question_images).strip()] if str(question_images).strip() else []

    def _question_with_image_caption(self, qa: Dict[str, Any], question: str) -> str:
        query = question.strip()
        if not query:
            return query
        image_caption = qa.get("image_caption")
        if not image_caption:
            return query
        if isinstance(image_caption, list):
            caption_text = " ".join(str(item).strip() for item in image_caption if str(item).strip())
        else:
            caption_text = str(image_caption).strip()
        if not caption_text:
            return query
        return f"{query}\nquestion image caption: {caption_text}"

    def answer_question(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure_initialized(dataset)
        if self._dataset is None:
            raise ValueError("SemanticRAGAgenticSystem requires a dataset before answering.")

        qa_images = question_images if question_images is not None else self._resolve_question_images(qa)
        retrieved_round_ids = select_round_ids_for_qa(self._dataset, qa, self.config, runtime_info=self.runtime_info)

        allowed_round_ids = set(retrieved_round_ids)
        history: List[Dict[str, Any]] = []
        for session_id in self._dataset.session_order():
            history.extend(
                history_from_round_ids(
                    self._dataset.get_session(session_id),
                    self._dataset.rounds,
                    allowed_round_ids,
                    modality="multimodal",
                )
            )

        mode = "mcq" if isinstance(qa.get("options"), (dict, list)) and bool(qa.get("options")) else "open"
        router = self._get_router(mode)
        query = self._question_with_image_caption(qa, question)
        return router.answer(history, query, question_images=qa_images)


class SemanticRAGAgenticMethod(HistoryMethod):
    """Semantic RAG multimodal 的 agentic 版本。

    该方法在初始化时构建一个“记忆库”，并在每次问题到来时从记忆库中检索相关轮次。
    """

    name = "semantic_rag_multimodal_agentic"
    fixed_modality = "multimodal"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._system: Optional[SemanticRAGAgenticSystem] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._system is not None and self._dataset_key == dataset_id:
            return
        model_cfg = dict(self.config.get("_model_cfg", {}))
        self._system = SemanticRAGAgenticSystem(self.config, model_cfg)
        self._system._ensure_initialized(dataset)
        self._dataset_key = dataset_id
        self.runtime_info.update(self._system.runtime_info)

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure_initialized(dataset)
        assert self._system is not None
        return self._system.answer_question(dataset, qa, question, question_images=question_images)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Agentic 方法在内部管理记忆，不走标准的 build_history 压缩流程。
        return []
