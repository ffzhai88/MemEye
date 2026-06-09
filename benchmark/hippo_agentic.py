from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from router import GeminiAPIRouter, OpenAIAPIRouter, QwenLocalRouter

from .dataset import MemoryBenchmarkDataset, build_caption_text
from .hippo import HippoRAG
from .methods import HistoryMethod

logger = logging.getLogger(__name__)


class HippoAgenticSystem:
    """Agentic wrapper around HippoRAG for MemEye datasets."""

    def __init__(self, method_config: Dict[str, Any], model_config: Dict[str, Any]) -> None:
        self.config = method_config or {}
        self.model_config = model_config or {}
        self.hippo: Optional[HippoRAG] = None
        self._dataset: Optional[MemoryBenchmarkDataset] = None
        self._dataset_key: Optional[int] = None
        self._routers: Dict[str, Any] = {}
        self._doc_round_ids: List[str] = []
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

    def _resolve_question_images(self, qa: Dict[str, Any]) -> List[str]:
        # 支持从 QA 中读取 question_images 或 question_image 字段
        question_images = qa.get("question_images") or qa.get("question_image") or []
        # 如果字段显式为 None，则直接返回空列表
        if question_images is None:
            return []
        # 如果是列表，则逐个归一化为字符串并去除空白项
        if isinstance(question_images, list):
            return [str(path).strip() for path in question_images if str(path).strip()]
        # 如果是单一值，则转成字符串并返回单元素列表
        return [str(question_images).strip()] if str(question_images).strip() else []

    def _question_with_image_caption(self, qa: Dict[str, Any], question: str) -> str:
        # 去除问题字符串两端空白
        query = question.strip()
        # 如果问题文本为空，则直接返回空字符串
        if not query:
            return query
        # 读取可能存在的 image_caption 字段
        image_caption = qa.get("image_caption")
        # 如果没有图片描述，则仅返回问题文本
        if not image_caption:
            return query
        # 如果图片描述是列表，则合并每个非空描述项
        if isinstance(image_caption, list):
            caption_text = " ".join(str(item).strip() for item in image_caption if str(item).strip())
        else:
            # 否则将单个描述转换成字符串并去除空白
            caption_text = str(image_caption).strip()
        # 如果最终描述为空，则仍然只返回问题文本
        if not caption_text:
            return query
        # 将问题文本与图片描述拼接为检索 query
        return f"{query}\nquestion image caption: {caption_text}"

    def _build_hippo(self) -> None:
        # 读取保存目录配置，优先使用 method config，再退回 runtime 路径
        save_dir = self.config.get("save_dir") or self.config.get("_runtime_paths", {}).get("run_dir") or "outputs"
        # 读取 LLM 模型名称配置
        llm_model_name = self.config.get("llm_model_name") or self.model_config.get("model")
        # 读取 LLM 基础 URL 配置
        llm_base_url = self.config.get("llm_base_url") or self.model_config.get("base_url")
        # 读取文本嵌入模型配置
        embedding_model_name = self.config.get("embedding_model_name") or self.config.get("text_embedding_model")
        # 读取嵌入服务基础 URL
        embedding_base_url = self.config.get("embedding_base_url")
        # 读取可选的 Azure endpoint
        azure_endpoint = self.config.get("azure_endpoint")
        # 读取可选的 Azure 嵌入 endpoint
        azure_embedding_endpoint = self.config.get("azure_embedding_endpoint")

        # 用配置构建 HippoRAG 实例
        self.hippo = HippoRAG(
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            embedding_base_url=embedding_base_url,
            azure_endpoint=azure_endpoint,
            azure_embedding_endpoint=azure_embedding_endpoint,
        )

    def _build_round_docs(self, dataset: MemoryBenchmarkDataset) -> List[str]:
        # 初始化文档列表和对应 round id 记录
        docs: List[str] = []
        self._doc_round_ids = []
        # 按 session 顺序遍历数据集
        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            for dialogue in session.get("dialogues", []) or []:
                # 获取轮次 id，并跳过无效值
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id:
                    continue
                # 获取用户问句和助手回复
                user_text = str(dialogue.get("user", "")).strip()
                assistant_text = str(dialogue.get("assistant", "")).strip()
                # 如果这一轮没有可用文本，则跳过
                if not user_text and not assistant_text:
                    continue
                # 构造文档基本信息
                parts = [f"Round: {round_id}", f"Session: {session_id}"]
                if user_text:
                    parts.append(f"User: {user_text}")
                if assistant_text:
                    parts.append(f"Assistant: {assistant_text}")
                # 读取本轮图像 caption 并追加到文档中
                caption_text = build_caption_text(dataset.rounds.get(round_id, {}))
                if caption_text:
                    parts.append(f"Image caption: {caption_text}")
                # 将部分拼接为单条文档，添加到结果列表
                docs.append("\n".join(parts))
                # 记录文档对应的 round id
                self._doc_round_ids.append(round_id)
        # 返回构造完成的 Hippo 文档列表
        return docs

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        # 通过 dataset id 判断是否需要重新初始化
        dataset_id = id(dataset)
        if self.hippo is not None and self._dataset_key == dataset_id:
            return
        # 初始化 Hippo 实例
        self._build_hippo()
        # 构造当前数据集的 round 文档
        docs = self._build_round_docs(dataset)
        if docs:
            # 如果有文档则索引到 Hippo
            logger.info(f"[HippoAgentic] Indexing {len(docs)} MemEye round documents into Hippo.")
            self.hippo.index(docs)
        else:
            # 如果没有文档则记录警告
            logger.warning("[HippoAgentic] No round documents available for Hippo indexing.")
        # 记录当前 dataset 以及 id
        self._dataset = dataset
        self._dataset_key = dataset_id
        # 更新 runtime 信息中的 memory size
        self.runtime_info["memory_size"] = len(docs)

    def answer_question(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        # 确保 Hippo 系统已经初始化并针对当前数据集建立索引
        self._ensure_initialized(dataset)
        # 防御性检查，保证 dataset 已经被设置
        if self._dataset is None:
            raise ValueError("HippoAgenticSystem requires a dataset before answering.")
        # 优先使用外部传入的问题图片，否则从 qa 中解析
        qa_images = question_images if question_images is not None else self._resolve_question_images(qa)
        # 构造带 image_caption 的检索 query
        query = self._question_with_image_caption(qa, question)
        # 如果 query 为空则直接报错
        if not query:
            raise ValueError("Question text is required for HippoAgenticSystem.answer_question.")
        # 获取检索 top_k 配置
        top_k = int(self.config.get("retrieval_top_k", 10))
        # 初始化检索结果列表
        retrieved_docs: List[str] = []
        try:
            # 先用 Hippo 图检索获取文档
            retrieved = self.hippo.retrieve([query], num_to_retrieve=top_k)
            retrieved_docs = retrieved[0].docs if retrieved else []
        except Exception as exc:
            # 图检索失败则降级到 DPR 检索
            logger.warning("[HippoAgentic] Hippo graph retrieval failed: %s. Falling back to DPR retrieval.", exc)
            try:
                retrieved = self.hippo.retrieve_dpr([query], num_to_retrieve=top_k)
                retrieved_docs = retrieved[0].docs if retrieved else []
            except Exception as exc2:
                # 两次检索失败时记录错误
                logger.error("[HippoAgentic] DPR fallback retrieval also failed: %s", exc2)
                retrieved_docs = []
        # 记录实际检索到的文档数量
        self.runtime_info["retrieved_docs"] = len(retrieved_docs)
        # 将检索文档转换为 router 需要的历史消息格式
        history: List[Dict[str, Any]] = []
        for doc in retrieved_docs:
            history.append({"role": "user", "text": doc, "images": []})
        # 根据是否含有 options 决定 mcq 还是 open 模式
        mode = "mcq" if isinstance(qa.get("options"), (dict, list)) and bool(qa.get("options")) else "open"
        # 获取对应模式的 router
        router = self._get_router(mode)
        # 调用 router 生成最终回答
        return router.answer(history, query, question_images=qa_images)


class HippoAgenticMethod(HistoryMethod):
    name = "hippo_agentic"
    fixed_modality = "multimodal"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._system: Optional[HippoAgenticSystem] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        # 通过 dataset id 判断是否为相同数据集，避免重复初始化
        dataset_id = id(dataset)
        # 如果已经创建过系统并且 dataset 未变，则直接返回
        if self._system is not None and self._dataset_key == dataset_id:
            return
        # 从配置中提取模型相关参数
        model_cfg = dict(self.config.get("_model_cfg", {}))
        # 创建 HippoAgenticSystem 实例
        self._system = HippoAgenticSystem(self.config, model_cfg)
        # 初始化 HippoAgenticSystem 的索引和内存
        self._system._ensure_initialized(dataset)
        # 保存当前 dataset id
        self._dataset_key = dataset_id
        # 同步运行时信息到外层方法
        self.runtime_info.update(self._system.runtime_info)

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        # 确保系统和索引已为当前数据集初始化
        self._ensure_initialized(dataset)
        # 断言系统存在，供静态类型检查使用
        assert self._system is not None
        # 委托 HippoAgenticSystem 执行实际回答流程
        return self._system.answer_question(dataset, qa, question, question_images=question_images)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
