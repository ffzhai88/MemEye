from __future__ import annotations

import hashlib
import json
import logging
import os
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
        # 检索结果缓存：key = retrieval_query, value = {"docs": List[str], "backend": str}
        self._retrieval_cache: Dict[str, Dict[str, Any]] = {}
        print("\n" + "=" * 80)
        print("[HippoAgentic::Debug] === HippoAgenticSystem initialized ===")
        print(f"[HippoAgentic::Debug] method_config keys: {list(self.config.keys())}")
        print(f"[HippoAgentic::Debug] model_config keys: {list(self.model_config.keys())}")
        print("[HippoAgentic::Debug] key config: "
              f"save_dir={self.config.get('save_dir')}, "
              f"llm_model_name={self.config.get('llm_model_name')}, "
              f"embedding_model_name={self.config.get('embedding_model_name')}, "
              f"retrieval_top_k={self.config.get('retrieval_top_k', 'default 10')}")
        print("=" * 80)

    def _load_sys_prompt(self, mode: str = "open") -> str:
        prompt_dir = Path(__file__).parent / "prompt"
        if mode not in {"open", "mcq"}:
            mode = "open"
        mode_file = prompt_dir / f"sys_prompt_{mode}.txt"
        if mode_file.exists():
            content = mode_file.read_text(encoding="utf-8").strip()
            print(f"[HippoAgentic::Debug] _load_sys_prompt(mode={mode}) -> loaded from {mode_file.name} "
                  f"({len(content)} chars)")
            return content
        fallback = (prompt_dir / "sys_prompt.txt")
        content = fallback.read_text(encoding="utf-8").strip()
        print(f"[HippoAgentic::Debug] _load_sys_prompt(mode={mode}) -> no {mode_file.name}, "
              f"fallback to {fallback.name} ({len(content)} chars)")
        return content

    def _instantiate_router(self, model_cfg: Dict[str, Any], system_prompt: str = ""):
        provider = str(model_cfg.get("provider", "qwen_local")).strip()
        print(f"[HippoAgentic::Debug] _instantiate_router: provider={provider}, "
              f"model={model_cfg.get('model', 'N/A')}, "
              f"max_new_tokens={model_cfg.get('max_new_tokens', 'default 128')}, "
              f"base_url={model_cfg.get('base_url', 'default')}")
        if provider == "qwen_local":
            router = QwenLocalRouter(
                model_path=str(model_cfg["model_path"]),
                max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
                system_prompt=system_prompt,
                max_time=model_cfg.get("max_time", 25),
            )
            print(f"[HippoAgentic::Debug] _instantiate_router -> QwenLocalRouter, "
                  f"model_path={model_cfg.get('model_path')}")
            return router
        if provider == "openai_api":
            router = OpenAIAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "OPENAI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://api.openai.com/v1")),
                max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
            print(f"[HippoAgentic::Debug] _instantiate_router -> OpenAIAPIRouter, "
                  f"model={model_cfg.get('model')}, base_url={model_cfg.get('base_url')}")
            return router
        if provider == "gemini_api":
            router = GeminiAPIRouter(
                model=str(model_cfg.get("model", "")),
                api_key=str(model_cfg.get("api_key", "")),
                api_key_env=str(model_cfg.get("api_key_env", "GEMINI_API_KEY")),
                base_url=str(model_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")),
                max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
                timeout=int(model_cfg.get("timeout", 90)),
                system_prompt=system_prompt,
            )
            print(f"[HippoAgentic::Debug] _instantiate_router -> GeminiAPIRouter, "
                  f"model={model_cfg.get('model')}")
            return router
        raise ValueError(f"Unsupported provider: {provider}")

    def _get_router(self, mode: str) -> Any:
        if mode in self._routers:
            print(f"[HippoAgentic::Debug] _get_router(mode={mode}) -> using cached router")
            return self._routers[mode]
        print(f"[HippoAgentic::Debug] _get_router(mode={mode}) -> creating new router")
        model_cfg = dict(self.model_config or {})
        system_prompt = self._load_sys_prompt(mode)
        print(f"[HippoAgentic::Debug] _get_router: system_prompt preview (first 200 chars): "
              f"{system_prompt[:200]}")
        router = self._instantiate_router(model_cfg, system_prompt=system_prompt)
        self._routers[mode] = router
        return router

    def _resolve_question_images(self, qa: Dict[str, Any]) -> List[str]:
        # 支持从 QA 中读取 question_images 或 question_image 字段
        question_images = qa.get("question_images") or qa.get("question_image") or []
        # 如果字段显式为 None，则直接返回空列表
        if question_images is None:
            print(f"[HippoAgentic::Debug] _resolve_question_images -> None, returning []")
            return []
        # 如果是列表，则逐个归一化为字符串并去除空白项
        if isinstance(question_images, list):
            result = [str(path).strip() for path in question_images if str(path).strip()]
            print(f"[HippoAgentic::Debug] _resolve_question_images -> {len(result)} image(s): {result}")
            return result
        # 如果是单一值，则转成字符串并返回单元素列表
        result = [str(question_images).strip()] if str(question_images).strip() else []
        print(f"[HippoAgentic::Debug] _resolve_question_images -> {len(result)} image(s): {result}")
        return result

    def _question_with_image_caption(self, qa: Dict[str, Any], question: str) -> str:
        # 去除问题字符串两端空白
        query = question.strip()
        # 如果问题文本为空，则直接返回空字符串
        if not query:
            print(f"[HippoAgentic::Debug] _question_with_image_caption -> empty question, returning empty")
            return query
        # 读取可能存在的 image_caption 字段
        image_caption = qa.get("image_caption")
        # 如果没有图片描述，则仅返回问题文本
        if not image_caption:
            print(f"[HippoAgentic::Debug] _question_with_image_caption -> no image_caption, "
                  f"query preview={query[:120]}...")
            return query
        # 如果图片描述是列表，则合并每个非空描述项
        if isinstance(image_caption, list):
            caption_text = " ".join(str(item).strip() for item in image_caption if str(item).strip())
        else:
            # 否则将单个描述转换成字符串并去除空白
            caption_text = str(image_caption).strip()
        # 如果最终描述为空，则仍然只返回问题文本
        if not caption_text:
            print(f"[HippoAgentic::Debug] _question_with_image_caption -> empty caption_text after process, "
                  f"query preview={query[:120]}...")
            return query
        result = f"{query}\nquestion image caption: {caption_text}"
        print(f"[HippoAgentic::Debug] _question_with_image_caption -> appended image_caption, "
              f"query preview (first 200 chars): {result[:200]}")
        return result

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

        print(f"\n[HippoAgentic::Debug] === _build_hippo ===")
        print(f"[HippoAgentic::Debug]   save_dir={save_dir}")
        print(f"[HippoAgentic::Debug]   llm_model_name={llm_model_name}")
        print(f"[HippoAgentic::Debug]   llm_base_url={llm_base_url}")
        print(f"[HippoAgentic::Debug]   embedding_model_name={embedding_model_name}")
        print(f"[HippoAgentic::Debug]   embedding_base_url={embedding_base_url}")
        print(f"[HippoAgentic::Debug]   azure_endpoint={azure_endpoint}")
        print(f"[HippoAgentic::Debug]   azure_embedding_endpoint={azure_embedding_endpoint}")

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
        print(f"[HippoAgentic::Debug] HippoRAG instance created: {self.hippo}")

    def _build_round_docs(self, dataset: MemoryBenchmarkDataset) -> List[str]:
        # 初始化文档列表和对应 round id 记录
        docs: List[str] = []
        self._doc_round_ids = []
        session_order = dataset.session_order()
        print(f"\n[HippoAgentic::Debug] === _build_round_docs: {len(session_order)} session(s) ===")
        total_dialogues = 0
        skipped_no_round = 0
        skipped_no_text = 0
        # 按 session 顺序遍历数据集
        for session_id in session_order:
            session = dataset.get_session(session_id)
            dialogues = session.get("dialogues", []) or []
            print(f"[HippoAgentic::Debug]   Session {session_id}: {len(dialogues)} dialogues")
            for dialogue in dialogues:
                total_dialogues += 1
                # 获取轮次 id，并跳过无效值
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id:
                    skipped_no_round += 1
                    continue
                # 获取用户问句和助手回复
                user_text = str(dialogue.get("user", "")).strip()
                assistant_text = str(dialogue.get("assistant", "")).strip()
                # 如果这一轮没有可用文本，则跳过
                if not user_text and not assistant_text:
                    skipped_no_text += 1
                    continue
                # 构造文档基本信息
                parts = [f"Round: {round_id}", f"Session: {session_id}"]
                if user_text:
                    parts.append(f"User: {user_text}")
                if assistant_text:
                    parts.append(f"Assistant: {assistant_text}")
                # 读取本轮图像 caption 并追加到文档中
                round_data = dataset.rounds.get(round_id, {})
                caption_text = build_caption_text(round_data)
                has_images = bool(round_data.get("images") or round_data.get("screenshots"))
                if caption_text:
                    parts.append(f"Image caption: {caption_text}")
                # 将部分拼接为单条文档，添加到结果列表
                docs.append("\n".join(parts))
                # 记录文档对应的 round id
                self._doc_round_ids.append(round_id)
        print(f"[HippoAgentic::Debug] _build_round_docs summary: "
              f"total_dialogues={total_dialogues}, "
              f"skipped_no_round={skipped_no_round}, "
              f"skipped_no_text={skipped_no_text}, "
              f"final_docs={len(docs)}")
        # 打印前 3 个和后 1 个文档的 round_id 供验证
        if docs:
            print(f"[HippoAgentic::Debug] _build_round_docs first 3 round_ids: "
                  f"{self._doc_round_ids[:3]}")
            print(f"[HippoAgentic::Debug] _build_round_docs last round_id: "
                  f"{self._doc_round_ids[-1] if len(self._doc_round_ids) > 1 else self._doc_round_ids[0]}")
            print(f"[HippoAgentic::Debug] _build_round_docs sample doc (first doc, first 300 chars): "
                  f"{docs[0][:300]}")
        # 返回构造完成的 Hippo 文档列表
        return docs

    def _get_index_cache_marker(self, docs: List[str]) -> Optional[str]:
        """Compute a cache marker file path from the round docs content hash.

        Returns the marker file path if docs is non-empty and hippo is ready,
        otherwise None.
        """
        if not docs or self.hippo is None or not self.hippo.working_dir:
            return None
        content_hash = hashlib.md5("\n===\n".join(docs).encode()).hexdigest()
        marker_path = os.path.join(self.hippo.working_dir, f"index_{content_hash}.complete")
        return marker_path

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        # 通过 dataset id 判断是否需要重新初始化（内存级缓存）
        dataset_id = id(dataset)
        if self.hippo is not None and self._dataset_key == dataset_id:
            print(f"[HippoAgentic::Debug] _ensure_initialized -> already initialized for dataset#{dataset_id}, skip")
            return
        print(f"[HippoAgentic::Debug] === _ensure_initialized: building Hippo memory for dataset#{dataset_id} ===")
        # 初始化 Hippo 实例（会自动加载已持久化的 graph/embeddings/openie 结果）
        self._build_hippo()
        # 构造文档列表：纯字符串操作，不含 LLM/embedding 调用，开销很小
        docs = self._build_round_docs(dataset)

        # 检查磁盘缓存：如果同样内容的文档已经索引过，则跳过昂贵的 index() 调用
        marker_file = self._get_index_cache_marker(docs)
        if marker_file is not None and os.path.exists(marker_file):
            print(f"[HippoAgentic::Debug] _ensure_initialized -> disk cache HIT ({marker_file}), "
                  f"skipping hippo.index() rebuild")
            logger.info(f"[HippoAgentic] Disk cache hit for {len(docs)} docs, skipping index rebuild.")
            self._dataset = dataset
            self._dataset_key = dataset_id
            self.runtime_info["memory_size"] = len(docs)
            print(f"[HippoAgentic::Debug] === _ensure_initialized done (cached): memory_size={len(docs)} ===")
            return

        if docs:
            # 没有缓存，执行实际的索引构建（耗时耗 token 的部分）
            print(f"[HippoAgentic::Debug] _ensure_initialized: disk cache MISS, "
                  f"calling hippo.index({len(docs)} docs) ...")
            logger.info(f"[HippoAgentic] Indexing {len(docs)} MemEye round documents into Hippo.")
            self.hippo.index(docs)
            print(f"[HippoAgentic::Debug] _ensure_initialized: hippo.index() completed")
            # 索引完成后写入磁盘缓存标记，下次跳过
            if marker_file is not None:
                os.makedirs(os.path.dirname(marker_file), exist_ok=True)
                with open(marker_file, "w") as f:
                    json.dump({"content_hash": os.path.basename(marker_file).replace("index_", "").replace(".complete", ""),
                               "doc_count": len(docs)}, f)
                print(f"[HippoAgentic::Debug] _ensure_initialized -> saved cache marker to {marker_file}")
        else:
            # 如果没有文档则记录警告
            print(f"[HippoAgentic::Debug] _ensure_initialized: WARNING - no round documents to index!")
            logger.warning("[HippoAgentic] No round documents available for Hippo indexing.")
        # 记录当前 dataset 以及 id
        self._dataset = dataset
        self._dataset_key = dataset_id
        # 更新 runtime 信息中的 memory size
        self.runtime_info["memory_size"] = len(docs)
        print(f"[HippoAgentic::Debug] === _ensure_initialized done: memory_size={len(docs)} ===")

    def answer_question(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        point = qa.get("point", "N/A")
        print(f"\n" + "=" * 80)
        print(f"[HippoAgentic::Debug] === answer_question (point={point}) ===")
        print(f"[HippoAgentic::Debug]   question preview: {question[:200]}")
        print(f"[HippoAgentic::Debug]   qa keys: {list(qa.keys())}")
        print(f"[HippoAgentic::Debug]   qa clue: {list(qa.get('clue', []) or [])}")
        print(f"[HippoAgentic::Debug]   qa options present: {'options' in qa}")

        # 确保 Hippo 系统已经初始化并针对当前数据集建立索引
        self._ensure_initialized(dataset)
        # 防御性检查，保证 dataset 已经被设置
        if self._dataset is None:
            raise ValueError("HippoAgenticSystem requires a dataset before answering.")
        # 优先使用外部传入的问题图片，否则从 qa 中解析
        qa_images = question_images if question_images is not None else self._resolve_question_images(qa)
        # 构造带 image_caption 的检索 query（只使用原始题干，不含选项/格式指令）
        qa_text = qa.get("question", "").strip()
        if not qa_text:
            qa_text = question.strip()  # fallback: 使用传入的 question
        retrieval_query = self._question_with_image_caption(qa, qa_text)
        # 构造完整的 answer query（含选项 + 格式指令）
        query = self._question_with_image_caption(qa, question)
        # 如果 query 为空则直接报错
        if not query:
            raise ValueError("Question text is required for HippoAgenticSystem.answer_question.")
        # 获取检索 top_k 配置
        top_k = int(self.config.get("retrieval_top_k", 10))

        # --- 检索阶段（带缓存） ---
        print(f"\n[HippoAgentic::Debug] --- Retrieval phase ---")
        print(f"[HippoAgentic::Debug]   top_k={top_k}")
        print(f"[HippoAgentic::Debug]   retrieval_query (from raw question, first 300 chars): {retrieval_query[:300]}")
        print(f"[HippoAgentic::Debug]   answer_query (with options, first 300 chars): {query[:300]}")

        # 检查缓存：相同检索 query 直接复用结果
        cache_key = retrieval_query
        if cache_key in self._retrieval_cache:
            cached = self._retrieval_cache[cache_key]
            retrieved_docs: List[str] = list(cached["docs"])
            retrieval_backend_used = cached["backend"]
            print(f"[HippoAgentic::Debug]   Cache HIT for retrieval_query -> "
                  f"reusing {len(retrieved_docs)} docs from {retrieval_backend_used}")
        else:
            # 缓存未命中，执行实际检索
            retrieved_docs = []
            retrieval_backend_used = "none"
            try:
                # 先用 Hippo 图检索获取文档
                print(f"[HippoAgentic::Debug]   Cache MISS, trying hippo.retrieve() (graph retrieval)...")
                retrieved = self.hippo.retrieve([retrieval_query], num_to_retrieve=top_k)
                retrieved_docs = retrieved[0].docs if retrieved else []
                retrieval_backend_used = "hippo_graph"
                print(f"[HippoAgentic::Debug]   hippo.retrieve() succeeded: {len(retrieved_docs)} docs returned")
            except Exception as exc:
                # 图检索失败则降级到 DPR 检索
                print(f"[HippoAgentic::Debug]   hippo.retrieve() FAILED: {exc}")
                logger.warning("[HippoAgentic] Hippo graph retrieval failed: %s. Falling back to DPR retrieval.", exc)
                try:
                    print(f"[HippoAgentic::Debug]   Trying hippo.retrieve_dpr() (DPR fallback)...")
                    retrieved = self.hippo.retrieve_dpr([retrieval_query], num_to_retrieve=top_k)
                    retrieved_docs = retrieved[0].docs if retrieved else []
                    retrieval_backend_used = "dpr_fallback"
                    print(f"[HippoAgentic::Debug]   hippo.retrieve_dpr() succeeded: {len(retrieved_docs)} docs returned")
                except Exception as exc2:
                    print(f"[HippoAgentic::Debug]   hippo.retrieve_dpr() also FAILED: {exc2}")
                    logger.error("[HippoAgentic] DPR fallback retrieval also failed: %s", exc2)
                    retrieved_docs = []
            # 写入缓存
            self._retrieval_cache[cache_key] = {
                "docs": list(retrieved_docs),
                "backend": retrieval_backend_used,
            }

        # 记录实际检索到的文档数量
        self.runtime_info["retrieved_docs"] = len(retrieved_docs)

        # 从检索结果中解析 round id 并计算 clue_hit_count（方便调试）
        retrieved_round_ids: List[str] = []
        for doc in retrieved_docs:
            for line in doc.splitlines():
                line_stripped = line.strip()
                if line_stripped.startswith("Round:"):
                    rid = line_stripped[len("Round:"):].strip()
                    if rid:
                        retrieved_round_ids.append(rid)
                    break
        clue_rounds = list(qa.get("clue", []) or [])
        clue_hit_count = sum(1 for rid in retrieved_round_ids if rid in clue_rounds)
        self.runtime_info["clue_rounds"] = clue_rounds
        self.runtime_info["retrieved_round_ids"] = retrieved_round_ids
        self.runtime_info["clue_hit_count"] = clue_hit_count
        print(f"[HippoAgentic::Debug]   Retrieved doc count: {len(retrieved_docs)}")
        print(f"[HippoAgentic::Debug]   Retrieved round IDs: {retrieved_round_ids}")
        print(f"[HippoAgentic::Debug]   Clue rounds (oracle): {clue_rounds}")
        print(f"[HippoAgentic::Debug]   clue_hit_count={clue_hit_count} / {len(clue_rounds)}")
        print(f"[HippoAgentic::Debug]   Retrieval backend: {retrieval_backend_used}")

        # 将检索文档转换为 router 需要的历史消息格式
        history: List[Dict[str, Any]] = []
        for idx, doc in enumerate(retrieved_docs):
            history.append({"role": "user", "text": doc, "images": []})
            if idx < 2:
                print(f"[HippoAgentic::Debug]   history[{idx}] doc preview: {doc[:200]}...")

        # 根据是否含有 options 决定 mcq 还是 open 模式
        mode = "mcq" if isinstance(qa.get("options"), (dict, list)) and bool(qa.get("options")) else "open"
        print(f"\n[HippoAgentic::Debug] --- Answer phase ---")
        print(f"[HippoAgentic::Debug]   mode={mode}, history_turns={len(history)}")
        # 获取对应模式的 router
        router = self._get_router(mode)
        # 调用 router 生成最终回答
        print(f"[HippoAgentic::Debug]   Calling router.answer(history={len(history)} turns, query_len={len(query)}) ...")
        answer = router.answer(history, query, question_images=qa_images)
        print(f"[HippoAgentic::Debug]   Router answer (first 300 chars): {answer[:300]}")
        print(f"[HippoAgentic::Debug] === answer_question done ===")
        return answer


class HippoAgenticMethod(HistoryMethod):
    name = "hippo_agentic"
    fixed_modality = "multimodal"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._system: Optional[HippoAgenticSystem] = None
        self._dataset_key: Optional[int] = None
        print(f"\n" + "=" * 80)
        print(f"[HippoAgentic::Debug] === HippoAgenticMethod initialized ===")
        print(f"[HippoAgentic::Debug] config keys: {list(self.config.keys()) if self.config else 'None'}")
        if self.config:
            print(f"[HippoAgentic::Debug]   name from config: {self.config.get('method', self.name)}")
            print(f"[HippoAgentic::Debug]   modality: {self.config.get('modality', 'not set')}")
            print(f"[HippoAgentic::Debug]   llm_model_name: {self.config.get('llm_model_name')}")
            print(f"[HippoAgentic::Debug]   embedding_model_name: {self.config.get('embedding_model_name')}")
            print(f"[HippoAgentic::Debug]   retrieval_top_k: {self.config.get('retrieval_top_k')}")
        print("=" * 80)

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        # 通过 dataset id 判断是否为相同数据集，避免重复初始化
        dataset_id = id(dataset)
        # 如果已经创建过系统并且 dataset 未变，则直接返回
        if self._system is not None and self._dataset_key == dataset_id:
            print(f"[HippoAgentic::Debug] HippoAgenticMethod._ensure_initialized -> "
                  f"already initialized for dataset#{dataset_id}, skip")
            return
        print(f"\n[HippoAgentic::Debug] === HippoAgenticMethod._ensure_initialized (dataset#{dataset_id}) ===")
        print(f"[HippoAgentic::Debug]   session_order={dataset.session_order()}")
        # 从配置中提取模型相关参数
        model_cfg = dict(self.config.get("_model_cfg", {}))
        print(f"[HippoAgentic::Debug]   model_cfg keys: {list(model_cfg.keys())}")
        # 创建 HippoAgenticSystem 实例
        print(f"[HippoAgentic::Debug]   Creating HippoAgenticSystem ...")
        self._system = HippoAgenticSystem(self.config, model_cfg)
        # 初始化 HippoAgenticSystem 的索引和内存
        self._system._ensure_initialized(dataset)
        # 保存当前 dataset id
        self._dataset_key = dataset_id
        # 同步运行时信息到外层方法
        self.runtime_info.update(self._system.runtime_info)
        print(f"[HippoAgentic::Debug]   runtime_info now: {dict(self.runtime_info)}")
        print(f"[HippoAgentic::Debug] === HippoAgenticMethod._ensure_initialized done ===")

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        point = qa.get("point", "N/A")
        print(f"[HippoAgentic::Debug] === HippoAgenticMethod.answer (point={point}) ===")
        # 确保系统和索引已为当前数据集初始化
        self._ensure_initialized(dataset)
        # 断言系统存在，供静态类型检查使用
        assert self._system is not None
        # 委托 HippoAgenticSystem 执行实际回答流程
        result = self._system.answer_question(dataset, qa, question, question_images=question_images)
        # 将 system 内部更新的 per-question runtime_info（如 retrieved_docs、clue_hit_count 等）
        # 同步回外层 HippoAgenticMethod.runtime_info，确保 runner 能正确捕获并写入 predictions.jsonl
        self.runtime_info.update(self._system.runtime_info)
        print(f"[HippoAgentic::Debug] === HippoAgenticMethod.answer done, final answer (first 300): {result[:300]}")
        print("=" * 80)
        return result

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"[HippoAgentic::Debug] HippoAgenticMethod.build_history called -> returning empty (agentic method)")
        return []
