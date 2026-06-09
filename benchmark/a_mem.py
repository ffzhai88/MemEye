import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from .common import REPO_ROOT, write_json
from .dataset import MemoryBenchmarkDataset
from .methods import HistoryMethod


_A_MEM_ROOT = (Path(__file__).resolve().parent / "a-mem" / "A-mem").resolve()
if str(_A_MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_A_MEM_ROOT))

if TYPE_CHECKING:
    from memory_layer import AgenticMemorySystem, LLMController  # type: ignore


def _load_a_mem_classes() -> Tuple[Any, Any]:
    """动态导入 A-MEM 的核心类。

    这里使用延迟导入，避免在环境尚未安装 A-MEM 依赖时就触发导入错误。
    如果依赖缺失，会抛出清晰的提示，指引用户安装对应的 requirements。
    """
    try:
        from memory_layer import AgenticMemorySystem, LLMController  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "A-MEM dependencies are not fully installed. Missing module: "
            f"{exc.name}. Install benchmark/a-mem/A-mem/requirements.txt in the active environment."
        ) from exc
    return AgenticMemorySystem, LLMController


def _resolve_openai_api_key(method_config: Dict[str, Any]) -> Optional[str]:
    """从方法配置或环境变量中解析 A-MEM 使用的 OpenAI API Key。

    优先读取方法配置里的显式密钥；如果未配置，再回退到环境变量。
    这样能兼容本地开发环境以及不同的运行配置。
    """
    raw = str(method_config.get("llm_api_key", "")).strip()
    if raw:
        return raw
    env_name = str(method_config.get("llm_api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
    return os.getenv(env_name)


def _resolve_openai_base_url(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> Optional[str]:
    """解析 OpenAI 兼容接口的基础 URL。

    优先使用方法配置中的 llm_base_url；否则回退到模型配置中的 base_url。
    便于支持本地代理、Azure/OpenAI 兼容服务或自建网关。
    """
    raw = str(method_config.get("llm_base_url", "")).strip()
    if raw:
        return raw
    raw = str(model_config.get("base_url", "")).strip()
    return raw or None


def _resolve_openai_timeout(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> int:
    """解析 API 请求超时时间，保证调用时有稳定的超时配置。

    先取方法配置中的 llm_timeout；若无则使用模型配置中的 timeout。
    如果解析失败，会回落到默认值 90 秒，避免异常中断。
    """
    try:
        return int(method_config.get("llm_timeout", model_config.get("timeout", 90)) or 90)
    except Exception:
        return 90


def _resolve_model_api_key(model_config: Dict[str, Any]) -> Optional[str]:
    """解析用于最终回答生成的模型 API Key。

    这部分用于 benchmark 的最终答复生成，优先读取模型配置中的 api_key，
    否则从环境变量中查找，方便在不同执行环境下复用同一套配置。
    """
    raw = str(model_config.get("api_key", "")).strip()
    if raw:
        return raw
    env_name = str(model_config.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
    return os.getenv(env_name)


def _normalize_backend(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    """标准化后端名称，统一为 A-MEM 内部可识别的 backend 字符串。

    这样无论配置中写的是 openai、openai_api、qwen_local 还是 sglang，
    都可以被 A-MEM 的运行时正确识别。
    """
    backend = str(method_config.get("backend", "")).strip().lower()
    if backend:
        return backend

    provider = str(model_config.get("provider", "")).strip().lower()
    if provider == "openai_api":
        return "openai"
    if provider in {"qwen_local", "sglang"}:
        return "sglang"
    return provider or "openai"


def _resolve_model_name(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    """解析要用于 A-MEM 推理的模型名称。

    方法配置中的 llm_model 优先级最高；若没有显式设置，则回退到模型配置。
    这样能在实验中灵活切换不同的 LLM。
    """
    explicit = str(method_config.get("llm_model", "")).strip()
    if explicit:
        return explicit
    return str(model_config.get("model", "")).strip() or str(model_config.get("name", "gpt-4o-mini")).strip()


def _resolve_sglang_host(method_config: Dict[str, Any]) -> str:
    """解析 SGLang 服务主机地址。

    当后端使用本地或远程 SGLang 服务时，需要知道访问地址。
    默认会指向本机的 localhost，便于开发调试。
    """
    return str(method_config.get("sglang_host", "http://localhost")).strip() or "http://localhost"


def _resolve_sglang_port(method_config: Dict[str, Any]) -> int:
    """解析 SGLang 服务监听端口。

    如果配置中未提供端口，则回退到默认值 30000，以保持和常见本地部署一致。
    """
    try:
        return int(method_config.get("sglang_port", 30000))
    except Exception:
        return 30000


def _dialogue_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    """把一轮对话内容拼成可读的文本片段。

    这里会提取 user 和 assistant 的文本，并按固定格式拼成一段自然语言。
    这段文本会被用于后续构造 A-MEM 的记忆条目。
    """
    parts: List[str] = []
    user_text = str(round_payload.get("user", "")).strip()
    assistant_text = str(round_payload.get("assistant", "")).strip()
    if user_text:
        parts.append(f"{speaker_a}: {user_text}")
    if assistant_text:
        parts.append(f"{speaker_b}: {assistant_text}")
    return "\n".join(parts).strip()


def _round_image_blocks(raw_dialogue: Dict[str, Any]) -> List[Tuple[str, str]]:
    """整理当前轮次中的图片信息，返回 image_id 与 caption 的配对列表。

    这一步把原始数据中的图片字段规范化成统一格式，便于后续写入记忆文本。
    如果某些图片缺少 caption，会保留空字符串，避免中断处理流程。
    """
    input_images = raw_dialogue.get("input_image", []) or []
    captions = raw_dialogue.get("image_caption", []) or []
    image_ids = raw_dialogue.get("image_id", []) or []
    blocks: List[Tuple[str, str]] = []
    for idx, _ in enumerate(input_images):
        image_id = ""
        if idx < len(image_ids):
            image_id = str(image_ids[idx]).strip()
        caption = ""
        if idx < len(captions):
            caption = str(captions[idx]).strip()
        blocks.append((image_id, caption))
    return blocks


def build_a_mem_note_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    """将一轮对话及其图片信息转成适合写入 A-MEM 的记忆文本。

    这一段文本会作为单条记忆被 A-MEM 存入系统，因此需要尽量保留对话内容
    以及可用的图片 ID / caption 信息，以提高后续检索效果。
    """
    raw_dialogue = round_payload.get("raw", {}) or {}
    text = _dialogue_text(round_payload, speaker_a, speaker_b)
    lines: List[str] = [text] if text else []
    for image_id, caption in _round_image_blocks(raw_dialogue):
        lines.extend(
            [
                "image:",
                f"image_id: {image_id}",
                f"image_caption: {caption}",
            ]
        )
    return "\n".join(lines).strip()


def _question_with_image_caption(qa: Dict[str, Any], question: str) -> str:
    """把问题图像的 caption 拼接到问题文本中，增强检索与回答质量。

    如果问题附带图片且已经有 caption，就把这些信息与原始问题合并。
    这样在检索阶段可以更容易利用视觉语义线索，而不是只看文本问句。
    """
    query = question.strip()
    question_image = str(qa.get("question_image", "")).strip()
    question_caption = qa.get("image_caption")
    if not question_image or not question_caption:
        return query

    if isinstance(question_caption, list):
        caption_text = " ".join(str(item).strip() for item in question_caption if str(item).strip())
    else:
        caption_text = str(question_caption).strip()
    if not caption_text:
        return query
    return f"{query}\nquestion's image:\nimage_caption: {caption_text}"


def _load_json_object(raw: str, fallback_key: str) -> str:
    """从模型响应中提取 JSON 字段内容，兼容多种返回格式。

    某些模型会返回标准 JSON 字符串，有些则可能直接返回普通文本。
    这里会优先尝试解析 JSON，再从指定字段中提取值，保证接口更稳定。
    """
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            value = payload.get(fallback_key, "")
            if isinstance(value, str):
                return value.strip()
    except Exception:
        pass
    return raw.strip()


class AMemAgent:
    """A-MEM 的运行时代理封装器。

    这个类负责初始化 A-MEM 的记忆系统、检索器、以及最终回答模型，
    并提供一套简洁的接口给 benchmark 调用。
    """

    def __init__(self, method_config: Dict[str, Any], model_config: Dict[str, Any]) -> None:
        """根据配置创建 A-MEM 运行时所需的所有组件。

        这里会初始化：
        1. 记忆系统，用于保存和检索记忆；
        2. LLMController，用于生成检索关键词；
        3. 最终问答客户端，用于生成最后的短答案。
        """
        agentic_memory_system_cls, llm_controller_cls = _load_a_mem_classes()
        embedding_model = str(method_config.get("embedding_model", "all-MiniLM-L6-v2")).strip() or "all-MiniLM-L6-v2"
        backend = _normalize_backend(method_config, model_config)
        model_name = _resolve_model_name(method_config, model_config)
        api_key = _resolve_openai_api_key(method_config) if backend == "openai" else None
        llm_base_url = _resolve_openai_base_url(method_config, model_config)
        llm_timeout = _resolve_openai_timeout(method_config, model_config)
        self.answer_model = str(model_config.get("model", "")).strip() or "gpt-4.1-nano"
        self.answer_provider = str(model_config.get("provider", "")).strip().lower() or "openai_api"
        self.answer_api_key = _resolve_model_api_key(model_config)
        self.answer_base_url = str(model_config.get("base_url", "https://api.openai.com/v1")).strip() or "https://api.openai.com/v1"
        self.answer_timeout = int(model_config.get("timeout", 90) or 90)
        self.retrieve_k = max(1, int(method_config.get("retrieve_k", 10)))
        self.temperature_c5 = float(method_config.get("temperature_c5", 0.5))
        self.memory_system = agentic_memory_system_cls(
            model_name=embedding_model,
            llm_backend=backend,
            llm_model=model_name,
            api_key=api_key,
            api_base=llm_base_url,
            api_timeout=llm_timeout,
            sglang_host=_resolve_sglang_host(method_config),
            sglang_port=_resolve_sglang_port(method_config),
        )
        self.retriever_llm = llm_controller_cls(
            backend=backend,
            model=model_name,
            api_key=api_key,
            api_base=llm_base_url,
            api_timeout=llm_timeout,
            sglang_host=_resolve_sglang_host(method_config),
            sglang_port=_resolve_sglang_port(method_config),
        )
        self.answer_client: Optional[OpenAI] = None
        if self.answer_provider in ("openai_api", "gemini_api"):
            if not self.answer_api_key:
                raise ValueError("API key not found for benchmark answer model.")
            self.answer_client = OpenAI(api_key=self.answer_api_key, base_url=self.answer_base_url, timeout=self.answer_timeout)
        self._answer_retryable_errors = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

    def _create_answer_with_retry(self, **kwargs: Any) -> Any:
        """带重试机制的回答生成调用。

        对于联网模型，偶尔会出现超时、限流或服务器异常。
        这里会按指数退避策略进行最多几次重试，尽量提升稳定性。
        """
        if self.answer_client is None:
            raise ValueError("OpenAI answer client is not initialized.")
        max_retries = 4
        for attempt in range(max_retries):
            try:
                return self.answer_client.chat.completions.create(**kwargs)
            except self._answer_retryable_errors:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.5 * (2 ** attempt))

    def add_memory(self, content: str, timestamp: Optional[str]) -> None:
        """把一条记忆文本写入 A-MEM 的记忆库。

        content 是已经格式化好的对话与图片信息；timestamp 用于记录时间线。
        这一步相当于把 benchmark 数据中每一轮对话都“记住”。
        """
        self.memory_system.add_note(content, time=timestamp)

    def retrieve_memory(self, query: str) -> str:
        """根据查询文本检索相关记忆片段。

        返回值是原始的检索结果字符串，后续会被拼装成 prompt 中的上下文。
        这里的 k 控制一次最多检索多少条记忆。
        """
        return self.memory_system.find_related_memories_raw(query, k=self.retrieve_k)

    def generate_query(self, question: str) -> str:
        """为问题生成一组检索关键词，作为记忆检索的输入。

        A-MEM 的检索过程依赖更高质量的 query；因此先让 LLM 提取关键词，
        再用这些关键词去召回更相关的历史记忆片段。
        """
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

Question: {question}

Format your response as a JSON object with a "keywords" field containing the selected text.

Example response format:
{{"keywords": "keyword1, keyword2, keyword3"}}"""
        response = self.retriever_llm.llm.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {"keywords": {"type": "string"}},
                        "required": ["keywords"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        return _load_json_object(response, "keywords")

    def answer_question(self, question: str) -> Tuple[str, str, str]:
        """执行完整的问答流程：关键词生成 → 记忆检索 → 最终答案生成。

        返回值包含：
        1. 最终答案；
        2. 供调试的 prompt；
        3. 检索到的上下文字符串。
        这个接口方便 benchmark 记录中间状态并排查问题。
        """
        # 先用 LLM 从问题中抽取一组检索关键词，用于召回相关记忆。
        # keywords是一组由逗号分隔的关键词字符串，例如 "keyword1, keyword2, keyword3"。A-MEM 内部会用这些关键词去匹配记忆库中的文本，找到最相关的历史片段。
        keywords = self.generate_query(question)
        # 再用这些关键词在 A-MEM 的记忆库中检索最相关的历史片段。
        context = self.retrieve_memory(keywords)
        # 组装给最终回答模型的用户提示词，并要求它尽量使用上下文中的原词。
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:
"""
        # 如果没有初始化可用的回答客户端，就直接报错，避免后续调用失败。
        if self.answer_client is None:
            raise ValueError(
                f"Unsupported answer model provider for A-MEM benchmark QA: {self.answer_provider}. "
                "Use an OpenAI API model config for final answer generation."
            )
        # 调用带重试机制的回答接口生成最终短答案。
        response = self._create_answer_with_retry(
            # 指定使用的模型名。
            model=self.answer_model,
            # 把系统提示和用户提示拼成消息列表，要求模型按 JSON 返回答案。
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": user_prompt},
            ],
            # 约束模型返回的 JSON schema，方便稳定解析 answer 字段。
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            # 关闭随机性，保证答案更稳定。
            temperature=0.0,
        )
        # 从模型响应中提取回答文本内容。
        raw_response = response.choices[0].message.content
        # 把 JSON 字符串解析成最终答案，并把 prompt 与检索结果一并返回给调用方。
        return _load_json_object(raw_response, "answer"), user_prompt, context


class AMemMethod(HistoryMethod):
    """A-MEM 记忆方法的 benchmark 适配器。

    这个类把 A-MEM 的记忆系统接入到 MemEye 的通用评测流程中，
    负责把历史对话写入记忆、在问答时做检索，并返回最终预测结果。
    """

    name = "a_mem"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化方法配置，并准备运行时状态容器。"""
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._agent: Optional[AMemAgent] = None
        self._speaker_a: str = "user"
        self._speaker_b: str = "assistant"
        self._debug_rows: List[Dict[str, Any]] = []

    def _ensure_caption_preprocessed(self, dataset: MemoryBenchmarkDataset) -> None:
        """检查数据是否已经具备 A-MEM 所需的图像 caption。

        A-MEM 的适配实现是文本化记忆，因此需要每轮对话的 image_caption 信息。
        如果缺少这些 caption，就会直接报错，避免后续检索结果不可信。
        """
        if not bool(self.config.get("caption_preprocessed", True)):
            return
        missing_rounds: List[str] = []
        for round_id, payload in dataset.rounds.items():
            raw = payload.get("raw", {}) or {}
            images = raw.get("input_image", []) or []
            if not images:
                continue
            captions = raw.get("image_caption", []) or []
            if len(captions) != len(images):
                missing_rounds.append(round_id)
        if missing_rounds:
            raise ValueError(
                "A-MEM text-only adaptation requires Mem-Gallery-style image captions. "
                "Run the caption preprocess first. Missing/invalid image_caption for rounds: "
                + ", ".join(missing_rounds[:10])
                + ("..." if len(missing_rounds) > 10 else "")
            )

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        """确定调试输出目录，便于保存中间执行轨迹。

        这些日志可以帮助研究者查看 A-MEM 在某个任务中写入了哪些记忆、
        检索了哪些上下文，以及最终答案是如何生成的。
        """
        task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
        safe_task = task_name.lower().replace(" ", "_").replace("/", "_")
        return (REPO_ROOT / "output" / safe_task / "a_mem").resolve()

    def _flush_debug(self, dataset: MemoryBenchmarkDataset) -> None:
        """把调试轨迹写入 JSON 文件，方便后续分析。

        这里保存的数据包括每一条写入的记忆、问答调用和检索结果。
        这样可以在实验结束后复盘 A-MEM 的真实行为。
        """
        if not self._debug_rows:
            return
        payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "rows": self._debug_rows,
        }
        write_json(self._debug_dir(dataset) / "debug_trace.json", payload)

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        """确保 A-MEM 实例只初始化一次，并把所有历史记忆写入系统。

        这一步会遍历整个数据集，把每轮对话转成可存储的记忆文本并写入 A-MEM。
        如果同一个 dataset 已经初始化过，会直接复用缓存，避免重复加载。
        """
        dataset_id = id(dataset)
        if self._agent is not None and self._dataset_key == dataset_id:
            return

        self._ensure_caption_preprocessed(dataset)
        self._debug_rows = []
        model_config = dict(self.config.get("_model_cfg", {}))
        self._agent = AMemAgent(self.config, model_config)

        character_profile = dataset.data.get("character_profile", {}) or {}
        speaker_name = str(character_profile.get("name", "")).strip()
        self._speaker_a = f"user ({speaker_name})" if speaker_name else "user"
        self._speaker_b = "assistant"

        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    continue
                round_payload = dataset.rounds[round_id]
                note_text = build_a_mem_note_text(round_payload, self._speaker_a, self._speaker_b)
                if not note_text:
                    continue
                timestamp = str(session.get("date", "")).strip() or None
                assert self._agent is not None
                self._agent.add_memory(note_text, timestamp)
                self._debug_rows.append(
                    {
                        "type": "stored_memory",
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "timestamp": timestamp or "",
                        "text": note_text,
                    }
                )

        self._dataset_key = dataset_id
        self.runtime_info["num_memories"] = len(self._debug_rows)
        self.runtime_info["debug_dir"] = str(self._debug_dir(dataset))
        self._flush_debug(dataset)

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        """为单个 QA 样本生成预测答案。

        流程包括：先确保记忆已经写入，再把当前问题增强为带 image caption 的查询，
        调用 A-MEM 检索并生成回答，最后把调试信息记录下来。

        qa：当前这道题的“题目对象”。它是从 dataset.iter_qas(...) 里取出来的一个字典。
        里面不仅有题干，还包含答案、选项、题目 ID、所属 session、clue、point 等元信息。
        qa["question"] = 题干
        qa["answer"] = 标准答案
        qa["options"] = 选项
        qa["session_id"] / qa["clue"] = 上下文来源信息
        question：当前这道题“实际发给模型的 prompt 文本”。它不是原始 qa 字典，而是由代码提前拼出来的字符串。例如在 MCQ 分支里，它会变成：“题干 + A. xxx + B. yyy + … + Answer with ONLY the option letter …”

        """
        # 先保证当前 dataset 的记忆已经初始化并写入 A-MEM。
        self._ensure_initialized(dataset)
        # 这里的 agent 一定已经创建完成，否则后续调用会出错。
        assert self._agent is not None
        # 把问题关联图片的 caption 拼接到question上，形成更完整的检索查询。
        query = _question_with_image_caption(qa, question)
        # 调用 A-MEM 的内部流程得到最终答案、提示词和检索到的上下文。
        answer, answer_prompt, retrieved = self._agent.answer_question(query)
        # 把这次问答的记录追加到调试日志中，方便复盘和分析。
        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": query,
                "retrieved_memories": retrieved,
                "answer_prompt": answer_prompt,
                "prediction": answer,
            }
        )
        # 把更新后的调试信息立即写入文件，避免中途丢失。
        self._flush_debug(dataset)
        # 返回模型给出的最终预测答案。
        return answer

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        """A-MEM 采用自主管理记忆，不走传统的 history 构造流程。

        因此这里返回空列表，表示不需要额外拼装历史上下文。
        实际的记忆检索与回答逻辑都在 A-MEM 自身内部完成。
        """
        return []
