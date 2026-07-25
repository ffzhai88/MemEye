import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import MemoryBenchmarkDataset, history_from_round_ids, validate_text_only_captions
from .retrieval import select_round_ids_for_qa


# ---------------------------------------------------------------------------
# Token estimation helpers for context-window truncation
# ---------------------------------------------------------------------------

# Rough chars-per-token ratio (conservative for English + mixed content)
_CHARS_PER_TOKEN = 4

# Estimated token cost per image, following Mem-Gallery (predefined token cost)
_IMAGE_TOKEN_COST = 765


def _normalize_modality(config: Dict[str, Any], method_name: str) -> str:
    raw = str(config.get("modality", "")).strip().lower()
    if raw in {"text_only", "multimodal", "no_visual"}:
        return raw
    if method_name in {"semantic_rag_multimodal", "full_context_multimodal"}:
        return "multimodal"
    if method_name in {"full_context_text_only", "semantic_rag_text_only"}:
        return "text_only"
    return "text_only"


def _estimate_turn_tokens(turn: Dict[str, Any]) -> int:
    """估算单条历史轮次的大致 token 开销，包括文本与图片两部分。"""
    # 把当前轮次的文本内容转成字符串，并按固定比例估算文本 token 数。
    text = str(turn.get("text", ""))
    text_tokens = max(1, len(text) // _CHARS_PER_TOKEN)
    # 把图片数量乘上预设的图片 token 开销，作为视觉内容的估计成本。
    image_tokens = len(turn.get("images", []) or []) * _IMAGE_TOKEN_COST
    # 返回文本和图片两部分的总估算 token 数。
    return text_tokens + image_tokens


def _truncate_history(history: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    """按“保留最近轮次、丢弃最老轮次”的方式截断历史上下文，使其不超过 token 上限。"""
    # 如果配置的 token 上限小于等于 0，就不做任何截断，直接返回原历史。
    if max_tokens <= 0:
        return history

    # 从最后一轮开始往前累加 token 开销，优先保留最近的历史。
    cumulative = 0
    # 默认保留全部历史；如果超限，再把过老的轮次裁掉。
    cutoff_idx = len(history)
    for i in range(len(history) - 1, -1, -1):
        # 估算当前轮次的 token 开销，并累加到总预算中。
        cumulative += _estimate_turn_tokens(history[i])
        # 一旦超出预算，就从当前位置之后开始保留，丢弃更早的轮次。
        if cumulative > max_tokens:
            cutoff_idx = i + 1
            break
    else:
        # 如果整个历史都没有超限，就直接返回原列表，不需要裁剪。
        return history

    # 返回保留最近轮次后的历史片段。
    return history[cutoff_idx:]


class HistoryMethod(ABC):
    name = "base"
    fixed_modality: Optional[str] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.runtime_info: Dict[str, Any] = {}
        self.modality = self.fixed_modality or _normalize_modality(self.config, self.name)

    # build_history 是所有方法必须实现的核心接口，负责根据数据集和 QA 信息构造适合模型输入的历史上下文列表。每个历史条目是一个字典，包含文本、图片等信息，具体格式由 history_from_round_ids 统一处理。
    @abstractmethod
    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _MemGalleryHistoryMethod(HistoryMethod):
    history_source = "history"

    def _validate_modality_inputs(self, dataset: MemoryBenchmarkDataset) -> None:
        # 只有 text_only 模式才需要额外校验 caption；如果当前方法是多模态模式，就跳过这一步。
        if self.modality != "text_only":
            return
        # 把 caption 校验结果写入 runtime_info，方便后续分析是否成功加载了文本替代信息。
        self.runtime_info.update(validate_text_only_captions(dataset.rounds))

    def _update_history_runtime(
        self,
        history: List[Dict[str, Any]],
        *,
        history_before_truncation: Optional[int] = None,
    ) -> None:
        # 先保留已有的 runtime 信息，避免覆盖别的统计字段。
        existing = dict(self.runtime_info)
        self.runtime_info.clear()
        self.runtime_info.update(existing)
        # 统一写入当前方法的基本运行时标签，方便后续结果汇总与日志分析。
        self.runtime_info.update(
            {
                "method_modality": self.modality,
                "history_source": self.history_source,
                "captions_loaded": self.modality == "text_only",
                "images_loaded": self.modality == "multimodal",
                "history_turns_after_truncation": len(history),
            }
        )
        # 如果有截断前的原始长度，就额外记录下来，便于观察上下文压缩比例。
        if history_before_truncation is not None:
            self.runtime_info["history_turns_before_truncation"] = history_before_truncation


class _MemGalleryFullContextMethod(_MemGalleryHistoryMethod):
    history_source = "full_context"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        """按全上下文方式构造历史记忆，支持文本和多模态两种模式。"""
        # 先检查当前模态是否需要额外的 caption 校验；text_only 模式下会补充 caption 相关运行信息。
        self._validate_modality_inputs(dataset)
        # 初始化一个空列表，用来保存所有会话轮次拼接后的历史上下文。
        history: List[Dict[str, Any]] = []
        # 遍历数据集中所有会话，按顺序把每轮对话加入历史。
        for sid in dataset.session_order():
            # 把当前会话中的轮次转换为可用于模型输入的历史条目。
            history.extend(
                history_from_round_ids(
                    dataset.get_session(sid),
                    dataset.rounds,
                    modality=self.modality,
                )
            )

        # 读取最大上下文长度，默认 128k tokens；如果配置中没有设置，就使用默认值。
        max_tokens = int(self.config.get("context_token_limit", 128_000))
        # 预留一部分 token 给系统提示、用户问题和生成答案，避免上下文把主对话挤爆。
        reserved = int(self.config.get("reserved_tokens", 1_000))
        # 用截断函数把历史压缩到可用 token 上限内，优先保留最近的轮次。
        truncated = _truncate_history(history, max_tokens - reserved)
        # 更新运行时统计信息：记录截断前后轮次数量，帮助分析上下文长度和模态开销。
        self._update_history_runtime(truncated, history_before_truncation=len(history))
        # 返回最终可用于推理的历史上下文列表。
        return truncated


class FullContextTextMethod(_MemGalleryFullContextMethod):
    """Local equivalent of Mem-Gallery FUMemory (text + captions, no images)."""

    name = "full_context_text_only"
    fixed_modality = "text_only"


class FullContextMultimodalMethod(_MemGalleryFullContextMethod):
    """Local equivalent of Mem-Gallery MMFUMemory."""

    name = "full_context_multimodal"
    fixed_modality = "multimodal"


class FullContextNoVisualMethod(_MemGalleryFullContextMethod):
    """Ablation: full dialogue text, no images, no captions. Tests context leakage."""

    name = "full_context_no_visual"
    fixed_modality = "no_visual"


class QuestionOnlyMethod(HistoryMethod):
    """Ablation: zero history, question + options only. Tests MCQ guessability."""

    name = "question_only"
    fixed_modality = "no_visual"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class TargetSessionContextMethod(_MemGalleryHistoryMethod):
    name = "target_session_context"
    history_source = "target_session_context"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._validate_modality_inputs(dataset)
        history: List[Dict[str, Any]] = []
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(
                history_from_round_ids(
                    dataset.get_session(sid),
                    dataset.rounds,
                    modality=self.modality,
                )
            )
        self._update_history_runtime(history)
        return history


class ClueOnlyContextMethod(_MemGalleryHistoryMethod):
    """Oracle retrieval: only include the exact rounds listed in QA clue field."""

    name = "clue_only_context"
    history_source = "clue_only"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._validate_modality_inputs(dataset)
        clue_round_ids = set()
        for clue in qa.get("clue", []):
            # clue format: "S1:4" or "D28:1" — the round_id is the full string
            clue_round_ids.add(clue)
        history: List[Dict[str, Any]] = []
        for sid in dataset.session_order():
            session = dataset.get_session(sid)
            for dialogue in session.get("dialogues", []):
                rid = dialogue.get("round", "")
                if rid in clue_round_ids:
                    round_payload = dataset.rounds.get(rid, {})
                    if round_payload:
                        history.extend(
                            history_from_round_ids(
                                {"dialogues": [dialogue]},
                                dataset.rounds,
                                modality=self.modality,
                            )
                        )
        self._update_history_runtime(history)
        return history


class _RetrievalHistoryMethod(_MemGalleryHistoryMethod):
    history_source = "retrieval"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 先做模态校验，text_only 需要检查 caption 是否可用，避免后续检索结果不完整。
        self._validate_modality_inputs(dataset)
        # 调用检索器，按当前题目 qa 选出相关的 round_id 列表。
        selected_round_ids = select_round_ids_for_qa(dataset, qa, self.config, runtime_info=self.runtime_info)
        # 如果检索结果为空，说明当前题目没有找到可用上下文，直接返回空历史。
        if not selected_round_ids:
            return []

        # 准备一个空列表，保存最终要送入模型的历史上下文。
        history: List[Dict[str, Any]] = []
        # 把检索到的 round_id 转成集合，便于后续快速匹配。
        allowed_round_ids = set(selected_round_ids)
        include_session_markers = bool(self.config.get("include_session_markers", False))
        # 遍历所有会话，把命中的轮次转成 history 条目。
        # history: 是一个list，每个元素是一个dict，包含role（user/assistant）、text（文本内容）、images（图片路径列表）等信息，具体格式由history_from_round_ids函数统一处理。
        for sid in dataset.session_order():
            history.extend(
                history_from_round_ids(
                    # 当前会话的原始对话结构。
                    dataset.get_session(sid),
                    # 全部 round 的元数据表。
                    dataset.rounds,
                    # 只保留检索命中的 round_id。
                    allowed_round_ids,
                    # 根据当前方法的模态决定是文本还是多模态历史。
                    modality=self.modality,
                    include_session_marker=include_session_markers,
                )
            )
        # 把检索结果的运行时信息写回 runtime_info，方便后续统计与分析。
        self.runtime_info.update(
            {
                # 当前方法的模态。
                "method_modality": self.modality,
                # 标识当前历史来源是检索而不是全量上下文。
                "history_source": self.history_source,
                # text_only 模式下是否加载了 caption。
                "captions_loaded": self.modality == "text_only",
                # multimodal 模式下是否加载了图像。
                "images_loaded": self.modality == "multimodal",
                # 检索后最终保留的历史轮次数量。
                "history_turns_after_truncation": len(history),
                "include_session_markers": include_session_markers,
            }
        )
        # 返回最终拼好的历史上下文列表，供 router.answer() 使用。
        return history


class SavedContextReplayMethod(_MemGalleryHistoryMethod):
    """Diagnostic-only replay of saved retrieval contexts and clue-oracle variants."""

    name = "saved_context_replay"
    fixed_modality = "multimodal"
    history_source = "saved_context_replay"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        source_path = Path(str(self.config.get("source_predictions_jsonl", ""))).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"Saved context predictions not found: {source_path}")
        self.variant = str(self.config.get("context_variant", "evi_replay_control")).strip()
        self.top_k = int(self.config.get("context_top_k", 10))
        self._rows_by_question: Dict[str, Dict[str, Any]] = {}
        self._rows_by_idx: Dict[int, Dict[str, Any]] = {}
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                question = str(row.get("question", "")).strip()
                if question:
                    self._rows_by_question[question] = row
                source_idx = int(row.get("idx", 0) or 0)
                if source_idx:
                    self._rows_by_idx[source_idx] = row

    @staticmethod
    def _unique(values: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            round_id = str(value).strip()
            if round_id and round_id not in seen:
                out.append(round_id)
                seen.add(round_id)
        return out

    def _select_round_ids(self, qa: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
        question = str(qa.get("question", "")).strip()
        benchmark_idx = int(qa.get("_benchmark_idx", 0) or 0)
        source = self._rows_by_idx.get(benchmark_idx) if benchmark_idx else None
        if source is None:
            source = self._rows_by_question.get(question)
        if source is None:
            raise KeyError(f"Question missing from saved context predictions: {question[:160]}")

        retrieved = self._unique(list(source.get("context_round_ids", []) or []))
        clues = self._unique(list(qa.get("clue", []) or []))
        clue_set = set(clues)
        capacity_exceeded = len(clues) > self.top_k

        if self.variant == "evi_replay_control":
            selected = retrieved[: self.top_k]
        elif self.variant == "retrieved_clues_only":
            selected = [round_id for round_id in retrieved if round_id in clue_set]
        elif self.variant == "oracle_complete_top10":
            if capacity_exceeded:
                selected = clues[: self.top_k]
            else:
                selected = clues + [
                    round_id for round_id in retrieved
                    if round_id not in clue_set
                ][: max(0, self.top_k - len(clues))]
        elif self.variant == "oracle_clue_only":
            selected = clues
        else:
            raise ValueError(f"Unsupported saved context variant: {self.variant}")

        selected = self._unique(selected)
        source_idx = int(source.get("idx", 0) or 0)
        return selected, {
            "context_variant": self.variant,
            "source_prediction_idx": source_idx,
            "source_context_round_ids": retrieved,
            "selected_round_ids": selected,
            "clue_round_ids": clues,
            "clue_capacity_exceeded": capacity_exceeded,
            "context_top_k": self.top_k,
            "removed_round_ids": [round_id for round_id in retrieved if round_id not in set(selected)],
            "added_round_ids": [round_id for round_id in selected if round_id not in set(retrieved)],
        }

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_round_ids, trace = self._select_round_ids(qa)
        allowed = set(selected_round_ids)
        history: List[Dict[str, Any]] = []
        for session_id in dataset.session_order():
            history.extend(
                history_from_round_ids(
                    dataset.get_session(session_id),
                    dataset.rounds,
                    allowed,
                    modality="multimodal",
                )
            )
        self.runtime_info.update(trace)
        self._update_history_runtime(history)
        print(
            f"[CONTEXT-REPLAY] variant={self.variant} "
            f"source_idx={trace['source_prediction_idx']} "
            f"selected={selected_round_ids} "
            f"added={trace['added_round_ids']} "
            f"removed={trace['removed_round_ids']} "
            f"clues={trace['clue_round_ids']} "
            f"capacity_exceeded={trace['clue_capacity_exceeded']}"
        )
        return history


class SavedRankingReplayMethod(_MemGalleryHistoryMethod):
    """Replay one saved retrieval ranking as raw multimodal QA context."""

    name = "saved_ranking_replay"
    fixed_modality = "multimodal"
    history_source = "saved_ranking_replay"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        source_path = Path(
            str(self.config.get("source_rankings_jsonl", ""))
        ).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Saved retrieval rankings not found: {source_path}"
            )
        self.source_path = source_path.resolve()
        self.source_benchmark = str(
            self.config.get("source_benchmark", "memeye")
        ).strip()
        self.source_dataset = str(
            self.config.get("source_dataset", "")
        ).strip()
        if not self.source_dataset:
            raise ValueError("source_dataset is required for saved ranking replay")
        self.ranking_strategy = str(
            self.config.get("ranking_strategy", "selective_vlm")
        ).strip()
        self.top_k = int(self.config.get("context_top_k", 10))
        self.include_session_markers = bool(
            self.config.get("include_session_markers", False)
        )
        if self.top_k <= 0:
            raise ValueError("context_top_k must be positive")

        self._rows_by_question_id: Dict[str, Dict[str, Any]] = {}
        with self.source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("benchmark", "")).strip() != self.source_benchmark:
                    continue
                if str(row.get("dataset", "")).strip() != self.source_dataset:
                    continue
                question_id = str(row.get("question_id", "")).strip()
                if not question_id:
                    continue
                if question_id in self._rows_by_question_id:
                    raise ValueError(
                        "Duplicate saved ranking row for "
                        f"{self.source_dataset}/{question_id}"
                    )
                self._rows_by_question_id[question_id] = row
        if not self._rows_by_question_id:
            raise ValueError(
                "No saved ranking rows found for "
                f"{self.source_benchmark}/{self.source_dataset}"
            )

    @staticmethod
    def _unique(values: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            round_id = str(value).strip()
            if round_id and round_id not in seen:
                out.append(round_id)
                seen.add(round_id)
        return out

    def build_history(
        self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        question_id = str(qa.get("question_id", "")).strip()
        if not question_id:
            raise KeyError("QA is missing question_id for saved ranking replay")
        source = self._rows_by_question_id.get(question_id)
        if source is None:
            raise KeyError(
                "Question missing from saved retrieval rankings: "
                f"{self.source_dataset}/{question_id}"
            )
        rankings = source.get("rankings")
        if not isinstance(rankings, dict):
            raise ValueError(
                f"Saved row has no rankings object: {self.source_dataset}/{question_id}"
            )
        ranking = rankings.get(self.ranking_strategy)
        if not isinstance(ranking, list):
            raise KeyError(
                f"Ranking strategy {self.ranking_strategy!r} missing for "
                f"{self.source_dataset}/{question_id}"
            )
        selected = self._unique(ranking)[: self.top_k]
        if len(selected) < self.top_k:
            raise ValueError(
                f"Saved ranking contains only {len(selected)} unique rounds for "
                f"{self.source_dataset}/{question_id}; expected {self.top_k}"
            )
        missing = [round_id for round_id in selected if round_id not in dataset.rounds]
        if missing:
            raise KeyError(
                f"Saved ranking references unknown rounds for "
                f"{self.source_dataset}/{question_id}: {missing}"
            )

        allowed = set(selected)
        history: List[Dict[str, Any]] = []
        for session_id in dataset.session_order():
            history.extend(
                history_from_round_ids(
                    dataset.get_session(session_id),
                    dataset.rounds,
                    allowed,
                    modality="multimodal",
                    include_session_marker=self.include_session_markers,
                )
            )
        self.runtime_info.update({
            "ranking_source": str(self.source_path),
            "ranking_strategy": self.ranking_strategy,
            "source_benchmark": self.source_benchmark,
            "source_dataset": self.source_dataset,
            "source_question_id": question_id,
            "selected_round_ids": selected,
            "context_top_k": self.top_k,
            "include_session_markers": self.include_session_markers,
        })
        self._update_history_runtime(history)
        print(
            f"[RANKING-REPLAY] dataset={self.source_dataset} "
            f"question_id={question_id} strategy={self.ranking_strategy} "
            f"selected={selected}"
        )
        return history


class SemanticRAGTextMethod(_RetrievalHistoryMethod):
    name = "semantic_rag_text_only"
    fixed_modality = "text_only"


class SemanticRAGMultimodalMethod(_RetrievalHistoryMethod):
    name = "semantic_rag_multimodal"
    fixed_modality = "multimodal"



class M2AAgentMethod(HistoryMethod):
    name = "m2a"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._system: Optional[Any] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._system is not None and self._dataset_key == dataset_id:
            return

        from .m2a import M2ASystem

        self._system = M2ASystem(self.config)
        sessions = dataset.session_order()
        print(f"[M2A] Building memory from {len(sessions)} session(s)...")
        self._system.process_all_sessions(dataset)
        self._dataset_key = dataset_id
        print(f"[M2A] Memory ready: {self._system.num_memories} semantic memories stored.")

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._system is not None
        return self._system.answer_question(question)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class MMAAgentMethod(HistoryMethod):
    """Confidence-aware multimodal memory agent (adapted from MMA)."""

    name = "mma"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._system: Optional[Any] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._system is not None and self._dataset_key == dataset_id:
            return

        from .mma import MMASystem

        self._system = MMASystem(self.config)
        sessions = dataset.session_order()
        print(f"[MMA] Building memory from {len(sessions)} session(s)...")
        self._system.process_all_sessions(dataset)
        self._dataset_key = dataset_id

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure_initialized(dataset)
        assert self._system is not None
        qa_images = question_images
        if qa_images is None:
            qa_images = qa.get("question_images") or qa.get("question_image") or None
        return self._system.answer_question(question, image_paths=qa_images)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class EVIMethod(HistoryMethod):
    """Evidence-grounded Visual Indexing: lightweight extraction + dual-track retrieval."""

    name = "evi"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._system: Optional[Any] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._system is not None and self._dataset_key == dataset_id:
            return
        from .evi import EVISystem

        self._system = EVISystem(self.config)
        sessions = dataset.session_order()
        print(f"[EVI] Processing {len(sessions)} session(s)...")
        self._system.process_all_sessions(dataset)
        self._dataset_key = dataset_id
        print(f"[EVI] Done: {self._system.num_indexed} vectors indexed.")

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure_initialized(dataset)
        assert self._system is not None
        answer = self._system.answer_question(question, qa=qa, question_images=question_images, dataset=dataset)
        self.runtime_info["final_context_round_ids"] = self._system.last_context_round_ids
        self.runtime_info["history_source"] = "evi_internal"
        self.runtime_info["history_turns_after_truncation"] = len(self._system.last_context_round_ids) * 2
        return answer

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


def get_method(method_name: str, config: Optional[Dict[str, Any]] = None) -> HistoryMethod:
    config = config or {}
    registry = {
        FullContextTextMethod.name: FullContextTextMethod,
        FullContextMultimodalMethod.name: FullContextMultimodalMethod,
        FullContextNoVisualMethod.name: FullContextNoVisualMethod,
        QuestionOnlyMethod.name: QuestionOnlyMethod,
        TargetSessionContextMethod.name: TargetSessionContextMethod,
        ClueOnlyContextMethod.name: ClueOnlyContextMethod,
        SavedContextReplayMethod.name: SavedContextReplayMethod,
        SavedRankingReplayMethod.name: SavedRankingReplayMethod,
        SemanticRAGTextMethod.name: SemanticRAGTextMethod,
        SemanticRAGMultimodalMethod.name: SemanticRAGMultimodalMethod,
        "semantic_rag_multimodal_session_markers": SemanticRAGMultimodalMethod,
        "semantic_rag_multimodal_nvembed": SemanticRAGMultimodalMethod,
        "semantic_rag_multimodal_nvembed_session_markers": SemanticRAGMultimodalMethod,
        M2AAgentMethod.name: M2AAgentMethod,
        MMAAgentMethod.name: MMAAgentMethod,
    }
    cls = registry.get(method_name)
    # 如果方法不再以上的registry中，就尝试按名称动态加载对应的类；如果仍然找不到，就默认使用 Mirix 方法，支持更多社区贡献的扩展方法。
    if cls is None:
        if method_name == "a_mem":
            from .a_mem import AMemMethod

            return AMemMethod(config=config)
        if method_name == "semantic_rag_multimodal_agentic":
            from .semantic_rag_agentic import SemanticRAGAgenticMethod

            return SemanticRAGAgenticMethod(config=config)
        if method_name == "evi":
            # EVIMethod is defined in this file
            return EVIMethod(config=config)
        if method_name == "memgpt":
            from .memgpt import MemGPTMethod

            return MemGPTMethod(config=config)
        if method_name == "gen_agents":
            from .gen_agents import GAMethod

            return GAMethod(config=config)
        if method_name == "evermemos":
            from .evermemos import EverMemOSMethod

            return EverMemOSMethod(config=config)
        if method_name == "reflexion":
            from .reflexion_method import ReflexionMethod

            return ReflexionMethod(config=config)
        if method_name == "simplemem":
            from .simplemem import SimpleMemMethod

            return SimpleMemMethod(config=config)
        if method_name == "memoryos":
            from .memoryos import MemoryOSMethod

            return MemoryOSMethod(config=config)
        from .mirix import get_mirix_method

        return get_mirix_method(method_name, config=config)
    return cls(config=config)
