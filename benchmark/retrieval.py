import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset import MemoryBenchmarkDataset, build_round_retrieval_text


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "into", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "with",
    "you", "your",
}

_RETRIEVER_CACHE: Dict[Tuple[Any, ...], "_BaseRetriever"] = {}


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _normalize_modality(config: Dict[str, Any]) -> str:
    raw = str(config.get("modality", "")).strip().lower()
    if raw in {"text_only", "multimodal"}:
        return raw

    method_name = str(config.get("name", "")).strip().lower()
    if method_name == "semantic_rag_multimodal":
        return "multimodal"
    return "text_only"


def _idf(documents: List[List[str]]) -> Dict[str, float]:
    num_docs = len(documents)
    doc_freq: Counter = Counter()
    for doc in documents:
        for token in set(doc):
            doc_freq[token] += 1
    return {
        token: math.log((1 + num_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {token: (count / total) * idf.get(token, 0.0) for token, count in counts.items()}


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dense_cosine(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(set(query_tokens) & set(doc_tokens)) / len(set(query_tokens))


def _normalize_backend(config: Dict[str, Any]) -> str:
    # 先看是否显式指定了 retrieval_backend；如果有，就直接使用它。
    backend = str(config.get("retrieval_backend", "")).strip().lower()
    # 如果没有显式指定，就根据方法名推断默认后端。
    if backend:
        return backend
    method_name = str(config.get("name", "")).strip().lower()
    # semantic_rag 系列默认走 dense_text，其他方法默认走 legacy_sparse。
    if method_name in {"semantic_rag_text_only", "semantic_rag_multimodal"}:
        return "dense_text"
    return "legacy_sparse"


def _normalize_corpus(config: Dict[str, Any]) -> str:
    # 默认检索语料是每轮对话文本；这里把配置值规范化成统一格式。
    corpus = str(config.get("retrieval_corpus", "round_text")).strip().lower()
    corpus = corpus or "round_text"
    # 当前实现只支持 round_text 这一种语料格式，避免混用不同的索引方式。
    if corpus != "round_text":
        raise ValueError(
            "Only retrieval_corpus=round_text is supported for the current benchmark methods. "
            f"Unsupported retrieval_corpus: {corpus}"
        )
    return corpus


def _dataset_round_order(dataset: MemoryBenchmarkDataset) -> List[str]:
    # 按 session 顺序把所有 round_id 组织成一个统一的遍历序列。
    ordered: List[str] = []
    for session_id in dataset.session_order():
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if round_id:
                ordered.append(round_id)
    return ordered


def _resolve_notes_path(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> Path:
    # 如果配置里显式指定了 notes 文件，就按它的路径解析；否则沿用默认命名规则。
    configured = str(config.get("retrieval_notes_json", "")).strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = (dataset.dialog_json_path.parent / path).resolve()
        return path
    return dataset.dialog_json_path.with_name(f"{dataset.dialog_json_path.stem}_notes.json")


def _load_note_texts(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> Tuple[Dict[str, str], Path]:
    # 先找到 notes 文件的实际路径，再读取它的内容。
    notes_path = _resolve_notes_path(dataset, config)
    if not notes_path.exists():
        raise FileNotFoundError(
            f"retrieval_corpus=notes requires a notes file at '{notes_path}'."
        )
    with notes_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_entries: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        notes_value = payload.get("notes", payload.get("entries", []))
        if isinstance(notes_value, dict):
            raw_entries = [
                {"round_id": round_id, "text": text}
                for round_id, text in notes_value.items()
            ]
        elif isinstance(notes_value, list):
            raw_entries = [entry for entry in notes_value if isinstance(entry, dict)]
    elif isinstance(payload, list):
        raw_entries = [entry for entry in payload if isinstance(entry, dict)]

    note_text_by_round: Dict[str, str] = {}
    for entry in raw_entries:
        round_id = str(entry.get("round_id", "")).strip()
        text = str(entry.get("text", "")).strip()
        if not round_id or not text or round_id not in dataset.rounds:
            continue
        note_text_by_round[round_id] = text
    return note_text_by_round, notes_path


def _build_corpus_rows(
    dataset: MemoryBenchmarkDataset,
    config: Dict[str, Any],
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    # 先规范化检索语料与模态，再据此构建候选文本列表。
    corpus = _normalize_corpus(config)
    modality = _normalize_modality(config)

    # rows 用来保存 (round_id, text) 对，后续给检索器做向量化或词频统计。
    rows: List[Tuple[str, str]] = []
    for round_id in _dataset_round_order(dataset):
        round_payload = dataset.rounds.get(round_id, {})
        text = build_round_retrieval_text(round_payload, modality=modality)
        if text:
            rows.append((round_id, text))
    return rows, {
        "retrieval_corpus": "round_text",
        "corpus_entry_count": len(rows),
        "modality": modality,
    }


def _expand_with_neighbors(
    dataset: MemoryBenchmarkDataset,
    seed_round_ids: List[str],
    session_ids: List[str],
    window: int,
) -> List[str]:
    # 如果不需要邻居扩展，就直接返回初始命中的 round_id。
    if window <= 0:
        return seed_round_ids

    # 用集合保存最终候选，避免重复。
    selected = set(seed_round_ids)
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        ordered_round_ids = [
            dialogue.get("round", "") for dialogue in session.get("dialogues", [])
        ]
        index_by_round_id = {rid: idx for idx, rid in enumerate(ordered_round_ids)}
        for round_id in list(seed_round_ids):
            if round_id not in index_by_round_id:
                continue
            idx = index_by_round_id[round_id]
            start = max(0, idx - window)
            end = min(len(ordered_round_ids), idx + window + 1)
            selected.update(ordered_round_ids[start:end])

    ordered: List[str] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if round_id in selected:
                ordered.append(round_id)
    return ordered


class _BaseRetriever:
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        # 记录数据集与检索配置，后续所有检索器都依赖这两部分信息。
        self.dataset = dataset
        self.config = config
        self.top_k = int(config.get("top_k", 10))
        self.neighbor_window = int(config.get("neighbor_window", 1))
        self.session_ids = dataset.session_order()
        self.corpus_rows, self.corpus_meta = _build_corpus_rows(dataset, config)

    def _build_debug_info(
        self,
        qa: Dict[str, Any],
        seed_round_ids: List[str],
        selected_round_ids: List[str],
        top_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # 把题目的 clue 也纳入调试信息，方便观察命中是否与 oracle 相关。
        clue_rounds = list(qa.get("clue", []) or [])
        debug: Dict[str, Any] = {
            "retrieval_backend": self.config.get("retrieval_backend", "legacy_sparse"),
            "retrieval_corpus": self.corpus_meta.get("retrieval_corpus", "round_text"),
            "method_modality": self.corpus_meta.get("modality", _normalize_modality(self.config)),
            "top_k": self.top_k,
            "neighbor_window": self.neighbor_window,
            "seed_round_ids": seed_round_ids,
            "selected_round_ids": selected_round_ids,
            "top_candidates": top_candidates,
            "clue_hit_count": sum(1 for rid in selected_round_ids if rid in clue_rounds),
            "corpus_entry_count": self.corpus_meta.get("corpus_entry_count", 0),
        }
        notes_path = self.corpus_meta.get("retrieval_notes_json")
        if notes_path:
            debug["retrieval_notes_json"] = notes_path
        return debug

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        raise NotImplementedError


class _SparseRetriever(_BaseRetriever):
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        # 先复用父类的基础初始化，再准备词频 / TF-IDF 相关参数。
        super().__init__(dataset, config)
        self.lexical_weight = float(config.get("lexical_weight", 0.35))
        self.semantic_weight = float(config.get("semantic_weight", 0.65))
        self.candidate_rows: List[Tuple[str, List[str]]] = []
        for round_id, text in self.corpus_rows:
            tokens = _tokenize(text)
            if tokens:
                self.candidate_rows.append((round_id, tokens))

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        # 把当前问题当作检索 query，提取词袋特征用于后续打分。
        query_text = str(qa.get("question", "")).strip()
        query_tokens = _tokenize(query_text)
        # 如果 query 为空，或者候选语料为空，就直接返回空结果。
        if not query_tokens or not self.candidate_rows:
            return [], self._build_debug_info(qa, [], [], [])

        # 先根据所有候选文档计算 IDF，再得到问题的 TF-IDF 向量。
        documents = [tokens for _, tokens in self.candidate_rows]
        idf = _idf(documents)
        query_vector = _tfidf_vector(query_tokens, idf)

        # 遍历候选轮次，计算每一轮与问题的综合得分。
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for round_id, tokens in self.candidate_rows:
            doc_vector = _tfidf_vector(tokens, idf)
            lexical_score = _keyword_overlap(query_tokens, tokens)
            semantic_score = _cosine_similarity(query_vector, doc_vector)
            score = (self.lexical_weight * lexical_score) + (self.semantic_weight * semantic_score)
            # 只保留分数大于 0 的候选，过滤掉明显不相关的轮次。
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    round_id,
                    {
                        "round_id": round_id,
                        "score": score,
                        "lexical_score": lexical_score,
                        "semantic_score": semantic_score,
                    },
                )
            )
        # 按得分从高到低排序，取前 top_k 个作为初始种子轮次。
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored:
            return [], self._build_debug_info(qa, [], [], [])

        seed_round_ids = [round_id for _, round_id, _ in scored[: max(1, self.top_k)]]
        # 在种子轮次周围扩展邻居窗口，补充上下文轮次。
        selected_round_ids = _expand_with_neighbors(
            self.dataset, seed_round_ids, self.session_ids, self.neighbor_window
        )
        return selected_round_ids, self._build_debug_info(
            qa,
            seed_round_ids,
            selected_round_ids,
            [row for _, _, row in scored[: max(5, self.top_k)]],
        )


class _DenseTextRetriever(_BaseRetriever):
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        # 先复用父类的基础检索上下文，再加载文本 embedding 模型。
        super().__init__(dataset, config)
        from .embeddings import TextEmbedder

        # 读取配置里指定的 embedding 模型名；默认使用项目内置的 text embedder。
        self.text_embedding_model = str(config.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL))
        # 初始化文本 embedding 器，用于把问题和候选轮次都转换成向量。
        text_embedding_kwargs = dict(config.get("text_embedding_kwargs") or {})
        self.text_embedder = TextEmbedder(
            self.text_embedding_model,
            **text_embedding_kwargs,
        )
        # 把候选语料从父类的 corpus_rows 复制出来，方便逐轮计算相似度。
        self.round_texts: List[Tuple[str, str]] = list(self.corpus_rows)
        # 把所有候选轮次的文本一次性编码成向量，避免每次检索都重复计算。
        texts = [text for _, text in self.round_texts]
        self.round_vectors = self.text_embedder.embed_batch(texts) if texts else []

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        # 把当前问题转成文本 embedding，作为检索时的查询向量。
        # query_text 是问题文本，query_vec 是问题的向量表示；如果问题文本为空，就直接返回空结果。
        query_text = str(qa.get("question", "")).strip()
        # 如果问题为空，或者没有可检索的候选轮次，就直接返回空结果。
        if not query_text or not self.round_texts:
            return [], self._build_debug_info(qa, [], [], [])

        # 生成问题向量后，依次和每个候选轮次的向量做余弦相似度打分。
        query_vec = self.text_embedder.embed_query(query_text)
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for (round_id, _), round_vec in zip(self.round_texts, self.round_vectors):
            score = _dense_cosine(query_vec, round_vec)
            scored.append(
                (
                    score,
                    round_id,
                    {
                        "round_id": round_id,
                        "score": score,
                        "text_dense_score": score,
                    },
                )
            )
        # 按相似度从高到低排序，取前 top_k 个作为检索种子。
        scored.sort(key=lambda item: (-item[0], item[1]))
        seed_round_ids = [round_id for _, round_id, _ in scored[: max(1, self.top_k)]]
        # 对种子轮次做邻居扩展，把上下文连续的几轮一并带进去。
        selected_round_ids = _expand_with_neighbors(
            self.dataset, seed_round_ids, self.session_ids, self.neighbor_window
        )
        # 组装调试信息，方便后续观察哪些轮次命中以及为什么命中。
        debug = self._build_debug_info(
            qa,
            seed_round_ids,
            selected_round_ids,
            [row for _, _, row in scored[: max(5, self.top_k)]],
        )
        debug["text_embedding_model"] = self.text_embedding_model
        debug["caption_text_included"] = self.corpus_meta.get("modality") == "text_only"
        debug["image_embeddings_built"] = False
        return selected_round_ids, debug


class _DenseMultimodalRetriever(_BaseRetriever):
    """Dense dialogue and raw-image retrieval with round-level score fusion."""

    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        super().__init__(dataset, config)
        from .embeddings import TextEmbedder
        from .image_retrieval import RawImageRoundIndex

        self.text_embedding_model = str(
            config.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL)
        )
        self.text_dense_weight = float(config.get("text_dense_weight", 1.0))
        self.image_dense_weight = float(config.get("image_dense_weight", 0.0))
        if self.text_dense_weight < 0 or self.image_dense_weight < 0:
            raise ValueError("Dense retrieval weights must be non-negative")
        if self.text_dense_weight == 0 and self.image_dense_weight == 0:
            raise ValueError("At least one dense retrieval weight must be positive")

        self.mm_model = str(
            config.get("multimodal_embedding_model", "siglip2-base-patch16-384")
        )
        text_embedding_kwargs = dict(config.get("text_embedding_kwargs") or {})
        self.text_embedder = (
            TextEmbedder(self.text_embedding_model, **text_embedding_kwargs)
            if self.text_dense_weight > 0
            else None
        )
        self.image_index = (
            RawImageRoundIndex(dataset, config) if self.image_dense_weight > 0 else None
        )

        text_vectors_by_round: Dict[str, List[float]] = {}
        if self.text_embedder is not None:
            text_round_ids = [round_id for round_id, text in self.corpus_rows if text]
            texts = [text for _, text in self.corpus_rows if text]
            if texts:
                vectors = self.text_embedder.embed_batch(texts)
                text_vectors_by_round = dict(zip(text_round_ids, vectors))

        self.round_rows: List[Tuple[str, Optional[List[float]]]] = []
        for round_id, corpus_text in self.corpus_rows:
            images = list(dataset.rounds.get(round_id, {}).get("images", []) or [])
            if corpus_text or images:
                self.round_rows.append((round_id, text_vectors_by_round.get(round_id)))

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        query_text = str(qa.get("question", "")).strip()
        if not query_text or not self.round_rows:
            return [], self._build_debug_info(qa, [], [], [])

        text_query_vec: Optional[List[float]] = None
        if self.text_embedder is not None:
            text_query_vec = self.text_embedder.embed_query(query_text)

        image_hits: List[Dict[str, Any]] = []
        image_meta: Dict[str, Any] = {}
        if self.image_index is not None:
            image_hits, image_meta = self.image_index.search(query_text, len(self.round_rows))
        image_hit_by_round = {str(item["round_id"]): item for item in image_hits}

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for round_id, text_vec in self.round_rows:
            text_score = _dense_cosine(text_query_vec or [], text_vec or [])
            image_hit = image_hit_by_round.get(round_id, {})
            image_score = float(image_hit.get("score", 0.0))
            score = (
                self.text_dense_weight * text_score
                + self.image_dense_weight * image_score
            )
            scored.append(
                (
                    score,
                    round_id,
                    {
                        "round_id": round_id,
                        "score": score,
                        "text_dense_score": text_score,
                        "image_dense_score": image_score,
                        "indexed_image_count": len(
                            self.dataset.rounds.get(round_id, {}).get("images", []) or []
                        ),
                        "best_image_path": str(image_hit.get("best_image_path", "")),
                        "image_rank": image_hit.get("rank"),
                    },
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        seed_round_ids = [round_id for _, round_id, _ in scored[: max(1, self.top_k)]]
        selected_round_ids = _expand_with_neighbors(
            self.dataset, seed_round_ids, self.session_ids, self.neighbor_window
        )
        debug = self._build_debug_info(
            qa,
            seed_round_ids,
            selected_round_ids,
            [row for _, _, row in scored[: max(5, self.top_k)]],
        )
        debug.update(
            {
                "text_embedding_model": self.text_embedding_model,
                "multimodal_embedding_model": self.mm_model,
                "text_dense_weight": self.text_dense_weight,
                "image_dense_weight": self.image_dense_weight,
                "caption_text_included": False,
                "image_embeddings_built": bool(image_meta.get("indexed_image_count", 0)),
                "multi_image_indexing_enabled": True,
                "image_ranking": image_hits[: max(5, self.top_k)],
                **image_meta,
            }
        )
        return selected_round_ids, debug

def _cache_key(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> Tuple[Any, ...]:
    # 先确定当前检索后端与语料类型，再基于这些配置生成缓存 key。
    backend = _normalize_backend(config)
    corpus = _normalize_corpus(config)
    relevant_keys = [
        "name",
        "top_k",
        "neighbor_window",
        "lexical_weight",
        "semantic_weight",
        "text_embedding_model",
        "multimodal_embedding_model",
        "multimodal_clip_fallback_model",
        "multimodal_clip_local_files_only",
        "use_image_embedding_cache",
        "image_embedding_cache_dir",
        "text_dense_weight",
        "image_dense_weight",
        "retrieval_backend",
        "retrieval_corpus",
        "retrieval_notes_json",
    ]
    dialog_mtime = dataset.dialog_json_path.stat().st_mtime_ns if dataset.dialog_json_path.exists() else None
    notes_path = None
    notes_mtime = None
    dataset_key = (
        str(dataset.dialog_json_path),
        str(dataset.image_root),
        dialog_mtime,
        str(notes_path) if notes_path is not None else "",
        notes_mtime,
    )
    return (dataset_key, backend, corpus) + tuple((key, config.get(key)) for key in relevant_keys)


def _get_retriever(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> _BaseRetriever:
    # 先用缓存 key 查找已经创建好的检索器，避免重复初始化开销。
    key = _cache_key(dataset, config)
    retriever = _RETRIEVER_CACHE.get(key)
    if retriever is not None:
        return retriever

    # semantic_rag 系列默认走 dense_text，其他方法默认走 legacy_sparse。
    backend = _normalize_backend(config)
    if backend == "dense_text":
        retriever = _DenseTextRetriever(dataset, config)
    elif backend == "dense_multimodal":
        retriever = _DenseMultimodalRetriever(dataset, config)
    else:
        retriever = _SparseRetriever(dataset, config)
    _RETRIEVER_CACHE[key] = retriever
    return retriever


def clear_retriever_cache(*, keep_embedding_models: bool = False) -> None:
    """Release cached indexes, optionally retaining shared embedding weights."""
    # 清空全局缓存，释放内存中的检索器和 embedding 向量。
    _RETRIEVER_CACHE.clear()
    if not keep_embedding_models:
        from .embeddings import clear_embedding_model_cache

        clear_embedding_model_cache()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available() and not keep_embedding_models:
            torch.cuda.empty_cache()
    except ImportError:
        pass


def select_round_ids_for_qa(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
    runtime_info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    # 先拿到当前数据集和配置对应的检索器实例。
    # semantic_rag 系列默认走 dense_text retriever，其他方法默认走 legacy_sparse retriever。
    retriever = _get_retriever(dataset, config)
    # 让检索器根据题目 q/a 选出候选 round_id。
    selected_round_ids, debug = retriever.select(qa)
    # 如果调用方传入了 runtime_info，就把检索调试信息写进去，方便追踪。
    if runtime_info is not None:
        runtime_info.clear()
        runtime_info.update(debug)
    # 返回最终要保留的 round_id 列表。
    return selected_round_ids
