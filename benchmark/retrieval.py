import json
import logging
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

log = logging.getLogger(__name__)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(_handler)
    log.propagate = False
log.setLevel(logging.INFO)


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

    def _candidate_log_limit(self) -> int:
        return max(5, self.top_k)

    def _candidate_rows(self, scored: List[Tuple[float, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [row for _, _, row in scored[: self._candidate_log_limit()]]

    def _clue_coverage(self, clue_rounds: List[str], round_ids: List[str]) -> Dict[str, Any]:
        round_set = set(round_ids)
        ranks = {round_id: idx + 1 for idx, round_id in enumerate(round_ids)}
        hits = [round_id for round_id in clue_rounds if round_id in round_set]
        misses = [round_id for round_id in clue_rounds if round_id not in round_set]
        return {
            "hit_count": len(hits),
            "total_count": len(clue_rounds),
            "hit_round_ids": hits,
            "missed_round_ids": misses,
            "hit_ranks": {round_id: ranks[round_id] for round_id in hits},
        }

    def _build_debug_info(
        self,
        qa: Dict[str, Any],
        seed_round_ids: List[str],
        selected_round_ids: List[str],
        top_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # 把题目的 clue 也纳入调试信息，方便观察命中是否与 oracle 相关。
        clue_rounds = list(qa.get("clue", []) or [])
        seed_coverage = self._clue_coverage(clue_rounds, seed_round_ids)
        selected_coverage = self._clue_coverage(clue_rounds, selected_round_ids)
        debug: Dict[str, Any] = {
            "retrieval_backend": self.config.get("retrieval_backend", "legacy_sparse"),
            "retrieval_corpus": self.corpus_meta.get("retrieval_corpus", "round_text"),
            "method_modality": self.corpus_meta.get("modality", _normalize_modality(self.config)),
            "top_k": self.top_k,
            "neighbor_window": self.neighbor_window,
            "seed_round_ids": seed_round_ids,
            "selected_round_ids": selected_round_ids,
            "top_candidates": top_candidates,
            "clue_round_ids": clue_rounds,
            "seed_clue_coverage": seed_coverage,
            "selected_clue_coverage": selected_coverage,
            "clue_hit_count": selected_coverage["hit_count"],
            "corpus_entry_count": self.corpus_meta.get("corpus_entry_count", 0),
        }
        notes_path = self.corpus_meta.get("retrieval_notes_json")
        if notes_path:
            debug["retrieval_notes_json"] = notes_path
        return debug

    def _log_retrieval_debug(self, qa: Dict[str, Any], debug: Dict[str, Any]) -> None:
        question = str(qa.get("question", "")).strip()
        seed_cov = debug.get("seed_clue_coverage", {})
        selected_cov = debug.get("selected_clue_coverage", {})
        log.info(
            "[retrieval] backend=%s modality=%s top_k=%s neighbor_window=%s question=%r",
            debug.get("retrieval_backend"),
            debug.get("method_modality"),
            debug.get("top_k"),
            debug.get("neighbor_window"),
            question,
        )
        log.info(
            "[retrieval] seed_rounds=%s selected_rounds=%s",
            debug.get("seed_round_ids", []),
            debug.get("selected_round_ids", []),
        )
        log.info(
            "[retrieval] clue_coverage seed=%s/%s hits=%s misses=%s ranks=%s | selected=%s/%s hits=%s misses=%s ranks=%s",
            seed_cov.get("hit_count", 0),
            seed_cov.get("total_count", 0),
            seed_cov.get("hit_round_ids", []),
            seed_cov.get("missed_round_ids", []),
            seed_cov.get("hit_ranks", {}),
            selected_cov.get("hit_count", 0),
            selected_cov.get("total_count", 0),
            selected_cov.get("hit_round_ids", []),
            selected_cov.get("missed_round_ids", []),
            selected_cov.get("hit_ranks", {}),
        )
        for rank, candidate in enumerate(debug.get("top_candidates", []), start=1):
            log.info("[retrieval] candidate_rank=%s %s", rank, candidate)

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
            self._candidate_rows(scored),
        )


class _DenseTextRetriever(_BaseRetriever):
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        # 先复用父类的基础检索上下文，再加载文本 embedding 模型。
        super().__init__(dataset, config)
        from .embeddings import TextEmbedder

        # 读取配置里指定的 embedding 模型名；默认使用项目内置的 text embedder。
        self.text_embedding_model = str(config.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL))
        # 初始化文本 embedding 器，用于把问题和候选轮次都转换成向量。
        self.text_embedder = TextEmbedder(self.text_embedding_model)
        # 把候选语料从父类的 corpus_rows 复制出来，方便逐轮计算相似度。
        self.round_texts: List[Tuple[str, str]] = list(self.corpus_rows)
        # 把所有候选轮次的文本一次性编码成向量，避免每次检索都重复计算。
        texts = [text for _, text in self.round_texts]
        self.round_vectors = self.text_embedder.embed_batch(texts) if texts else []

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        # 把当前问题转成文本 embedding，作为检索时的查询向量。
        # query_text 是问题文本，query_vec 是问题的向量表示；如果问题文本为空，就直接返回空结果。
        query_text = str(qa.get("question", "")).strip()
        log.info("====== Dense Text Retrieving and embedding for question: %s =======", query_text)
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
            self._candidate_rows(scored),
        )
        debug["text_embedding_model"] = self.text_embedding_model
        debug["caption_text_included"] = self.corpus_meta.get("modality") == "text_only"
        debug["image_embeddings_built"] = False
        return selected_round_ids, debug


class _DenseMultimodalRetriever(_BaseRetriever):
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        # 先调用父类初始化，把数据集、配置和基础语料信息都准备好。
        super().__init__(dataset, config)
        # 从 embeddings 模块导入文本与多模态 embedding 的具体实现。
        from .embeddings import TextEmbedder, get_multimodal_embedder

        # 读取文本 embedding 模型名；如果配置里没写，就用默认值。
        self.text_embedding_model = str(config.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL))
        # 读取文本相似度与图像相似度的权重，用于后续混合打分。
        self.text_dense_weight = float(config.get("text_dense_weight", 1.0))
        self.image_dense_weight = float(config.get("image_dense_weight", 0.0))
        # 初始化文本 embedding 器，用来把问题和候选文本转成向量。
        self.text_embedder = TextEmbedder(self.text_embedding_model)
        # 读取多模态 embedding 模型名，默认使用 siglip2。
        self.mm_model = str(config.get("multimodal_embedding_model", "siglip2-base-patch16-384"))
        # 初始化多模态 embedding 器，用来把图像转成向量。
        self.mm_embedder = get_multimodal_embedder(self.mm_model)
        # 如果多模态 embedding 器不可用，就直接抛错，避免后面检索失败。
        if self.mm_embedder is None:
            raise RuntimeError(
                "semantic_rag dense_multimodal requires a working multimodal embedder; "
                "neither vLLM SigLIP2 nor local CLIP is available."
            )

        # 这个列表用于保存每个 round 的最终检索信息：round_id、文本向量、图像向量列表。
        self.round_rows: List[Tuple[str, Optional[List[float]], List[Tuple[str, List[float]]]]] = []
        # 收集需要批量文本编码的 round_id 和文本内容。
        text_batch_round_ids: List[str] = []
        text_batch_texts: List[str] = []
        # 收集需要逐张图像编码的任务。
        image_jobs: List[Tuple[str, str]] = []
        # 遍历所有候选 round，准备文本和图像的 embedding 任务。
        for round_id, corpus_text in self.corpus_rows:
            # 从数据集里取出当前 round 的原始信息，读取其中的图片路径。
            round_payload = dataset.rounds.get(round_id, {})
            images = list(round_payload.get("images", []) or [])
            # 如果当前 round 既没有文本也没有图像，就跳过它。
            if not corpus_text and not images:
                continue
            # 先把这一轮的占位项加入 round_rows，后面再补文本和图像向量。
            self.round_rows.append((round_id, None, []))
            # 如果有文本内容，就把它加入批量文本编码列表。
            if corpus_text:
                text_batch_round_ids.append(round_id)
                text_batch_texts.append(corpus_text)
            # 如果有图像，就把图片路径加入图像编码任务列表。
            if images:
                for image_path in images:
                    image_jobs.append((round_id, image_path))

        # 把所有文本候选一次性编码成向量，并按 round_id 建立映射。
        text_vectors_by_round: Dict[str, List[float]] = {}
        if text_batch_texts:
            for round_id, vec in zip(text_batch_round_ids, self.text_embedder.embed_batch(text_batch_texts)):
                text_vectors_by_round[round_id] = vec

        # 把所有图像候选逐张编码成向量，并按 round_id 分组存储。
        image_vectors_by_round: Dict[str, List[Tuple[str, List[float]]]] = {}
        for round_id, image_path in image_jobs:
            image_vectors_by_round.setdefault(round_id, []).append(
                (image_path, self.mm_embedder.embed_image(image_path))
            )

        # 把前面准备好的文本向量和图像向量合并回最终的 round_rows 结构。
        self.round_rows = [
            (round_id, text_vectors_by_round.get(round_id), image_vectors_by_round.get(round_id, []))
            for round_id, _, _ in self.round_rows
        ]

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        # 先取出当前问题文本，作为文本和图像两种 embedding 的查询输入。
        query_text = str(qa.get("question", "")).strip()
        log.info("====== Dense Multimodal retrieving for question: %s =======", query_text)
        # 如果问题为空，或者没有可用的候选轮次，就直接返回空结果。
        if not query_text or not self.round_rows:
            return [], self._build_debug_info(qa, [], [], [])

        # 初始化两个查询向量，分别用于文本相似度和图像相似度。
        text_query_vec: Optional[List[float]] = None
        image_query_vec: Optional[List[float]] = None
        # 只有在配置允许时，才生成文本 query 向量。
        if self.text_dense_weight > 0:
            text_query_vec = self.text_embedder.embed_query(query_text)
            log.info("[retrieval] text_dense_weight=%s built_text_query_vec_len=%s", self.text_dense_weight, len(text_query_vec) if text_query_vec else 0)
        # 只有在配置允许时，才生成图像 query 向量。
        if self.image_dense_weight > 0:
            image_query_vec = self.mm_embedder.embed_text(query_text)
            log.info("[retrieval] image_dense_weight=%s built_image_query_vec_len=%s", self.image_dense_weight, len(image_query_vec) if image_query_vec else 0)

        # 用于保存所有候选 round 的打分结果，后续按分数排序。
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        # 统计一共索引了多少张图像，用于 debug 信息。
        total_indexed_images = 0
        log.info("[retrieval] round_rows_count=%s top_k=%s neighbor_window=%s", len(self.round_rows), self.top_k, self.neighbor_window)
        # 遍历每个候选 round，分别计算文本得分和图像得分。
        for round_id, text_vec, image_items in self.round_rows:
            # 计算文本相似度分数；如果没有文本向量，就用 0 分。
            text_score = _dense_cosine(text_query_vec or [], text_vec or [])
            total_indexed_images += len(image_items)
            # 先默认没有最佳图像，图像得分为 0。
            best_image_path = ""
            image_score = 0.0
            # 在这一轮的所有图像里，找出与 query 最相似的一张。
            for image_path, image_vec in image_items:
                score = _dense_cosine(image_query_vec or [], image_vec or [])
                if score >= image_score:
                    image_score = score
                    best_image_path = image_path
            # 把文本分数和图像分数按权重混合成最终得分。
            score = (self.text_dense_weight * text_score) + (self.image_dense_weight * image_score)
            scored.append(
                (
                    score,
                    round_id,
                    {
                        "round_id": round_id,
                        "score": score,
                        "text_dense_score": text_score,
                        "image_dense_score": image_score,
                        "indexed_image_count": len(image_items),
                        "best_image_path": best_image_path,
                    },
                )
            )
        # 按最终得分从高到低排序，取前 top_k 个作为初始候选种子。
        # top 候选的详细内容改由统一日志函数打印。
        scored.sort(key=lambda item: (-item[0], item[1]))
        seed_round_ids = [round_id for _, round_id, _ in scored[: max(1, self.top_k)]]
        # 对种子轮次做邻居扩展，补充连续的上下文历史轮次。
        selected_round_ids = _expand_with_neighbors(
            self.dataset, seed_round_ids, self.session_ids, self.neighbor_window
        )
        # 组装 debug 信息，方便观察命中的轮次和打分细节。
        debug = self._build_debug_info(
            qa,
            seed_round_ids,
            selected_round_ids,
            self._candidate_rows(scored),
        )
        debug["text_embedding_model"] = self.text_embedding_model
        debug["multimodal_embedding_model"] = self.mm_model
        debug["text_dense_weight"] = self.text_dense_weight
        debug["image_dense_weight"] = self.image_dense_weight
        debug["caption_text_included"] = False
        debug["image_embeddings_built"] = total_indexed_images > 0
        debug["multi_image_indexing_enabled"] = True
        debug["indexed_image_count"] = total_indexed_images
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


def clear_retriever_cache() -> None:
    """Release all cached retrievers and their embedding vectors."""
    # 清空全局缓存，释放内存中的检索器和 embedding 向量。
    _RETRIEVER_CACHE.clear()


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
    # 用 log.info 打印检索后（含邻居扩展）的最终候选轮次，便于调试观察哪些轮次被选中。
    retriever._log_retrieval_debug(qa, debug)
    # 如果调用方传入了 runtime_info，就把检索调试信息写进去，方便追踪。
    if runtime_info is not None:
        runtime_info.clear()
        runtime_info.update(debug)
    # 返回最终要保留的 round_id 列表。
    return selected_round_ids
