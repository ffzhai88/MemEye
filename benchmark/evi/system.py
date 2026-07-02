"""
EVI v2: Multi-vector temporal indexing for multimodal long-term memory.

Two-phase lifecycle:
  1. process_all_sessions(dataset) — build vector index
  2. answer_question(question, qa) — retrieve + temporally assemble + VLM

QDMO mode (interaction_mode="qdmo"):
  Adds query-conditioned memory-to-memory interaction and emergent clustering
  before VLM reasoning:
    query → soft activation → active subset → interaction → clustering → LLM
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from .extractor import describe_image
from .indexes import VectorIndex, embed_text, sigmoid, softmax
from .schemas import MemoryCluster, VectorRecord
from .summarizer import load_all_summaries, summarize_session
from .vlm import VLMCallable, make_vlm_callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt: LLM directory retrieval — picks candidate sessions from summaries
# ---------------------------------------------------------------------------

DIRECTORY_RETRIEVAL_PROMPT = """You are a session directory analyst. Below is a list of conversation sessions, each with a date and summary.

{summary_block}

Question: {question}

Which of the above sessions are likely to contain information relevant to answering this question?
This is a coarse-grained pre-filter — it is much better to over-select and include extra sessions than to miss something relevant. When in doubt, include it.

Return ONLY a JSON object in the following format (no other text):
{{"candidate_sessions": ["SESSION_ID_1", "SESSION_ID_2"]}}"""

# ---------------------------------------------------------------------------
# EVISystem
# ---------------------------------------------------------------------------


class EVISystem:
    """Multi-vector temporal indexing system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._cfg = cfg  # full merged config (method + model + task)
        self._model_cfg = dict(cfg.get("_model_cfg", {}))

        # Vector index (single, stores both dialogue + image nodes)
        self._vector_index = VectorIndex()

        # Round metadata for temporal assembly
        self._round_order: List[str] = []           # ordered round_ids
        self._round_session: Dict[str, str] = {}    # round_id -> session_id
        self._round_text: Dict[str, str] = {}       # round_id -> dialogue text
        self._round_images: Dict[str, str] = {}     # round_id -> image_path (first only)
        self._session_dates: Dict[str, str] = {}    # session_id -> date
        self._session_summaries: Dict[str, str] = {}    # session_id -> summary text

        # VLM callable
        self._vlm: Optional[VLMCallable] = None
        self._embedder: Optional[Any] = None

        # Config flags
        self._use_directory_retrieval: bool = cfg.get("use_directory_retrieval", True)
        self._search_top_k: int = int(cfg.get("search_top_k", 20))  # legacy alias for max_rounds
        self._raw_search_k: int = int(cfg.get("raw_search_k", max(self._search_top_k * 3, 60)))
        self._max_rounds: int = int(cfg.get("max_rounds", self._search_top_k))
        self._diversity_boost: float = float(cfg.get("diversity_boost", 0.03))

        # ---- QDMO interaction mode ----
        self._interaction_mode: str = str(cfg.get("interaction_mode", "standard"))
        self._active_k: int = int(cfg.get("active_k", 30))
        self._softmax_temperature: float = float(cfg.get("softmax_temperature", 0.3))
        self._num_cluster_seeds: int = int(cfg.get("num_cluster_seeds", 5))
        self._cluster_threshold: float = float(cfg.get("cluster_threshold", 0.65))

        self._initialized = False

    # ---- lazy init ----

    def _ensure(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._vlm = make_vlm_callable(self._model_cfg)

        # Text embedder — configurable via config dict.
        # Priority: method config (self._cfg) → model config (self._model_cfg) → default.
        from ..embeddings import TextEmbedder

        embed_model = (
            self._cfg.get("text_embedding_model")
            or self._model_cfg.get("text_embedding_model")
            or TextEmbedder.DEFAULT_MODEL
        )
        embed_kwargs = dict(
            self._cfg.get("text_embedding_kwargs")
            or self._model_cfg.get("text_embedding_kwargs")
            or {}
        )
        self._embedder = TextEmbedder(embed_model, **embed_kwargs)

        if self._embedder.is_available:
            log.info("Embedder: %s loaded.", embed_model)
        else:
            log.warning("Embedder: %s not available — disabling retrieval", embed_model)
            self._embedder = None

    def _embed(self, text: str) -> List[float]:
        return embed_text(text, self._embedder)

    # ---- Phase 1: Indexing ----

    def process_all_sessions(self, dataset: Any) -> None:
        """
        构建向量索引：预处理全量对话数据，为后续检索做准备。

        ===== 核心流程（Phase 1）=====

        遍历所有 session（按数据集定义的顺序），对每个 session 内的每个 round 做两件事：

        1. 对话节点（Dialogue Node）：
           - 将 round 中的 user 文本和 assistant 文本拼接成一条纯文本
           - 如果该 round 有图片且数据中提供了 image_caption，则追加到文本后面
           - 用配置的 text embedding 模型（默认 all-MiniLM-L6-v2，也支持 NV-Embed-v2 等）将该文本转为 embedding → 存入向量索引
           - 节点类型标记为 "dialogue"

        2. 图片节点（Image Node）— 三轮 VLM 调用构建微型知识图谱，结果合并缓存：
           a) 图片本身 → 不固定数量的自由文本描述（VLM 自行决定从哪些角度描述）
              每个描述独立嵌入，类型标记为 "image_visual"
           b) 图片在对话中的角色 + 命名 → 基于完整上下文（当前 round + 该 session 之前所有 round），
              生成图片的语义名称（image_name）和上下文描述（context_description）
              image_name 独立嵌入，类型标记为 "image_name"
              context_desc 独立嵌入，类型标记为 "image_context"
           c) 三元组提取 → 以 image_name 为中心节点，提取结构化事实三元组
              每个事实独立嵌入，类型标记为 "image_fact"
           - 同一个图片可能对应 N+3+ 个向量（N 个 image_visual + 1 个 image_name + 1 个 image_context + M 个 image_fact）

        同时记录：
        - _round_order: 所有 round 的全局顺序（按 session + round 序排列）
        - _round_session: round_id → session_id 的映射
        - _round_text: round_id → 对话文本
        - _round_images: round_id → 图片路径
        - _session_dates: session_id → 日期

        ===== 关键设计决策 =====

        - Phase 1 完全 blind：不依赖任何问题信息。只做"尽可能全面的描述"
        - 每个图片生成 3 种描述而不是 1 种：不同的描述覆盖不同的检索需求
          （纯视觉匹配 vs 语义语境匹配 vs 名称匹配 vs 精确事实匹配）
        - 对话节点和图片节点放在同一个向量索引中：检索时可以同时命中
        - VLM 调用次数 = 有图片的 round 数量 × 3（已被缓存，重复运行不产生费用）
        """
        self._ensure()
        vlm = self._vlm
        if vlm is None:
            raise RuntimeError("VLM not initialized")

        session_ids = list(dataset.session_order())
        total_img = 0
        log.info("Processing %d sessions ...", len(session_ids))

        for sid in session_ids:
            session = dataset.get_session(sid)
            date = str(session.get("date", "")).strip() or "unknown"
            self._session_dates[sid] = date
            dialogues = session.get("dialogues", [])

            # Gather visible rounds (those with text)
            rounds_in_session: List[str] = []
            for d in dialogues:
                rid = d.get("round", "")
                if not rid:
                    continue
                rp = dataset.rounds.get(rid, {})
                ut = str(rp.get("user", "")).strip()
                at = str(rp.get("assistant", "")).strip()
                if ut or at:
                    rounds_in_session.append(rid)
                    self._round_order.append(rid)
                    self._round_session[rid] = sid
                    self._round_text[rid] = f"User: {ut}\nAssistant: {at}"

            if not rounds_in_session:
                continue

            log.info("  Session %s: %d rounds, date=%s", sid, len(rounds_in_session), date)

            # Process each round in the session
            prior_rounds_text: List[str] = []  # accumulate for context prompt
            for rid in rounds_in_session:
                rp = dataset.rounds.get(rid, {})
                images: List[str] = rp.get("images", []) or []

                # Build dialogue node text (round text + optional caption)
                dialogue_text = self._round_text.get(rid, "")
                raw = rp.get("raw", {})
                captions_raw = raw.get("image_caption", []) or []
                if captions_raw:
                    caption_text = str(captions_raw[0])
                    dialogue_text += f"\n[Image: {caption_text}]"

                # Embed dialogue node
                d_vec = self._embed(dialogue_text)
                if d_vec:
                    self._vector_index.add(VectorRecord(
                        id=f"dialogue_{rid}",
                        round_id=rid,
                        session_id=sid,
                        text=dialogue_text,
                        vector=d_vec,
                        node_type="dialogue",
                    ))

                # Process images
                for img_path in images:
                    self._round_images[rid] = img_path
                    log.info("    Round %s: 1 image, generating descriptions ...", rid)

                    # Two-stage extraction: image-only descriptions + context-aware description
                    img_node = describe_image(
                        image_path=img_path,
                        user_text=self._round_text.get(rid, ""),
                        prior_rounds_text="\n---\n".join(prior_rounds_text),
                        vlm_callable=vlm,
                    )
                    if img_node is None:
                        continue

                    # Embed each image description as a separate vector
                    for idx, desc_text in enumerate(img_node.image_descs):
                        if not desc_text:
                            continue
                        i_vec = self._embed(desc_text)
                        if i_vec:
                            self._vector_index.add(VectorRecord(
                                id=f"image_{rid}_desc{idx}",
                                round_id=rid,
                                session_id=sid,
                                text=desc_text,
                                vector=i_vec,
                                node_type="image_visual",
                                image_path=img_path,
                            ))

                    # Embed context-aware description
                    if img_node.context_desc:
                        i_vec = self._embed(img_node.context_desc)
                        if i_vec:
                            self._vector_index.add(VectorRecord(
                                id=f"image_{rid}_ctx",
                                round_id=rid,
                                session_id=sid,
                                text=img_node.context_desc,
                                vector=i_vec,
                                node_type="image_context",
                                image_path=img_path,
                            ))

                    # Embed image_name as a node-level vector
                    if img_node.image_name:
                        name_text = img_node.image_name
                        n_vec = self._embed(name_text)
                        if n_vec:
                            self._vector_index.add(VectorRecord(
                                id=f"image_{rid}_name",
                                round_id=rid,
                                session_id=sid,
                                text=name_text,
                                vector=n_vec,
                                node_type="image_name",
                                image_path=img_path,
                            ))

                    # Embed each fact as a separate vector
                    for fidx, fact in enumerate(img_node.facts):
                        f_text = fact.to_text()
                        f_vec = self._embed(f_text)
                        if f_vec:
                            self._vector_index.add(VectorRecord(
                                id=f"image_{rid}_fact{fidx}",
                                round_id=rid,
                                session_id=sid,
                                text=f_text,
                                vector=f_vec,
                                node_type="image_fact",
                                image_path=img_path,
                            ))

                    total_img += 1

                # After processing this round, add it to prior context for next round
                prior_rounds_text.append(self._round_text.get(rid, ""))

            # ---- Build session summary for LLM directory retrieval ----
            session_text_parts = [self._round_text.get(rid, "") for rid in rounds_in_session]
            session_text = "\n---\n".join(session_text_parts)

            session_node = summarize_session(
                session_id=sid,
                session_text=session_text,
                date=date,
                vlm_callable=vlm,
                use_cache=True,
            )
            if session_node is not None and session_node.summary:
                self._session_summaries[sid] = session_node.summary
                log.info("  Session summary cached: %d chars", len(session_node.summary))

        n_dialogue = len([r for r in self._vector_index._records if r.node_type == "dialogue"])
        n_visual = len([r for r in self._vector_index._records if r.node_type == "image_visual"])
        n_context = len([r for r in self._vector_index._records if r.node_type == "image_context"])
        n_name = len([r for r in self._vector_index._records if r.node_type == "image_name"])
        n_fact = len([r for r in self._vector_index._records if r.node_type == "image_fact"])
        log.info(
            "Indexing done: %d dialogue, %d visual, %d context, %d names, %d facts, "
            "%d session summaries — %d total vectors",
            n_dialogue, n_visual, n_context, n_name, n_fact,
            len(self._session_summaries),
            len(self._vector_index),
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON from LLM response, handling markdown fences."""
        import re
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r'(\{.*\})', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return None

    def _directory_retrieve(self, question_stem: str) -> Optional[set[str]]:
        """Stage 1: LLM directory retrieval.

        Loads all cached session summaries and asks the LLM to select
        candidate sessions relevant to the question. Returns a set of
        session_ids, or None if the directory cache is empty / the LLM
        response cannot be parsed (caller falls back to searching all).
        """
        summaries = load_all_summaries()
        if not summaries:
            log.info("  [DIR] No session summaries cached — skipping directory retrieval")
            return None

        # Format the summary block
        lines: List[str] = []
        for sid, info in summaries.items():
            summary_text = info.get("summary", "") if isinstance(info, dict) else str(info)
            sdate = (info.get("date", "") if isinstance(info, dict) else "")
            sdate_str = f" ({sdate})" if sdate else ""
            lines.append(f"Session: {sid}{sdate_str}")
            lines.append(f"Summary: {summary_text}")
            lines.append("")

        summary_block = "\n".join(lines)
        prompt = DIRECTORY_RETRIEVAL_PROMPT.format(summary_block=summary_block, question=question_stem)

        log.info("  [DIR] Asking LLM to select relevant sessions from %d summaries ...", len(summaries))
        raw = self._vlm("", prompt, []) if self._vlm else ""
        if not raw:
            log.warning("  [DIR] LLM returned empty response")
            return None

        log.info("  [DIR] Raw LLM response:\n%s", raw)
        parsed = self._extract_json(raw)
        if parsed is None:
            log.warning("  [DIR] Could not parse LLM response: %s ...", raw[:200])
            return None

        candidates = parsed.get("candidate_sessions", [])
        if not isinstance(candidates, list):
            log.warning("  [DIR] Unexpected format (candidate_sessions not a list)")
            return None

        candidate_set = {str(s).strip() for s in candidates if str(s).strip()}
        log.info("  [DIR] LLM selected %d candidate session(s): %s", len(candidate_set), sorted(candidate_set))
        return candidate_set

    # ================================================================
    # QDMO Pipeline (Step 2–6): query-conditioned interaction + clustering
    # ================================================================

    def _soft_activation(
        self,
        q_vec: List[float],
    ) -> List[Tuple[float, VectorRecord]]:
        """Step 2: Soft activation — compute relevance scores between query and ALL memory nodes.

        Unlike standard topK retrieval, this assigns a weight to every node
        via softmax over cosine similarities. The resulting distribution
        represents "how much each memory is awakened" by the query.

        Returns:
            List of (softmax_weight, VectorRecord) sorted descending by weight.
        """
        scored = self._vector_index.compute_all_scores(q_vec)
        if not scored:
            return []

        scores_only = [s for s, _ in scored]
        weights = softmax(scores_only, temperature=self._softmax_temperature)

        result = list(zip(weights, [rec for _, rec in scored]))
        result.sort(key=lambda x: x[0], reverse=True)

        log.info("  [QDMO ACTIVATION] %d nodes scored, temperature=%.2f", len(result), self._softmax_temperature)
        log.info("  [QDMO ACTIVATION] weight range: [%.6f, %.6f], top-5 weights: %s",
                 result[-1][0] if result else 0,
                 result[0][0] if result else 0,
                 [f"{w:.4f}" for w, _ in result[:5]])

        # Distribution bands
        bands = {"0.0-0.01": 0, "0.01-0.1": 0, "0.1-0.5": 0, "0.5-1.0": 0}
        for w, _ in result:
            if w < 0.01: bands["0.0-0.01"] += 1
            elif w < 0.1: bands["0.01-0.1"] += 1
            elif w < 0.5: bands["0.1-0.5"] += 1
            else: bands["0.5-1.0"] += 1
        log.info("  [QDMO ACTIVATION] weight distribution: %s",
                 " | ".join(f"{k}:{v}" for k, v in bands.items() if v > 0))

        # Node type composition in top weights
        top_types: dict = {}
        for _, rec in result[:50]:
            top_types[rec.node_type] = top_types.get(rec.node_type, 0) + 1
        log.info("  [QDMO ACTIVATION] top-50 node_type composition: %s",
                 " | ".join(f"{k}:{v}" for k, v in sorted(top_types.items())))

        return result

    def _select_active_subset(
        self,
        scored_nodes: List[Tuple[float, VectorRecord]],
    ) -> Tuple[List[VectorRecord], List[float]]:
        """Step 3: Active subset selection — pick top-K nodes for interaction.

        Keeps only the most activated nodes to control interaction complexity.
        The subset represents "which memories are allowed to influence each other."

        Args:
            scored_nodes: (weight, VectorRecord) list from _soft_activation.

        Returns:
            Tuple of (active_records, active_weights) both in the same order.
        """
        subset = scored_nodes[:self._active_k]
        active_records = [rec for _, rec in subset]
        active_weights = [w for w, _ in subset]

        log.info("  [QDMO SUBSET] Selected %d active nodes (max_active=%d)",
                 len(active_records), self._active_k)
        log.info("  [QDMO SUBSET] Weight range in active subset: [%.6f, %.6f]",
                 min(active_weights), max(active_weights))

        type_counts: dict = {}
        for rec in active_records:
            type_counts[rec.node_type] = type_counts.get(rec.node_type, 0) + 1
        log.info("  [QDMO SUBSET] Active subset composition: %s",
                 " | ".join(f"{k}:{v}" for k, v in sorted(type_counts.items())))

        return active_records, active_weights

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def _memory_interaction(
        self,
        q_vec: List[float],
        active_records: List[VectorRecord],
    ) -> Dict[str, List[float]]:
        """Step 4: Memory-to-memory interaction (query-conditioned).

        This is the core QDMO step. Each memory node is updated based on
        other nodes it is related to — but only when both are relevant to
        the query (query gate mechanism).

        For each pair (i, j):
            1. sim_ij = cosine(emb_i, emb_j)       — base similarity
            2. gate_i = sigmoid(cosine(q, emb_i))   — query relevance of i
               gate_j = sigmoid(cosine(q, emb_j))   — query relevance of j
            3. g_ij = gate_i * gate_j               — query gate (both must be relevant)
               (Simplified from sigmoid(gate_i + gate_j) for numerical stability)
            4. α_ij = g_ij * sim_ij                 — interaction weight
            5. h_i' = Σ_j α_ij * emb_j / Σ_j α_ij   — weighted update

        Args:
            q_vec: Query embedding.
            active_records: Records in the active subset.

        Returns:
            Dict mapping record_id -> updated embedding vector (interaction result).
        """
        n = len(active_records)
        if n == 0:
            return {}

        # Precompute query gates for all active nodes
        query_gates: Dict[str, float] = {}
        for rec in active_records:
            gate = sigmoid(self._cosine(q_vec, rec.vector))
            query_gates[rec.id] = gate

        log.info("  [QDMO INTERACT] %d active nodes, computing pairwise interactions ...", n)
        log.info("  [QDMO INTERACT] Query gate stats — min=%.4f max=%.4f mean=%.4f",
                 min(query_gates.values()), max(query_gates.values()),
                 sum(query_gates.values()) / len(query_gates))

        # Gate distribution bands
        gate_bands = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-1.0": 0}
        for g in query_gates.values():
            if g < 0.3: gate_bands["0.0-0.3"] += 1
            elif g < 0.5: gate_bands["0.3-0.5"] += 1
            elif g < 0.7: gate_bands["0.5-0.7"] += 1
            else: gate_bands["0.7-1.0"] += 1
        log.info("  [QDMO INTERACT] Gate distribution: %s",
                 " | ".join(f"{k}:{v}" for k, v in gate_bands.items() if v > 0))

        # Precompute base similarity matrix (n x n)
        sim_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_matrix[i][j] = 1.0
                else:
                    sim_matrix[i][j] = self._cosine(active_records[i].vector, active_records[j].vector)

        # Compute interaction weights and update each node
        updated_embeddings: Dict[str, List[float]] = {}
        interaction_weight_log: List[Tuple[str, str, float]] = []  # (src_id, tgt_id, weight)

        for i, rec_i in enumerate(active_records):
            weights: List[float] = []
            vectors: List[List[float]] = []

            for j, rec_j in enumerate(active_records):
                # Query-conditioned gate: both i and j must be relevant to query
                g_ij = query_gates[rec_i.id] * query_gates[rec_j.id]
                # Interaction weight = gate * similarity
                alpha = g_ij * sim_matrix[i][j]
                if alpha > 0.001:  # only keep meaningful interactions
                    weights.append(alpha)
                    vectors.append(rec_j.vector)
                    if i != j and len(interaction_weight_log) < 20:  # sample top pairs
                        interaction_weight_log.append((rec_i.id, rec_j.id, alpha))

            # Weighted sum
            if weights and vectors:
                total_w = sum(weights)
                updated = [
                    sum(w * v[idx] for w, v in zip(weights, vectors)) / total_w
                    for idx in range(len(vectors[0]))
                ]
            else:
                updated = list(rec_i.vector)  # fallback: unchanged

            updated_embeddings[rec_i.id] = updated

        # Log interaction pattern summary
        log.info("  [QDMO INTERACT] Interaction weights (sample top-20 non-self pairs):")
        for src, tgt, w in sorted(interaction_weight_log, key=lambda x: x[2], reverse=True)[:20]:
            log.info("    [%.4f] %s → %s", w, src, tgt)

        # Measure shift magnitude
        shifts = []
        for rec in active_records:
            old = rec.vector
            new = updated_embeddings[rec.id]
            shift = math.sqrt(sum((a - b) ** 2 for a, b in zip(old, new)))
            shifts.append(shift)
        log.info("  [QDMO INTERACT] Embedding shift — min=%.4f max=%.4f mean=%.4f",
                 min(shifts), max(shifts), sum(shifts) / len(shifts))

        # Shift distribution
        shift_bands = {"0.0-0.01": 0, "0.01-0.05": 0, "0.05-0.1": 0, "0.1+": 0}
        for s in shifts:
            if s < 0.01: shift_bands["0.0-0.01"] += 1
            elif s < 0.05: shift_bands["0.01-0.05"] += 1
            elif s < 0.1: shift_bands["0.05-0.1"] += 1
            else: shift_bands["0.1+"] += 1
        log.info("  [QDMO INTERACT] Shift distribution: %s",
                 " | ".join(f"{k}:{v}" for k, v in shift_bands.items() if v > 0))

        return updated_embeddings

    def _emergent_clustering(
        self,
        updated_embeddings: Dict[str, List[float]],
        scored_nodes: List[Tuple[float, VectorRecord]],
    ) -> List[MemoryCluster]:
        """Step 5: Emergent clustering — find semantic groups in interaction-updated space.

        After memory-to-memory interaction, nodes that have moved closer together
        in the embedding space form natural clusters. Uses seed-based grouping:
        - Seeds = top-K nodes by original activation weight
        - Cluster = all nodes whose updated embedding has cosine > threshold to seed
        - Each node joins the first seed it matches (greedy assignment)

        Args:
            updated_embeddings: Record id -> updated embedding vector from interaction.
            scored_nodes: (weight, VectorRecord) from soft_activation.

        Returns:
            List of MemoryCluster objects, ordered by seed activation weight descending.
        """
        # Get seed candidates: top-K by original activation weight
        seed_candidates = [rec for _, rec in scored_nodes[:self._num_cluster_seeds * 3]]

        # Filter to ones with updated embeddings
        seeds: List[VectorRecord] = []
        for rec in seed_candidates:
            if rec.id in updated_embeddings:
                seeds.append(rec)
            if len(seeds) >= self._num_cluster_seeds:
                break

        if not seeds:
            log.warning("  [QDMO CLUSTER] No seeds found — skipping clustering")
            return []

        # Greedy assignment: assign each node to the first seed it matches
        assigned: set = set()
        clusters: List[MemoryCluster] = []

        for seed in seeds:
            if seed.id in assigned:
                continue

            seed_emb = updated_embeddings.get(seed.id, seed.vector)
            members: List[VectorRecord] = [seed]
            assigned.add(seed.id)

            # Find nodes similar to this seed in the updated space
            for _, rec in scored_nodes:
                if rec.id in assigned or rec.id not in updated_embeddings:
                    continue
                node_emb = updated_embeddings[rec.id]
                sim = self._cosine(seed_emb, node_emb)
                if sim >= self._cluster_threshold:
                    members.append(rec)
                    assigned.add(rec.id)

            # Compute centroid and coherence
            if len(members) > 0:
                centroid = [
                    sum(updated_embeddings.get(m.id, m.vector)[d] for m in members) / len(members)
                    for d in range(len(seed_emb))
                ]
                coherence = sum(
                    self._cosine(centroid, updated_embeddings.get(m.id, m.vector))
                    for m in members
                ) / len(members)

                cluster = MemoryCluster(
                    seed_record=seed,
                    members=members,
                    centroid=centroid,
                    coherence_score=coherence,
                )
                clusters.append(cluster)

        log.info("  [QDMO CLUSTER] Formed %d clusters from %d seeds (threshold=%.2f)",
                 len(clusters), self._num_cluster_seeds, self._cluster_threshold)

        for idx, cl in enumerate(clusters):
            types_in_cluster: dict = {}
            for m in cl.members:
                types_in_cluster[m.node_type] = types_in_cluster.get(m.node_type, 0) + 1
            log.info("  [QDMO CLUSTER]   Cluster %d: %d members, coherence=%.4f, types=%s",
                     idx, len(cl.members), cl.coherence_score,
                     " | ".join(f"{k}:{v}" for k, v in sorted(types_in_cluster.items())))

        # Log unassigned ratio
        total_in_scored = len(scored_nodes)
        unassigned = total_in_scored - len(assigned)
        log.info("  [QDMO CLUSTER] %d / %d nodes assigned to clusters (%d unassigned)",
                 len(assigned), total_in_scored, unassigned)

        return clusters

    @staticmethod
    def _build_cluster_context(
        clusters: List[MemoryCluster],
    ) -> str:
        """Step 6: Build LLF context from clusters.

        Organizes memory nodes by cluster (semantic group) instead of by
        chronological order. Each cluster represents a distinct "aspect"
        of information relevant to the query.

        Args:
            clusters: List of MemoryCluster from emergent clustering.

        Returns:
            Formatted context string grouped by cluster.
        """
        parts: List[str] = []
        for idx, cl in enumerate(clusters):
            # Separate members by type for structured output
            dialogue_members = [m for m in cl.members if m.node_type == "dialogue"]
            image_name_members = [m for m in cl.members if m.node_type == "image_name"]
            fact_members = [m for m in cl.members if m.node_type == "image_fact"]
            desc_members = [m for m in cl.members if m.node_type in ("image_visual", "image_context")]

            parts.append(f"Relevant Information Group {idx + 1} "
                         f"(coherence: {cl.coherence_score:.2f}, "
                         f"members: {len(cl.members)}):")

            if dialogue_members:
                for m in dialogue_members:
                    parts.append(f"  [Conversation] {m.text}")

            if image_name_members:
                for m in image_name_members:
                    parts.append(f"  [Image] {m.text}")

            if fact_members:
                parts.append("  [Facts from this image]:")
                for m in fact_members:
                    parts.append(f"    - {m.text}")

            if desc_members:
                for m in desc_members:
                    parts.append(f"  [{m.node_type}] {m.text}")

            parts.append("")  # blank line between groups

        return "\n".join(parts)

    def _qdmo_answer(
        self,
        question: str,
        qa: Optional[Dict[str, Any]],
        question_images: Optional[List[str]],
    ) -> str:
        """Run the full QDMO pipeline (Steps 2-7) and return VLM answer.

        Steps:
          2. Soft activation: relevance scores for ALL nodes
          3. Active subset: top-K nodes for interaction
          4. Memory-to-memory interaction: query-conditioned attention
          5. Emergent clustering: seed-based grouping in updated space
          6. Context construction: cluster-organized text
          7. VLM reasoning: answer based on cluster structure
        """
        _ = question_images
        vlm = self._vlm
        if vlm is None:
            raise RuntimeError("VLM not initialized")

        log.info("")
        log.info("========== EVI QDMO Answer ==========")
        log.info("Q: %s", question)

        question_stem = (qa or {}).get("question", "").strip() or question
        is_mcq = bool(qa and qa.get("options"))

        # Step 2: Soft activation
        log.info("")
        log.info("--- [QDMO Step 2] Soft Activation ---")
        q_vec = self._embed(question_stem)
        if not q_vec:
            log.warning("  [QDMO] Embedding returned empty — aborting")
            return ""

        scored_nodes = self._soft_activation(q_vec)
        if not scored_nodes:
            log.warning("  [QDMO] No nodes scored — returning empty")
            return ""

        # Step 3: Active subset selection
        log.info("")
        log.info("--- [QDMO Step 3] Active Subset ---")
        active_records, active_weights = self._select_active_subset(scored_nodes)
        if not active_records:
            log.warning("  [QDMO] Active subset empty — returning empty")
            return ""

        # Step 4: Memory-to-memory interaction
        log.info("")
        log.info("--- [QDMO Step 4] Memory Interaction ---")
        updated_embs = self._memory_interaction(q_vec, active_records)
        if not updated_embs:
            log.warning("  [QDMO] Interaction produced no updates — returning empty")
            return ""

        # Step 5: Emergent clustering
        log.info("")
        log.info("--- [QDMO Step 5] Emergent Clustering ---")
        clusters = self._emergent_clustering(updated_embs, scored_nodes)

        if not clusters:
            log.warning("  [QDMO] No clusters formed — using standard context assembly")
            # Fallback to interaction-updated nodes as flat context
            context_text = "\n".join(
                f"  [{rec.node_type}] {rec.text}"
                for _, rec in scored_nodes[:self._max_rounds]
            )
        else:
            # Step 6: Build cluster-organized context
            log.info("")
            log.info("--- [QDMO Step 6] Context Construction ---")
            context_text = self._build_cluster_context(clusters)

        # Step 7: VLM reasoning
        log.info("")
        log.info("--- [QDMO Step 7] VLM Reasoning ---")
        prompt = f"""Below is information organized by relevant groups, each representing a distinct aspect of the conversation history.

{context_text}

Question: {question}"""

        log.info("  [QDMO VLM] Prompt %d chars, MCQ=%s", len(prompt), is_mcq)
        log.debug("----- Final Prompt to VLM -----\n%s\n----- End Prompt -----", prompt)

        if is_mcq:
            answer = vlm("", prompt, [])
        else:
            # Collect images from the most relevant nodes (top-5 by activation weight)
            image_paths: List[str] = []
            seen_paths: set = set()
            for _, rec in scored_nodes:
                if rec.image_path and rec.image_path not in seen_paths:
                    image_paths.append(rec.image_path)
                    seen_paths.add(rec.image_path)
                    if len(image_paths) >= 5:
                        break
            log.info("  [QDMO VLM] Collected %d unique images for VLM", len(image_paths))
            answer = vlm("", prompt, image_paths)

        if answer:
            log.info("  [QDMO VLM] Response: %d chars", len(answer))
            log.info("  [QDMO VLM] First 500 chars: %s", answer[:500])
        else:
            log.warning("  [QDMO VLM] Empty response")

        log.info("====================================\n")
        return answer

    # ---- Phase 2: Retrieval + Temporal Assembly ----

    def answer_question(
        self,
        question: str,
        qa: Optional[Dict[str, Any]] = None,
        question_images: Optional[List[str]] = None,
    ) -> str:
        """
        回答一个问题：LLM 目录检索 → 向量细粒度检索 → 时序排序 → 上下文组装 → VLM 生成答案

        ===== 核心流程（Phase 2）=====

        Step 1 — LLM 目录检索（Directory Retrieval）：
          - 从缓存中加载所有 session 的综合摘要
          - 构造目录提示词，让 LLM 基于各 session 的摘要判断哪些包含回答问题的线索
          - LLM 返回候选 session_id 列表（模糊定位）

        Step 2 — 向量细粒度检索（Vector Search within Candidates）：
          - 使用配置的 text embedding 模型（默认 all-MiniLM-L6-v2，也支持 NV-Embed-v2 等）将问题文本转为向量
          - 注意：使用 qa["question"]（题干，不含选项）作为检索源，
            避免选项文本污染检索相关性
          - 仅在候选 session 内的节点上搜索 Top-20

        Step 3 — 按 round 聚合：
          - 同一个 round 可能命中多个节点（dialogue + image_visual + image_name + image_context + image_fact 等）
          - 每个 round 取最高分作为该 round 的相关性分数
          - 保留该 round 命中的所有节点类型（为后面组装上下文使用）

        Step 4 — 时序排序：
          - 按 _round_order（整体对话顺序）对命中的 round 排序
          - 截取 Top-8

        Step 5 — 上下文组装：
          - 按 session 分组，在每个 session 前插入显式的时间/主题标题
            （例如 "--- Session BRAND_S5 (2024-01-16) ---"）
          - 对每个 round：
            a) 对话文本
            b) image_name（微图谱根节点名称）
            c) image_facts（结构化三元组，即微图谱边）
            d) 图片的多维描述（视觉描述 + 上下文描述）
            e) 收集高清原图路径（给 VLM 使用）
          - 最终生成一个"时序显式化"的上下文文本块

        Step 6 — VLM 回答：
          - 构造 prompt：时序上下文 + 问题完整文本（含 MCQ 选项）
          - MCQ 模式：不传图（选项文本已包含足够信息）
          - 非 MCQ 模式：传 Top-5 张原图让 VLM 看图回答

        ===== 关键设计决策 =====

        - 两阶段检索：LLM 目录检索做粗筛 → 向量检索做精召，避免在全量数据上做 embedding 搜索
        - 检索问题和最终问题分离：题干用于检索（去噪声），完整问题给 VLM（含选项）
        - 时序排序而不是相关性排序：避免打乱对话顺序导致 VLM 误解
        - 显式的 session 标题：帮助 VLM 理解"这是哪天聊的"
        - 图片路径在上下文组装时就收集好：VLM 调用时不需再查数据库
        - MCQ 模式不传图：选项已经是文字，无需看图判断
        - 目录检索失败时（无缓存 / LLM 解析失败）自动 fallback 到全量检索，保证鲁棒性
        - 可通过配置 `use_directory_retrieval: false` 跳过 LLM 目录检索，直接全量向量搜索
        """
        _ = question_images
        self._ensure()
        vlm = self._vlm
        if vlm is None:
            raise RuntimeError("VLM not initialized")

        # ---- QDMO mode: use memory interaction + emergent clustering pipeline ----
        if self._interaction_mode == "qdmo":
            return self._qdmo_answer(question, qa, question_images)

        log.info("")
        log.info("========== EVI Answer ==========")
        log.info("Q: %s", question)

        # Dump config for debugging
        log.info("  [CFG] directory_retrieval=%s, raw_search_k=%d, max_rounds=%d, diversity_boost=%.2f, total_vectors=%d",
                  self._use_directory_retrieval, self._raw_search_k, self._max_rounds, self._diversity_boost, len(self._vector_index))
        log.debug("  [CFG] round_order=%d rounds, %d sessions indexed",
                  len(self._round_order), len(self._session_dates))
        node_type_counts = {}
        for rec in self._vector_index._records:
            node_type_counts[rec.node_type] = node_type_counts.get(rec.node_type, 0) + 1
        log.debug("  [CFG] vector index composition: %s",
                  " | ".join(f"{k}:{v}" for k, v in sorted(node_type_counts.items())))

        # Use the raw question text (without MCQ options) for embedding
        question_stem = (qa or {}).get("question", "").strip() or question
        log.info("Question (for retrieval): %s", question_stem)

        is_mcq = bool(qa and qa.get("options"))
        if is_mcq:
            log.info("MCQ mode")

        # ================================================================
        # Stage 1: LLM Directory Retrieval — pick candidate sessions
        # (can be disabled via use_directory_retrieval: false in config)
        # ================================================================
        candidate_sessions = None
        if self._use_directory_retrieval:
            candidate_sessions = self._directory_retrieve(question_stem)
        #else:
        #    log.info("  [DIR] Directory retrieval disabled by config — searching all sessions")

        # ================================================================
        # Stage 2: Vector search — limited to candidate sessions
        # ================================================================
        q_vec = self._embed(question_stem)
        if not q_vec:
            log.warning("  [EMBED] Embedding returned empty — aborting")
            return ""

        search_kwargs: Dict[str, Any] = dict(top_k=self._raw_search_k)
        if candidate_sessions is not None:
            search_kwargs["session_ids"] = candidate_sessions

        log.info("  [SEARCH] Retrieving top-%d raw vectors ...", self._raw_search_k)
        all_results = self._vector_index.search(q_vec, **search_kwargs)

        log.info("  [SEARCH] Returned %d results in %s",
                 len(all_results),
                 f"{len(candidate_sessions)} candidate session(s)" if candidate_sessions else "all sessions")

        log.info("  [SEARCH] Raw results (top 30 shown):")
        for rec in all_results[:30]:
            log.info("    [%.4f] %-30s round=%-12s type=%-14s session=%s",
                      rec.score, rec.id, rec.round_id, rec.node_type, rec.session_id)

        # Log clue rounds hit/miss in raw search
        clue_rounds = (qa or {}).get("clue", [])
        if clue_rounds:
            hit = {rec.round_id for rec in all_results}
            log.info("  [CLUE RAW] %d clue rounds:", len(clue_rounds))
            for cr in clue_rounds:
                status = "HIT" if cr in hit else "MISS"
                log.info("    [CLUE RAW] %s: %s", status, cr)

        # ================================================================
        # Stage 3: Aggregate results by round — max score per round
        # ================================================================
        log.info("  [AGGREGATE] Aggregating %d raw results by round ...", len(all_results))
        round_scores: Dict[str, float] = {}
        round_node_types: Dict[str, set] = {}
        for rec in all_results:
            rid = rec.round_id
            if rid not in round_scores or rec.score > round_scores[rid]:
                round_scores[rid] = rec.score
            if rid not in round_node_types:
                round_node_types[rid] = set()
            round_node_types[rid].add(rec.node_type)
        round_nodes: Dict[str, List[VectorRecord]] = {}
        for rec in all_results:
            rid = rec.round_id
            if rid not in round_nodes:
                round_nodes[rid] = []
            round_nodes[rid].append(rec)

        # log.info("  [AGGREGATE] %d unique rounds from %d raw results", len(round_scores), len(all_results))
        # log.info("  [AGGREGATE] All unique rounds sorted by score:")
        # for rid in sorted(round_scores.keys(), key=lambda r: round_scores[r], reverse=True):
        #     types_fmt = ",".join(sorted(round_node_types.get(rid, [])))
        #     log.info("    [%.4f] %-12s types=[%s]", round_scores[rid], rid, types_fmt)

        # Type contribution analysis: which node_types dominate retrieval
        type_freq: dict = {}
        type_max_score: dict = {}
        type_avg_scores: dict = {}
        for rec in all_results:
            t = rec.node_type
            type_freq[t] = type_freq.get(t, 0) + 1
            if t not in type_max_score or rec.score > type_max_score[t]:
                type_max_score[t] = rec.score
            type_avg_scores.setdefault(t, []).append(rec.score)
        log.info("  [AGGREGATE] Type contribution in raw results:")
        for t in sorted(type_freq.keys()):
            avg = sum(type_avg_scores[t]) / len(type_avg_scores[t])
            log.info("      %-14s count=%-3d max_score=%.4f avg_score=%.4f",
                      t, type_freq[t], type_max_score[t], avg)

        # ================================================================
        # Stage 3.5: Diversity boost — reward rounds with richer node_type coverage
        #   A round that matches from multiple angles (e.g. dialogue + image_name + facts)
        #   is more likely to be truly relevant than one matching on only one type.
        #   boost = base_score * (1.0 + (num_extra_types * diversity_boost))
        # ================================================================
        log.info("  [BOOST] Diversity boost: +%.0f%% per extra node_type", self._diversity_boost * 100)
        boosted_scores: Dict[str, float] = {}
        for rid, base_score in round_scores.items():
            types = round_node_types.get(rid, set())
            extra_types = len(types) - 1  # at least 1 (dialogue) is always present
            if extra_types > 0:
                boost = 1.0 + extra_types * self._diversity_boost
                boosted = base_score * boost
                boosted_scores[rid] = boosted
                log.info("    [BOOST] %-12s base=%.4f types=%d boost=%.2fx → %.4f",
                          rid, base_score, len(types), boost, boosted)
            else:
                boosted_scores[rid] = base_score  # no boost for single-type rounds

        # Show top-5 before/after for comparison
        sorted_before = sorted(round_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        sorted_after = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        log.info("  [BOOST] Top-5 BEFORE boost:")
        for rid, s in sorted_before:
            log.info("    [%.4f] %-12s types=[%s]", s, rid, ",".join(sorted(round_node_types.get(rid, []))))
        log.info("  [BOOST] Top-5 AFTER boost:")
        for rid, s in sorted_after:
            log.info("    [%.4f] %-12s types=[%s]", s, rid, ",".join(sorted(round_node_types.get(rid, []))))

        # ================================================================
        # Stage 4: Temporal ordering — boosted top-N → chronological
        # ================================================================
        top_by_score = sorted(boosted_scores.keys(), key=lambda r: boosted_scores[r], reverse=True)[:self._max_rounds]
        ordered_rounds = [rid for rid in self._round_order if rid in top_by_score]

        log.info("  [ORDER] After chronological sort: %d rounds (max_rounds=%d)",
                 len(ordered_rounds), self._max_rounds)

        # Show boosted top-N with scores and types
        log.info("  [ORDER] Top-%d by boosted score:", self._max_rounds)
        for rid in top_by_score[:self._max_rounds]:
            types_fmt = ",".join(sorted(round_node_types.get(rid, [])))
            log.info("    [%.4f] %-12s types=[%s] (base=%.4f)",
                      boosted_scores.get(rid, 0), rid, types_fmt, round_scores.get(rid, 0))

        if ordered_rounds:
            first_rid, last_rid = ordered_rounds[0], ordered_rounds[-1]
            log.debug("  [ORDER] Chronological range: %s → %s", first_rid, last_rid)
            sids_in_order = {self._round_session.get(rid, "?") for rid in ordered_rounds}
            log.debug("  [ORDER] Sessions covered: %s", sorted(sids_in_order))
            chrono_scores = [round_scores.get(rid, 0) for rid in ordered_rounds]
            log.debug("  [ORDER] Score range chrono: min=%.4f max=%.4f",
                      min(chrono_scores), max(chrono_scores))

        # Log CLUE hit/miss after chronological sort
        if clue_rounds:
            final_hit = {rid for rid in ordered_rounds}
            log.info("  [CLUE FINAL] %d clue rounds after chrono sort:", len(clue_rounds))
            for cr in clue_rounds:
                status = "HIT" if cr in final_hit else "MISS"
                log.info("    [CLUE FINAL] %s: %s", status, cr)

            # Show which clue rounds were missed and why
            missed = [cr for cr in clue_rounds if cr not in final_hit]
            if missed:
                log.info("  [CLUE FINAL] Missed rounds analysis:")
                for cr in missed:
                    if cr in boosted_scores:
                        rank = sorted(boosted_scores.keys(), key=lambda r: boosted_scores[r], reverse=True).index(cr)
                        score = round_scores[cr]
                        log.info("      %s: base=%.4f boosted=%.4f (rank=%d in boosted, dropped by chrono max_rounds=%d)",
                                  cr, score, boosted_scores[cr], rank, self._max_rounds)
                    elif cr not in self._round_order:
                        log.info("      %s: not in round_order (missing from dataset)", cr)
                    else:
                        log.info("      %s: not found in any vector results", cr)

        # ================================================================
        # Stage 5: Context assembly — build temporally ordered prompt
        # ================================================================
        log.info("  [ASSEMBLE] Assembling context from %d rounds in %d sessions ...",
                 len(ordered_rounds),
                 len({self._round_session.get(rid, "?") for rid in ordered_rounds}))
        image_paths: List[str] = []
        context_parts: List[str] = []
        last_sid = ""

        for rid in ordered_rounds:
            sid = self._round_session.get(rid, "?")

            # Insert session header when session changes
            if sid != last_sid:
                date = self._session_dates.get(sid, "")
                context_parts.append(
                    f"--- Session {sid} ({date}) ---"
                )
                log.debug("  [ASSEMBLE]   session header: %s (%s)", sid, date)
                last_sid = sid

            # Round text
            context_parts.append(f"  Round {rid}:")
            context_parts.append(f"    {self._round_text.get(rid, '')}")
            log.debug("  [ASSEMBLE]   Round %s (session=%s, score=%.4f):",
                      rid, sid, round_scores.get(rid, 0))

            # Image information from vector results — structured by type
            image_records = [rec for rec in round_nodes.get(rid, []) if rec.node_type != "dialogue"]
            n_img_items = 0

            # Show image_name first (the micro-KG root)
            name_rec = next((r for r in image_records if r.node_type == "image_name"), None)
            if name_rec:
                context_parts.append(f"    [IMAGE] {name_rec.text}")
                n_img_items += 1

            # Show facts grouped (micro-KG edges)
            fact_recs = [r for r in image_records if r.node_type == "image_fact"]
            for r in fact_recs:
                context_parts.append(f"      ── {r.text}")
                n_img_items += 1

            # Show descriptions last (visual + context)
            desc_recs = [r for r in image_records if r.node_type in ("image_visual", "image_context")]
            for r in desc_recs:
                context_parts.append(f"    [{r.node_type}] {r.text}")
                n_img_items += 1

            log.debug("  [ASSEMBLE]     -> %d image items added (%s)",
                      n_img_items,
                      "name=" + ("yes" if name_rec else "no") +
                      " facts=" + str(len(fact_recs)) +
                      " descs=" + str(len(desc_recs)))

            # Collect original image
            img_path = self._round_images.get(rid)
            if img_path:
                image_paths.append(img_path)
                log.debug("  [ASSEMBLE]     -> collected image: %s", img_path)

        log.info("  [ASSEMBLE] Done: %d lines of context, %d images collected",
                 len(context_parts), len(image_paths))

        # ================================================================
        # Stage 6: Build final prompt and call VLM
        # ================================================================
        context_text = "\n".join(context_parts)

        prompt = f"""Below is a conversation history organized by session and round, in chronological order.

{context_text}

Question: {question}"""

        log.info("  [VLM] Prompt built: %d chars, %d images for VLM, MCQ=%s",
                 len(prompt),
                 len(image_paths[:5]) if not is_mcq else 0,
                 is_mcq)
        log.debug("  [VLM] Context breakdown: text=%d chars, system_prefix=%d chars",
                  len(context_text), len(prompt) - len(context_text))
        log.debug("  [VLM] Images passed: %s",
                  image_paths[:5] if not is_mcq else "(MCQ mode — no images passed to VLM)")
        if len(image_paths) > 5:
            log.debug("  [VLM]   (+ %d more images not sent due to limit)", len(image_paths) - 5)

        log.info("----- Final Prompt to VLM -----\n%s\n----- End Prompt -----", prompt)

        # 7. MCQ mode: just return the LLM response (no image needed for routing)
        log.info("  [VLM] Calling VLM (mode=%s) ...", "MCQ (no images)" if is_mcq else f"open-ended ({len(image_paths[:5])} images)")
        if is_mcq:
            answer = vlm("", prompt, [])
        else:
            answer = vlm("", prompt, image_paths[:5])

        if answer:
            log.info("  [VLM] Response: %d chars", len(answer))
            log.info("  [VLM] First 500 chars: %s", answer[:500])
            # Log condensed version for quick scan
            answer_one_line = answer.replace("\n", "\\n").replace("\r", "")
            log.debug("  [VLM] Full response (one-line): %s",
                      answer_one_line[:1000] + ("..." if len(answer_one_line) > 1000 else ""))
        else:
            log.warning("  [VLM] Empty response from VLM")

        log.info("==============================\n")
        return answer

    @property
    def num_indexed(self) -> int:
        return len(self._vector_index)
