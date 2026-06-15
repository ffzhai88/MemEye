"""
EVI v2: Multi-vector temporal indexing for multimodal long-term memory.

Two-phase lifecycle:
  1. process_all_sessions(dataset) — build vector index
  2. answer_question(question, qa) — retrieve + temporally assemble + VLM
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .extractor import describe_image
from .indexes import VectorIndex, embed_text
from .schemas import VectorRecord
from .vlm import VLMCallable, make_vlm_callable

log = logging.getLogger(__name__)


class EVISystem:
    """Multi-vector temporal indexing system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._model_cfg = dict(cfg.get("_model_cfg", {}))

        # Vector index (single, stores both dialogue + image nodes)
        self._vector_index = VectorIndex()

        # Round metadata for temporal assembly
        self._round_order: List[str] = []           # ordered round_ids
        self._round_session: Dict[str, str] = {}    # round_id -> session_id
        self._round_text: Dict[str, str] = {}       # round_id -> dialogue text
        self._round_images: Dict[str, str] = {}     # round_id -> image_path (first only)
        self._session_dates: Dict[str, str] = {}    # session_id -> date

        # VLM callable
        self._vlm: Optional[VLMCallable] = None
        self._embedder: Optional[Any] = None
        self._initialized = False

    # ---- lazy init ----

    def _ensure(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._vlm = make_vlm_callable(self._model_cfg)

        # Text embedder
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("Embedder: all-MiniLM-L6-v2 loaded.")
        except ImportError:
            log.warning("sentence-transformers not available")
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
           - 用 all-MiniLM-L6-v2 将该文本转为 embedding → 存入向量索引
           - 节点类型标记为 "dialogue"

        2. 图片节点（Image Node）— 两次 VLM 调用，结果合并缓存：
           a) 图片本身 → 不固定数量的自由文本描述（VLM 自行决定从哪些角度描述）
              每个描述独立嵌入，类型标记为 "image_visual"
           b) 图片在对话中的角色 → 基于完整上下文（当前 round + 该 session 之前所有 round）
              生成一段描述，类型标记为 "image_context"
           - 同一个图片可能对应 N+1 个向量（N 个 image_visual + 1 个 image_context）

        同时记录：
        - _round_order: 所有 round 的全局顺序（按 session + round 序排列）
        - _round_session: round_id → session_id 的映射
        - _round_text: round_id → 对话文本
        - _round_images: round_id → 图片路径
        - _session_dates: session_id → 日期

        ===== 关键设计决策 =====

        - Phase 1 完全 blind：不依赖任何问题信息。只做"尽可能全面的描述"
        - 每个图片生成 3 种描述而不是 1 种：不同的描述覆盖不同的检索需求
          （纯视觉匹配 vs 语义语境匹配 vs session 级宏观匹配）
        - 对话节点和图片节点放在同一个向量索引中：检索时可以同时命中
        - VLM 调用次数 = 有图片的 round 数量（已被缓存，重复运行不产生费用）
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
                    total_img += 1

                # After processing this round, add it to prior context for next round
                prior_rounds_text.append(self._round_text.get(rid, ""))

        log.info(
            "Indexing done: %d dialogue nodes, %d image vectors, %d total",
            len([r for r in self._vector_index._records if r.node_type == "dialogue"]),
            len([r for r in self._vector_index._records if r.node_type != "dialogue"]),
            len(self._vector_index),
        )

    # ---- Phase 2: Retrieval + Temporal Assembly ----

    def answer_question(
        self,
        question: str,
        qa: Optional[Dict[str, Any]] = None,
        question_images: Optional[List[str]] = None,
    ) -> str:
        """
        回答一个问题：向量检索 → 时序排序 → 上下文组装 → VLM 生成答案

        ===== 核心流程（Phase 2）=====

        Step 1 — 向量检索：
          - 使用 all-MiniLM-L6-v2 将问题文本转为向量
          - 注意：使用 qa["question"]（题干，不含选项）作为检索源，
            避免选项文本污染检索相关性
          - 在所有节点（对话 + 图片）上搜索 Top-20

        Step 2 — 按 round 聚合：
          - 同一个 round 可能命中多个节点（dialogue + image_visual + image_context 等）
          - 每个 round 取最高分作为该 round 的相关性分数
          - 保留该 round 命中的所有节点类型（为后面组装上下文使用）

        Step 3 — 时序排序：
          - 按 _round_order（整体对话顺序）对命中的 round 排序
          - 截取 Top-8

        Step 4 — 上下文组装：
          - 按 session 分组，在每个 session 前插入显式的时间/主题标题
            （例如 "--- Session BRAND_S5 (2024-01-16) ---"）
          - 对每个 round：
            a) 对话文本
            b) 图片的多维描述（视觉描述 + 上下文描述）
            c) 收集高清原图路径（给 VLM 使用）
          - 最终生成一个"时序显式化"的上下文文本块

        Step 5 — VLM 回答：
          - 构造 prompt：时序上下文 + 问题完整文本（含 MCQ 选项）
          - MCQ 模式：不传图（选项文本已包含足够信息）
          - 非 MCQ 模式：传 Top-5 张原图让 VLM 看图回答

        ===== 关键设计决策 =====

        - 检索问题和最终问题分离：题干用于检索（去噪声），完整问题给 VLM（含选项）
        - 时序排序而不是相关性排序：避免打乱对话顺序导致 VLM 误解
        - 显式的 session 标题：帮助 VLM 理解"这是哪天聊的"
        - 图片路径在上下文组装时就收集好：VLM 调用时不需再查数据库
        - MCQ 模式不传图：选项已经是文字，无需看图判断
        """
        _ = question_images
        self._ensure()
        vlm = self._vlm
        if vlm is None:
            raise RuntimeError("VLM not initialized")

        log.info("")
        log.info("========== EVI Answer ==========")
        log.info("Q: %s", question)

        # Use the raw question text (without MCQ options) for embedding
        question_stem = (qa or {}).get("question", "").strip() or question
        log.info("Question (for retrieval): %s", question_stem)

        is_mcq = bool(qa and qa.get("options"))
        if is_mcq:
            log.info("MCQ mode")

        # 1. Embed question stem (without options)
        q_vec = self._embed(question_stem)
        if not q_vec:
            return ""

        # 2. Search ALL node types
        all_results = self._vector_index.search(q_vec, top_k=20)
        log.info("Search returned %d results", len(all_results))
        # Log top-5 by type
        for rec in all_results:
            log.info("  [%.3f] %s | round=%s | type=%s",
                     rec.score, rec.id, rec.round_id, rec.node_type)

        # Log clue rounds hit/miss in top 20
        clue_rounds = (qa or {}).get("clue", [])
        if clue_rounds:
            hit = {rec.round_id for rec in all_results}
            for cr in clue_rounds:
                status = "HIT" if cr in hit else "MISS"
                log.info("  [CLUE] %s %s", status, cr)

        # Build round_scores
        round_scores: Dict[str, float] = {}
        for rec in all_results:
            rid = rec.round_id
            if rid not in round_scores or rec.score > round_scores[rid]:
                round_scores[rid] = rec.score
        round_nodes: Dict[str, List[VectorRecord]] = {}
        for rec in all_results:
            rid = rec.round_id
            if rid not in round_nodes:
                round_nodes[rid] = []
            round_nodes[rid].append(rec)

        # 4. Take top-8 by relevance score, then sort chronologically
        top_by_score = sorted(round_scores.keys(), key=lambda r: round_scores[r], reverse=True)[:8]
        ordered_rounds = [rid for rid in self._round_order if rid in top_by_score]

        # Log type breakdown for selected rounds
        log.info("Top %d rounds by relevance:", len(ordered_rounds))
        for rid in ordered_rounds:
            types = [r.node_type for r in round_nodes.get(rid, [])]
            log.info("  %s score=%.3f types=%s", rid, round_scores.get(rid, 0), types)

        # 5. Assemble temporally ordered context
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
                last_sid = sid

            # Round text
            context_parts.append(f"  Round {rid}:")
            context_parts.append(f"    {self._round_text.get(rid, '')}")

            # Image descriptions from vector results
            img_descs = []
            for rec in round_nodes.get(rid, []):
                if rec.node_type != "dialogue" and rec.text:
                    img_descs.append(f"    [{rec.node_type}] {rec.text}")
            if img_descs:
                context_parts.extend(img_descs[:3])

            # Collect original image
            img_path = self._round_images.get(rid)
            if img_path:
                image_paths.append(img_path)

        # 6. Build final prompt
        context_text = "\n".join(context_parts)

        prompt = f"""Below is a conversation history organized by session and round, in chronological order.

{context_text}

Question: {question}"""

        log.info("Context: %d chars, %d images", len(context_text), len(image_paths))
        log.info("----- Final Prompt to VLM -----\n%s\n----- End Prompt -----", prompt)

        # 7. MCQ mode: just return the LLM response (no image needed for routing)
        if is_mcq:
            answer = vlm("", prompt, [])
        else:
            answer = vlm("", prompt, image_paths[:5])

        log.info("Answer: %s", answer[:300] if answer else "(empty)")
        log.info("==============================\n")
        return answer

    @property
    def num_indexed(self) -> int:
        return len(self._vector_index)
