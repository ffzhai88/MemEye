import json
import re
import statistics as _stats
import string
import time
from collections import Counter
from itertools import product as _product
from typing import Any, Dict, List, Optional, Tuple

import nltk

for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_stemmer = PorterStemmer()
_ARTICLES = re.compile(r"\b(a|an|the)\s+(?=\w)")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Exact port of Mem-Gallery's normalize_answer_universal.

    Uses fixed-length lookbehind (?<=\\d) which re supports, so no extra
    spaces are inserted around the placeholders.
    """
    s = str(s).lower()
    # Protect decimal points: lookbehind/lookahead leave the digits in place
    s = re.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", s)
    # Protect underscores (no surrounding spaces)
    s = s.replace("_", "UNDERSCOREPLACEHOLDER")
    s = _ARTICLES.sub(" ", s)
    # Replace each punctuation character with a space (char-by-char, same as Mem-Gallery)
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = s.replace("DOTPLACEHOLDER", ".").replace("UNDERSCOREPLACEHOLDER", "_")
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# MCQ helpers (unchanged)
# ---------------------------------------------------------------------------

def to_mcq(question: str) -> str:
    return (
        f"{question}\n"
        "Choose the best option.\n"
        "A) Correct answer exactly matches the key fact.\n"
        "B) Partially correct but misses key detail.\n"
        "C) Incorrect.\n"
        "Reply with only A, B, or C."
    )


def extract_choice(text: str, valid_keys: Optional[set] = None) -> str:
    """Extract a single option letter from model output.

    Matching priority:
      1. Entire output is a single valid key (e.g. "B")
      2. Output starts with a valid key followed by punctuation (e.g. "B. ...")
      3. "answer is X" or "answer: X" pattern anywhere
      4. First standalone valid key in the text

    Args:
        text: Raw model prediction.
        valid_keys: Set of valid option letters (e.g. {"A","B","C","D"}).
                    Defaults to all uppercase letters if not provided.
    """
    t = text.strip().upper()
    if valid_keys is None:
        valid_keys = set(string.ascii_uppercase)
    # 1. Exact single letter
    if t in valid_keys:
        return t
    keys_pattern = "|".join(re.escape(k) for k in sorted(valid_keys))
    # 2. Starts with option letter + punctuation/space (e.g. "B." "B)" "B ")
    m = re.match(rf"^({keys_pattern})[\.\)\:\s,]", t)
    if m:
        return m.group(1)
    # 3. "answer is B" / "answer: B" pattern
    m = re.search(rf"(?:answer|choice)\s*(?:is|:)\s*({keys_pattern})\b", t)
    if m:
        return m.group(1)
    # 4. First standalone valid key (word boundary)
    m = re.search(rf"\b({keys_pattern})\b", t)
    return m.group(1) if m else "INVALID"


# ---------------------------------------------------------------------------
# Open-mode exact / contains scoring
# ---------------------------------------------------------------------------

def score_open(pred: str, gt: str) -> Tuple[bool, bool]:
    p = normalize(pred)
    g = normalize(gt)
    exact = p == g and p != ""
    contains = g in p if g else False
    return exact, contains


# ---------------------------------------------------------------------------
# F1 (token-level with Porter stemming)
# ---------------------------------------------------------------------------

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with Porter stemming (Mem-Gallery formula)."""
    p_tokens = [_stemmer.stem(w) for w in normalize(prediction).split()]
    g_tokens = [_stemmer.stem(w) for w in normalize(ground_truth).split()]
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# BLEU (NLTK, smoothing method 1)
# ---------------------------------------------------------------------------

def bleu_score(
    prediction: str,
    ground_truth: str,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """BLEU with NLTK smoothing method 1. Pass weights=(1,0,0,0) for BLEU-1."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(ground_truth).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    return sentence_bleu(
        [ref_tokens],
        pred_tokens,
        weights=weights,
        smoothing_function=SmoothingFunction().method1,
    )


# ---------------------------------------------------------------------------
# BERT Score — module-level cached scorer (loads roberta-large once per run)
# ---------------------------------------------------------------------------

_bert_scorer = None  # BERTScorer instance, initialised on first call


def bert_score_metric(prediction: str, ground_truth: str) -> float:
    """Semantic similarity via BERTScore F1, rescaled. Returns 0.0 on error.

    roberta-large is loaded once and reused for all subsequent calls.
    """
    global _bert_scorer
    try:
        if _bert_scorer is None:
            from bert_score import BERTScorer
            _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        p = normalize(prediction)
        g = normalize(ground_truth)
        _, _, F1 = _bert_scorer.score([p], [g])
        return max(0.0, float(F1[0].item()))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

_VALID_JUDGE_SCORES = {0.0, 0.5, 1.0}


def _coerce_valid_judge_score(raw_score: Any) -> Optional[float]:
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    if score < 0.25:
        return 0.0
    if score < 0.75:
        return 0.5
    return 1.0


def _extract_judge_result(response_text: str) -> Optional[Dict[str, Any]]:
    parsed = parse_judge_response(response_text)
    if parsed is None:
        return None
    score = _coerce_valid_judge_score(parsed.get("score"))
    if score is None:
        return None
    return {
        "score": score,
        "reasoning": str(parsed.get("reasoning", "")),
    }


def parse_judge_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract {'score': float, 'reasoning': str} from LLM judge output.
    Tries JSON extraction first, then regex fallback.
    Returns None if both fail.
    """
    # Primary: find a JSON object containing "score"
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"score"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            raw = _coerce_valid_judge_score(result.get("score"))
            if raw is not None:
                result["score"] = raw
                result.setdefault("reasoning", "")
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: regex score extraction
    for pattern in (r'"score"\s*:\s*([0-9.]+)', r'[Ss]core\s*:\s*([0-9.]+)'):
        m = re.search(pattern, response_text)
        if m:
            try:
                raw = _coerce_valid_judge_score(m.group(1))
                if raw is not None:
                    return {
                        "score": raw,
                        "reasoning": response_text[:200],
                    }
            except ValueError:
                continue

    return None


def llm_judge_score(
    question: str,
    ground_truth: str,
    model_output: str,
    client: Any,           # openai.OpenAI instance
    model_name: str,
    prompt_template: str,
    max_retries: int = 3,
    timeout: int = 60,
    delay_base: float = 1.0,
) -> Dict[str, Any]:
    """
    Call LLM judge with exponential-backoff retries.

    Args:
        client: An openai.OpenAI (or compatible) client instance.
        model_name: Judge model, e.g. 'gpt-4.1-nano'.
        prompt_template: Contents of llm_judge.txt.
        max_retries: Retry attempts on parse failure or API error.

    Returns:
        {'score': float in {0,0.25,0.5,0.75,1}, 'reasoning': str}

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    prompt = (
        prompt_template
        .replace("{{question}}", question)
        .replace("{{ground_truth}}", ground_truth)
        .replace("{{model_output}}", model_output)
    )
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=timeout,
            )
            content = response.choices[0].message.content
            text = content.strip() if isinstance(content, str) else str(content).strip()
            result = _extract_judge_result(text)
            if result is not None:
                return result
            last_err = ValueError(f"Unparseable judge response: {text[:120]}")
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                # Parse retry-after from 429 body if available
                exc_str = str(exc)
                m = re.search(r"try again in ([\d.]+)s", exc_str)
                wait = (float(m.group(1)) + 0.5) if m else (delay_base * (2 ** attempt))
                if "429" in exc_str or "rate_limit" in exc_str:
                    print(f"[WARN] Judge 429 rate limit — waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
    raise RuntimeError(
        f"LLM judge failed after {max_retries} attempts: {last_err}"
    )


# ---------------------------------------------------------------------------
# MemEye matrix coordinate parsing
# ---------------------------------------------------------------------------

def parse_matrix_coords(point: Any) -> List[Tuple[str, str]]:
    """
    Parse MemEye point field → list of (Xi, Yj) cell tuples.
    Input: [["X1", "X3"], ["Y1"]] → [("X1","Y1"), ("X3","Y1")]
    Returns [] for unrecognised formats.
    """
    if not isinstance(point, list) or len(point) != 2:
        return []
    xs, ys = point[0], point[1]
    if not isinstance(xs, list) or not isinstance(ys, list):
        return []
    return list(_product(xs, ys))


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

# 需要汇总的核心指标集合；这里统一覆盖 EM、包含率、F1、BLEU、BERTScore 和 LLM Judge。
_OPEN_METRICS = ("em", "contains_gt", "f1", "bleu", "bleu_1", "bleu_2", "bert", "judge")


def _mean(vals: List[float]) -> float:
    # 对一组数值取平均；如果为空列表，就返回 0，避免除零或空结果导致异常。
    return _stats.mean(vals) if vals else 0.0


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    把逐题结果汇总成全局、按 X 分组、按 Y 分组和按矩阵单元分组的统计信息。
    对于某些指标未启用的题目（例如 bert=None），会在汇总时自动跳过，不影响平均值计算。
    """
    # 先把原始结果复制一份，作为“总题数”统计的基准。
    open_rows = list(results)
    # 只挑出 MCQ 题，用于单独统计正确率、有效选择率和旋转位置偏置。
    mcq_rows  = [r for r in results if r.get("mode") == "mcq"]
    # 只保留至少有一个有效度量值的行，避免把纯文本记录也纳入数值聚合。
    metric_rows = [r for r in results if any(r.get(m) is not None for m in _OPEN_METRICS)]

    # 初始化四类汇总容器：全局、按 X、按 Y、按单元格。
    overall: Dict[str, List[float]] = {m: [] for m in _OPEN_METRICS}
    by_x:    Dict[str, Dict[str, List[float]]] = {}
    by_y:    Dict[str, Dict[str, List[float]]] = {}
    by_cell: Dict[str, Dict[str, List[float]]] = {}
    # MCQ 专用的汇总容器，记录 MCQ 题的平均分和有效选择率。
    mcq_overall: Dict[str, List[float]] = {m: [] for m in _OPEN_METRICS}

    # 辅助函数：把某一行的有效数值指标追加到指定分组桶中。
    def _add_to(bucket: Dict[str, Dict[str, List[float]]], key: str, row: Dict) -> None:
        # 如果这个分组尚未存在，就先初始化空列表。
        b = bucket.setdefault(key, {m: [] for m in _OPEN_METRICS})
        for m in _OPEN_METRICS:
            # 只收集非 None 的数值，避免把未启用的指标（如 bert=None）带进来。
            val = row.get(m)
            if val is not None:
                b[m].append(float(val))

    # 遍历所有有数值的结果行，先做全局汇总，再做 X/Y/Cell 分组汇总。
    for row in metric_rows:
        # 把这行的有效指标追加到全局指标列表里。
        for m in _OPEN_METRICS:
            val = row.get(m)
            if val is not None:
                overall[m].append(float(val))

        # 解析题目中的矩阵坐标，例如 point=[['X1','X3'],['Y1']]。
        cells = parse_matrix_coords(row.get("point"))
        seen_x: set = set()
        seen_y: set = set()
        for (xi, yj) in cells:
            # 第一次遇到某个 X 标签时，把这一行加到 by_x 中。
            if xi not in seen_x:
                _add_to(by_x, xi, row)
                seen_x.add(xi)
            # 第一次遇到某个 Y 标签时，把这一行加到 by_y 中。
            if yj not in seen_y:
                _add_to(by_y, yj, row)
                seen_y.add(yj)
            # 每个 (Xi, Yj) 组合都单独形成一个 cell bucket。
            _add_to(by_cell, f"{xi}_{yj}", row)

    # 把中间的分组列表转换成“平均值字典”，只保留真正有数值的项。
    def _collapse(bucket: Dict) -> Dict:
        return {k: {m: _mean(vs) for m, vs in mv.items() if vs} for k, mv in bucket.items()}

    # 组装最终的汇总结果：包括总题数、全局平均值，以及 X/Y/Cell 的分组平均值。
    summary: Dict[str, Any] = {
        # open_count 表示总共处理了多少题（包括 MCQ 和 open-ended），用于后续统计覆盖率。
        "open_count": len(open_rows),
        # overall 是全局平均值，按每个指标分别计算。
        "overall":    {m: _mean(vs) for m, vs in overall.items() if vs},
        # by_x / by_y / by_cell 分别按实验矩阵的行、列、单元做聚合，方便定位表现差异。
        "by_x":       _collapse(by_x),
        "by_y":       _collapse(by_y),
        "by_cell":    _collapse(by_cell),
    }
    # 如果存在 MCQ 题，再补充 MCQ 专用的统计信息。
    if mcq_rows:
        # 记录本次结果里一共有多少条 MCQ 题。
        summary["mcq_count"] = len(mcq_rows)
        # 计算有效选择率：模型是否成功给出合法选项字母的比例。
        summary["mcq_valid_rate"] = (
            sum(1 for r in mcq_rows if r.get("valid_choice")) / len(mcq_rows)
        )
        # 把 MCQ 题的数值指标也一并收集，用于计算 MCQ 平均得分。
        for row in mcq_rows:
            for m in _OPEN_METRICS:
                val = row.get(m)
                if val is not None:
                    mcq_overall[m].append(float(val))
        # 把 MCQ 题的平均指标写回 summary，方便单独查看 MCQ 表现。
        summary["mcq_overall"] = {m: _mean(vs) for m, vs in mcq_overall.items() if vs}

        # 对旋转 MCQ 做额外统计：按正确答案位置（A/B/C/D）统计各位置的准确率。
        rotation_rows = [r for r in mcq_rows if r.get("rotations")]
        if rotation_rows:
            per_position_em: Dict[str, List[float]] = {}
            for row in rotation_rows:
                # 每个旋转版本都记录了其正确位置和这一次的 EM，因此可以统计“某位置下的平均命中率”。
                for rot in row["rotations"]:
                    pos = rot["correct_position"]
                    per_position_em.setdefault(pos, []).append(float(rot["em"]))
            # 按字母顺序输出每个位置的平均 EM，便于发现位置偏置。
            summary["mcq_per_position_accuracy"] = {
                pos: _mean(vals) for pos, vals in sorted(per_position_em.items())
            }
    return summary
