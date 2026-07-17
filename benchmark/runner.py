import datetime as dt
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from router import GeminiAPIRouter, OpenAIAPIRouter, QwenLocalRouter

from .common import (
    REPO_ROOT,
    SCRIPT_DIR,
    get_git_commit,
    load_yaml,
    resolve_config_path,
    resolve_dataset_path,
    write_json,
    write_jsonl,
)
from .dataset import MemoryBenchmarkDataset
from .evaluator import (
    bert_score_metric,
    bleu_score,
    extract_choice,
    f1_score,
    llm_judge_score,
    score_open,
    summarize_results,
    to_mcq,
)
from .methods import get_method


@dataclass
class LegacyRunOptions:
    config_path: str
    dialog_json: str = ""
    image_root: str = ""
    model_path: str = ""
    max_new_tokens: int = 0
    output_json: str = ""
    mode: str = ""
    max_questions: int = 0


def merge_legacy_config(opts: LegacyRunOptions) -> Dict[str, Any]:
    # 兼容旧版命令行入口：先读取 YAML 基础配置，再用 CLI 参数覆盖，形成统一的运行配置。
    base: Dict[str, Any] = {}
    if opts.config_path:
        try:
            base = load_yaml(resolve_config_path(opts.config_path))
        except Exception:
            pass

    # Build merged config; CLI opts override file-based config
    dataset = dict(base.get("dataset", {}))
    if opts.dialog_json:
        dataset["dialog_json"] = opts.dialog_json
    if opts.image_root:
        dataset["image_root"] = opts.image_root

    model = dict(base.get("model", {}))
    if opts.model_path:
        model.update({"provider": "qwen_local", "name": "legacy_model", "model_path": opts.model_path})
    if opts.max_new_tokens:
        model["max_new_tokens"] = opts.max_new_tokens
    model.setdefault("provider", "qwen_local")
    model.setdefault("name", "legacy_model")
    model.setdefault("max_new_tokens", 128)

    eval_cfg = dict(base.get("eval", {}))
    if opts.mode:
        eval_cfg["mode"] = opts.mode
    if opts.max_questions:
        eval_cfg["max_questions"] = opts.max_questions
    eval_cfg.setdefault("mode", "open")
    eval_cfg.setdefault("max_questions", 0)

    run_cfg = dict(base.get("run", {}))
    if opts.output_json:
        run_cfg["output_root"] = str(Path(opts.output_json).parent)

    return {
        "task": base.get("task", {"name": "legacy"}),
        "dataset": dataset,
        "eval": eval_cfg,
        "model": model,
        "method": base.get("method", {"name": "full_context_multimodal"}),
        "run": run_cfg,
    }



def _effective_method_name(method_cfg: Dict[str, Any]) -> str:
    # 统一规范化方法名，兼容旧配置中的 modality 变体，避免结果目录和日志混乱。
    method_name = str(method_cfg.get("name", "method")).strip() or "method"
    if method_name in {"full_context_multimodal", "full_context_text_only",
                       "full_context_no_visual", "question_only",
                       "semantic_rag_multimodal", "semantic_rag_text_only"}:
        return method_name
    modality = str(method_cfg.get("modality", "")).strip().lower()
    if modality in {"text_only", "multimodal", "no_visual"}:
        return f"{method_name}__{modality}"
    return method_name


def load_sys_prompt(mode: str = "open", method_cfg: Optional[Dict[str, Any]] = None) -> str:
    """Load MemEye system prompt for the given evaluation mode and modality.

    Uses sys_prompt_mcq.txt for MCQ mode, sys_prompt_open.txt for open mode.
    For text_only modality, uses sys_prompt_text_only.txt if it exists.
    Falls back to sys_prompt.txt if the mode-specific file is missing.
    """
    prompt_dir = Path(__file__).parent / "prompt"
    modality = str((method_cfg or {}).get("modality", "")).strip().lower()
    if modality == "text_only":
        text_only_file = prompt_dir / "sys_prompt_text_only.txt"
        if text_only_file.exists():
            return text_only_file.read_text(encoding="utf-8").strip()
    mode_file = prompt_dir / f"sys_prompt_{mode}.txt"
    if mode_file.exists():
        return mode_file.read_text(encoding="utf-8").strip()
    fallback = prompt_dir / "sys_prompt.txt"
    return fallback.read_text(encoding="utf-8").strip()


def instantiate_router(model_cfg: Dict[str, Any], system_prompt: str = ""):
    # 根据模型提供方选择不同的推理路由器，统一封装 OpenAI / Gemini / 本地 Qwen 的调用入口。
    provider = model_cfg.get("provider", "qwen_local")
    if provider == "qwen_local":
        return QwenLocalRouter(
            model_path=str(model_cfg["model_path"]),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            system_prompt=system_prompt,
            max_time=model_cfg.get("max_time", 25),
        )
    if provider == "openai_api":
        return OpenAIAPIRouter(
            model=str(model_cfg["model"]),
            api_key=str(model_cfg.get("api_key", "")),
            api_key_env=str(model_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=str(model_cfg.get("base_url", "https://api.openai.com/v1")),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            timeout=int(model_cfg.get("timeout", 90)),
            system_prompt=system_prompt,
        )
    if provider == "gemini_api":
        return GeminiAPIRouter(
            model=str(model_cfg["model"]),
            api_key=str(model_cfg.get("api_key", "")),
            api_key_env=str(model_cfg.get("api_key_env", "GEMINI_API_KEY")),
            base_url=str(model_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            timeout=int(model_cfg.get("timeout", 90)),
            system_prompt=system_prompt,
        )
    raise ValueError(f"Unsupported provider: {provider}")



def compose_modular_config(
    task_config_path: str,
    model_config_path: str,
    method_config_path: str,
    output_root: str = "",
    mode: str = "",
    max_questions: int = 0,
) -> Dict[str, Any]:
    task_cfg = load_yaml(resolve_config_path(task_config_path))
    model_cfg = load_yaml(resolve_config_path(model_config_path))
    method_cfg = load_yaml(resolve_config_path(method_config_path))
    cfg = {
        "task": task_cfg,
        "dataset": task_cfg.get("dataset", {}),
        "eval": task_cfg.get("eval", {}),
        "model": model_cfg,
        "method": method_cfg,
        "run": {},
    }
    if output_root:
        cfg["run"]["output_root"] = output_root
    if mode:
        cfg["eval"]["mode"] = mode
    if max_questions:
        cfg["eval"]["max_questions"] = max_questions
    return cfg


def resolve_runtime_paths(cfg: Dict[str, Any], config_dir: Path) -> Dict[str, Path]:
    # 解析运行时需要的真实路径，包括数据文件、图片目录和输出目录，保证实验在不同目录下可复现。
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("eval", {})
    task_name = str(cfg.get("task", {}).get("name", "task")).strip() or "task"
    dialog_json = resolve_dataset_path(str(dataset_cfg["dialog_json"]), config_dir)
    image_root_raw = str(dataset_cfg.get("image_root", "")).strip()
    image_root = resolve_dataset_path(image_root_raw, config_dir) if image_root_raw else None

    output_root_raw = str(cfg.get("run", {}).get("output_root", "")).strip()
    if output_root_raw:
        output_root_path = Path(output_root_raw)
        output_root = output_root_path if output_root_path.is_absolute() else (SCRIPT_DIR / output_root_path)
        output_root = output_root.resolve()
    else:
        output_root = (SCRIPT_DIR / "runs").resolve()
    output_json_raw = str(eval_cfg.get("output_json", "")).strip()
    if output_json_raw:
        oj = Path(output_json_raw)
        if oj.is_absolute():
            output_json = oj
        else:
            output_json = (SCRIPT_DIR / oj).resolve()
            output_root_dir = (SCRIPT_DIR / "output").resolve()
            if output_json.parent == output_root_dir:
                output_json = output_root_dir / task_name / output_json.name
    else:
        output_json = None

    return {
        "dialog_json": dialog_json,
        "image_root": image_root,
        "output_root": output_root,
        "output_json": output_json,
    }


def default_run_dir(cfg: Dict[str, Any], output_root: Path) -> Path:
    task_name = str(cfg.get("task", {}).get("name", "task"))
    model_name = str(cfg.get("model", {}).get("name", "model"))
    method_name = _effective_method_name(cfg.get("method", {}))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / task_name / f"{ts}_{model_name}_{method_name}"



def build_payload(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    run_dir: Path,
    dataset: MemoryBenchmarkDataset,
    results: List[Dict[str, Any]],
    method_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = summarize_results(results)
    evidence_summary = summarize_evidence_context(results)
    if evidence_summary:
        summary["evidence_context"] = evidence_summary
    model_ref = cfg["model"].get("model_path") or cfg["model"].get("model", "")
    payload = {
        "task_name": cfg.get("task", {}).get("name", "task"),
        "model_name": cfg.get("model", {}).get("name", "qwen_local"),
        "model_path": model_ref,
        "method_name": _effective_method_name(cfg.get("method", {})),
        "base_method_name": cfg.get("method", {}).get("name", "full_context_multimodal"),
        "method_modality": cfg.get("method", {}).get("modality", ""),
        "mode": "mcq" if all(isinstance(q.get("options"), (dict, list)) and q.get("options") for q in dataset.qas) else cfg["eval"].get("mode", "open"),
        "num_qas": len(dataset.qas),
        "num_qas_run": len({r["idx"] for r in results}),
        "dialog_json": str(paths["dialog_json"]),
        "image_root": str(paths["image_root"]) if paths["image_root"] else "",
        "run_dir": str(run_dir),
        "git_commit": get_git_commit(REPO_ROOT),
        "summary": summary,
        "results": results,
    }
    if method_runtime:
        payload["method_runtime"] = method_runtime
    return payload



def _unique_round_ids(values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    if not isinstance(values, list):
        return out
    for value in values:
        rid = str(value or "").strip()
        if not rid or rid in seen:
            continue
        out.append(rid)
        seen.add(rid)
    return out


def _round_ids_from_history(history: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in history or []:
        rid = str(item.get("round_id") or "").strip()
        if not rid or rid in seen:
            continue
        out.append(rid)
        seen.add(rid)
    return out


def _context_round_ids(history: List[Dict[str, Any]], runtime_info: Dict[str, Any]) -> List[str]:
    for key in ("final_context_round_ids", "selected_round_ids", "retrieved_round_ids"):
        round_ids = _unique_round_ids(runtime_info.get(key))
        if round_ids:
            return round_ids
    return _round_ids_from_history(history)


def _evidence_context_metrics(qa: Dict[str, Any], context_round_ids: List[str]) -> Dict[str, Any]:
    clue_rounds = _unique_round_ids(qa.get("clue", []))
    context_round_ids = _unique_round_ids(context_round_ids)
    context_set = set(context_round_ids)
    hits = [rid for rid in clue_rounds if rid in context_set]
    clue_count = len(clue_rounds)
    context_count = len(context_round_ids)
    recall = (len(hits) / clue_count) if clue_count else None
    precision = (len(hits) / context_count) if context_count else (None if clue_count else 0.0)
    if recall is None or precision is None or recall + precision == 0:
        f1 = None if recall is None or precision is None else 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return {
        "context_round_count": context_count,
        "clue_round_count": clue_count,
        "clue_round_hits": len(hits),
        "clue_round_hit_ids": hits,
        "clue_round_missed_ids": [rid for rid in clue_rounds if rid not in context_set],
        "clue_round_recall_at_context": recall,
        "clue_round_precision_at_context": precision,
        "clue_round_f1_at_context": f1,
        "full_clue_coverage": bool(clue_count and len(hits) == clue_count),
        "context_available": bool(context_round_ids),
    }


def summarize_evidence_context(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [
        _evidence_context_metrics({"clue": r.get("clue_rounds", [])}, r.get("context_round_ids", []))
        for r in results
    ]
    rows = [r for r in rows if r.get("clue_round_count", 0) > 0]
    if not rows:
        return {}
    total_clues = sum(int(r.get("clue_round_count", 0) or 0) for r in rows)
    total_hits = sum(int(r.get("clue_round_hits", 0) or 0) for r in rows)
    total_context = sum(int(r.get("context_round_count", 0) or 0) for r in rows)
    recalls = [float(r["clue_round_recall_at_context"]) for r in rows if r.get("clue_round_recall_at_context") is not None]
    precisions = [float(r["clue_round_precision_at_context"]) for r in rows if r.get("clue_round_precision_at_context") is not None]
    f1s = [float(r["clue_round_f1_at_context"]) for r in rows if r.get("clue_round_f1_at_context") is not None]
    available = [r for r in rows if r.get("context_available")]
    return {
        "num_qas_with_clues": len(rows),
        "num_qas_with_context": len(available),
        "clue_round_recall_at_context_micro": (total_hits / total_clues) if total_clues else None,
        "clue_round_recall_at_context_macro": (sum(recalls) / len(recalls)) if recalls else None,
        "clue_round_precision_at_context_micro": (total_hits / total_context) if total_context else None,
        "clue_round_precision_at_context_macro": (sum(precisions) / len(precisions)) if precisions else None,
        "clue_round_f1_at_context_macro": (sum(f1s) / len(f1s)) if f1s else None,
        "full_clue_coverage_rate": sum(1 for r in rows if r.get("full_clue_coverage")) / len(rows),
        "context_round_count_mean": total_context / len(rows),
        "total_clue_rounds": total_clues,
        "total_clue_round_hits": total_hits,
        "total_context_rounds": total_context,
    }


def _format_options_block(question: str, options_dict: Dict[str, str]) -> str:
    """Append MCQ options to a question string."""
    option_keys = [k for k in sorted(options_dict.keys()) if k != "answer"]
    lines = [question, ""]
    for key in option_keys:
        lines.append(f"{key}. {options_dict[key]}")
    lines.append("")
    valid = ", ".join(option_keys)
    lines.append(f"Answer with ONLY the option letter ({valid}). Do not explain.")
    return "\n".join(lines)


def format_question(qa: Dict[str, Any]) -> str:
    """Build the final question string.

    Handles both legacy format (options as dict) and rotation format
    (options as list of dicts).  For rotation format, uses the *last*
    rotation (answer at D) as the default — the rotation loop calls
    ``_format_options_block`` directly for each rotation.
    """
    question = qa.get("question", "")
    options = qa.get("options")
    if options and isinstance(options, list):
        # Rotation format — use last rotation as canonical for non-rotation callers
        return _format_options_block(question, options[-1])
    if options and isinstance(options, dict):
        return _format_options_block(question, options)
    return question


def is_rotation_mcq(qa: Dict[str, Any]) -> bool:
    """Check if a QA uses the rotation MCQ format (options is a list)."""
    return isinstance(qa.get("options"), list) and len(qa["options"]) > 0


def run_benchmark(
    cfg: Dict[str, Any],
    config_dir: Path,
    enable_bert_score: bool = False,
    enable_llm_judge: bool = False,
    judge_config: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    执行一次完整的 benchmark 跑通流程。

    主要职责包括：
    1. 根据配置解析真实的数据路径与输出路径；
    2. 实例化目标 memory method 与模型 router；
    3. 遍历所有 QA 样本，调用模型生成答案并计算指标；
    4. 记录运行时信息、汇总结果和逐题预测结果，写入 runs/ 或 output/。
    """
    # 第一步：把配置中的相对路径转换成实际可访问的文件路径，并确定实验输出目录。
    paths = resolve_runtime_paths(cfg, config_dir)
    mode = str(cfg.get("eval", {}).get("mode", "open"))
    max_questions = int(cfg.get("eval", {}).get("max_questions", 0))
    # 第二步：生成这次运行的独立目录，避免不同实验结果互相覆盖。
    run_dir = run_dir or default_run_dir(cfg, paths["output_root"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # 第三步：加载任务数据集，准备题目列表与图像路径解析能力。
    dataset = MemoryBenchmarkDataset(paths["dialog_json"], paths["image_root"])
    # 第四步：把模型配置、运行时路径和评测配置注入到 method 中，便于 method 自己读取。
    method_cfg = dict(cfg.get("method", {}))
    method_cfg["_model_cfg"] = dict(cfg.get("model", {}))
    method_cfg["_runtime_paths"] = {
        "output_root": str(paths["output_root"]),
        "output_json": str(paths.get("output_json") or ""),
        "run_dir": str(run_dir),
    }
    method_cfg["_eval_cfg"] = dict(cfg.get("eval", {}))
    # 获取具体 memory method 实例；agentic 方法会自行处理推理流程，非 agentic 方法走统一的 router.answer。
    method = get_method(
        str(method_cfg.get("name", "full_context_multimodal")),
        config=method_cfg,
    )
    # Agentic methods (for example M2A) own end-to-end inference via answer().
    # They bypass build_history() + router.answer() and may keep internal runtime state.
    is_agentic = hasattr(method, "answer") and callable(getattr(method, "answer"))
    router = None
    if not is_agentic:
        sys_prompt = load_sys_prompt(mode, method_cfg)
        router = instantiate_router(cfg["model"], system_prompt=sys_prompt)

    # 第五步：如果启用了 LLM-as-a-Judge，则提前初始化 judge 客户端，避免每题重复创建连接。
    _judge_client = None
    _judge_template = None
    _judge_model = None
    if enable_llm_judge:
        if not judge_config:
            raise ValueError("judge_config must be provided when enable_llm_judge=True")
        from openai import OpenAI
        _judge_client = OpenAI(
            api_key=judge_config.get("api_key") or None,
            base_url=judge_config.get("base_url") or None,
        )
        _judge_model = judge_config["model"]
        prompt_path = Path(__file__).parent / "llm_judge.txt"
        _judge_template = prompt_path.read_text(encoding="utf-8")

    # 第六步：得到待评测的 QA 列表；如果配置了 max_questions，则只跑前 N 题，方便调试和小规模验证。
    qas = dataset.iter_qas(limit=max_questions)
    results: List[Dict[str, Any]] = []

    # 对于与问题无关的全量历史方法，缓存 build_history 的结果，避免每个 QA 都重复拼接长上下文。
    # 这能显著减少重复计算，并让模型调用时更容易复用 prompt 前缀缓存。
    _cached_history: Optional[List[Dict[str, Any]]] = None
    _history_is_qa_independent = (
        not is_agentic
        and hasattr(method, "history_source")
        and method.history_source in ("full_context",)
    )

    # 第七步：逐题执行推理与评分。
    # 这里一共分成三类评测分支，分别对应不同的题型与评分方式：
    # 1) 旋转 MCQ 分支：当 QA 的 options 是一个列表时，会把每个候选位置都轮换一次。
    #    每次都调用模型做一次选择，最后统计平均 EM 和位置偏置（position bias）。
    #    这能检测模型是否因为选项顺序而“偏爱某个字母”。
    # 2) 普通 MCQ 分支：当 QA 只有一个 options 字典时，直接把题目拼成标准 MCQ prompt，
    #    用 extract_choice 抽取模型返回的选项字母，再计算 EM / exact_match。
    # 3) Open-ended 分支：当题目没有选项时，模型直接生成自由文本答案，
    #    再用 F1、BLEU、BERTScore、LLM Judge 等指标做语义与文本相似度评估。
    for i, qa in enumerate(qas, start=1):
        question_text = qa.get("question", "")
        gt = qa.get("answer", "")
        has_options = isinstance(qa.get("options"), (dict, list)) and bool(qa.get("options"))
        rotation_mode = is_rotation_mcq(qa)

        # 根据题目类型决定当前 QA 的评测模式：
        # - 有选项题统一按 MCQ 处理；
        # - 无选项题按配置的 open/mcq 模式执行。
        qa_mode = "mcq" if has_options else mode

        if is_agentic:
            history: List[Dict[str, Any]] = []
        elif _history_is_qa_independent and _cached_history is not None:
            history = _cached_history
        else:
            history = method.build_history(dataset, qa)
            if _history_is_qa_independent:
                _cached_history = history
        current_method_runtime = dict(getattr(method, "runtime_info", {}) or {})

        # 解析当前题目的图片路径；如果当前 method 是 text_only / no_visual，则不传图，避免无关视觉输入。
        question_image_paths = dataset.resolve_question_images(qa)
        if getattr(method, "modality", "") in ("text_only", "no_visual"):
            question_image_paths = []

        # --- 分支 1：旋转 MCQ（rotation MCQ） ---
        # 这里的核心思路不是“只跑一次”，而是把同一道题的正确选项位置轮换多次。
        # 这样可以观察模型是否受选项顺序影响，并把多个轮次的结果聚合成一个去偏后的 EM。
        if qa_mode == "mcq" and rotation_mode:
            # 取出这道题的所有旋转版本，每个版本都包含一套不同顺序的选项。
            rotations = qa["options"]
            # 记录一共有多少个旋转版本，后面用来做平均 EM 和位置偏置统计。
            n_rot = len(rotations)
            # 打印这道题的基本信息，方便在日志中追踪当前题的历史长度和旋转数。
            print(
                f"[INFO] QA {i}/{len(qas)} point={qa.get('point')} "
                f"method={method.name} mode=mcq rotations={n_rot} history_turns={len(history)}"
            )
            # 用于保存每个旋转版本的结果，后面再汇总成一个去偏 EM。
            rotation_results = []
            for r_idx, rot in enumerate(rotations):
                # 当前旋转版本的正确答案字母（如 A/B/C/D）。
                rot_answer = rot["answer"]
                # 去掉 answer 字段后，保留真正的选项内容，用于拼装 prompt。
                rot_options = {k: v for k, v in rot.items() if k != "answer"}
                # 把题干和当前旋转后的选项拼成最终输入文本。
                question = _format_options_block(question_text, rot_options)
                # 提取模型返回值时允许的合法选项集合。
                valid_keys = set(rot_options.keys())

                # 记录开始时间，用于计算这一次推理的耗时。
                t0 = dt.datetime.now()
                if is_agentic:
                    answer_kwargs: Dict[str, Any] = {}
                    try:
                        answer_signature = inspect.signature(method.answer)
                    except (TypeError, ValueError):
                        answer_signature = None
                    # 对于 agentic 方法，如果它的 answer() 支持 question_images，就把题图传进去。
                    if answer_signature is not None and "question_images" in answer_signature.parameters:
                        answer_kwargs["question_images"] = question_image_paths
                    
                    
                    # 调用 method 自己的 answer() 完成推理。
                    '''
                    qa：当前这道题的“题目对象”。它是从 dataset.iter_qas(...) 里取出来的一个字典。
                    里面不仅有题干，还包含答案、选项、题目 ID、所属 session、clue、point 等元信息。
                    qa["question"] = 题干
                    qa["answer"] = 标准答案
                    qa["options"] = 选项
                    qa["session_id"] / qa["clue"] = 上下文来源信息
                    question：当前这道题“实际发给模型的 prompt 文本”。它不是原始 qa 字典，而是由代码提前拼出来的字符串。例如在 MCQ 分支里，它会变成：“题干 + A. xxx + B. yyy + … + Answer with ONLY the option letter …”
                    '''
                    pred = method.answer(dataset, qa, question, **answer_kwargs)
                else:
                    # 对于普通方法，走统一的 router.answer(history, question, question_images=...)。
                    pred = router.answer(history, question, question_images=question_image_paths)
                # 计算单轮推理耗时，单位为毫秒。
                latency_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)

                # 从模型输出里提取选项字母，并和正确答案做对照。
                choice = extract_choice(pred, valid_keys=valid_keys)
                # 只有抽取出的选项字母与正确答案一致时，EM 才为 1。 
                em = 1.0 if choice == rot_answer.strip().upper() else 0.0
                # 记录 token 用量（如果 router 提供了 last_usage）。
                usage = {}
                if router is not None and hasattr(router, "last_usage"):
                    usage = dict(router.last_usage or {})
                # 把这一次旋转版本的结果追加到列表中，等所有旋转跑完后再汇总。
                rotation_results.append({
                    "rotation_idx": r_idx,
                    "correct_position": rot_answer,
                    "pred": pred,
                    "choice": choice,
                    "em": em,
                    "latency_ms": latency_ms,
                    "usage": usage,
                })
                print(
                    f"  [ROT {rot_answer}][{i}] choice={choice} gt={rot_answer} "
                    f"em={em:.0f} latency_ms={latency_ms}"
                )
                break

            # 把所有旋转版本的 EM 求平均，得到去偏后的整体得分。
            debiased_em = sum(r["em"] for r in rotation_results) / n_rot
            # 把每个旋转版本耗时加总，得到这道题的总延迟。
            total_latency = sum(r["latency_ms"] for r in rotation_results)
            # 统计模型实际选择了哪些选项字母，观察位置偏置（A/B/C/D 选择分布）。
            position_counts: Dict[str, int] = {}
            for r in rotation_results:
                c = r["choice"]
                if c != "INVALID":
                    position_counts[c] = position_counts.get(c, 0) + 1
            result = {
                "idx": i,
                "point": qa.get("point"),
                "mode": "mcq",
                "question": question_text,
                "gt": gt,
                "pred": rotation_results[-1]["pred"],  # last rotation pred for compat
                "choice": rotation_results[-1]["choice"],
                "valid_choice": all(r["choice"] != "INVALID" for r in rotation_results),
                "exact_match": debiased_em == 1.0,
                "em": debiased_em,
                "contains_gt": rotation_results[-1]["choice"] == gt.strip().upper(),
                "f1": debiased_em,
                "bleu": None,
                "bleu_1": None,
                "bleu_2": None,
                "bert": None,
                "judge": None,
                "judge_reasoning": None,
                "latency_ms": total_latency,
                "method_name": method.name,
                "effective_method_name": _effective_method_name(method_cfg),
                "method_modality": getattr(method, "modality", method_cfg.get("modality", "")),
                "history_turns": len(history),
                "source_sessions": qa.get("session_id", []),
                "clue_rounds": qa.get("clue", []),
                "rotations": rotation_results,
                "debiased_em": debiased_em,
                "position_bias": position_counts,
                "usage": {
                    "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rotation_results),
                    "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rotation_results),
                    "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in rotation_results),
                },
            }
            print(f"[MCQ][{i}] debiased_em={debiased_em:.2f} position_bias={position_counts} total_latency={total_latency}ms")

        # --- 分支 2：普通 MCQ（single options dict） ---
        # 这类题目已经是标准的单一选项集合，直接把题干与选项拼成 prompt 即可。
        # 模型输出通常是一段文本，代码会用 extract_choice 从中提取真正的选项字母 A/B/C/D。
        elif qa_mode == "mcq":
            question = format_question(qa)
            print(
                f"[INFO] QA {i}/{len(qas)} point={qa.get('point')} "
                f"method={method.name} mode=mcq history_turns={len(history)}"
            )
            t0 = dt.datetime.now()
            if is_agentic:
                answer_kwargs = {}
                try:
                    answer_signature = inspect.signature(method.answer)
                except (TypeError, ValueError):
                    answer_signature = None
                if answer_signature is not None and "question_images" in answer_signature.parameters:
                    answer_kwargs["question_images"] = question_image_paths
                # 如果 agentic method 支持图像参数，就把题图传入其 answer()。
                pred = method.answer(dataset, qa, question, **answer_kwargs)
            else:
                # 普通方法走统一的 router.answer 路径。
                pred = router.answer(history, question, question_images=question_image_paths)
            # 计算这次回答的耗时，并把结果写回结果对象，方便后续做性能分析。
            latency_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
            # 只保留选项字母作为有效答案集合；这里的 keys 通常是 A/B/C/D。
            valid_keys = set(qa.get("options", {}).keys())
            # 从模型输出中抽取选项字母，并和标准答案做比较。
            choice = extract_choice(pred, valid_keys=valid_keys)
            # 当抽取结果与正确答案一致时，EM 为 1，否则为 0。
            em = 1.0 if choice == gt.strip().upper() else 0.0
            result = {
                "idx": i,
                "point": qa.get("point"),
                "mode": "mcq",
                "question": question,
                "gt": gt,
                "pred": pred,
                "choice": choice,
                "valid_choice": choice != "INVALID",
                "exact_match": em == 1.0,
                "em": em,
                "contains_gt": choice == gt.strip().upper(),
                "f1": em,
                "bleu": None,
                "bleu_1": None,
                "bleu_2": None,
                "bert": None,
                "judge": None,
                "judge_reasoning": None,
                "latency_ms": latency_ms,
                "method_name": method.name,
                "effective_method_name": _effective_method_name(method_cfg),
                "method_modality": getattr(method, "modality", method_cfg.get("modality", "")),
                "history_turns": len(history),
                "source_sessions": qa.get("session_id", []),
                "clue_rounds": qa.get("clue", []),
            }
            print(f"[MCQ][{i}] choice={choice} gt={gt} em={em} latency_ms={latency_ms}")

        else:
            # --- 分支 3：Open-ended QA（开放式回答） ---
            # 这类题目没有固定选项，模型需要自己生成自然语言答案。
            # 因此评分不仅看是否精确匹配，还会用 F1 / BLEU / BERTScore / LLM Judge 做更全面的比较。
            question = format_question(qa)
            print(
                f"[INFO] QA {i}/{len(qas)} point={qa.get('point')} "
                f"method={method.name} mode=open history_turns={len(history)}"
            )
            t0 = dt.datetime.now()
            if is_agentic:
                answer_kwargs = {}
                try:
                    answer_signature = inspect.signature(method.answer)
                except (TypeError, ValueError):
                    answer_signature = None
                if answer_signature is not None and "question_images" in answer_signature.parameters:
                    answer_kwargs["question_images"] = question_image_paths
                # agentic 方法会自行管理记忆与推理流程，因此这里直接调用其 answer()。
                pred = method.answer(dataset, qa, question, **answer_kwargs)
            else:
                # 普通方法则依赖 router 来完成最终生成。
                pred = router.answer(history, question, question_images=question_image_paths)
            # 计算这种自由文本回答的耗时。
            latency_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
            # 如果 router 暴露了 token 用量，就记录下来，用于后续分析成本。
            _open_usage = {}
            if router is not None and hasattr(router, "last_usage"):
                _open_usage = dict(router.last_usage or {})

            # 先做基础的文本匹配判断：exact / contains。
            exact, contains = score_open(pred, gt)
            # 再计算更细的文本相似度指标。
            _f1 = f1_score(pred, gt)
            _bleu = bleu_score(pred, gt)
            _bleu1 = bleu_score(pred, gt, weights=(1, 0, 0, 0))
            _bleu2 = bleu_score(pred, gt, weights=(0.5, 0.5, 0, 0))
            # 如果启用了 BERTScore，就额外计算语义相似度分数。
            _bert  = bert_score_metric(pred, gt) if enable_bert_score else None

            # 如果启用了 LLM Judge，则额外调用 judge 模型给出打分与推理解释。
            _judge: Optional[float] = None
            _judge_reasoning: Optional[str] = None
            if enable_llm_judge and _judge_client is not None:
                try:
                    jr = llm_judge_score(
                        question=question,
                        ground_truth=gt,
                        model_output=pred,
                        client=_judge_client,
                        model_name=_judge_model,
                        prompt_template=_judge_template,
                        max_retries=judge_config.get("max_retries", 3),
                        timeout=judge_config.get("timeout", 60),
                    )
                    _judge = jr["score"]
                    _judge_reasoning = jr.get("reasoning", "")
                except RuntimeError as exc:
                    print(f"[WARN] LLM judge failed for QA {i}: {exc}")
            result = {
                "idx": i,
                "point": qa.get("point"),
                "mode": "open",
                "question": question,
                "gt": gt,
                "pred": pred,
                "exact_match": exact,
                "em": 1.0 if exact else 0.0,
                "contains_gt": contains,
                "f1": _f1,
                "bleu": _bleu,
                "bleu_1": _bleu1,
                "bleu_2": _bleu2,
                "bert": _bert,
                "judge": _judge,
                "judge_reasoning": _judge_reasoning,
                "latency_ms": latency_ms,
                "method_name": method.name,
                "effective_method_name": _effective_method_name(method_cfg),
                "method_modality": getattr(method, "modality", method_cfg.get("modality", "")),
                "history_turns": len(history),
                "source_sessions": qa.get("session_id", []),
                "clue_rounds": qa.get("clue", []),
                "usage": _open_usage,
            }
            print(
                f"[OPEN][{i}] em={exact} f1={_f1:.3f} bleu={_bleu:.3f}"
                + (f" bert={_bert:.3f}" if _bert is not None else "")
                + (f" judge={_judge}" if _judge is not None else "")
                + f" latency_ms={latency_ms}"
            )

        post_method_runtime = dict(getattr(method, "runtime_info", {}) or {})
        context_round_ids = _context_round_ids(history, post_method_runtime)
        result["context_round_ids"] = context_round_ids
        result["context_round_count"] = len(context_round_ids)
        if post_method_runtime:
            result["method_runtime"] = post_method_runtime
        results.append(result)

    # 第八步：所有题目跑完后，汇总 method 的运行时信息，并生成最终的 payload。
    method_runtime = dict(getattr(method, "runtime_info", {}) or {})
    payload = build_payload(cfg, paths, run_dir, dataset, results, method_runtime=method_runtime)

    cfg_to_write = dict(cfg)
    run_cfg = dict(cfg_to_write.get("run", {}))
    if method_runtime:
        run_cfg["method_runtime"] = method_runtime
    cfg_to_write["run"] = run_cfg

    # 第九步：把实验配置、指标摘要、逐题结果写入磁盘，供后续复现、对比和可视化使用。
    write_json(run_dir / "config.json", cfg_to_write)
    write_json(run_dir / "metrics.json", {k: payload[k] for k in payload if k != "results"})
    write_jsonl(run_dir / "predictions.jsonl", results)
    print(f"[INFO] Saved run artifacts: {run_dir}")

    output_json = paths.get("output_json")
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        model_name = cfg.get("model", {}).get("name", "model")
        method_name = _effective_method_name(cfg.get("method", {}))
        stem = output_json.stem
        suffix = output_json.suffix or ".json"
        tagged_path = output_json.parent / f"{stem}__{model_name}__{method_name}{suffix}"
        write_json(tagged_path, {k: payload[k] for k in payload if k != "results"})
        print(f"[INFO] Saved output summary: {tagged_path}")

    return payload



def run_modular_benchmark(
    task_config_path: str,
    model_config_path: str,
    method_config_path: str,
    output_root: str = "",
    mode: str = "",
    max_questions: int = 0,
    enable_bert_score: bool = False,
    enable_llm_judge: bool = False,
    judge_config: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = compose_modular_config(
        task_config_path=task_config_path,
        model_config_path=model_config_path,
        method_config_path=method_config_path,
        output_root=output_root,
        mode=mode,
        max_questions=max_questions,
    )
    config_dir = resolve_config_path(task_config_path).parent
    return run_benchmark(
        cfg,
        config_dir,
        enable_bert_score=enable_bert_score,
        enable_llm_judge=enable_llm_judge,
        judge_config=judge_config,
        run_dir=run_dir,
    )


def run_legacy_benchmark(
    opts: LegacyRunOptions,
    enable_bert_score: bool = False,
    enable_llm_judge: bool = False,
    judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = merge_legacy_config(opts)
    if opts.dialog_json:
        config_dir = Path(opts.dialog_json).parent
    else:
        config_dir = resolve_config_path(opts.config_path).parent
    return run_benchmark(
        cfg,
        config_dir,
        enable_bert_score=enable_bert_score,
        enable_llm_judge=enable_llm_judge,
        judge_config=judge_config,
    )
