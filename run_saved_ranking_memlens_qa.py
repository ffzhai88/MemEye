"""Run MEMLENS QA from a ranking saved by an offline joint analyzer."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
from pathlib import Path
from types import SimpleNamespace

import yaml

from benchmark.common import SCRIPT_DIR, resolve_config_path
from benchmark.memlens.suite import run_suite
from run_saved_ranking_qa_suite import _acquire_run_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rankings", required=True)
    parser.add_argument(
        "--manifest",
        default="data/memlens/converted_32k_agent195/manifest.json",
    )
    parser.add_argument("--image-root", default="")
    parser.add_argument(
        "--model-config",
        default="config/models/qwen3_vl_8B_ali.yaml",
    )
    parser.add_argument("--ranking-strategy", default="selective_vlm")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--clear-cache-every", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--official-dir", default="third_party/MEMLENS")
    parser.add_argument("--questions-file", default="data/memlens/dataset_32k.json")
    parser.add_argument(
        "--judge-model",
        default="qwen3-vl-235b-a22b-instruct",
    )
    parser.add_argument(
        "--judge-base-url",
        default=(
            "https://llm-owockiqa6c46tlmv.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
    )
    parser.add_argument("--judge-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-workers", type=int, default=1)
    args = parser.parse_args()

    source_rankings = Path(args.source_rankings).expanduser().resolve()
    if not source_rankings.is_file():
        parser.error(f"Saved rankings not found: {source_rankings}")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = SCRIPT_DIR / output_root
    output_root = output_root.resolve()
    run_lock = _acquire_run_lock(
        output_root,
        source_rankings,
        args.model_config,
        args.ranking_strategy,
        args.top_k,
        scope="MEMLENS",
    )

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            output_root
            / "MEMLENS"
            / (
                f"{timestamp}_{resolve_config_path(args.model_config).stem}_"
                f"saved_{args.ranking_strategy}_top{args.top_k}"
            )
        )
        run_dir.mkdir(parents=True, exist_ok=False)
    method_config = run_dir / "method_config.yaml"
    method_config.write_text(
        yaml.safe_dump({
            "method": "saved_ranking_replay",
            "name": f"saved_ranking_replay_{args.ranking_strategy}",
            "modality": "multimodal",
            "source_rankings_jsonl": str(source_rankings),
            "source_benchmark": "memlens",
            "source_dataset": "memlens",
            "ranking_strategy": args.ranking_strategy,
            "context_top_k": args.top_k,
        }, sort_keys=False),
        encoding="utf-8",
    )
    suite_args = SimpleNamespace(
        manifest=args.manifest,
        image_root=args.image_root,
        model_config=args.model_config,
        method_config=str(method_config),
        output_root=str(output_root),
        run_dir=str(run_dir),
        max_questions=args.max_questions,
        clear_cache_every=args.clear_cache_every,
        unload_embedding_models=False,
        fail_fast=args.fail_fast,
        skip_judge=args.skip_judge,
        official_dir=args.official_dir,
        questions_file=args.questions_file,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_key_env=args.judge_key_env,
        judge_workers=args.judge_workers,
    )
    try:
        completed_dir = run_suite(suite_args)
        print(f"Saved-ranking MEMLENS QA written to: {completed_dir}")
    finally:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
        run_lock.close()


if __name__ == "__main__":
    main()
