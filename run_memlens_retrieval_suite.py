"""Evaluate retrieval only over the converted MEMLENS subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.memlens.retrieval_suite import run_memlens_retrieval_suite


def parse_ks(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--ks requires comma-separated positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/memlens/converted_32k_agent195/manifest.json")
    parser.add_argument("--image-root", default="data/memlens")
    parser.add_argument("--model-config", default="config/models/qwen3_vl_8b_openrouter.yaml")
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--ks", type=parse_ks, default=[1, 3, 5, 10, 20])
    parser.add_argument("--clear-cache-every", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    run_memlens_retrieval_suite(
        manifest_path=Path(args.manifest), image_root=Path(args.image_root),
        model_config=args.model_config, method_config=args.method_config,
        output_root=Path(args.output_root), max_questions=args.max_questions,
        k_values=args.ks, run_dir=Path(args.run_dir) if args.run_dir else None,
        clear_cache_every=args.clear_cache_every, fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
