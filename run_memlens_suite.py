"""Run MemEye methods on converted MEMLENS items, then invoke its official judge."""

from __future__ import annotations

import argparse
import os

from benchmark.memlens.suite import run_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/memlens/converted_32k_agent195/manifest.json")
    parser.add_argument("--image-root", default="", help="Defaults to runtime_image_root in the manifest.")
    parser.add_argument("--model-config", default="config/models/gpt_4_1_nano.yaml")
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-dir", default="", help="Existing directory to resume, or an explicit new directory.")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--clear-cache-every", type=int, default=1, help="0 disables per-item retriever cleanup.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--official-dir", default="third_party/MEMLENS")
    parser.add_argument("--questions-file", default="data/memlens/dataset_32k.json")
    parser.add_argument("--judge-model", default=os.environ.get("MEMLENS_JUDGE_MODEL", ""))
    parser.add_argument("--judge-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = run_suite(args)
    print(f"MEMLENS run written to: {run_dir}")


if __name__ == "__main__":
    main()
