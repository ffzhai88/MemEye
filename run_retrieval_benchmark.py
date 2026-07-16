import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.retrieval_eval import run_modular_retrieval_benchmark


def parse_k_values(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks must be a comma-separated list of positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--ks must contain at least one positive integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality without calling the final QA model.")
    parser.add_argument("--task-config", default="config/tasks/brand_memory_test.yaml")
    parser.add_argument("--model-config", default="config/models/gpt_4_1_nano.yaml")
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--ks", type=parse_k_values, default=[1, 3, 5, 10, 20])
    args = parser.parse_args()
    run_modular_retrieval_benchmark(
        task_config_path=args.task_config,
        model_config_path=args.model_config,
        method_config_path=args.method_config,
        output_root=args.output_root,
        max_questions=args.max_questions,
        k_values=args.ks,
    )


if __name__ == "__main__":
    main()