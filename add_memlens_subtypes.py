from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.memlens.subtypes import add_question_subtypes


def main() -> None:
    parser = argparse.ArgumentParser(description="Add official question_subtype metadata.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    args = parser.parse_args()
    result = add_question_subtypes(args.dataset, args.converted_dir)
    print(f"Updated {result['updated']} records; missing={len(result['missing'])}")


if __name__ == "__main__":
    main()
