"""MEMLENS QA runner with dated EVI retrieval queries."""

from run_memlens_suite import build_parser
from benchmark.memlens.runtime_fixes import install_evi_question_date_fix


def main() -> None:
    install_evi_question_date_fix()
    from benchmark.memlens.suite import run_suite

    run_dir = run_suite(build_parser().parse_args())
    print(f"MEMLENS run written to: {run_dir}")


if __name__ == "__main__":
    main()
