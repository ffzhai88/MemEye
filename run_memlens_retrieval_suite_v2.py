"""MEMLENS retrieval runner with corrected round-K answer-session coverage."""

from benchmark.memlens.runtime_fixes import install_retrieval_session_metric_fix


if __name__ == "__main__":
    install_retrieval_session_metric_fix()
    from run_memlens_retrieval_suite import main

    main()
