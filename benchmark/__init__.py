"""MemEye benchmark package with lazy public entry points."""
from __future__ import annotations

from typing import Any

__all__ = ["LegacyRunOptions", "run_legacy_benchmark", "run_modular_benchmark", "run_benchmark_matrix"]


def __getattr__(name: str) -> Any:
    if name == "run_benchmark_matrix":
        from .matrix import run_benchmark_matrix
        return run_benchmark_matrix
    if name in {"LegacyRunOptions", "run_legacy_benchmark", "run_modular_benchmark"}:
        from .runner import LegacyRunOptions, run_legacy_benchmark, run_modular_benchmark
        return {
            "LegacyRunOptions": LegacyRunOptions,
            "run_legacy_benchmark": run_legacy_benchmark,
            "run_modular_benchmark": run_modular_benchmark,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
