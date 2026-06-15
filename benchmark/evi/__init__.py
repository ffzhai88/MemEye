"""
EVI v2: Multi-dimensional vector indexing with temporal assembly.
"""
import logging
import sys

_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("[EVI] %(message)s"))

_evi_logger = logging.getLogger("benchmark.evi")
_evi_logger.addHandler(_handler)
_evi_logger.setLevel(logging.INFO)
_evi_logger.propagate = False

from .system import EVISystem

__all__ = ["EVISystem"]
