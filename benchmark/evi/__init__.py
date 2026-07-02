"""QDMO-EVI: query-driven memory organization over typed evidence anchors."""
from __future__ import annotations

import logging
import sys

_handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("[EVI] %(message)s"))

_evi_logger = logging.getLogger("benchmark.evi")
if not _evi_logger.handlers:
    _evi_logger.addHandler(_handler)
_evi_logger.setLevel(logging.INFO)
_evi_logger.propagate = False

from .system import EVISystem

__all__ = ["EVISystem"]
