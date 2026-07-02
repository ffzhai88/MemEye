"""
EVI v2: Multi-dimensional vector indexing with temporal assembly.

Two retrieval modes (config: interaction_mode):
  - "standard" (default): LLM directory retrieval → vector topK → diversity boost → temporal assembly
  - "qdmo":           soft activation → active subset → memory interaction → emergent clustering → cluster context

QDMO adds query-conditioned memory-to-memory interaction and emergent clustering,
where memories dynamically reorganize based on the query rather than being retrieved as independent items.
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
from .summarizer import load_all_summaries, summarize_session

__all__ = ["EVISystem", "load_all_summaries", "summarize_session"]
