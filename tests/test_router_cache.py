from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from router.base import BaseRouter
from router.cache import CachedAnswerRouter


class FakeRouter(BaseRouter):
    def __init__(self) -> None:
        self.model = "fake-model"
        self.base_url = "https://example.invalid/v1"
        self.max_new_tokens = 32
        self.system_prompt = "system"
        self.calls = 0
        self.last_usage = {}

    def answer(self, history_messages, question, question_images=None):
        self.calls += 1
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }
        return f"answer:{question}"


class CachedAnswerRouterTests(unittest.TestCase):
    def test_exact_input_is_reused_and_cache_hit_cost_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
            os.environ,
            {"MEMEYE_QA_CACHE_DIR": cache_dir},
        ):
            backend = FakeRouter()
            router = CachedAnswerRouter(backend)
            history = [{"role": "user", "text": "memory", "images": []}]

            first = router.answer(history, "question A")
            first_usage = dict(router.last_usage)
            second = router.answer(history, "question A")

            self.assertEqual(first, second)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(first_usage["total_tokens"], 12)
            self.assertTrue(router.last_cache_hit)
            self.assertEqual(router.last_usage["total_tokens"], 0)
            self.assertEqual(router.last_cached_usage["total_tokens"], 12)

    def test_rotation_question_text_gets_a_distinct_key(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
            os.environ,
            {"MEMEYE_QA_CACHE_DIR": cache_dir},
        ):
            backend = FakeRouter()
            router = CachedAnswerRouter(backend)
            history = [{"role": "user", "text": "memory", "images": []}]

            router.answer(history, """A. one
B. two""")
            router.answer(history, """A. two
B. one""")

            self.assertEqual(backend.calls, 2)
            self.assertFalse(router.last_cache_hit)


if __name__ == "__main__":
    unittest.main()