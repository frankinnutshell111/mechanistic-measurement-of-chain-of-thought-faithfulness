import unittest

from cot_faithfulness.config import DecodingConfig
from cot_faithfulness.generation import (
    detect_repetitive_loop,
    resolve_end_think_id,
    trace_from_generated_ids,
)

from .helpers import CharacterTokenizer


class GenerationTests(unittest.TestCase):
    def test_resolves_verified_end_think_id(self):
        self.assertEqual(resolve_end_think_id(CharacterTokenizer()), 2)

    def test_preserves_reasoning_ids_and_removes_closing_tag(self):
        tokenizer = CharacterTokenizer()
        ids = tokenizer.encode("reason") + [resolve_end_think_id(tokenizer)]
        trace = trace_from_generated_ids(tokenizer, ids, DecodingConfig())
        self.assertEqual(trace.text, "reason")
        self.assertEqual(trace.token_ids, tuple(tokenizer.encode("reason")))
        self.assertTrue(trace.completed)
        self.assertFalse(trace.truncated)

    def test_marks_token_cap_without_closing_tag(self):
        tokenizer = CharacterTokenizer()
        config = DecodingConfig(max_reasoning_tokens=4)
        trace = trace_from_generated_ids(tokenizer, [101, 102, 103, 104], config)
        self.assertFalse(trace.completed)
        self.assertTrue(trace.truncated)
        self.assertEqual(trace.stop_reason, "token_cap")

    def test_detects_exact_suffix_loop(self):
        self.assertTrue(detect_repetitive_loop([7, 8] * 4))
        self.assertFalse(detect_repetitive_loop([7, 8, 7, 9, 7, 8, 7, 10]))


if __name__ == "__main__":
    unittest.main()
