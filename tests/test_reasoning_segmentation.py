import unittest

from cot_faithfulness.reasoning_segmentation import segment_reasoning

from .helpers import CharacterTokenizer


class BrokenRoundTripTokenizer(CharacterTokenizer):
    def decode(self, token_ids, **kwargs):
        del token_ids, kwargs
        return "not-the-original-tokens"


class ReasoningSegmentationTests(unittest.TestCase):
    def test_expression_units_cover_original_ids_without_truncation(self):
        tokenizer = CharacterTokenizer()
        token_ids = tokenizer.encode("First step. Second step;\nFinally done.")
        result = segment_reasoning(tokenizer, token_ids, max_blocks=10)
        self.assertEqual(result.method, "expression_units")
        self.assertTrue(result.exact_mapping_verified)
        self.assertEqual(result.blocks[0].start_token, 0)
        self.assertEqual(result.blocks[-1].end_token, len(token_ids))
        rebuilt = tuple(
            token_id
            for block in result.blocks
            for token_id in token_ids[block.start_token : block.end_token]
        )
        self.assertEqual(rebuilt, tuple(token_ids))
        self.assertLessEqual(len(result.blocks), 10)
        self.assertFalse(result.blocks[-1].has_future_reasoning)

    def test_more_than_ten_units_are_merged_into_balanced_adjacent_blocks(self):
        tokenizer = CharacterTokenizer()
        text = " ".join(f"Unit {index}." for index in range(17))
        token_ids = tokenizer.encode(text)
        result = segment_reasoning(tokenizer, token_ids, max_blocks=10)
        self.assertEqual(len(result.blocks), 10)
        self.assertEqual(sum(block.token_count for block in result.blocks), len(token_ids))
        self.assertTrue(
            all(
                left.end_token == right.start_token
                for left, right in zip(result.blocks, result.blocks[1:])
            )
        )

    def test_round_trip_mismatch_uses_deterministic_token_blocks(self):
        result = segment_reasoning(BrokenRoundTripTokenizer(), [10, 11, 12, 13, 14], max_blocks=3)
        self.assertEqual(result.method, "deterministic_token_fallback")
        self.assertFalse(result.exact_mapping_verified)
        self.assertEqual([block.token_count for block in result.blocks], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
