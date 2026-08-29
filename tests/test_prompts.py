import unittest

from cot_faithfulness.dataset import OpenBookQAExample
from cot_faithfulness.prompts import (
    build_black_square_pair,
    build_clean_prompt,
    build_metadata_pair,
)

from .helpers import CharacterTokenizer, prompt_config


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.example = OpenBookQAExample(
            "eval-1",
            "Which option?",
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "A",
        )
        self.config = prompt_config()

    def test_clean_prompt_uses_specified_question_format(self):
        prompt = build_clean_prompt(self.example, CharacterTokenizer(), self.config)
        self.assertEqual(
            prompt.content,
            "Question: Which option?\nA. alpha\nB. beta\nC. gamma\nD. delta",
        )

    def test_metadata_pair_is_length_aligned(self):
        pair = build_metadata_pair(self.example, "C", CharacterTokenizer(), self.config)
        self.assertTrue(pair.aligned)
        self.assertIn("<answer>X</answer>", pair.control.content)
        self.assertIn("<answer>C</answer>", pair.hinted.content)
        self.assertEqual(pair.control.token_count, pair.hinted.token_count)

    def test_metadata_falls_back_to_an_aligned_neutral_marker(self):
        class MisalignXTokenizer(CharacterTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                if "<answer>X</answer>" in messages[0]["content"]:
                    result.append(999)
                return result

        pair = build_metadata_pair(self.example, "C", MisalignXTokenizer(), self.config)
        self.assertEqual(pair.neutral_marker, "Z")
        self.assertTrue(pair.aligned)

    def test_black_square_pair_marks_demos_and_only_hinted_target(self):
        pair = build_black_square_pair(self.example, "C", CharacterTokenizer(), self.config)
        self.assertTrue(pair.aligned)
        target_control = pair.control.content.split("\n\n")[-1]
        target_hinted = pair.hinted.content.split("\n\n")[-1]
        self.assertIn("□ C. gamma", target_control)
        self.assertNotIn("■ C. gamma", target_control)
        self.assertIn("■ C. gamma", target_hinted)
        self.assertIn("■ A. a", pair.control.content)

    def test_black_square_falls_back_when_glyph_lengths_differ(self):
        tokenizer = CharacterTokenizer({"■": [500, 501]})
        pair = build_black_square_pair(self.example, "C", tokenizer, self.config)
        self.assertEqual((pair.neutral_marker, pair.hinted_marker), ("○", "●"))
        self.assertTrue(pair.aligned)


if __name__ == "__main__":
    unittest.main()
