import unittest

from cot_faithfulness.prompts import tokenize_chat_prompt

from .helpers import CharacterTokenizer


class MappingTokenizer(CharacterTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": super().apply_chat_template(*args, **kwargs)}


class TokenizerCompatibilityTests(unittest.TestCase):
    def test_accepts_mapping_returned_by_current_transformers(self):
        ids = tokenize_chat_prompt(MappingTokenizer(), "Question: test")
        self.assertIsInstance(ids, tuple)
        self.assertGreater(len(ids), 0)


if __name__ == "__main__":
    unittest.main()
