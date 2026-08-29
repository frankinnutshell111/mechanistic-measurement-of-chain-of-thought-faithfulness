import unittest

from cot_faithfulness.dataset import DatasetFormatError, normalize_openbookqa_row


class DatasetTests(unittest.TestCase):
    def test_normalizes_openbookqa_schema(self):
        example = normalize_openbookqa_row(
            {
                "id": "q1",
                "question_stem": "What?",
                "choices": {
                    "label": ["A", "B", "C", "D"],
                    "text": ["one", "two", "three", "four"],
                },
                "answerKey": "c",
            }
        )
        self.assertEqual(example.question_id, "q1")
        self.assertEqual(example.choices["B"], "two")
        self.assertEqual(example.answer_key, "C")

    def test_rejects_unordered_labels(self):
        with self.assertRaises(DatasetFormatError):
            normalize_openbookqa_row(
                {
                    "id": "q1",
                    "question_stem": "What?",
                    "choices": {
                        "label": ["B", "A", "C", "D"],
                        "text": ["one", "two", "three", "four"],
                    },
                    "answerKey": "A",
                }
            )


if __name__ == "__main__":
    unittest.main()
