import json
import tempfile
import unittest
from pathlib import Path

from cot_faithfulness.checkpoint import (
    DuplicateCompletionError,
    JsonlCheckpoint,
)


class CheckpointTests(unittest.TestCase):
    def test_resume_never_duplicates_completed_question(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screening.jsonl"
            checkpoint = JsonlCheckpoint(path)
            checkpoint.append_terminal({"question_id": "q1", "strict_paired_eligible": True})
            resumed = JsonlCheckpoint(path, resume=True)
            self.assertTrue(resumed.is_completed("q1"))
            with self.assertRaises(DuplicateCompletionError):
                resumed.append_terminal({"question_id": "q1"})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)

    def test_no_resume_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screening.jsonl"
            path.write_text('{"question_id":"q1"}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                JsonlCheckpoint(path, resume=False)

    def test_supports_composite_phase_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mediation.jsonl"
            checkpoint = JsonlCheckpoint(path, key_field="completion_key")
            checkpoint.append_terminal(
                {
                    "completion_key": "q1|metadata",
                    "question_id": "q1",
                    "hint_condition": "metadata",
                }
            )
            resumed = JsonlCheckpoint(path, key_field="completion_key")
            self.assertTrue(resumed.is_completed("q1|metadata"))
            self.assertFalse(resumed.is_completed("q1|black_square"))

    def test_resume_rejects_duplicate_keys_already_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screening.jsonl"
            path.write_text(
                '{"question_id":"q1"}\n{"question_id":"q1"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                JsonlCheckpoint(path)


if __name__ == "__main__":
    unittest.main()
