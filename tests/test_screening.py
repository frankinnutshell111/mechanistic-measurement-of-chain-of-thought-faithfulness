import unittest

from cot_faithfulness.answer_scoring import AnswerScores
from cot_faithfulness.generation import ReasoningTrace
from cot_faithfulness.prompts import Prompt
from cot_faithfulness.screening import (
    EvaluatedPrompt,
    choose_hint_label,
    classify_screening,
)


def evaluated(answer: str, *, completed: bool = True) -> EvaluatedPrompt:
    return EvaluatedPrompt(
        prompt=Prompt("p", (1,)),
        trace=ReasoningTrace(
            token_ids=(10,),
            text="r",
            completed=completed,
            truncated=False,
            repetitive=False,
            stop_reason="end_think" if completed else "eos_before_end_think",
            generated_token_count=2,
        ),
        answer_scores=AnswerScores(
            predicted_answer=answer,
            label_logprobs={"A": -1.0, "B": -2.0, "C": -3.0, "D": -4.0},
            label_token_ids={"A": (1,), "B": (2,), "C": (3,), "D": (4,)},
        ),
        scoring_context_token_count=3,
    )


class ScreeningTests(unittest.TestCase):
    def test_hint_selection_is_stable_and_excludes_clean_answer(self):
        first = choose_hint_label("A", ("A", "B", "C", "D"), base_seed=7, question_id="q1")
        second = choose_hint_label("A", ("A", "B", "C", "D"), base_seed=7, question_id="q1")
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], "A")

    def test_strict_paired_classification(self):
        result = classify_screening(
            clean=evaluated("A"),
            metadata_control=evaluated("A"),
            metadata_hinted=evaluated("C"),
            black_square_control=evaluated("A"),
            black_square_hinted=evaluated("C"),
            hint_label="C",
            metadata_aligned=True,
            black_square_aligned=True,
        )
        self.assertTrue(result["strict_paired_eligible"])
        self.assertTrue(result["condition_eligible"]["metadata"])
        self.assertEqual(result["rejection_reasons"], [])

    def test_incomplete_trace_is_rejected_with_reason(self):
        result = classify_screening(
            clean=evaluated("A"),
            metadata_control=evaluated("A", completed=False),
            metadata_hinted=evaluated("C"),
            black_square_control=evaluated("A"),
            black_square_hinted=evaluated("C"),
            hint_label="C",
            metadata_aligned=True,
            black_square_aligned=True,
        )
        self.assertFalse(result["strict_paired_eligible"])
        self.assertIn("metadata_control:missing_end_think", result["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
