import unittest

from cot_faithfulness.answer_scoring import AnswerScores
from cot_faithfulness.mechanistic_metrics import (
    compute_mechanistic_effects,
    validate_null_intervention,
)


def scores(offset=0.0):
    return AnswerScores(
        predicted_answer="A",
        label_logprobs={"A": -0.1 + offset, "B": -1.0, "C": -2.0, "D": -3.0},
        label_token_ids={"A": (1,), "B": (2,), "C": (3,), "D": (4,)},
    )


class MechanisticMetricsTests(unittest.TestCase):
    def test_direct_mediated_and_total_effects_are_additive(self):
        result = compute_mechanistic_effects(y00=-2.0, y10=-0.5, y11=0.25)
        self.assertEqual(result.direct, 1.5)
        self.assertEqual(result.mediated, 0.75)
        self.assertEqual(result.total, 2.25)
        self.assertTrue(result.additivity_passed)

    def test_null_requires_exact_suffixes_and_tolerant_answer_scores(self):
        result = validate_null_intervention(
            baseline_suffix_ids=[7, 8],
            unbatched_suffix_ids=[7, 8],
            batched_suffix_ids=[[7, 8], [7, 8]],
            baseline_scores=scores(),
            unbatched_scores=scores(1e-7),
            batched_scores=[scores(), scores(-1e-7)],
            absolute_tolerance=1e-6,
            relative_tolerance=0.0,
        )
        self.assertTrue(result.passed)
        failed = validate_null_intervention(
            baseline_suffix_ids=[7, 8],
            unbatched_suffix_ids=[7, 9],
            batched_suffix_ids=[[7, 8], [7, 8]],
            baseline_scores=scores(),
            unbatched_scores=scores(),
            batched_scores=[scores(), scores()],
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )
        self.assertFalse(failed.passed)


if __name__ == "__main__":
    unittest.main()
