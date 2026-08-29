import unittest

from cot_faithfulness.config import MechanisticConfig
from cot_faithfulness.perturbations import (
    apply_activation_perturbation,
    build_perturbation_directions,
    structured_direction,
)

try:
    import torch
except ImportError:  # pragma: no cover - local offline Python may omit torch
    torch = None


@unittest.skipUnless(torch is not None, "torch is required for activation tests")
class PerturbationTests(unittest.TestCase):
    def test_structured_direction_is_hint_minus_neutral(self):
        neutral = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        hinted = torch.tensor([[2.0, 0.0], [5.0, 8.0]])
        self.assertTrue(torch.equal(structured_direction(neutral, hinted), hinted - neutral))

    def test_random_directions_are_deterministic_orthogonal_and_norm_matched(self):
        neutral = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10
        hinted = neutral + torch.linspace(-1.0, 1.0, 24).reshape(4, 6)
        kwargs = {
            "question_id": "q1",
            "hint_type": "metadata",
            "layer_idx": 9,
            "block_idx": 2,
            "base_seed": 17,
            "config": MechanisticConfig(),
        }
        first = build_perturbation_directions(neutral, hinted, **kwargs)
        second = build_perturbation_directions(neutral, hinted, **kwargs)
        self.assertTrue(first.detectable)
        self.assertEqual(first.num_directions, 5)
        self.assertTrue(torch.equal(first.values, second.values))
        flattened = first.values.flatten(1)
        norms = torch.linalg.vector_norm(flattened, dim=1)
        self.assertTrue(torch.allclose(norms, norms[0].expand_as(norms), rtol=1e-5, atol=1e-6))
        normalized = flattened / norms[:, None]
        gram = normalized @ normalized.T
        self.assertTrue(torch.allclose(gram, torch.eye(5), rtol=1e-5, atol=1e-5))
        self.assertEqual(
            [item.seed for item in first.metadata],
            [item.seed for item in second.metadata],
        )

    def test_negligible_structured_direction_skips_random_normalization(self):
        neutral = torch.ones((2, 3))
        result = build_perturbation_directions(
            neutral,
            neutral.clone(),
            question_id="q1",
            hint_type="black_square",
            layer_idx=19,
            block_idx=0,
            base_seed=4,
            config=MechanisticConfig(),
        )
        self.assertFalse(result.detectable)
        self.assertEqual(result.num_directions, 1)
        self.assertEqual(result.exclusion_reason, "structured_direction_norm_below_threshold")

    def test_alpha_zero_is_exact_identity(self):
        baseline = torch.randn(2, 3, 4)
        direction = torch.randn(2, 3, 4)
        result = apply_activation_perturbation(baseline, direction, alpha=0.0)
        self.assertTrue(torch.equal(result, baseline))
        self.assertNotEqual(result.data_ptr(), baseline.data_ptr())

    def test_structured_replacement_matches_within_bfloat16_tolerance(self):
        neutral = torch.randn(3, 5).to(torch.bfloat16)
        hinted = torch.randn(3, 5).to(torch.bfloat16)
        direction = structured_direction(neutral, hinted)
        result = apply_activation_perturbation(
            neutral.unsqueeze(0),
            direction,
            alpha=1.0,
        )
        self.assertTrue(torch.allclose(result.float()[0], hinted.float(), rtol=0.02, atol=0.02))

    def test_replacement_operation_interpolates_to_target(self):
        baseline = torch.zeros(1, 2, 3)
        target = torch.full((2, 3), 4.0)
        result = apply_activation_perturbation(
            baseline,
            target,
            alpha=1.0,
            operation="replacement",
        )
        self.assertTrue(torch.equal(result, target.unsqueeze(0)))


if __name__ == "__main__":
    unittest.main()
