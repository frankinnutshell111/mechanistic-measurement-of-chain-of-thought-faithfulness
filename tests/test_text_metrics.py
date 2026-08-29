import unittest

from cot_faithfulness.text_metrics import (
    answer_contrast,
    compute_text_mediation_effects,
)


class TextMetricsTests(unittest.TestCase):
    def test_answer_contrast_uses_hint_against_baseline(self):
        value = answer_contrast({"A": -2.5, "C": -0.5}, "C", "A")
        self.assertEqual(value, 2.0)

    def test_crossed_context_effects_and_fraction(self):
        effects = compute_text_mediation_effects(
            y_control_control=-2.0,
            y_hint_control=1.0,
            y_control_hint=2.0,
            epsilon=1e-8,
            minimum_effect_magnitude=1e-6,
        )
        self.assertEqual(effects.direct_text, 3.0)
        self.assertEqual(effects.mediated_text, 4.0)
        self.assertAlmostEqual(effects.fraction_text_mediated, 4.0 / (7.0 + 1e-8))
        self.assertTrue(effects.effect_above_floor)

    def test_negligible_effect_is_flagged(self):
        effects = compute_text_mediation_effects(
            y_control_control=0.0,
            y_hint_control=0.0,
            y_control_hint=0.0,
            epsilon=1e-8,
            minimum_effect_magnitude=1e-6,
        )
        self.assertEqual(effects.fraction_text_mediated, 0.0)
        self.assertFalse(effects.effect_above_floor)


if __name__ == "__main__":
    unittest.main()
