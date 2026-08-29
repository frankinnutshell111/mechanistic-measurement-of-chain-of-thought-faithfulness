import unittest

from cot_faithfulness.config import (
    ConfigError,
    apply_overrides,
    apply_mechanistic_smoke_overrides,
    apply_text_mediation_overrides,
    config_from_mapping,
)

from .helpers import experiment_config


class ConfigTests(unittest.TestCase):
    def test_round_trips_resolved_mapping(self):
        config = experiment_config()
        restored = config_from_mapping(config.to_dict())
        self.assertEqual(restored, config)

    def test_cli_overrides_are_local(self):
        config = experiment_config()
        updated = apply_overrides(
            config,
            model_id="Qwen/Qwen3-0.6B",
            layers=[3, 7, 11],
            output_path="smoke.jsonl",
            limit=1,
        )
        self.assertEqual(updated.model.model_id, "Qwen/Qwen3-0.6B")
        self.assertEqual(updated.model.layers, (3, 7, 11))
        self.assertEqual(updated.screening.output_path, "smoke.jsonl")
        self.assertEqual(updated.screening.limit, 1)
        self.assertEqual(config.model.model_id, "Qwen/Qwen3-14B")

    def test_unknown_keys_fail_loudly(self):
        raw = experiment_config().to_dict()
        raw["model"]["mystery_option"] = "none"
        with self.assertRaises(ConfigError):
            config_from_mapping(raw)

    def test_text_mediation_overrides_do_not_change_screening_output(self):
        config = experiment_config()
        updated = apply_text_mediation_overrides(
            config,
            input_path="phase1.jsonl",
            output_path="phase2.jsonl",
            limit=2,
        )
        self.assertEqual(updated.text_mediation.input_path, "phase1.jsonl")
        self.assertEqual(updated.text_mediation.output_path, "phase2.jsonl")
        self.assertEqual(updated.text_mediation.limit, 2)
        self.assertEqual(updated.screening.output_path, config.screening.output_path)

    def test_quantization_is_rejected(self):
        raw = experiment_config().to_dict()
        raw["model"]["quantization"] = "4bit"
        with self.assertRaises(ConfigError):
            config_from_mapping(raw)

    def test_duplicate_layer_indices_are_rejected(self):
        raw = experiment_config().to_dict()
        raw["model"]["layers"] = [9, 9]
        with self.assertRaises(ConfigError):
            config_from_mapping(raw)

    def test_mechanistic_settings_are_typed_and_validated(self):
        raw = experiment_config().to_dict()
        raw["mechanistic"]["random_distribution"] = "gaussian"
        restored = config_from_mapping(raw)
        self.assertEqual(restored.mechanistic.random_distribution, "gaussian")

    def test_mechanistic_smoke_override_selects_one_layer_without_touching_phase1(self):
        config = experiment_config()
        updated = apply_mechanistic_smoke_overrides(
            config,
            input_path="phase1.jsonl",
            output_path="phase4.json",
            layer_idx=4,
            block_idx=2,
            alpha=0.5,
        )
        self.assertIn(4, updated.model.layers)
        self.assertEqual(updated.mechanistic_smoke.layer_idx, 4)
        self.assertEqual(updated.mechanistic_smoke.block_idx, 2)
        self.assertEqual(updated.mechanistic.alpha_primary, 0.5)
        self.assertEqual(updated.screening.output_path, config.screening.output_path)


if __name__ == "__main__":
    unittest.main()
