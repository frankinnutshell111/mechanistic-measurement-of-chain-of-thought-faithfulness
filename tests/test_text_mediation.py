import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cot_faithfulness.answer_scoring import AnswerScores
from cot_faithfulness.text_mediation import (
    build_crossed_scoring_contexts,
    build_text_mediation_record,
    completion_key,
    run_text_mediation,
    summarize_text_mediation,
)

from .helpers import CharacterTokenizer, experiment_config


def serialized_scores(predicted, *, a, c):
    return AnswerScores(
        predicted_answer=predicted,
        label_logprobs={"A": a, "B": -5.0, "C": c, "D": -6.0},
        label_token_ids={"A": (1,), "B": (2,), "C": (3,), "D": (4,)},
    ).to_dict()


def screening_record():
    def evaluated(prompt_ids, reasoning_ids, answer_scores):
        return {
            "prompt": {"input_ids": prompt_ids},
            "trace": {"reasoning_token_ids": reasoning_ids},
            "answer_scores": answer_scores,
        }

    pair = {
        "aligned": True,
        "control": evaluated([10, 11], [30, 31], serialized_scores("A", a=-0.1, c=-2.1)),
        "hinted": evaluated([20, 21], [40, 41], serialized_scores("C", a=-3.0, c=-0.5)),
    }
    return {
        "schema_version": "phase1.screening.v1",
        "status": "complete",
        "question_id": "q1",
        "answer_key": "B",
        "hint_label": "C",
        "model_id": "Qwen/Qwen3-14B",
        "condition_eligible": {"metadata": True, "black_square": True},
        "strict_paired_eligible": True,
        "prompt_pairs": {"metadata": pair, "black_square": pair},
    }


def summary_record(question_id, condition, direct, mediated):
    magnitude = abs(direct) + abs(mediated)
    return {
        "question_id": question_id,
        "hint_condition": condition,
        "metrics": {
            "D_text": direct,
            "C_text": mediated,
            "absolute_D_text": abs(direct),
            "absolute_C_text": abs(mediated),
            "F_text": abs(mediated) / magnitude,
            "effect_above_floor": True,
        },
    }


class TextMediationTests(unittest.TestCase):
    def test_crossed_contexts_swap_prompt_and_reasoning_ids(self):
        tokenizer = CharacterTokenizer()
        direct, mediated = build_crossed_scoring_contexts(
            tokenizer,
            screening_record(),
            "metadata",
            experiment_config(),
        )
        self.assertEqual(direct[:5], (20, 21, 30, 31, 2))
        self.assertEqual(mediated[:5], (10, 11, 40, 41, 2))

    def test_record_computes_proposal_metrics(self):
        direct = AnswerScores.from_dict(serialized_scores("C", a=-1.0, c=0.0))
        mediated = AnswerScores.from_dict(serialized_scores("C", a=-2.0, c=0.0))
        record = build_text_mediation_record(
            screening_record(),
            "metadata",
            direct,
            mediated,
            experiment_config(),
            {},
            direct_context_token_count=10,
            mediated_context_token_count=10,
        )
        self.assertEqual(record["completion_key"], completion_key("q1", "metadata"))
        self.assertAlmostEqual(record["metrics"]["D_text"], 3.0)
        self.assertAlmostEqual(record["metrics"]["C_text"], 4.0)
        self.assertAlmostEqual(record["metrics"]["F_text"], 4.0 / (7.0 + 1e-8))

    def test_summary_reports_expected_group_pattern_and_pairs(self):
        records = [
            summary_record("q1", "metadata", 4.0, 1.0),
            summary_record("q2", "metadata", 3.0, 1.0),
            summary_record("q1", "black_square", 1.0, 4.0),
            summary_record("q2", "black_square", 1.0, 3.0),
        ]
        summary = summarize_text_mediation(records, epsilon=1e-8)
        self.assertEqual(summary["paired"]["n"], 2)
        self.assertTrue(summary["expected_pattern"]["all_criteria_observed"])
        self.assertGreater(
            summary["by_condition"]["black_square"]["mean_F_text"],
            summary["by_condition"]["metadata"]["mean_F_text"],
        )

    def test_runner_checkpoints_each_condition_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_config = experiment_config()
            config = replace(
                base_config,
                text_mediation=replace(
                    base_config.text_mediation,
                    input_path=str(root / "screening.jsonl"),
                    output_path=str(root / "mediation.jsonl"),
                    summary_json_path=str(root / "summary.json"),
                    summary_csv_path=str(root / "summary.csv"),
                ),
            )
            direct = AnswerScores.from_dict(serialized_scores("C", a=-1.0, c=0.0))
            mediated = AnswerScores.from_dict(serialized_scores("C", a=-2.0, c=0.0))

            def fake_evaluate(model, tokenizer, source, condition, active_config, metadata):
                return build_text_mediation_record(
                    source,
                    condition,
                    direct,
                    mediated,
                    active_config,
                    metadata,
                    direct_context_token_count=10,
                    mediated_context_token_count=10,
                )

            with (
                patch(
                    "cot_faithfulness.text_mediation.evaluate_text_condition",
                    side_effect=fake_evaluate,
                ) as evaluate,
                patch("cot_faithfulness.text_mediation.runtime_metadata", return_value={}),
            ):
                first = run_text_mediation(
                    config,
                    object(),
                    object(),
                    screening_records=[screening_record()],
                )
                second = run_text_mediation(
                    config,
                    object(),
                    object(),
                    screening_records=[screening_record()],
                )

            self.assertEqual(first["run_counts"]["processed_this_run"], 2)
            self.assertEqual(second["run_counts"]["processed_this_run"], 0)
            self.assertEqual(second["run_counts"]["resumed_conditions"], 2)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(len((root / "mediation.jsonl").read_text().splitlines()), 2)
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
