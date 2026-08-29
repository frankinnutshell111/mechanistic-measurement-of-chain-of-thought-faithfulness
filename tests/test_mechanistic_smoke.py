import types
import unittest
from dataclasses import replace

from cot_faithfulness.answer_scoring import (
    build_scoring_context,
    score_answer_labels_batched,
)
from cot_faithfulness.config import MechanisticSmokeConfig
from cot_faithfulness.generation import resolve_end_think_id
from cot_faithfulness.mechanistic_smoke import (
    evaluate_mechanistic_smoke,
    select_smoke_record,
)
from cot_faithfulness.reasoning_segmentation import segment_reasoning

from .helpers import CharacterTokenizer, experiment_config

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - local offline Python may omit torch
    torch = None
    nn = None


if torch is not None:

    class ContextLayer(nn.Module):
        def forward(self, hidden_states):
            return (hidden_states + hidden_states[:, :1, :], None)

    class TinyMechanisticModel(nn.Module):
        def __init__(self, vocab_size=512, hidden_size=12):
            super().__init__()
            torch.manual_seed(7)
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([ContextLayer()])
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.config = types.SimpleNamespace(
                _commit_hash=None,
                output_hidden_states=False,
            )
            self.continuation = (2,)

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            hidden_states = self.embedding(input_ids)
            hidden_states = self.model.layers[0](hidden_states)[0]
            return types.SimpleNamespace(logits=self.lm_head(hidden_states))

        def generate(self, input_ids, attention_mask, **kwargs):
            del kwargs
            self.forward(input_ids, attention_mask, use_cache=True)
            continuation = torch.tensor(
                [self.continuation], dtype=torch.long, device=input_ids.device
            ).expand(input_ids.shape[0], -1)
            return torch.cat((input_ids, continuation), dim=1)


def _record(control_prompt, hinted_prompt, reasoning, stored_scores, hint_label):
    trace = {
        "reasoning_token_ids": list(reasoning),
        "eligible": True,
        "completed": True,
        "truncated": False,
        "repetitive": False,
    }
    control = {
        "prompt": {"input_ids": list(control_prompt)},
        "trace": trace,
        "answer_scores": stored_scores.to_dict(),
    }
    hinted = {
        "prompt": {"input_ids": list(hinted_prompt)},
        "trace": trace,
        "answer_scores": stored_scores.to_dict(),
    }
    return {
        "schema_version": "phase1.screening.v1",
        "status": "complete",
        "question_id": "toy-question",
        "hint_label": hint_label,
        "model_id": "Qwen/Qwen3-14B",
        "condition_eligible": {"metadata": True, "black_square": False},
        "prompt_pairs": {
            "metadata": {
                "aligned": True,
                "control": control,
                "hinted": hinted,
            }
        },
    }


class MechanisticSmokeSelectionTests(unittest.TestCase):
    def test_selects_first_eligible_record_or_requested_question(self):
        records = [
            {"question_id": "bad", "condition_eligible": {"metadata": False}},
            {"question_id": "good", "condition_eligible": {"metadata": True}},
        ]
        self.assertEqual(
            select_smoke_record(records, hint_condition="metadata", question_id=None)[
                "question_id"
            ],
            "good",
        )
        with self.assertRaisesRegex(ValueError, "not eligible"):
            select_smoke_record(records, hint_condition="metadata", question_id="bad")


@unittest.skipUnless(torch is not None, "torch is required for mechanistic smoke tests")
class MechanisticSmokeIntegrationTests(unittest.TestCase):
    def test_synthetic_smoke_computes_y00_y10_y11_and_null_gate(self):
        tokenizer = CharacterTokenizer()
        model = TinyMechanisticModel()
        base = experiment_config()
        config = replace(
            base,
            model=replace(base.model, layers=(0,), device="cpu", dtype="float32"),
            decoding=replace(base.decoding, max_reasoning_tokens=50),
            mechanistic=replace(base.mechanistic, activation_storage_dtype="float32"),
            mechanistic_smoke=MechanisticSmokeConfig(
                hint_condition="metadata",
                layer_idx=0,
                block_idx=0,
                null_batch_size=2,
            ),
        )
        control_prompt = (10, 11)
        hinted_prompt = (12, 11)
        reasoning = tuple(tokenizer.encode("First step. More reasoning."))
        segmentation = segment_reasoning(tokenizer, reasoning, max_blocks=10)
        block = segmentation.blocks[0]
        model.continuation = (*reasoning[block.end_token :], resolve_end_think_id(tokenizer))
        baseline_context = build_scoring_context(
            tokenizer,
            control_prompt,
            reasoning,
            resolve_end_think_id(tokenizer),
            config.decoding.answer_cue,
        )
        stored_scores = score_answer_labels_batched(
            model,
            tokenizer,
            (baseline_context,),
            labels=config.prompts.labels,
            label_prefix=config.decoding.answer_label_prefix,
        )[0]
        hint_label = next(
            label for label in config.prompts.labels if label != stored_scores.predicted_answer
        )
        result = evaluate_mechanistic_smoke(
            config,
            model,
            tokenizer,
            _record(control_prompt, hinted_prompt, reasoning, stored_scores, hint_label),
            run_metadata={"model_revision": None},
        )
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["null_validation"]["passed"])
        self.assertIn("Y00", result["baseline"])
        self.assertIn("Y10", result["structured"])
        self.assertIn("Y11", result["structured"])
        self.assertEqual(result["metrics"]["T"], result["metrics"]["D"] + result["metrics"]["C"])
        self.assertFalse(result["phase_scope"]["phase5_batch_experiment_enabled"])


if __name__ == "__main__":
    unittest.main()
