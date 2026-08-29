import types
import unittest

from cot_faithfulness.answer_scoring import (
    build_scoring_context,
    encode_answer_labels,
    score_answer_labels,
    score_answer_labels_batched,
)

from .helpers import CharacterTokenizer

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - local offline Python may omit torch
    torch = None
    nn = None


if torch is not None:

    class DeterministicCausalModel(nn.Module):
        def __init__(self, vocab_size=400):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.vocab_size = vocab_size

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            vocabulary = torch.arange(self.vocab_size, device=input_ids.device).float()
            centers = (input_ids + 17).remainder(self.vocab_size).float().unsqueeze(-1)
            logits = -((vocabulary - centers) ** 2) / 100.0 + self.anchor
            return types.SimpleNamespace(logits=logits)


class AnswerScoringTests(unittest.TestCase):
    def test_retains_multi_token_candidate_sequences(self):
        tokenizer = CharacterTokenizer({" A": [20], " B": [21, 22], " C": [23], " D": [24]})
        candidates = encode_answer_labels(tokenizer)
        self.assertEqual(candidates["A"], (20,))
        self.assertEqual(candidates["B"], (21, 22))

    def test_scoring_context_uses_original_reasoning_ids(self):
        tokenizer = CharacterTokenizer()
        context = build_scoring_context(
            tokenizer,
            prompt_input_ids=[10, 11],
            reasoning_token_ids=[50, 51],
            end_think_id=2,
            answer_cue="Answer:",
        )
        self.assertEqual(context[:5], (10, 11, 50, 51, 2))
        self.assertEqual(context[5:], tuple(tokenizer.encode("Answer:")))


@unittest.skipUnless(torch is not None, "torch is required for scoring tests")
class BatchedAnswerScoringTests(unittest.TestCase):
    def test_single_token_batched_matches_individual_scoring(self):
        tokenizer = CharacterTokenizer(
            {" A": [20], " B": [21], " C": [22], " D": [23]}
        )
        model = DeterministicCausalModel()
        contexts = ((5, 6, 7), (8, 9))
        batched = score_answer_labels_batched(model, tokenizer, contexts)
        individual = tuple(
            score_answer_labels(model, tokenizer, context) for context in contexts
        )
        self.assertEqual(
            [score.predicted_answer for score in batched],
            [score.predicted_answer for score in individual],
        )
        for batched_score, individual_score in zip(batched, individual):
            for label in ("A", "B", "C", "D"):
                self.assertAlmostEqual(
                    batched_score.label_logprobs[label],
                    individual_score.label_logprobs[label],
                    places=6,
                )

    def test_multi_token_batched_matches_individual_scoring(self):
        tokenizer = CharacterTokenizer(
            {" A": [20, 30], " B": [21], " C": [22, 32], " D": [23]}
        )
        model = DeterministicCausalModel()
        contexts = ((5, 6, 7), (8, 9))
        batched = score_answer_labels_batched(model, tokenizer, contexts)
        individual = tuple(
            score_answer_labels(model, tokenizer, context, sequence_batch_size=4)
            for context in contexts
        )
        for batched_score, individual_score in zip(batched, individual):
            for label in ("A", "B", "C", "D"):
                self.assertAlmostEqual(
                    batched_score.label_logprobs[label],
                    individual_score.label_logprobs[label],
                    places=6,
                )


if __name__ == "__main__":
    unittest.main()
