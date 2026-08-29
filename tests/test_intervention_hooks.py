import types
import unittest

from cot_faithfulness.config import MechanisticConfig
from cot_faithfulness.intervention_hooks import intervene_residual_stream
from cot_faithfulness.perturbations import build_perturbation_directions

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - local offline Python may omit torch
    torch = None
    nn = None


if torch is not None:

    class TupleIdentityLayer(nn.Module):
        def forward(self, hidden_states):
            return (hidden_states, "preserved-cache")

    class HookModel(nn.Module):
        def __init__(self, num_layers=3):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TupleIdentityLayer() for _ in range(num_layers)])

    class ToyGreedyModel(HookModel):
        def __init__(self):
            super().__init__(num_layers=1)
            self.embedding = nn.Embedding(8, 5)
            self.lm_head = nn.Linear(5, 8, bias=False)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.arange(40).reshape(8, 5) / 20)
                self.lm_head.weight.copy_(torch.arange(40).reshape(8, 5) / 30)

        def forward(self, input_ids):
            hidden_states = self.embedding(input_ids)
            hidden_states = self.model.layers[0](hidden_states)[0]
            return types.SimpleNamespace(logits=self.lm_head(hidden_states))

        def generate(self, input_ids, max_new_tokens):
            sequence = input_ids.clone()
            for _ in range(max_new_tokens):
                next_token = self(sequence).logits[:, -1].argmax(dim=-1, keepdim=True)
                sequence = torch.cat((sequence, next_token), dim=1)
            return sequence


def run_layer(model, layer_idx, hidden_states):
    return model.model.layers[layer_idx](hidden_states)


@unittest.skipUnless(torch is not None, "torch is required for hook tests")
class InterventionHookTests(unittest.TestCase):
    def test_only_target_layer_positions_and_batch_elements_are_modified(self):
        model = HookModel()
        hidden = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        original = hidden.clone()
        directions = torch.stack((torch.ones(2, 3), torch.full((2, 3), 2.0)))
        with intervene_residual_stream(
            model,
            layer_idx=1,
            token_positions=(1, 3),
            perturbation=directions,
            alpha=1.0,
        ) as intervention:
            self.assertFalse(model.model.layers[0]._forward_hooks)
            self.assertEqual(len(model.model.layers[1]._forward_hooks), 1)
            output, cache = run_layer(model, 1, hidden)
        self.assertEqual(cache, "preserved-cache")
        self.assertEqual(intervention.state.apply_count, 1)
        self.assertTrue(torch.equal(hidden, original))
        self.assertTrue(torch.equal(output[:, (0, 2, 4), :], original[:, (0, 2, 4), :]))
        self.assertTrue(torch.equal(output[0, (1, 3), :], original[0, (1, 3), :] + 1))
        self.assertTrue(torch.equal(output[1, (1, 3), :], original[1, (1, 3), :] + 2))
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_alpha_zero_preserves_outputs_exactly(self):
        model = HookModel()
        hidden = torch.randn(1, 4, 3)
        with intervene_residual_stream(
            model,
            layer_idx=0,
            token_positions=(1, 2),
            perturbation=torch.randn(2, 3),
            alpha=0.0,
        ):
            output = run_layer(model, 0, hidden)[0]
        self.assertTrue(torch.equal(output, hidden))

    def test_tensor_layer_output_is_supported(self):
        model = HookModel()
        model.model.layers[0] = nn.Identity()
        hidden = torch.zeros(1, 3, 2)
        with intervene_residual_stream(
            model,
            layer_idx=0,
            token_positions=(1,),
            perturbation=torch.ones(1, 2),
            alpha=1.0,
        ):
            output = run_layer(model, 0, hidden)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.equal(output[0, 1], torch.ones(2)))

    def test_structured_alpha_one_replaces_neutral_block(self):
        model = HookModel()
        neutral = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        hinted = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        directions = build_perturbation_directions(
            neutral,
            hinted,
            question_id="q1",
            hint_type="metadata",
            layer_idx=0,
            block_idx=0,
            base_seed=9,
            config=MechanisticConfig(num_random_directions=0),
        )
        with intervene_residual_stream(
            model,
            layer_idx=0,
            token_positions=(0, 1),
            perturbation=directions.values[0],
            alpha=1.0,
        ):
            output = run_layer(model, 0, neutral.unsqueeze(0))[0]
        self.assertTrue(torch.allclose(output[0], hinted, rtol=0, atol=1e-6))

    def test_batched_and_sequential_interventions_agree(self):
        model = HookModel()
        hidden = torch.randn(2, 4, 3)
        directions = torch.randn(2, 2, 3)
        with intervene_residual_stream(
            model,
            layer_idx=2,
            token_positions=(1, 2),
            perturbation=directions,
            alpha=0.5,
        ):
            batched = run_layer(model, 2, hidden)[0]
        sequential = []
        for batch_index in range(2):
            with intervene_residual_stream(
                model,
                layer_idx=2,
                token_positions=(1, 2),
                perturbation=directions[batch_index],
                alpha=0.5,
            ):
                sequential.append(run_layer(model, 2, hidden[batch_index : batch_index + 1])[0])
        self.assertTrue(torch.allclose(batched, torch.cat(sequential), rtol=0, atol=0))

    def test_prefill_is_patched_once_and_cached_step_is_not_patched(self):
        model = HookModel()
        direction = torch.ones(2, 3)
        with intervene_residual_stream(
            model,
            layer_idx=0,
            token_positions=(2, 3),
            perturbation=direction,
            alpha=1.0,
        ) as intervention:
            prefill = run_layer(model, 0, torch.zeros(1, 5, 3))[0]
            cached_step = run_layer(model, 0, torch.zeros(1, 1, 3))[0]
        self.assertEqual(intervention.state.apply_count, 1)
        self.assertTrue(torch.equal(prefill[0, (2, 3)], torch.ones(2, 3)))
        self.assertTrue(torch.equal(cached_step, torch.zeros_like(cached_step)))

    def test_null_intervention_reproduces_greedy_continuation(self):
        model = ToyGreedyModel()
        prompt = torch.tensor([[1, 2, 3]])
        baseline = model.generate(prompt, max_new_tokens=4)
        with intervene_residual_stream(
            model,
            layer_idx=0,
            token_positions=(1, 2),
            perturbation=torch.randn(2, 5),
            alpha=0.0,
        ):
            null_intervention = model.generate(prompt, max_new_tokens=4)
        self.assertTrue(torch.equal(null_intervention, baseline))

    def test_hook_is_removed_after_exception(self):
        model = HookModel()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with intervene_residual_stream(
                model,
                layer_idx=1,
                token_positions=(1,),
                perturbation=torch.ones(1, 3),
                alpha=1.0,
            ):
                raise RuntimeError("boom")
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_hook_is_removed_when_perturbation_application_raises(self):
        model = HookModel()
        with self.assertRaisesRegex(ValueError, "shapes do not match"):
            with intervene_residual_stream(
                model,
                layer_idx=1,
                token_positions=(1,),
                perturbation=torch.ones(2, 3),
                alpha=1.0,
            ):
                run_layer(model, 1, torch.zeros(1, 3, 3))
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))


if __name__ == "__main__":
    unittest.main()
