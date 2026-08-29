import types
import unittest

from cot_faithfulness.activation_capture import (
    ResidualStreamCapture,
    build_teacher_forced_reasoning_input,
    capture_matched_residual_streams,
    capture_residual_streams,
)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - local offline Python may omit torch
    torch = None
    nn = None


if torch is not None:

    class AddLayer(nn.Module):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount

        def forward(self, hidden_states):
            return (hidden_states + self.amount, "unchanged-cache")

    class TinyCaptureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 4)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.arange(128).reshape(32, 4) / 10)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([AddLayer(1.0), AddLayer(2.0), AddLayer(3.0)])
            self.forward_calls = 0

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            self.forward_calls += 1
            hidden_states = self.embedding(input_ids)
            for layer in self.model.layers:
                hidden_states = layer(hidden_states)[0]
            return types.SimpleNamespace(last_hidden_state=hidden_states)


@unittest.skipUnless(torch is not None, "torch is required for activation tests")
class ActivationCaptureTests(unittest.TestCase):
    def test_teacher_forced_positions_follow_prompt(self):
        value = build_teacher_forced_reasoning_input([10, 11], [20, 21, 22])
        self.assertEqual(value.input_ids, (10, 11, 20, 21, 22))
        self.assertEqual(value.reasoning_positions, (2, 3, 4))

    def test_selected_layers_are_captured_in_one_forward_and_stored_on_cpu(self):
        model = TinyCaptureModel()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = capture_residual_streams(
            model,
            input_ids,
            layer_indices=(0, 2),
            token_positions=(1, 3),
            storage_dtype="bfloat16",
        )
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(set(result.tensors), {0, 2})
        self.assertEqual(result.tensors[0].shape, (1, 2, 4))
        self.assertEqual(result.tensors[0].device.type, "cpu")
        self.assertEqual(result.tensors[0].dtype, torch.bfloat16)
        expected_layer_zero = model.embedding(input_ids)[:, (1, 3), :] + 1.0
        self.assertTrue(torch.allclose(result.tensors[0].float(), expected_layer_zero, atol=0.05))
        self.assertEqual(result.block(2, 0, 1).shape, (1, 4))
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_matched_capture_uses_one_forward_per_condition(self):
        model = TinyCaptureModel()
        result = capture_matched_residual_streams(
            model,
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 2, 3]]),
            layer_indices=(1,),
            reasoning_positions=(1, 2),
            storage_dtype="float32",
        )
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(result.neutral.tensors[1].shape, result.hinted.tensors[1].shape)

    def test_matched_capture_rejects_different_reasoning_tokens(self):
        model = TinyCaptureModel()
        with self.assertRaisesRegex(ValueError, "identical visible reasoning"):
            capture_matched_residual_streams(
                model,
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[4, 9, 3]]),
                layer_indices=(1,),
                reasoning_positions=(1, 2),
            )
        self.assertEqual(model.forward_calls, 0)

    def test_capture_hooks_are_removed_after_exception(self):
        model = TinyCaptureModel()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with ResidualStreamCapture(
                model,
                layer_indices=(0, 2),
                token_positions=(1,),
            ):
                raise RuntimeError("boom")
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_global_hidden_state_materialization_is_rejected(self):
        model = TinyCaptureModel()
        with self.assertRaisesRegex(ValueError, "must not enable"):
            capture_residual_streams(
                model,
                torch.tensor([[1, 2, 3]]),
                layer_indices=(0,),
                token_positions=(1,),
                model_kwargs={"output_hidden_states": True},
            )
        self.assertEqual(model.forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
