"""Safe, temporary residual-stream intervention hooks for Qwen decoder blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .model_loader import decoder_layers, verify_decoder_layers
from .perturbations import apply_activation_perturbation


def _hidden_state_from_output(output: Any) -> Any:
    hidden_state = output[0] if isinstance(output, tuple) else output
    if not hasattr(hidden_state, "ndim") or hidden_state.ndim != 3:
        raise RuntimeError(
            "Decoder output must be a tensor or tuple beginning with [batch, sequence, hidden]"
        )
    return hidden_state


def _replace_hidden_state(output: Any, hidden_state: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden_state, *output[1:])
    return hidden_state


@dataclass
class InterventionState:
    apply_count: int = 0
    skipped_short_sequence_calls: int = 0


class ResidualStreamIntervention:
    """Context manager applying one batch-aware perturbation at one layer."""

    def __init__(
        self,
        model: Any,
        *,
        layer_idx: int,
        token_positions: tuple[int, ...],
        perturbation: Any,
        alpha: float,
        operation: str = "additive",
        prefill_only: bool = True,
        require_applied: bool = True,
    ) -> None:
        if not token_positions:
            raise ValueError("At least one intervention token position is required")
        if tuple(sorted(set(token_positions))) != token_positions or token_positions[0] < 0:
            raise ValueError(
                "Intervention token positions must be increasing, unique, and non-negative"
            )
        if not math.isfinite(float(alpha)):
            raise ValueError("alpha must be finite")
        if operation not in {"additive", "replacement"}:
            raise ValueError("operation must be additive or replacement")
        verify_decoder_layers(model, (layer_idx,))
        self.model = model
        self.layer_idx = layer_idx
        self.token_positions = token_positions
        self.perturbation = perturbation
        self.alpha = float(alpha)
        self.operation = operation
        self.prefill_only = prefill_only
        self.require_applied = require_applied
        self.state = InterventionState()
        self._handle: Any | None = None

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        if self.prefill_only and self.state.apply_count:
            return output
        hidden_state = _hidden_state_from_output(output)
        if self.token_positions[-1] >= hidden_state.shape[1]:
            self.state.skipped_short_sequence_calls += 1
            return output
        baseline_block = hidden_state[:, self.token_positions, :]
        perturbed_block = apply_activation_perturbation(
            baseline_block,
            self.perturbation,
            alpha=self.alpha,
            operation=self.operation,
        )
        modified_hidden_state = hidden_state.clone()
        modified_hidden_state[:, self.token_positions, :] = perturbed_block
        self.state.apply_count += 1
        return _replace_hidden_state(output, modified_hidden_state)

    def __enter__(self) -> ResidualStreamIntervention:
        if self._handle is not None:
            raise RuntimeError("An intervention context cannot be entered twice")
        layer = decoder_layers(self.model)[self.layer_idx]
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def _remove_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc, traceback
        self._remove_hook()
        if exc_type is None and self.require_applied and self.state.apply_count == 0:
            raise RuntimeError(
                "Intervention never reached its source positions; check the prefill input"
            )
        return False


def intervene_residual_stream(
    model: Any,
    *,
    layer_idx: int,
    token_positions: tuple[int, ...],
    perturbation: Any,
    alpha: float,
    operation: str = "additive",
    prefill_only: bool = True,
    require_applied: bool = True,
) -> ResidualStreamIntervention:
    """Build a context manager for one temporary residual-stream intervention."""

    return ResidualStreamIntervention(
        model,
        layer_idx=layer_idx,
        token_positions=token_positions,
        perturbation=perturbation,
        alpha=alpha,
        operation=operation,
        prefill_only=prefill_only,
        require_applied=require_applied,
    )
