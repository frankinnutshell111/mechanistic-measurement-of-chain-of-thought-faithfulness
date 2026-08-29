"""Targeted residual-stream capture without global hidden-state materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_loader import decoder_layers, model_input_device, verify_decoder_layers


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install torch before capturing activations") from exc
    return torch


def _storage_dtype(name: str) -> Any:
    torch = _torch()
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return supported[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation storage dtype: {name}") from exc


def _hidden_state_from_output(output: Any) -> Any:
    hidden_state = output[0] if isinstance(output, tuple) else output
    if not hasattr(hidden_state, "ndim") or hidden_state.ndim != 3:
        raise RuntimeError(
            "Decoder output must be a tensor or tuple beginning with [batch, sequence, hidden]"
        )
    return hidden_state


@dataclass(frozen=True)
class TeacherForcedReasoningInput:
    input_ids: tuple[int, ...]
    reasoning_positions: tuple[int, ...]


def build_teacher_forced_reasoning_input(
    prompt_input_ids: tuple[int, ...] | list[int],
    reasoning_token_ids: tuple[int, ...] | list[int],
) -> TeacherForcedReasoningInput:
    """Append exact reasoning IDs and return their absolute sequence positions."""

    prompt = tuple(int(token_id) for token_id in prompt_input_ids)
    reasoning = tuple(int(token_id) for token_id in reasoning_token_ids)
    if not prompt:
        raise ValueError("Prompt token IDs cannot be empty")
    if not reasoning:
        raise ValueError("Reasoning token IDs cannot be empty")
    start = len(prompt)
    return TeacherForcedReasoningInput(
        input_ids=prompt + reasoning,
        reasoning_positions=tuple(range(start, start + len(reasoning))),
    )


@dataclass(frozen=True)
class CapturedResidualStreams:
    layer_indices: tuple[int, ...]
    token_positions: tuple[int, ...]
    tensors: dict[int, Any]
    storage_dtype: str

    def block(
        self,
        layer_idx: int,
        block_start: int,
        block_end: int,
        *,
        batch_index: int = 0,
    ) -> Any:
        """Return one reasoning-relative [block_length, hidden_size] slice."""

        if layer_idx not in self.tensors:
            raise KeyError(f"Layer {layer_idx} was not captured")
        if not 0 <= block_start < block_end <= len(self.token_positions):
            raise ValueError("Block bounds are outside the captured reasoning positions")
        tensor = self.tensors[layer_idx]
        if not 0 <= batch_index < tensor.shape[0]:
            raise IndexError("batch_index is outside the captured activation batch")
        return tensor[batch_index, block_start:block_end]


@dataclass(frozen=True)
class MatchedResidualStreams:
    neutral: CapturedResidualStreams
    hinted: CapturedResidualStreams


class ResidualStreamCapture:
    """Context manager registering temporary read-only hooks on selected layers."""

    def __init__(
        self,
        model: Any,
        *,
        layer_indices: tuple[int, ...],
        token_positions: tuple[int, ...],
        storage_dtype: str = "bfloat16",
    ) -> None:
        if not layer_indices:
            raise ValueError("At least one layer must be selected")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("Layer indices must be unique")
        if not token_positions:
            raise ValueError("At least one token position must be selected")
        if any(position < 0 for position in token_positions):
            raise ValueError("Token positions must be non-negative")
        if tuple(sorted(set(token_positions))) != token_positions:
            raise ValueError("Token positions must be strictly increasing and unique")
        verify_decoder_layers(model, layer_indices)
        self.model = model
        self.layer_indices = layer_indices
        self.token_positions = token_positions
        self.storage_dtype = storage_dtype
        self.captured: dict[int, Any] = {}
        self._handles: list[Any] = []

    def _capture_hook(self, layer_idx: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            hidden_state = _hidden_state_from_output(output)
            if self.token_positions[-1] >= hidden_state.shape[1]:
                raise RuntimeError(
                    f"Capture position {self.token_positions[-1]} is outside sequence "
                    f"length {hidden_state.shape[1]}"
                )
            if layer_idx in self.captured:
                raise RuntimeError(f"Layer {layer_idx} produced more than one capture")
            self.captured[layer_idx] = (
                hidden_state[:, self.token_positions, :]
                .detach()
                .to(device="cpu", dtype=_storage_dtype(self.storage_dtype))
                .clone()
            )

        return hook

    def __enter__(self) -> ResidualStreamCapture:
        layers = decoder_layers(self.model)
        try:
            for layer_idx in self.layer_indices:
                handle = layers[layer_idx].register_forward_hook(self._capture_hook(layer_idx))
                self._handles.append(handle)
        except Exception:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc, traceback
        self._remove_hooks()
        if exc_type is None:
            missing = set(self.layer_indices) - set(self.captured)
            if missing:
                raise RuntimeError(f"Selected layers did not run during capture: {sorted(missing)}")
        return False

    def result(self) -> CapturedResidualStreams:
        missing = set(self.layer_indices) - set(self.captured)
        if missing:
            raise RuntimeError(f"Capture is incomplete for layers: {sorted(missing)}")
        return CapturedResidualStreams(
            layer_indices=self.layer_indices,
            token_positions=self.token_positions,
            tensors=dict(self.captured),
            storage_dtype=self.storage_dtype,
        )


def capture_residual_streams(
    model: Any,
    input_ids: Any,
    *,
    layer_indices: tuple[int, ...],
    token_positions: tuple[int, ...],
    attention_mask: Any | None = None,
    storage_dtype: str = "bfloat16",
    model_kwargs: dict[str, Any] | None = None,
) -> CapturedResidualStreams:
    """Capture requested positions at every selected layer in one forward pass."""

    torch = _torch()
    device = model_input_device(model)
    input_ids = input_ids.to(device=device)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device=device)
    kwargs = dict(model_kwargs or {})
    if kwargs.get("output_hidden_states"):
        raise ValueError("Targeted capture must not enable output_hidden_states")
    if getattr(getattr(model, "config", None), "output_hidden_states", False):
        raise ValueError("Disable model.config.output_hidden_states before targeted capture")
    kwargs.pop("output_hidden_states", None)
    kwargs["use_cache"] = False
    with ResidualStreamCapture(
        model,
        layer_indices=layer_indices,
        token_positions=token_positions,
        storage_dtype=storage_dtype,
    ) as capture:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    return capture.result()


def capture_matched_residual_streams(
    model: Any,
    neutral_input_ids: Any,
    hinted_input_ids: Any,
    *,
    layer_indices: tuple[int, ...],
    reasoning_positions: tuple[int, ...],
    neutral_attention_mask: Any | None = None,
    hinted_attention_mask: Any | None = None,
    storage_dtype: str = "bfloat16",
) -> MatchedResidualStreams:
    """Capture neutral and hinted prompts in one forward pass each."""

    torch = _torch()
    if tuple(neutral_input_ids.shape) != tuple(hinted_input_ids.shape):
        raise ValueError("Matched teacher-forced inputs must have identical shapes")
    if not reasoning_positions or reasoning_positions[-1] >= neutral_input_ids.shape[1]:
        raise ValueError("Reasoning positions are outside the matched teacher-forced inputs")
    if not torch.equal(
        neutral_input_ids[:, reasoning_positions],
        hinted_input_ids[:, reasoning_positions],
    ):
        raise ValueError(
            "Neutral and hinted captures must teacher-force identical visible reasoning tokens"
        )
    neutral = capture_residual_streams(
        model,
        neutral_input_ids,
        layer_indices=layer_indices,
        token_positions=reasoning_positions,
        attention_mask=neutral_attention_mask,
        storage_dtype=storage_dtype,
    )
    hinted = capture_residual_streams(
        model,
        hinted_input_ids,
        layer_indices=layer_indices,
        token_positions=reasoning_positions,
        attention_mask=hinted_attention_mask,
        storage_dtype=storage_dtype,
    )
    for layer_idx in layer_indices:
        if tuple(neutral.tensors[layer_idx].shape) != tuple(hinted.tensors[layer_idx].shape):
            raise RuntimeError(f"Matched activation shapes differ at layer {layer_idx}")
    return MatchedResidualStreams(neutral=neutral, hinted=hinted)
