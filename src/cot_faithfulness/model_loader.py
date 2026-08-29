"""Qwen causal language-model loading with explicit experiment invariants."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any


def _torch_dtype(torch: Any, name: str) -> Any:
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return supported[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def decoder_layers(model: Any) -> Sequence[Any]:
    """Return Qwen decoder blocks after verifying the expected public path."""

    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Expected decoder blocks at model.model.layers")
    return layers


def verify_decoder_layers(model: Any, requested_layers: tuple[int, ...]) -> None:
    """Fail loudly if requested zero-based decoder indices are invalid."""

    layers = decoder_layers(model)
    num_layers = len(layers)
    invalid = [idx for idx in requested_layers if idx < 0 or idx >= num_layers]
    if invalid:
        raise RuntimeError(
            f"Requested layer indices {invalid} are outside model.model.layers (n={num_layers})"
        )


def load_model_and_tokenizer(config: ModelConfig) -> LoadedModel:
    """Load an unquantized model for inference and verify the decoder layout."""

    if config.quantization != "none":
        raise ValueError("The primary experiment does not support quantization")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError(
            "Install torch, transformers, and accelerate before loading the model"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=_torch_dtype(torch, config.dtype),
        device_map={"": config.device},
        attn_implementation=config.attention_implementation,
        low_cpu_mem_usage=config.low_cpu_mem_usage,
    )
    model.eval()
    model.requires_grad_(False)
    verify_decoder_layers(model, config.layers)
    return LoadedModel(model=model, tokenizer=tokenizer)


def model_input_device(model: Any) -> Any:
    """Return the single-device input placement used by this experiment."""

    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("Unable to determine model input device") from exc
