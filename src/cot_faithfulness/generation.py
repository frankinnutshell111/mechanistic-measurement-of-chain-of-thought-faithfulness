"""Deterministic reasoning generation with exact token preservation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import DecodingConfig
from .model_loader import model_input_device


@dataclass(frozen=True)
class ReasoningTrace:
    token_ids: tuple[int, ...]
    text: str
    completed: bool
    truncated: bool
    repetitive: bool
    stop_reason: str
    generated_token_count: int

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    @property
    def eligible(self) -> bool:
        return self.completed and not self.truncated and not self.repetitive and not self.empty

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_token_ids": list(self.token_ids),
            "reasoning_text": self.text,
            "reasoning_token_count": len(self.token_ids),
            "generated_token_count": self.generated_token_count,
            "completed": self.completed,
            "truncated": self.truncated,
            "repetitive": self.repetitive,
            "empty": self.empty,
            "eligible": self.eligible,
            "stop_reason": self.stop_reason,
        }


def resolve_end_think_id(tokenizer: Any) -> int:
    """Resolve and verify Qwen's closing thinking token without hard-coding it."""

    token = "</think>"
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError("Tokenizer does not expose a valid </think> token ID")
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if list(encoded) != [int(token_id)]:
        raise RuntimeError(
            "</think> must be represented by the verified token returned by convert_tokens_to_ids"
        )
    return int(token_id)


def _strip_prefix(tokens: Sequence[int], prefix: Sequence[int]) -> tuple[int, ...]:
    if prefix and list(tokens[: len(prefix)]) == list(prefix):
        return tuple(int(token) for token in tokens[len(prefix) :])
    return tuple(int(token) for token in tokens)


def strip_optional_open_think(tokenizer: Any, token_ids: Sequence[int]) -> tuple[int, ...]:
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    if hasattr(opening, "tolist"):
        opening = opening.tolist()
    return _strip_prefix(token_ids, opening)


def detect_repetitive_loop(
    token_ids: Sequence[int],
    *,
    min_repeats: int = 4,
    min_cycle_tokens: int = 2,
    max_cycle_tokens: int = 64,
) -> bool:
    """Detect an exact repeated suffix cycle, a conservative loop signal."""

    if min_repeats < 2 or min_cycle_tokens < 1:
        raise ValueError("Repetition thresholds are invalid")
    total = len(token_ids)
    max_cycle = min(max_cycle_tokens, total // min_repeats)
    for cycle_length in range(min_cycle_tokens, max_cycle + 1):
        suffix_length = cycle_length * min_repeats
        suffix = list(token_ids[total - suffix_length :])
        cycle = suffix[:cycle_length]
        if cycle * min_repeats == suffix:
            return True
    return False


def trace_from_generated_ids(
    tokenizer: Any,
    generated_ids: Sequence[int],
    config: DecodingConfig,
) -> ReasoningTrace:
    """Convert raw generated IDs into the exact inside-<think> reasoning trace."""

    end_think_id = resolve_end_think_id(tokenizer)
    raw = tuple(int(token) for token in generated_ids)
    completed = end_think_id in raw
    if completed:
        close_index = raw.index(end_think_id)
        inside = raw[:close_index]
        stop_reason = "end_think"
    else:
        inside = raw
        stop_reason = (
            "token_cap" if len(raw) >= config.max_reasoning_tokens else "eos_before_end_think"
        )
    inside = strip_optional_open_think(tokenizer, inside)
    text = tokenizer.decode(
        list(inside),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    repetitive = detect_repetitive_loop(
        inside,
        min_repeats=config.repetition_min_repeats,
        min_cycle_tokens=config.repetition_min_cycle_tokens,
        max_cycle_tokens=config.repetition_max_cycle_tokens,
    )
    return ReasoningTrace(
        token_ids=inside,
        text=text,
        completed=completed,
        truncated=not completed and len(raw) >= config.max_reasoning_tokens,
        repetitive=repetitive,
        stop_reason=stop_reason,
        generated_token_count=len(raw),
    )


def all_stop_token_ids(tokenizer: Any, end_think_id: int) -> list[int]:
    """Return the verified thinking terminator followed by tokenizer EOS IDs."""

    result = [end_think_id]
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    for token_id in eos_ids:
        if token_id is not None and int(token_id) not in result:
            result.append(int(token_id))
    return result


def generate_reasoning(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Sequence[int],
    config: DecodingConfig,
) -> ReasoningTrace:
    """Generate one greedy reasoning trace and stop when </think> is emitted."""

    if config.do_sample:
        raise ValueError("Primary reasoning generation must be deterministic")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install torch before generating reasoning") from exc

    end_think_id = resolve_end_think_id(tokenizer)
    device = model_input_device(model)
    input_ids = torch.tensor([list(prompt_input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=config.max_reasoning_tokens,
            eos_token_id=all_stop_token_ids(tokenizer, end_think_id),
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    generated = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
    return trace_from_generated_ids(tokenizer, generated, config)
