"""Conditional sequence log-probability scoring for answer labels A-D."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .model_loader import model_input_device


@dataclass(frozen=True)
class AnswerScores:
    predicted_answer: str
    label_logprobs: Mapping[str, float]
    label_token_ids: Mapping[str, tuple[int, ...]]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnswerScores:
        """Restore scores stored in a Phase 1 JSONL record."""

        try:
            predicted_answer = str(value["predicted_answer"])
            label_logprobs = {
                str(label): float(logprob) for label, logprob in value["label_logprobs"].items()
            }
            label_token_ids = {
                str(label): tuple(int(token_id) for token_id in token_ids)
                for label, token_ids in value["label_token_ids"].items()
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid serialized answer scores") from exc
        if predicted_answer not in label_logprobs:
            raise ValueError("Predicted answer is missing from label log-probabilities")
        if set(label_logprobs) != set(label_token_ids):
            raise ValueError("Answer-score labels and token-ID labels do not match")
        return cls(predicted_answer, label_logprobs, label_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_answer": self.predicted_answer,
            "label_logprobs": dict(self.label_logprobs),
            "label_token_ids": {
                label: list(token_ids) for label, token_ids in self.label_token_ids.items()
            },
        }


def encode_answer_labels(
    tokenizer: Any,
    labels: Sequence[str] = ("A", "B", "C", "D"),
    *,
    prefix: str = " ",
) -> dict[str, tuple[int, ...]]:
    """Encode complete candidate strings, retaining multi-token labels when needed."""

    result: dict[str, tuple[int, ...]] = {}
    for label in labels:
        encoded = tokenizer.encode(f"{prefix}{label}", add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        token_ids = tuple(int(token_id) for token_id in encoded)
        if not token_ids:
            raise RuntimeError(f"Answer label {label!r} tokenized to an empty sequence")
        result[label] = token_ids
    return result


def build_scoring_context(
    tokenizer: Any,
    prompt_input_ids: Sequence[int],
    reasoning_token_ids: Sequence[int],
    end_think_id: int,
    answer_cue: str,
) -> tuple[int, ...]:
    """Append the exact reasoning IDs, closing tag, and fixed answer cue."""

    cue_ids = tokenizer.encode(answer_cue, add_special_tokens=False)
    if hasattr(cue_ids, "tolist"):
        cue_ids = cue_ids.tolist()
    if not cue_ids:
        raise RuntimeError("The answer cue tokenized to an empty sequence")
    return tuple(
        [*map(int, prompt_input_ids), *map(int, reasoning_token_ids), int(end_think_id)]
        + [int(token_id) for token_id in cue_ids]
    )


def _single_token_scores(
    model: Any,
    context_ids: Sequence[int],
    candidates: Mapping[str, tuple[int, ...]],
) -> dict[str, float]:
    import torch

    device = model_input_device(model)
    input_ids = torch.tensor([list(context_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    final_logprobs = torch.log_softmax(logits[0, -1].float(), dim=-1)
    return {
        label: float(final_logprobs[token_ids[0]].detach().cpu())
        for label, token_ids in candidates.items()
    }


def _sequence_scores(
    model: Any,
    tokenizer: Any,
    context_ids: Sequence[int],
    candidates: Mapping[str, tuple[int, ...]],
    *,
    batch_size: int,
) -> dict[str, float]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = model_input_device(model)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0]
    if pad_token_id is None:
        raise RuntimeError("Tokenizer needs a pad or EOS token for batched scoring")

    items = list(candidates.items())
    scores: dict[str, float] = {}
    context_length = len(context_ids)
    if context_length == 0:
        raise ValueError("Scoring context must be nonempty")

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        sequences = [list(context_ids) + list(candidate) for _, candidate in chunk]
        max_length = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), max_length),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, sequence in enumerate(sequences):
            input_ids[row_idx, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )
            attention_mask[row_idx, : len(sequence)] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

        for row_idx, (label, candidate) in enumerate(chunk):
            positions = torch.arange(
                context_length - 1,
                context_length + len(candidate) - 1,
                device=device,
            )
            target_ids = torch.tensor(candidate, dtype=torch.long, device=device)
            token_logprobs = torch.log_softmax(logits[row_idx, positions].float(), dim=-1).gather(
                1, target_ids[:, None]
            )
            scores[label] = float(token_logprobs.sum().detach().cpu())
    return scores


def score_answer_labels(
    model: Any,
    tokenizer: Any,
    context_ids: Sequence[int],
    *,
    labels: Sequence[str] = ("A", "B", "C", "D"),
    label_prefix: str = " ",
    sequence_batch_size: int = 1,
) -> AnswerScores:
    """Score A-D exactly, using a fast path only when all labels are one token."""

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install torch before scoring answers") from exc

    candidates = encode_answer_labels(tokenizer, labels, prefix=label_prefix)
    if all(len(token_ids) == 1 for token_ids in candidates.values()):
        scores = _single_token_scores(model, context_ids, candidates)
    else:
        scores = _sequence_scores(
            model,
            tokenizer,
            context_ids,
            candidates,
            batch_size=sequence_batch_size,
        )
    predicted = max(labels, key=lambda label: scores[label])
    return AnswerScores(
        predicted_answer=predicted,
        label_logprobs=scores,
        label_token_ids=candidates,
    )


def score_answer_labels_batched(
    model: Any,
    tokenizer: Any,
    contexts: Sequence[Sequence[int]],
    *,
    labels: Sequence[str] = ("A", "B", "C", "D"),
    label_prefix: str = " ",
) -> tuple[AnswerScores, ...]:
    """Score one or more contexts in exactly one model forward pass.

    This is the scorer used by activation interventions.  A single forward is
    important there: the temporary hook must see the entire scoring batch once,
    including every candidate row when an answer label has multiple tokens.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install torch before scoring answers") from exc

    normalized_contexts = tuple(tuple(int(token_id) for token_id in value) for value in contexts)
    if not normalized_contexts:
        raise ValueError("At least one scoring context is required")
    if any(not value for value in normalized_contexts):
        raise ValueError("Scoring contexts must be nonempty")
    candidates = encode_answer_labels(tokenizer, labels, prefix=label_prefix)
    single_token = all(len(token_ids) == 1 for token_ids in candidates.values())

    if single_token:
        rows = [list(context) for context in normalized_contexts]
        row_metadata = [(context_idx, None) for context_idx in range(len(rows))]
    else:
        rows = []
        row_metadata = []
        for context_idx, context in enumerate(normalized_contexts):
            for label in labels:
                rows.append([*context, *candidates[label]])
                row_metadata.append((context_idx, label))

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0]
    if pad_token_id is None:
        raise RuntimeError("Tokenizer needs a pad or EOS token for batched scoring")

    device = model_input_device(model)
    max_length = max(map(len, rows))
    input_ids = torch.full(
        (len(rows), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, row in enumerate(rows):
        input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[row_idx, : len(row)] = 1

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    scores_by_context: list[dict[str, float]] = [dict() for _ in normalized_contexts]
    if single_token:
        for row_idx, (context_idx, _) in enumerate(row_metadata):
            position = len(normalized_contexts[context_idx]) - 1
            logprobs = torch.log_softmax(logits[row_idx, position].float(), dim=-1)
            scores_by_context[context_idx] = {
                label: float(logprobs[token_ids[0]].detach().cpu())
                for label, token_ids in candidates.items()
            }
    else:
        for row_idx, (context_idx, label) in enumerate(row_metadata):
            assert label is not None
            candidate = candidates[label]
            context_length = len(normalized_contexts[context_idx])
            positions = torch.arange(
                context_length - 1,
                context_length + len(candidate) - 1,
                device=device,
            )
            target_ids = torch.tensor(candidate, dtype=torch.long, device=device)
            token_logprobs = torch.log_softmax(logits[row_idx, positions].float(), dim=-1).gather(
                1, target_ids[:, None]
            )
            scores_by_context[context_idx][label] = float(token_logprobs.sum().detach().cpu())

    return tuple(
        AnswerScores(
            predicted_answer=max(labels, key=lambda label: scores[label]),
            label_logprobs=scores,
            label_token_ids=candidates,
        )
        for scores in scores_by_context
    )
