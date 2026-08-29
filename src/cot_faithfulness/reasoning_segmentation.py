"""Exact, non-truncating segmentation of stored visible-reasoning token IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_UNIT_END = re.compile(r"(?:[.!?](?:[\"')\]]*)|;|\n+)(?=\s|$)")
_TRANSITION = re.compile(
    r"(?im)(?:(?<=\n)|^)[ \t]*(?:first|second|third|next|then|finally|therefore|thus|however)\b"
)


@dataclass(frozen=True)
class ReasoningBlock:
    index: int
    start_token: int
    end_token: int
    text: str
    has_future_reasoning: bool

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
            "text": self.text,
            "has_future_reasoning": self.has_future_reasoning,
        }


@dataclass(frozen=True)
class ReasoningSegmentation:
    blocks: tuple[ReasoningBlock, ...]
    method: str
    exact_mapping_verified: bool
    fallback_reason: str | None
    original_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "exact_mapping_verified": self.exact_mapping_verified,
            "fallback_reason": self.fallback_reason,
            "original_token_count": self.original_token_count,
            "block_count": len(self.blocks),
            "blocks": [block.to_dict() for block in self.blocks],
        }


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    values = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(token_id) for token_id in values)


def _decode(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _balanced_token_spans(token_count: int, block_count: int) -> tuple[tuple[int, int], ...]:
    base, remainder = divmod(token_count, block_count)
    spans = []
    start = 0
    for index in range(block_count):
        length = base + int(index < remainder)
        spans.append((start, start + length))
        start += length
    return tuple(spans)


def _merge_units(
    units: tuple[tuple[int, int], ...],
    max_blocks: int,
) -> tuple[tuple[int, int], ...]:
    """Merge adjacent units into approximately token-balanced contiguous blocks."""

    if len(units) <= max_blocks:
        return units
    result: list[tuple[int, int]] = []
    unit_index = 0
    while len(result) < max_blocks - 1:
        groups_left = max_blocks - len(result)
        units_left = len(units) - unit_index
        tokens_left = units[-1][1] - units[unit_index][0]
        target = tokens_left / groups_left
        start = units[unit_index][0]
        end = units[unit_index][1]
        unit_index += 1
        while unit_index < len(units) - (groups_left - 1):
            current = end - start
            next_end = units[unit_index][1]
            if current >= target or abs(current - target) <= abs(next_end - start - target):
                break
            end = next_end
            unit_index += 1
        result.append((start, end))
    result.append((units[unit_index][0], units[-1][1]))
    return tuple(result)


def _build_result(
    tokenizer: Any,
    token_ids: tuple[int, ...],
    spans: tuple[tuple[int, int], ...],
    *,
    method: str,
    exact_mapping_verified: bool,
    fallback_reason: str | None,
) -> ReasoningSegmentation:
    blocks = tuple(
        ReasoningBlock(
            index=index,
            start_token=start,
            end_token=end,
            text=_decode(tokenizer, token_ids[start:end]),
            has_future_reasoning=end < len(token_ids),
        )
        for index, (start, end) in enumerate(spans)
    )
    if not blocks or blocks[0].start_token != 0 or blocks[-1].end_token != len(token_ids):
        raise RuntimeError("Reasoning segmentation did not cover the full token sequence")
    if any(left.end_token != right.start_token for left, right in zip(blocks, blocks[1:])):
        raise RuntimeError("Reasoning segmentation contains a gap or overlap")
    return ReasoningSegmentation(
        blocks=blocks,
        method=method,
        exact_mapping_verified=exact_mapping_verified,
        fallback_reason=fallback_reason,
        original_token_count=len(token_ids),
    )


def segment_reasoning(
    tokenizer: Any,
    reasoning_token_ids: tuple[int, ...] | list[int],
    *,
    max_blocks: int = 10,
) -> ReasoningSegmentation:
    """Segment complete expression units, retaining exact original token spans.

    Text boundaries are accepted only when a cumulative decoded token prefix
    retokenizes to precisely the same original IDs.  If the full reasoning fails
    that round trip, deterministic token blocks are used instead.
    """

    token_ids = tuple(int(token_id) for token_id in reasoning_token_ids)
    if not token_ids:
        raise ValueError("Reasoning token IDs cannot be empty")
    if max_blocks <= 0:
        raise ValueError("max_blocks must be positive")
    text = _decode(tokenizer, token_ids)
    if _encode(tokenizer, text) != token_ids:
        spans = _balanced_token_spans(len(token_ids), min(max_blocks, len(token_ids)))
        return _build_result(
            tokenizer,
            token_ids,
            spans,
            method="deterministic_token_fallback",
            exact_mapping_verified=False,
            fallback_reason="full_reasoning_round_trip_mismatch",
        )

    candidate_char_boundaries = {match.end() for match in _UNIT_END.finditer(text)}
    candidate_char_boundaries.update(
        match.start() for match in _TRANSITION.finditer(text) if match.start() > 0
    )
    candidate_char_boundaries.discard(len(text))
    token_boundaries: list[int] = []
    for token_end in range(1, len(token_ids)):
        prefix_text = _decode(tokenizer, token_ids[:token_end])
        if len(prefix_text) not in candidate_char_boundaries:
            continue
        if (
            text.startswith(prefix_text)
            and _encode(tokenizer, prefix_text) == token_ids[:token_end]
        ):
            token_boundaries.append(token_end)
    boundaries = (0, *sorted(set(token_boundaries)), len(token_ids))
    units_list: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start >= end:
            continue
        if not _decode(tokenizer, token_ids[start:end]).strip() and units_list:
            previous_start, _ = units_list[-1]
            units_list[-1] = (previous_start, end)
        else:
            units_list.append((start, end))
    units = tuple(units_list)
    spans = _merge_units(units, max_blocks)
    return _build_result(
        tokenizer,
        token_ids,
        spans,
        method="expression_units",
        exact_mapping_verified=True,
        fallback_reason=None,
    )
