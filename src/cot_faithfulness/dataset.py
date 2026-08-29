"""OpenBookQA loading and strict normalization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import DatasetConfig


class DatasetFormatError(ValueError):
    """Raised when an OpenBookQA row does not match the expected schema."""


@dataclass(frozen=True)
class OpenBookQAExample:
    question_id: str
    question_stem: str
    choices: Mapping[str, str]
    answer_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_stem": self.question_stem,
            "choices": dict(self.choices),
            "answer_key": self.answer_key,
        }


def normalize_openbookqa_row(row: Mapping[str, Any]) -> OpenBookQAExample:
    """Normalize one Hugging Face OpenBookQA row and validate A-D ordering."""

    try:
        raw_choices = row["choices"]
        labels = list(raw_choices["label"])
        texts = list(raw_choices["text"])
        question_id = str(row["id"])
        question_stem = str(row["question_stem"]).strip()
        answer_key = str(row["answerKey"]).strip().upper()
    except (KeyError, TypeError) as exc:
        raise DatasetFormatError(f"Malformed OpenBookQA row: {row!r}") from exc

    if labels != ["A", "B", "C", "D"]:
        raise DatasetFormatError(
            f"Expected ordered labels A-D for {question_id}, received {labels!r}"
        )
    if len(texts) != 4 or any(not str(text).strip() for text in texts):
        raise DatasetFormatError(f"Expected four nonempty choices for {question_id}")
    if answer_key not in labels:
        raise DatasetFormatError(f"Invalid answerKey {answer_key!r} for {question_id}")
    if not question_id or not question_stem:
        raise DatasetFormatError("Question ID and stem must be nonempty")

    return OpenBookQAExample(
        question_id=question_id,
        question_stem=question_stem,
        choices={label: str(text).strip() for label, text in zip(labels, texts, strict=True)},
        answer_key=answer_key,
    )


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[OpenBookQAExample]:
    for row in rows:
        yield normalize_openbookqa_row(row)


def load_openbookqa(config: DatasetConfig) -> Sequence[Mapping[str, Any]]:
    """Load the configured OpenBookQA split from the Hugging Face Hub."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install the 'datasets' package to load OpenBookQA") from exc

    return load_dataset(
        config.dataset_id,
        config.subset,
        split=config.split,
        revision=config.revision,
    )
