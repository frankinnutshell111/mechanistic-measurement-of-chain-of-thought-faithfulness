"""Phase 1 behavioral screening pipeline."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_scoring import AnswerScores, build_scoring_context, score_answer_labels
from .checkpoint import JsonlCheckpoint
from .config import ExperimentConfig
from .dataset import OpenBookQAExample, load_openbookqa, normalize_rows
from .generation import ReasoningTrace, generate_reasoning, resolve_end_think_id
from .prompts import (
    Prompt,
    PromptAlignmentError,
    PromptPair,
    build_black_square_pair,
    build_clean_prompt,
    build_metadata_pair,
)
from .utils import atomic_write_json, runtime_metadata, stable_seed

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluatedPrompt:
    prompt: Prompt
    trace: ReasoningTrace
    answer_scores: AnswerScores
    scoring_context_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.to_dict(),
            "trace": self.trace.to_dict(),
            "answer_scores": self.answer_scores.to_dict(),
            "scoring_context_token_count": self.scoring_context_token_count,
        }


def choose_hint_label(
    clean_answer: str,
    labels: tuple[str, ...],
    *,
    base_seed: int,
    question_id: str,
) -> tuple[str, int]:
    alternatives = [label for label in labels if label != clean_answer]
    if len(alternatives) != len(labels) - 1:
        raise ValueError(f"Invalid clean answer: {clean_answer}")
    seed = stable_seed(base_seed, "hint", question_id)
    return random.Random(seed).choice(alternatives), seed


def evaluate_prompt(
    model: Any,
    tokenizer: Any,
    prompt: Prompt,
    config: ExperimentConfig,
) -> EvaluatedPrompt:
    trace = generate_reasoning(model, tokenizer, prompt.input_ids, config.decoding)
    end_think_id = resolve_end_think_id(tokenizer)
    context_ids = build_scoring_context(
        tokenizer,
        prompt.input_ids,
        trace.token_ids,
        end_think_id,
        config.decoding.answer_cue,
    )
    answer_scores = score_answer_labels(
        model,
        tokenizer,
        context_ids,
        labels=config.prompts.labels,
        label_prefix=config.decoding.answer_label_prefix,
        sequence_batch_size=config.decoding.answer_scoring_batch_size,
    )
    return EvaluatedPrompt(
        prompt=prompt,
        trace=trace,
        answer_scores=answer_scores,
        scoring_context_token_count=len(context_ids),
    )


def _trace_reasons(name: str, trace: ReasoningTrace) -> list[str]:
    reasons: list[str] = []
    if not trace.completed:
        reasons.append(f"{name}:missing_end_think")
    if trace.truncated:
        reasons.append(f"{name}:token_cap")
    if trace.repetitive:
        reasons.append(f"{name}:repetitive")
    if trace.empty:
        reasons.append(f"{name}:empty_reasoning")
    return reasons


def classify_screening(
    *,
    clean: EvaluatedPrompt,
    metadata_control: EvaluatedPrompt,
    metadata_hinted: EvaluatedPrompt,
    black_square_control: EvaluatedPrompt,
    black_square_hinted: EvaluatedPrompt,
    hint_label: str,
    metadata_aligned: bool,
    black_square_aligned: bool,
) -> dict[str, Any]:
    named = {
        "clean": clean,
        "metadata_control": metadata_control,
        "metadata_hinted": metadata_hinted,
        "black_square_control": black_square_control,
        "black_square_hinted": black_square_hinted,
    }
    reasons: list[str] = []
    for name, evaluated in named.items():
        reasons.extend(_trace_reasons(name, evaluated.trace))
    if not metadata_aligned:
        reasons.append("metadata:prompt_length_mismatch")
    if not black_square_aligned:
        reasons.append("black_square:prompt_length_mismatch")

    clean_answer = clean.answer_scores.predicted_answer
    metadata_control_answer = metadata_control.answer_scores.predicted_answer
    metadata_hinted_answer = metadata_hinted.answer_scores.predicted_answer
    black_control_answer = black_square_control.answer_scores.predicted_answer
    black_hinted_answer = black_square_hinted.answer_scores.predicted_answer

    if metadata_control_answer != clean_answer:
        reasons.append("metadata:control_answer_differs_from_clean")
    if black_control_answer != clean_answer:
        reasons.append("black_square:control_answer_differs_from_clean")
    if metadata_hinted_answer != hint_label:
        reasons.append("metadata:hinted_answer_not_target")
    if black_hinted_answer != hint_label:
        reasons.append("black_square:hinted_answer_not_target")

    metadata_eligible = (
        metadata_aligned
        and metadata_control.trace.eligible
        and metadata_hinted.trace.eligible
        and metadata_control_answer != hint_label
        and metadata_hinted_answer == hint_label
    )
    black_square_eligible = (
        black_square_aligned
        and black_square_control.trace.eligible
        and black_square_hinted.trace.eligible
        and black_control_answer != hint_label
        and black_hinted_answer == hint_label
    )
    strict_eligible = (
        not any(_trace_reasons(name, evaluated.trace) for name, evaluated in named.items())
        and metadata_aligned
        and black_square_aligned
        and metadata_control_answer == black_control_answer == clean_answer
        and metadata_hinted_answer == black_hinted_answer == hint_label
    )
    return {
        "strict_paired_eligible": strict_eligible,
        "condition_eligible": {
            "metadata": metadata_eligible,
            "black_square": black_square_eligible,
        },
        "rejection_reasons": [] if strict_eligible else sorted(set(reasons)),
    }


def _base_record(
    example: OpenBookQAExample,
    config: ExperimentConfig,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "phase1.screening.v1",
        **example.to_dict(),
        "seed": config.seed,
        "model_id": config.model.model_id,
        "configured_model_revision": config.model.revision,
        "dataset_id": config.dataset.dataset_id,
        "dataset_subset": config.dataset.subset,
        "dataset_split": config.dataset.split,
        "runtime": dict(run_metadata),
    }


def evaluate_example(
    model: Any,
    tokenizer: Any,
    example: OpenBookQAExample,
    config: ExperimentConfig,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate and score clean, control, and hinted traces for one question."""

    record = _base_record(example, config, run_metadata)
    clean_prompt = build_clean_prompt(example, tokenizer, config.prompts)
    clean = evaluate_prompt(model, tokenizer, clean_prompt, config)
    hint_label, hint_seed = choose_hint_label(
        clean.answer_scores.predicted_answer,
        config.prompts.labels,
        base_seed=config.seed,
        question_id=example.question_id,
    )
    record.update(
        {
            "hint_label": hint_label,
            "hint_seed": hint_seed,
            "hint_differs_from_ground_truth": hint_label != example.answer_key,
            "clean": clean.to_dict(),
        }
    )

    try:
        metadata_pair = build_metadata_pair(example, hint_label, tokenizer, config.prompts)
        black_square_pair = build_black_square_pair(example, hint_label, tokenizer, config.prompts)
    except PromptAlignmentError as exc:
        record.update(
            {
                "status": "rejected_prompt_alignment",
                "strict_paired_eligible": False,
                "condition_eligible": {"metadata": False, "black_square": False},
                "rejection_reasons": [f"prompt_alignment:{exc}"],
            }
        )
        return record

    metadata_control = evaluate_prompt(model, tokenizer, metadata_pair.control, config)
    metadata_hinted = evaluate_prompt(model, tokenizer, metadata_pair.hinted, config)
    black_square_control = evaluate_prompt(model, tokenizer, black_square_pair.control, config)
    black_square_hinted = evaluate_prompt(model, tokenizer, black_square_pair.hinted, config)
    classification = classify_screening(
        clean=clean,
        metadata_control=metadata_control,
        metadata_hinted=metadata_hinted,
        black_square_control=black_square_control,
        black_square_hinted=black_square_hinted,
        hint_label=hint_label,
        metadata_aligned=metadata_pair.aligned,
        black_square_aligned=black_square_pair.aligned,
    )
    record.update(
        {
            "status": "complete",
            "prompt_pairs": {
                "metadata": _evaluated_pair_dict(metadata_pair, metadata_control, metadata_hinted),
                "black_square": _evaluated_pair_dict(
                    black_square_pair, black_square_control, black_square_hinted
                ),
            },
            **classification,
        }
    )
    return record


def _evaluated_pair_dict(
    pair: PromptPair,
    control: EvaluatedPrompt,
    hinted: EvaluatedPrompt,
) -> dict[str, Any]:
    return {
        "aligned": pair.aligned,
        "neutral_marker": pair.neutral_marker,
        "hinted_marker": pair.hinted_marker,
        "control": control.to_dict(),
        "hinted": hinted.to_dict(),
    }


def run_screening(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Run a resumable screen and flush one terminal JSONL record per example."""

    output_path = Path(config.screening.output_path)
    checkpoint = JsonlCheckpoint(output_path, resume=config.screening.resume)
    resolved_config_path = output_path.with_suffix(output_path.suffix + ".config.json")
    atomic_write_json(resolved_config_path, config.to_dict())
    run_metadata = runtime_metadata(model)

    if rows is None:
        rows = load_openbookqa(config.dataset)
    examples = normalize_rows(rows)
    strict_found = sum(
        bool(record.get("strict_paired_eligible")) for record in checkpoint.completed_records
    )
    processed = 0
    skipped = 0

    for example in examples:
        if checkpoint.is_completed(example.question_id):
            skipped += 1
            continue
        if config.screening.limit is not None and processed >= config.screening.limit:
            break
        if (
            not config.dataset.screen_all_examples
            and strict_found >= config.dataset.target_paired_examples
        ):
            break

        LOGGER.info("Screening question %s", example.question_id)
        record = evaluate_example(
            model,
            tokenizer,
            example,
            config,
            run_metadata,
        )
        checkpoint.append_terminal(record)
        processed += 1
        strict_found += int(bool(record.get("strict_paired_eligible")))
        LOGGER.info(
            "Saved question %s (strict=%s, total strict=%d)",
            example.question_id,
            record.get("strict_paired_eligible"),
            strict_found,
        )

    return {"processed": processed, "skipped": skipped, "strict_paired": strict_found}
