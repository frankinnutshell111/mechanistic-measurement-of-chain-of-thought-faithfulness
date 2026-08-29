"""Phase 2 crossed-context text-level mediation experiment."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .answer_scoring import AnswerScores, build_scoring_context, score_answer_labels
from .checkpoint import JsonlCheckpoint
from .config import ExperimentConfig
from .generation import resolve_end_think_id
from .text_metrics import answer_contrast, compute_text_mediation_effects
from .utils import atomic_write_json, atomic_write_text, runtime_metadata

LOGGER = logging.getLogger(__name__)
HINT_CONDITIONS = ("metadata", "black_square")
SCREENING_SCHEMA = "phase1.screening.v1"
TEXT_MEDIATION_SCHEMA = "phase2.text_mediation.v1"


class TextMediationInputError(ValueError):
    """Raised when Phase 1 data cannot be safely reused for Phase 2."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TextMediationInputError(f"{name} must be a mapping")
    return value


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TextMediationInputError(f"{name} must be a token-ID sequence")
    try:
        result = tuple(int(token_id) for token_id in value)
    except (TypeError, ValueError) as exc:
        raise TextMediationInputError(f"{name} contains an invalid token ID") from exc
    if not result:
        raise TextMediationInputError(f"{name} cannot be empty")
    return result


def read_screening_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Phase 1 screening output does not exist: {source}. Run screening first."
        )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TextMediationInputError(f"Invalid JSON at {source}:{line_number}") from exc
            record = dict(_mapping(value, f"screening record at line {line_number}"))
            if record.get("schema_version") != SCREENING_SCHEMA:
                raise TextMediationInputError(
                    f"Unsupported screening schema at {source}:{line_number}: "
                    f"{record.get('schema_version')!r}"
                )
            try:
                question_id = str(record["question_id"])
            except KeyError as exc:
                raise TextMediationInputError(
                    f"Missing question_id at {source}:{line_number}"
                ) from exc
            if question_id in seen_ids:
                raise TextMediationInputError(
                    f"Duplicate Phase 1 question_id {question_id!r} at {source}:{line_number}"
                )
            seen_ids.add(question_id)
            records.append(record)
    if not records:
        raise TextMediationInputError(f"Phase 1 screening output is empty: {source}")
    return tuple(records)


def validate_screening_model(
    records: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> None:
    mismatches = sorted(
        {
            str(record.get("model_id"))
            for record in records
            if record.get("model_id") != config.model.model_id
        }
    )
    if mismatches:
        raise TextMediationInputError(
            "Phase 2 must score with the same model as Phase 1. "
            f"Configured {config.model.model_id!r}, but the input contains {mismatches!r}. "
            "Pass the Phase 1 model with --model-id."
        )


def validate_screening_sidecar(source: str | Path, config: ExperimentConfig) -> None:
    source = Path(source)
    sidecar = source.with_suffix(source.suffix + ".config.json")
    if not sidecar.exists():
        LOGGER.warning(
            "No Phase 1 resolved-config sidecar found at %s; validating model ID from JSONL only",
            sidecar,
        )
        return
    try:
        stored = _mapping(json.loads(sidecar.read_text(encoding="utf-8")), "Phase 1 sidecar")
        stored_model = _mapping(stored["model"], "Phase 1 sidecar model")
        stored_decoding = _mapping(stored["decoding"], "Phase 1 sidecar decoding")
        stored_prompts = _mapping(stored["prompts"], "Phase 1 sidecar prompts")
    except (json.JSONDecodeError, KeyError) as exc:
        raise TextMediationInputError(f"Invalid Phase 1 config sidecar: {sidecar}") from exc
    checks = {
        "model.model_id": (stored_model.get("model_id"), config.model.model_id),
        "model.revision": (stored_model.get("revision"), config.model.revision),
        "decoding.answer_cue": (
            stored_decoding.get("answer_cue"),
            config.decoding.answer_cue,
        ),
        "decoding.answer_label_prefix": (
            stored_decoding.get("answer_label_prefix"),
            config.decoding.answer_label_prefix,
        ),
        "prompts.labels": (
            tuple(stored_prompts.get("labels", ())),
            config.prompts.labels,
        ),
    }
    mismatches = [
        f"{name}: Phase 1={old!r}, Phase 2={new!r}"
        for name, (old, new) in checks.items()
        if old != new
    ]
    if mismatches:
        raise TextMediationInputError(
            "Phase 1 and Phase 2 scoring configurations differ: " + "; ".join(mismatches)
        )


def condition_is_available(record: Mapping[str, Any], hint_condition: str) -> bool:
    pairs = record.get("prompt_pairs")
    return (
        record.get("status") == "complete"
        and isinstance(pairs, Mapping)
        and isinstance(pairs.get(hint_condition), Mapping)
    )


def condition_is_eligible(record: Mapping[str, Any], hint_condition: str) -> bool:
    eligibility = record.get("condition_eligible")
    return isinstance(eligibility, Mapping) and bool(eligibility.get(hint_condition))


def completion_key(question_id: str, hint_condition: str) -> str:
    if hint_condition not in HINT_CONDITIONS:
        raise ValueError(f"Unknown hint condition: {hint_condition}")
    return f"{question_id}|{hint_condition}"


def screening_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Identify the exact Phase 1 source record independently of its file path."""

    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pair_components(
    record: Mapping[str, Any],
    hint_condition: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    question_id = str(record.get("question_id", "<missing>"))
    pairs = _mapping(record.get("prompt_pairs"), f"{question_id}.prompt_pairs")
    pair = _mapping(pairs.get(hint_condition), f"{question_id}.{hint_condition}")
    if not bool(pair.get("aligned")):
        raise TextMediationInputError(f"{question_id}.{hint_condition} prompts are not aligned")
    control = _mapping(pair.get("control"), f"{question_id}.{hint_condition}.control")
    hinted = _mapping(pair.get("hinted"), f"{question_id}.{hint_condition}.hinted")
    return pair, control, hinted


def _prompt_and_trace_ids(
    evaluated: Mapping[str, Any],
    name: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prompt = _mapping(evaluated.get("prompt"), f"{name}.prompt")
    trace = _mapping(evaluated.get("trace"), f"{name}.trace")
    return (
        _token_ids(prompt.get("input_ids"), f"{name}.prompt.input_ids"),
        _token_ids(trace.get("reasoning_token_ids"), f"{name}.trace.reasoning_token_ids"),
    )


def _stored_answer_scores(evaluated: Mapping[str, Any], name: str) -> AnswerScores:
    try:
        return AnswerScores.from_dict(
            _mapping(evaluated.get("answer_scores"), f"{name}.answer_scores")
        )
    except ValueError as exc:
        raise TextMediationInputError(f"Invalid {name}.answer_scores") from exc


def build_crossed_scoring_contexts(
    tokenizer: Any,
    record: Mapping[str, Any],
    hint_condition: str,
    config: ExperimentConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build P_hint/R_control and P_control/R_hint from exact stored token IDs."""

    _, control, hinted = _pair_components(record, hint_condition)
    control_prompt, control_trace = _prompt_and_trace_ids(
        control, f"{record.get('question_id')}.{hint_condition}.control"
    )
    hinted_prompt, hinted_trace = _prompt_and_trace_ids(
        hinted, f"{record.get('question_id')}.{hint_condition}.hinted"
    )
    if len(control_prompt) != len(hinted_prompt):
        raise TextMediationInputError(
            f"{record.get('question_id')}.{hint_condition} prompt token counts differ"
        )
    end_think_id = resolve_end_think_id(tokenizer)
    direct_context = build_scoring_context(
        tokenizer,
        hinted_prompt,
        control_trace,
        end_think_id,
        config.decoding.answer_cue,
    )
    mediated_context = build_scoring_context(
        tokenizer,
        control_prompt,
        hinted_trace,
        end_think_id,
        config.decoding.answer_cue,
    )
    return direct_context, mediated_context


def build_text_mediation_record(
    screening_record: Mapping[str, Any],
    hint_condition: str,
    direct_scores: AnswerScores,
    mediated_scores: AnswerScores,
    config: ExperimentConfig,
    run_metadata: Mapping[str, Any],
    *,
    direct_context_token_count: int,
    mediated_context_token_count: int,
) -> dict[str, Any]:
    """Combine stored natural scores and newly scored crossed contexts."""

    _, control, hinted = _pair_components(screening_record, hint_condition)
    control_scores = _stored_answer_scores(
        control, f"{screening_record.get('question_id')}.{hint_condition}.control"
    )
    hinted_scores = _stored_answer_scores(
        hinted, f"{screening_record.get('question_id')}.{hint_condition}.hinted"
    )
    hint_label = str(screening_record["hint_label"])
    baseline_answer = control_scores.predicted_answer
    y_control_control = answer_contrast(control_scores.label_logprobs, hint_label, baseline_answer)
    y_hint_control = answer_contrast(direct_scores.label_logprobs, hint_label, baseline_answer)
    y_control_hint = answer_contrast(mediated_scores.label_logprobs, hint_label, baseline_answer)
    y_hint_hint = answer_contrast(hinted_scores.label_logprobs, hint_label, baseline_answer)
    effects = compute_text_mediation_effects(
        y_control_control=y_control_control,
        y_hint_control=y_hint_control,
        y_control_hint=y_control_hint,
        epsilon=config.text_mediation.epsilon,
        minimum_effect_magnitude=config.text_mediation.minimum_effect_magnitude,
    )
    metrics = effects.to_dict()
    natural_total = y_hint_hint - y_control_control
    metrics.update(
        {
            "Y_hint_hint": y_hint_hint,
            "natural_total_effect": natural_total,
            "interaction_residual": natural_total - effects.direct_text - effects.mediated_text,
        }
    )
    question_id = str(screening_record["question_id"])
    return {
        "schema_version": TEXT_MEDIATION_SCHEMA,
        "source_screening_fingerprint": screening_record_fingerprint(screening_record),
        "completion_key": completion_key(question_id, hint_condition),
        "question_id": question_id,
        "hint_condition": hint_condition,
        "hint_label": hint_label,
        "baseline_answer": baseline_answer,
        "ground_truth": screening_record.get("answer_key"),
        "condition_eligible": condition_is_eligible(screening_record, hint_condition),
        "strict_paired_eligible": bool(screening_record.get("strict_paired_eligible")),
        "model_id": config.model.model_id,
        "configured_model_revision": config.model.revision,
        "epsilon": config.text_mediation.epsilon,
        "minimum_effect_magnitude": config.text_mediation.minimum_effect_magnitude,
        "natural_control_answer": control_scores.predicted_answer,
        "natural_hinted_answer": hinted_scores.predicted_answer,
        "answer_scores": {
            "control_prompt_control_reasoning": control_scores.to_dict(),
            "hint_prompt_control_reasoning": direct_scores.to_dict(),
            "control_prompt_hint_reasoning": mediated_scores.to_dict(),
            "hint_prompt_hint_reasoning": hinted_scores.to_dict(),
        },
        "scoring_context_token_counts": {
            "hint_prompt_control_reasoning": direct_context_token_count,
            "control_prompt_hint_reasoning": mediated_context_token_count,
        },
        "metrics": metrics,
        "runtime": dict(run_metadata),
    }


def evaluate_text_condition(
    model: Any,
    tokenizer: Any,
    screening_record: Mapping[str, Any],
    hint_condition: str,
    config: ExperimentConfig,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    direct_context, mediated_context = build_crossed_scoring_contexts(
        tokenizer, screening_record, hint_condition, config
    )
    scoring_kwargs = {
        "labels": config.prompts.labels,
        "label_prefix": config.decoding.answer_label_prefix,
        "sequence_batch_size": config.decoding.answer_scoring_batch_size,
    }
    direct_scores = score_answer_labels(model, tokenizer, direct_context, **scoring_kwargs)
    mediated_scores = score_answer_labels(model, tokenizer, mediated_context, **scoring_kwargs)
    return build_text_mediation_record(
        screening_record,
        hint_condition,
        direct_scores,
        mediated_scores,
        config,
        run_metadata,
        direct_context_token_count=len(direct_context),
        mediated_context_token_count=len(mediated_context),
    )


def _condition_summary(
    records: Sequence[Mapping[str, Any]],
    epsilon: float,
) -> dict[str, Any]:
    metrics = [_mapping(record["metrics"], "Phase 2 metrics") for record in records]
    if not metrics:
        return {
            "n": 0,
            "mean_D_text": None,
            "mean_C_text": None,
            "mean_absolute_D_text": None,
            "mean_absolute_C_text": None,
            "mean_F_text": None,
            "median_F_text": None,
            "pooled_F_text": None,
            "effect_above_floor_n": 0,
        }
    direct = [float(item["D_text"]) for item in metrics]
    mediated = [float(item["C_text"]) for item in metrics]
    absolute_direct = [float(item["absolute_D_text"]) for item in metrics]
    absolute_mediated = [float(item["absolute_C_text"]) for item in metrics]
    fractions = [float(item["F_text"]) for item in metrics]
    sum_absolute_direct = sum(absolute_direct)
    sum_absolute_mediated = sum(absolute_mediated)
    return {
        "n": len(metrics),
        "mean_D_text": statistics.fmean(direct),
        "mean_C_text": statistics.fmean(mediated),
        "mean_absolute_D_text": statistics.fmean(absolute_direct),
        "mean_absolute_C_text": statistics.fmean(absolute_mediated),
        "mean_F_text": statistics.fmean(fractions),
        "median_F_text": statistics.median(fractions),
        "pooled_F_text": sum_absolute_mediated
        / (sum_absolute_mediated + sum_absolute_direct + epsilon),
        "effect_above_floor_n": sum(bool(item["effect_above_floor"]) for item in metrics),
    }


def summarize_text_mediation(
    records: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
) -> dict[str, Any]:
    """Aggregate conditions and check the proposal's expected group-level pattern."""

    grouped = {
        condition: [record for record in records if record.get("hint_condition") == condition]
        for condition in HINT_CONDITIONS
    }
    by_condition = {
        condition: _condition_summary(condition_records, epsilon)
        for condition, condition_records in grouped.items()
    }
    by_question: dict[str, dict[str, float]] = {}
    for record in records:
        question_id = str(record["question_id"])
        condition = str(record["hint_condition"])
        metrics = _mapping(record["metrics"], "Phase 2 metrics")
        by_question.setdefault(question_id, {})[condition] = float(metrics["F_text"])
    paired_differences = [
        values["black_square"] - values["metadata"]
        for values in by_question.values()
        if set(HINT_CONDITIONS).issubset(values)
    ]
    paired = {
        "n": len(paired_differences),
        "mean_F_text_difference_black_square_minus_metadata": (
            statistics.fmean(paired_differences) if paired_differences else None
        ),
        "median_F_text_difference_black_square_minus_metadata": (
            statistics.median(paired_differences) if paired_differences else None
        ),
    }
    metadata = by_condition["metadata"]
    black_square = by_condition["black_square"]
    enough_data = bool(metadata["n"] and black_square["n"])
    metadata_direct = (
        metadata["mean_absolute_D_text"] > metadata["mean_absolute_C_text"] if enough_data else None
    )
    black_mediated = (
        black_square["mean_absolute_C_text"] > black_square["mean_absolute_D_text"]
        if enough_data
        else None
    )
    fraction_ordering = (
        black_square["mean_F_text"] > metadata["mean_F_text"] if enough_data else None
    )
    expected = {
        "description": (
            "Metadata is direct-effect dominated, Black-Square is text-mediated-effect "
            "dominated, and Black-Square has the larger mean F_text."
        ),
        "metadata_direct_effect_dominates": metadata_direct,
        "black_square_text_mediated_effect_dominates": black_mediated,
        "black_square_mean_F_text_exceeds_metadata": fraction_ordering,
        "all_criteria_observed": (
            bool(metadata_direct and black_mediated and fraction_ordering) if enough_data else None
        ),
    }
    return {
        "schema_version": "phase2.text_mediation.summary.v1",
        "by_condition": by_condition,
        "paired": paired,
        "expected_pattern": expected,
    }


def _summary_csv(summary: Mapping[str, Any]) -> str:
    fieldnames = [
        "hint_condition",
        "n",
        "mean_D_text",
        "mean_C_text",
        "mean_absolute_D_text",
        "mean_absolute_C_text",
        "mean_F_text",
        "median_F_text",
        "pooled_F_text",
        "effect_above_floor_n",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    by_condition = _mapping(summary["by_condition"], "summary.by_condition")
    for condition in HINT_CONDITIONS:
        row = dict(_mapping(by_condition[condition], f"summary.{condition}"))
        writer.writerow({"hint_condition": condition, **row})
    return stream.getvalue()


def _validate_resumed_records(
    records: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
    screening_records: Sequence[Mapping[str, Any]],
) -> None:
    source_fingerprints = {
        completion_key(str(screening_record["question_id"]), condition): (
            screening_record_fingerprint(screening_record)
        )
        for screening_record in screening_records
        for condition in HINT_CONDITIONS
        if condition_is_available(screening_record, condition)
    }
    for record in records:
        if record.get("schema_version") != TEXT_MEDIATION_SCHEMA:
            raise TextMediationInputError(
                f"Existing Phase 2 output has unsupported schema: {record.get('schema_version')!r}"
            )
        if record.get("model_id") != config.model.model_id:
            raise TextMediationInputError(
                "Existing Phase 2 output was produced by a different model; "
                "choose a new output path"
            )
        try:
            epsilon_matches = math.isclose(
                float(record["epsilon"]),
                config.text_mediation.epsilon,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TextMediationInputError("Existing Phase 2 output has invalid epsilon") from exc
        if not epsilon_matches:
            raise TextMediationInputError(
                "Existing Phase 2 output uses a different epsilon; choose a new output path"
            )
        try:
            floor_matches = math.isclose(
                float(record["minimum_effect_magnitude"]),
                config.text_mediation.minimum_effect_magnitude,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TextMediationInputError(
                "Existing Phase 2 output has an invalid minimum effect magnitude"
            ) from exc
        if not floor_matches:
            raise TextMediationInputError(
                "Existing Phase 2 output uses a different minimum effect magnitude; "
                "choose a new output path"
            )
        try:
            record_key = str(record["completion_key"])
            expected_fingerprint = source_fingerprints[record_key]
        except KeyError as exc:
            raise TextMediationInputError(
                "Existing Phase 2 output does not belong to the current Phase 1 input; "
                "choose a new output path"
            ) from exc
        if record.get("source_screening_fingerprint") != expected_fingerprint:
            raise TextMediationInputError(
                "A Phase 1 source record changed since this Phase 2 output was created; "
                "choose a new output path"
            )
        if config.text_mediation.eligible_only and not bool(record.get("condition_eligible")):
            raise TextMediationInputError(
                "Existing Phase 2 output contains ineligible conditions; either use "
                "--no-eligible-only or choose a new output path"
            )


def validate_loaded_model_revision(
    screening_records: Sequence[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
) -> None:
    source_revisions = {
        str(revision)
        for record in screening_records
        if isinstance(record.get("runtime"), Mapping)
        if (revision := record["runtime"].get("model_revision")) is not None
    }
    if len(source_revisions) > 1:
        raise TextMediationInputError(
            f"Phase 1 input mixes model revisions: {sorted(source_revisions)!r}"
        )
    if source_revisions:
        current_revision = run_metadata.get("model_revision")
        if current_revision not in source_revisions:
            raise TextMediationInputError(
                "The loaded model revision does not match the Phase 1 runtime revision: "
                f"Phase 1={next(iter(source_revisions))!r}, Phase 2={current_revision!r}"
            )


def run_text_mediation(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    *,
    screening_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score crossed text contexts, checkpoint each condition, and aggregate results."""

    phase_config = config.text_mediation
    source_path = Path(phase_config.input_path)
    if screening_records is None:
        screening_records = read_screening_records(source_path)
    validate_screening_sidecar(source_path, config)
    validate_screening_model(screening_records, config)
    run_metadata = runtime_metadata(model)
    validate_loaded_model_revision(screening_records, run_metadata)

    output_path = Path(phase_config.output_path)
    checkpoint = JsonlCheckpoint(
        output_path,
        resume=phase_config.resume,
        key_field="completion_key",
    )
    _validate_resumed_records(checkpoint.completed_records, config, screening_records)
    atomic_write_json(
        output_path.with_suffix(output_path.suffix + ".config.json"), config.to_dict()
    )
    processed = 0
    resumed = 0
    unavailable = 0
    excluded_ineligible = 0

    stop = False
    for screening_record in screening_records:
        for hint_condition in HINT_CONDITIONS:
            if not condition_is_available(screening_record, hint_condition):
                unavailable += 1
                continue
            if phase_config.eligible_only and not condition_is_eligible(
                screening_record, hint_condition
            ):
                excluded_ineligible += 1
                continue
            question_id = str(screening_record["question_id"])
            key = completion_key(question_id, hint_condition)
            if checkpoint.is_completed(key):
                resumed += 1
                continue
            if phase_config.limit is not None and processed >= phase_config.limit:
                stop = True
                break
            LOGGER.info("Scoring Phase 2 crossed contexts for %s (%s)", question_id, hint_condition)
            record = evaluate_text_condition(
                model,
                tokenizer,
                screening_record,
                hint_condition,
                config,
                run_metadata,
            )
            checkpoint.append_terminal(record)
            processed += 1
        if stop:
            break

    result_records = checkpoint.completed_records
    summary = summarize_text_mediation(result_records, epsilon=phase_config.epsilon)
    summary["run_counts"] = {
        "screening_records": len(screening_records),
        "processed_this_run": processed,
        "resumed_conditions": resumed,
        "unavailable_conditions": unavailable,
        "excluded_ineligible_conditions": excluded_ineligible,
        "total_completed_conditions": len(result_records),
        "limit_reached": stop,
    }
    summary["input_path"] = str(source_path)
    summary["output_path"] = str(output_path)
    atomic_write_json(phase_config.summary_json_path, summary)
    atomic_write_text(phase_config.summary_csv_path, _summary_csv(summary))
    return summary
