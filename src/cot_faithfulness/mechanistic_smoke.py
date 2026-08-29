"""Phase 4: one-example mechanistic mediation smoke runner.

The module intentionally implements only one eligible question, hint condition,
layer, reasoning block, and structured direction.  Phase 5 batching, random
controls, checkpoint resumption, and aggregation do not live here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .activation_capture import (
    build_teacher_forced_reasoning_input,
    capture_matched_residual_streams,
)
from .answer_scoring import (
    AnswerScores,
    build_scoring_context,
    score_answer_labels_batched,
)
from .config import DecodingConfig, ExperimentConfig
from .generation import all_stop_token_ids, detect_repetitive_loop, resolve_end_think_id
from .intervention_hooks import intervene_residual_stream
from .mechanistic_metrics import compute_mechanistic_effects, validate_null_intervention
from .model_loader import model_input_device
from .perturbations import build_perturbation_directions
from .reasoning_segmentation import ReasoningBlock, segment_reasoning
from .text_mediation import (
    TextMediationInputError,
    condition_is_eligible,
    read_screening_records,
    screening_record_fingerprint,
    validate_loaded_model_revision,
    validate_screening_model,
    validate_screening_sidecar,
)
from .text_metrics import answer_contrast
from .utils import atomic_write_json, runtime_metadata


PHASE4_SCHEMA = "phase4.mechanistic_smoke.v1"


@dataclass(frozen=True)
class SmokeExample:
    source_record: Mapping[str, Any]
    question_id: str
    hint_condition: str
    hint_label: str
    control_prompt_ids: tuple[int, ...]
    hinted_prompt_ids: tuple[int, ...]
    baseline_reasoning_ids: tuple[int, ...]
    stored_baseline_scores: AnswerScores


@dataclass(frozen=True)
class CounterfactualSuffix:
    token_ids: tuple[int, ...]
    text: str
    completed: bool
    truncated: bool
    repetitive: bool
    stop_reason: str
    generated_token_count: int

    @property
    def eligible(self) -> bool:
        return self.completed and not self.truncated and not self.repetitive

    def to_dict(self) -> dict[str, Any]:
        return {
            "future_reasoning_token_ids": list(self.token_ids),
            "future_reasoning_text": self.text,
            "future_reasoning_token_count": len(self.token_ids),
            "completed": self.completed,
            "truncated": self.truncated,
            "repetitive": self.repetitive,
            "eligible": self.eligible,
            "stop_reason": self.stop_reason,
            "generated_token_count": self.generated_token_count,
        }


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


def select_smoke_record(
    records: Sequence[Mapping[str, Any]],
    *,
    hint_condition: str,
    question_id: str | None,
) -> Mapping[str, Any]:
    """Select one eligible Phase 1 record deterministically."""

    requested_seen = False
    for record in records:
        current_id = str(record.get("question_id", ""))
        if question_id is not None and current_id != question_id:
            continue
        requested_seen = True
        if condition_is_eligible(record, hint_condition):
            return record
    if question_id is not None and requested_seen:
        raise TextMediationInputError(
            f"Question {question_id!r} is not eligible for {hint_condition!r}"
        )
    if question_id is not None:
        raise TextMediationInputError(f"Question {question_id!r} is absent from Phase 1 output")
    raise TextMediationInputError(
        f"No Phase 1 record is eligible for the {hint_condition!r} condition"
    )


def prepare_smoke_example(
    record: Mapping[str, Any],
    *,
    hint_condition: str,
) -> SmokeExample:
    """Extract the exact Phase 1 prompts and neutral reasoning token IDs."""

    question_id = str(record.get("question_id", "<missing>"))
    pairs = _mapping(record.get("prompt_pairs"), f"{question_id}.prompt_pairs")
    pair = _mapping(pairs.get(hint_condition), f"{question_id}.{hint_condition}")
    if not bool(pair.get("aligned")):
        raise TextMediationInputError(f"{question_id}.{hint_condition} prompts are not aligned")
    control = _mapping(pair.get("control"), f"{question_id}.{hint_condition}.control")
    hinted = _mapping(pair.get("hinted"), f"{question_id}.{hint_condition}.hinted")
    control_prompt = _mapping(control.get("prompt"), "control.prompt")
    hinted_prompt = _mapping(hinted.get("prompt"), "hinted.prompt")
    control_trace = _mapping(control.get("trace"), "control.trace")
    if not bool(control_trace.get("eligible")):
        raise TextMediationInputError("Selected neutral reasoning trace is not eligible")
    control_prompt_ids = _token_ids(control_prompt.get("input_ids"), "control.prompt.input_ids")
    hinted_prompt_ids = _token_ids(hinted_prompt.get("input_ids"), "hinted.prompt.input_ids")
    if len(control_prompt_ids) != len(hinted_prompt_ids):
        raise TextMediationInputError("Control and hinted prompt token counts differ")
    try:
        stored_scores = AnswerScores.from_dict(
            _mapping(control.get("answer_scores"), "control.answer_scores")
        )
    except ValueError as exc:
        raise TextMediationInputError("Invalid stored neutral answer scores") from exc
    hint_label = str(record.get("hint_label", ""))
    if hint_label not in stored_scores.label_logprobs:
        raise TextMediationInputError("Hint label is absent from stored answer scores")
    if hint_label == stored_scores.predicted_answer:
        raise TextMediationInputError("Hint label must differ from the neutral baseline answer")
    return SmokeExample(
        source_record=record,
        question_id=question_id,
        hint_condition=hint_condition,
        hint_label=hint_label,
        control_prompt_ids=control_prompt_ids,
        hinted_prompt_ids=hinted_prompt_ids,
        baseline_reasoning_ids=_token_ids(
            control_trace.get("reasoning_token_ids"),
            "control.trace.reasoning_token_ids",
        ),
        stored_baseline_scores=stored_scores,
    )


def _parse_counterfactual_suffix(
    tokenizer: Any,
    generated_ids: Sequence[int],
    decoding: DecodingConfig,
    *,
    max_new_tokens: int,
) -> CounterfactualSuffix:
    end_think_id = resolve_end_think_id(tokenizer)
    raw = tuple(int(token_id) for token_id in generated_ids)
    eos_before_end_think = False
    if end_think_id in raw:
        close_index = raw.index(end_think_id)
        inside = raw[:close_index]
        completed = True
        stop_reason = "end_think"
    else:
        eos = getattr(tokenizer, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        eos_indices = [index for index, token_id in enumerate(raw) if token_id in eos_ids]
        eos_before_end_think = bool(eos_indices)
        inside = raw[: min(eos_indices)] if eos_before_end_think else raw
        completed = False
        stop_reason = "eos_before_end_think" if eos_before_end_think else "token_cap"
    text = tokenizer.decode(
        list(inside),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    repetitive = detect_repetitive_loop(
        inside,
        min_repeats=decoding.repetition_min_repeats,
        min_cycle_tokens=decoding.repetition_min_cycle_tokens,
        max_cycle_tokens=decoding.repetition_max_cycle_tokens,
    )
    return CounterfactualSuffix(
        token_ids=inside,
        text=text,
        completed=completed,
        truncated=not completed and not eos_before_end_think and len(raw) >= max_new_tokens,
        repetitive=repetitive,
        stop_reason=stop_reason,
        generated_token_count=len(raw),
    )


def generate_counterfactual_suffixes(
    model: Any,
    tokenizer: Any,
    prefix_ids: Sequence[int],
    *,
    layer_idx: int,
    source_positions: tuple[int, ...],
    direction: Any,
    alpha: float,
    batch_size: int,
    reasoning_prefix_token_count: int,
    decoding: DecodingConfig,
) -> tuple[CounterfactualSuffix, ...]:
    """Greedily generate only the future reasoning under one source intervention."""

    if decoding.do_sample:
        raise ValueError("Phase 4 counterfactual generation must be deterministic")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if reasoning_prefix_token_count < 0:
        raise ValueError("reasoning_prefix_token_count cannot be negative")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install torch before generating counterfactual reasoning") from exc

    prefix = tuple(int(token_id) for token_id in prefix_ids)
    if not prefix:
        raise ValueError("Counterfactual generation prefix cannot be empty")
    remaining_reasoning = max(0, decoding.max_reasoning_tokens - reasoning_prefix_token_count)
    max_new_tokens = remaining_reasoning + 1  # reserve one token for </think>
    end_think_id = resolve_end_think_id(tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0]

    device = model_input_device(model)
    input_ids = (
        torch.tensor([list(prefix)], dtype=torch.long, device=device)
        .expand(batch_size, -1)
        .clone()
    )
    attention_mask = torch.ones_like(input_ids)
    with intervene_residual_stream(
        model,
        layer_idx=layer_idx,
        token_positions=source_positions,
        perturbation=direction,
        alpha=alpha,
        operation="additive",
        prefill_only=True,
    ):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=all_stop_token_ids(tokenizer, end_think_id),
                pad_token_id=pad_token_id,
                use_cache=True,
            )
    generated = output_ids[:, input_ids.shape[1] :].detach().cpu().tolist()
    return tuple(
        _parse_counterfactual_suffix(
            tokenizer,
            row,
            decoding,
            max_new_tokens=max_new_tokens,
        )
        for row in generated
    )


def score_contexts_with_intervention(
    model: Any,
    tokenizer: Any,
    contexts: Sequence[Sequence[int]],
    *,
    layer_idx: int,
    source_positions: tuple[int, ...],
    direction: Any,
    alpha: float,
    config: ExperimentConfig,
) -> tuple[AnswerScores, ...]:
    """Score all contexts in one forward while applying the same source patch."""

    with intervene_residual_stream(
        model,
        layer_idx=layer_idx,
        token_positions=source_positions,
        perturbation=direction,
        alpha=alpha,
        operation="additive",
        prefill_only=True,
    ):
        return score_answer_labels_batched(
            model,
            tokenizer,
            contexts,
            labels=config.prompts.labels,
            label_prefix=config.decoding.answer_label_prefix,
        )


def _counterfactual_context(
    tokenizer: Any,
    example: SmokeExample,
    block: ReasoningBlock,
    suffix: CounterfactualSuffix,
    config: ExperimentConfig,
) -> tuple[int, ...]:
    reasoning = example.baseline_reasoning_ids[: block.end_token] + suffix.token_ids
    return build_scoring_context(
        tokenizer,
        example.control_prompt_ids,
        reasoning,
        resolve_end_think_id(tokenizer),
        config.decoding.answer_cue,
    )


def _base_output(
    example: SmokeExample,
    config: ExperimentConfig,
    runtime: Mapping[str, Any],
    *,
    layer_idx: int,
    block: ReasoningBlock,
    segmentation: Mapping[str, Any],
    direction_metadata: Mapping[str, Any],
    source_positions: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": PHASE4_SCHEMA,
        "phase_scope": {
            "questions": 1,
            "hint_conditions": 1,
            "layers": 1,
            "reasoning_blocks": 1,
            "directions": ["structured"],
            "phase5_batch_experiment_enabled": False,
        },
        "source_screening_fingerprint": screening_record_fingerprint(example.source_record),
        "question_id": example.question_id,
        "hint_condition": example.hint_condition,
        "hint_label": example.hint_label,
        "baseline_answer": example.stored_baseline_scores.predicted_answer,
        "layer_idx": layer_idx,
        "block_idx": block.index,
        "source_positions": list(source_positions),
        "segmentation": dict(segmentation),
        "direction": dict(direction_metadata),
        "alpha": config.mechanistic.alpha_primary,
        "runtime": dict(runtime),
    }


def evaluate_mechanistic_smoke(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    screening_record: Mapping[str, Any],
    *,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the full Phase 4 smoke case in memory."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install torch before running the Phase 4 smoke test") from exc

    smoke = config.mechanistic_smoke
    example = prepare_smoke_example(screening_record, hint_condition=smoke.hint_condition)
    layer_idx = smoke.layer_idx if smoke.layer_idx is not None else config.model.layers[0]
    segmentation = segment_reasoning(
        tokenizer,
        example.baseline_reasoning_ids,
        max_blocks=config.mechanistic.max_reasoning_blocks,
    )
    if smoke.block_idx >= len(segmentation.blocks):
        raise ValueError(
            f"Requested block {smoke.block_idx}, but segmentation produced "
            f"{len(segmentation.blocks)} blocks"
        )
    block = segmentation.blocks[smoke.block_idx]
    neutral = build_teacher_forced_reasoning_input(
        example.control_prompt_ids, example.baseline_reasoning_ids
    )
    hinted = build_teacher_forced_reasoning_input(
        example.hinted_prompt_ids, example.baseline_reasoning_ids
    )
    if neutral.reasoning_positions != hinted.reasoning_positions:
        raise RuntimeError("Matched prompts do not place neutral reasoning at identical positions")
    device = model_input_device(model)
    captures = capture_matched_residual_streams(
        model,
        torch.tensor([neutral.input_ids], dtype=torch.long, device=device),
        torch.tensor([hinted.input_ids], dtype=torch.long, device=device),
        layer_indices=(layer_idx,),
        reasoning_positions=neutral.reasoning_positions,
        storage_dtype=config.mechanistic.activation_storage_dtype,
    )
    # Phase 4 constructs only the structured direction. Random controls begin in Phase 5.
    phase4_direction_config = replace(config.mechanistic, num_random_directions=0)
    directions = build_perturbation_directions(
        captures.neutral.block(layer_idx, block.start_token, block.end_token),
        captures.hinted.block(layer_idx, block.start_token, block.end_token),
        question_id=example.question_id,
        hint_type=example.hint_condition,
        layer_idx=layer_idx,
        block_idx=block.index,
        base_seed=config.seed,
        config=phase4_direction_config,
    )
    direction = directions.values[0]
    source_positions = tuple(
        range(
            len(example.control_prompt_ids) + block.start_token,
            len(example.control_prompt_ids) + block.end_token,
        )
    )
    runtime = dict(run_metadata or runtime_metadata(model))
    result = _base_output(
        example,
        config,
        runtime,
        layer_idx=layer_idx,
        block=block,
        segmentation=segmentation.to_dict(),
        direction_metadata=directions.to_metadata_dict(),
        source_positions=source_positions,
    )

    end_think_id = resolve_end_think_id(tokenizer)
    baseline_context = build_scoring_context(
        tokenizer,
        example.control_prompt_ids,
        example.baseline_reasoning_ids,
        end_think_id,
        config.decoding.answer_cue,
    )
    baseline_scores = score_answer_labels_batched(
        model,
        tokenizer,
        (baseline_context,),
        labels=config.prompts.labels,
        label_prefix=config.decoding.answer_label_prefix,
    )[0]
    y00 = answer_contrast(
        baseline_scores.label_logprobs,
        example.hint_label,
        example.stored_baseline_scores.predicted_answer,
    )
    result["baseline"] = {
        "Y00": y00,
        "stored_answer_scores": example.stored_baseline_scores.to_dict(),
        "live_answer_scores": baseline_scores.to_dict(),
        "answer_matches_phase1": (
            baseline_scores.predicted_answer == example.stored_baseline_scores.predicted_answer
        ),
        "prompt_token_count": len(example.control_prompt_ids),
        "reasoning_token_count": len(example.baseline_reasoning_ids),
        "scoring_context_token_count": len(baseline_context),
    }
    if baseline_scores.predicted_answer != example.stored_baseline_scores.predicted_answer:
        result["status"] = "baseline_answer_mismatch"
        result["structured"] = None
        return result

    prefix_ids = (
        example.control_prompt_ids + example.baseline_reasoning_ids[: block.end_token]
    )
    baseline_suffix = example.baseline_reasoning_ids[block.end_token :]
    null_unbatched = generate_counterfactual_suffixes(
        model,
        tokenizer,
        prefix_ids,
        layer_idx=layer_idx,
        source_positions=source_positions,
        direction=direction,
        alpha=0.0,
        batch_size=1,
        reasoning_prefix_token_count=block.end_token,
        decoding=config.decoding,
    )[0]
    null_batched = generate_counterfactual_suffixes(
        model,
        tokenizer,
        prefix_ids,
        layer_idx=layer_idx,
        source_positions=source_positions,
        direction=direction,
        alpha=0.0,
        batch_size=smoke.null_batch_size,
        reasoning_prefix_token_count=block.end_token,
        decoding=config.decoding,
    )
    null_contexts = tuple(
        _counterfactual_context(tokenizer, example, block, suffix, config)
        for suffix in (null_unbatched, *null_batched)
    )
    null_scores = score_contexts_with_intervention(
        model,
        tokenizer,
        null_contexts,
        layer_idx=layer_idx,
        source_positions=source_positions,
        direction=direction,
        alpha=0.0,
        config=config,
    )
    null_validation = validate_null_intervention(
        baseline_suffix_ids=baseline_suffix,
        unbatched_suffix_ids=null_unbatched.token_ids,
        batched_suffix_ids=[suffix.token_ids for suffix in null_batched],
        baseline_scores=baseline_scores,
        unbatched_scores=null_scores[0],
        batched_scores=null_scores[1:],
        absolute_tolerance=smoke.null_absolute_tolerance,
        relative_tolerance=smoke.null_relative_tolerance,
        unbatched_generation_valid=null_unbatched.eligible,
        batched_generations_valid=all(suffix.eligible for suffix in null_batched),
    )
    result["null_validation"] = {
        **null_validation.to_dict(),
        "alpha": 0.0,
        "baseline_future_reasoning_token_ids": list(baseline_suffix),
        "unbatched_generation": null_unbatched.to_dict(),
        "batched_generations": [suffix.to_dict() for suffix in null_batched],
        "unbatched_answer_scores": null_scores[0].to_dict(),
        "batched_answer_scores": [score.to_dict() for score in null_scores[1:]],
    }
    if not null_validation.passed:
        result["status"] = "null_intervention_failed"
        result["structured"] = None
        return result
    if not directions.detectable:
        result["status"] = "structured_direction_below_threshold"
        result["structured"] = None
        return result

    y10_scores = score_contexts_with_intervention(
        model,
        tokenizer,
        (baseline_context,),
        layer_idx=layer_idx,
        source_positions=source_positions,
        direction=direction,
        alpha=config.mechanistic.alpha_primary,
        config=config,
    )[0]
    y10 = answer_contrast(
        y10_scores.label_logprobs,
        example.hint_label,
        example.stored_baseline_scores.predicted_answer,
    )
    structured_suffix = generate_counterfactual_suffixes(
        model,
        tokenizer,
        prefix_ids,
        layer_idx=layer_idx,
        source_positions=source_positions,
        direction=direction,
        alpha=config.mechanistic.alpha_primary,
        batch_size=1,
        reasoning_prefix_token_count=block.end_token,
        decoding=config.decoding,
    )[0]
    structured: dict[str, Any] = {
        "alpha": config.mechanistic.alpha_primary,
        "Y10": y10,
        "Y10_answer_scores": y10_scores.to_dict(),
        "counterfactual_generation": structured_suffix.to_dict(),
        "counterfactual_reasoning_token_ids": list(
            example.baseline_reasoning_ids[: block.end_token] + structured_suffix.token_ids
        ),
    }
    result["structured"] = structured
    if not structured_suffix.eligible:
        result["status"] = "counterfactual_reasoning_rejected"
        return result

    y11_context = _counterfactual_context(tokenizer, example, block, structured_suffix, config)
    y11_scores = score_contexts_with_intervention(
        model,
        tokenizer,
        (y11_context,),
        layer_idx=layer_idx,
        source_positions=source_positions,
        direction=direction,
        alpha=config.mechanistic.alpha_primary,
        config=config,
    )[0]
    y11 = answer_contrast(
        y11_scores.label_logprobs,
        example.hint_label,
        example.stored_baseline_scores.predicted_answer,
    )
    effects = compute_mechanistic_effects(
        y00=y00,
        y10=y10,
        y11=y11,
        tolerance=smoke.additivity_tolerance,
    )
    structured.update(
        {
            "Y11": y11,
            "Y11_answer_scores": y11_scores.to_dict(),
            "Y11_scoring_context_token_count": len(y11_context),
        }
    )
    result["metrics"] = effects.to_dict()
    result["status"] = "complete"
    return result


def run_mechanistic_smoke(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    *,
    screening_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select, evaluate, and atomically save exactly one Phase 4 smoke case."""

    source_path = Path(config.mechanistic_smoke.input_path)
    if screening_records is None:
        screening_records = read_screening_records(source_path)
    validate_screening_model(screening_records, config)
    validate_screening_sidecar(source_path, config)
    runtime = runtime_metadata(model)
    validate_loaded_model_revision(screening_records, runtime)
    selected = select_smoke_record(
        screening_records,
        hint_condition=config.mechanistic_smoke.hint_condition,
        question_id=config.mechanistic_smoke.question_id,
    )
    result = evaluate_mechanistic_smoke(
        config,
        model,
        tokenizer,
        selected,
        run_metadata=runtime,
    )
    output_path = Path(config.mechanistic_smoke.output_path)
    atomic_write_json(output_path, result)
    atomic_write_json(
        output_path.with_suffix(output_path.suffix + ".config.json"),
        config.to_dict(),
    )
    return result
