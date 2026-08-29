"""Phase 4 answer-contrast effects and null-intervention validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .answer_scoring import AnswerScores


@dataclass(frozen=True)
class MechanisticEffects:
    y00: float
    y10: float
    y11: float
    direct: float
    mediated: float
    total: float
    additivity_error: float
    additivity_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "Y00": self.y00,
            "Y10": self.y10,
            "Y11": self.y11,
            "D": self.direct,
            "C": self.mediated,
            "T": self.total,
            "additivity_error": self.additivity_error,
            "additivity_passed": self.additivity_passed,
        }


def compute_mechanistic_effects(
    *,
    y00: float,
    y10: float,
    y11: float,
    tolerance: float = 1e-8,
) -> MechanisticEffects:
    """Compute D=Y10-Y00, C=Y11-Y10, T=Y11-Y00 and check T=D+C."""

    values = (float(y00), float(y10), float(y11))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Mechanistic outcome values must be finite")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    direct = values[1] - values[0]
    mediated = values[2] - values[1]
    total = values[2] - values[0]
    error = total - (direct + mediated)
    return MechanisticEffects(
        y00=values[0],
        y10=values[1],
        y11=values[2],
        direct=direct,
        mediated=mediated,
        total=total,
        additivity_error=error,
        additivity_passed=abs(error) <= tolerance,
    )


@dataclass(frozen=True)
class NullValidation:
    passed: bool
    unbatched_generation_valid: bool
    batched_generations_valid: bool
    unbatched_suffix_exact: bool
    batched_suffixes_exact: bool
    unbatched_scores_match: bool
    batched_scores_match: bool
    maximum_logprob_difference: float
    absolute_tolerance: float
    relative_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "unbatched_generation_valid": self.unbatched_generation_valid,
            "batched_generations_valid": self.batched_generations_valid,
            "unbatched_suffix_exact": self.unbatched_suffix_exact,
            "batched_suffixes_exact": self.batched_suffixes_exact,
            "unbatched_scores_match": self.unbatched_scores_match,
            "batched_scores_match": self.batched_scores_match,
            "maximum_logprob_difference": self.maximum_logprob_difference,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }


def _scores_match(
    baseline: AnswerScores,
    observed: AnswerScores,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[bool, float]:
    if set(baseline.label_logprobs) != set(observed.label_logprobs):
        return False, math.inf
    differences = [
        abs(float(observed.label_logprobs[label]) - float(baseline.label_logprobs[label]))
        for label in baseline.label_logprobs
    ]
    matched = baseline.predicted_answer == observed.predicted_answer and all(
        math.isclose(
            float(observed.label_logprobs[label]),
            float(baseline.label_logprobs[label]),
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        )
        for label in baseline.label_logprobs
    )
    return matched, max(differences, default=0.0)


def validate_null_intervention(
    *,
    baseline_suffix_ids: Sequence[int],
    unbatched_suffix_ids: Sequence[int],
    batched_suffix_ids: Sequence[Sequence[int]],
    baseline_scores: AnswerScores,
    unbatched_scores: AnswerScores,
    batched_scores: Sequence[AnswerScores],
    absolute_tolerance: float,
    relative_tolerance: float,
    unbatched_generation_valid: bool = True,
    batched_generations_valid: bool = True,
) -> NullValidation:
    """Require alpha-zero generation and scoring to reproduce baseline behavior."""

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Null tolerances cannot be negative")
    baseline_suffix = tuple(int(token_id) for token_id in baseline_suffix_ids)
    unbatched_suffix_exact = tuple(map(int, unbatched_suffix_ids)) == baseline_suffix
    batch_suffixes_exact = bool(batched_suffix_ids) and all(
        tuple(map(int, suffix)) == baseline_suffix for suffix in batched_suffix_ids
    )
    unbatched_match, unbatched_difference = _scores_match(
        baseline_scores,
        unbatched_scores,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    batch_matches = [
        _scores_match(
            baseline_scores,
            score,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for score in batched_scores
    ]
    batched_scores_match = bool(batch_matches) and all(match for match, _ in batch_matches)
    maximum_difference = max(
        [unbatched_difference, *(difference for _, difference in batch_matches)],
        default=0.0,
    )
    passed = (
        unbatched_generation_valid
        and batched_generations_valid
        and unbatched_suffix_exact
        and batch_suffixes_exact
        and unbatched_match
        and batched_scores_match
    )
    return NullValidation(
        passed=passed,
        unbatched_generation_valid=unbatched_generation_valid,
        batched_generations_valid=batched_generations_valid,
        unbatched_suffix_exact=unbatched_suffix_exact,
        batched_suffixes_exact=batch_suffixes_exact,
        unbatched_scores_match=unbatched_match,
        batched_scores_match=batched_scores_match,
        maximum_logprob_difference=maximum_difference,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
