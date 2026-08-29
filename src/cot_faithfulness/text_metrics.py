"""Pure text-level mediation metrics used by Phase 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def answer_contrast(
    label_logprobs: Mapping[str, float],
    hint_label: str,
    baseline_answer: str,
) -> float:
    """Return log p(hint) - log p(baseline answer) for one scored context."""

    if hint_label == baseline_answer:
        raise ValueError("Hint label and baseline answer must differ")
    try:
        return float(label_logprobs[hint_label]) - float(label_logprobs[baseline_answer])
    except KeyError as exc:
        raise ValueError(f"Missing answer-label score: {exc.args[0]}") from exc


@dataclass(frozen=True)
class TextMediationEffects:
    y_control_control: float
    y_hint_control: float
    y_control_hint: float
    direct_text: float
    mediated_text: float
    fraction_text_mediated: float
    absolute_direct_text: float
    absolute_mediated_text: float
    total_effect_magnitude: float
    effect_above_floor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "Y_control_control": self.y_control_control,
            "Y_hint_control": self.y_hint_control,
            "Y_control_hint": self.y_control_hint,
            "D_text": self.direct_text,
            "C_text": self.mediated_text,
            "F_text": self.fraction_text_mediated,
            "absolute_D_text": self.absolute_direct_text,
            "absolute_C_text": self.absolute_mediated_text,
            "total_effect_magnitude": self.total_effect_magnitude,
            "effect_above_floor": self.effect_above_floor,
        }


def compute_text_mediation_effects(
    *,
    y_control_control: float,
    y_hint_control: float,
    y_control_hint: float,
    epsilon: float,
    minimum_effect_magnitude: float = 0.0,
) -> TextMediationEffects:
    """Compute direct and reasoning-mediated crossed-context effects."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if minimum_effect_magnitude < 0:
        raise ValueError("minimum_effect_magnitude cannot be negative")
    direct = float(y_hint_control) - float(y_control_control)
    mediated = float(y_control_hint) - float(y_control_control)
    absolute_direct = abs(direct)
    absolute_mediated = abs(mediated)
    magnitude = absolute_direct + absolute_mediated
    return TextMediationEffects(
        y_control_control=float(y_control_control),
        y_hint_control=float(y_hint_control),
        y_control_hint=float(y_control_hint),
        direct_text=direct,
        mediated_text=mediated,
        fraction_text_mediated=absolute_mediated / (magnitude + epsilon),
        absolute_direct_text=absolute_direct,
        absolute_mediated_text=absolute_mediated,
        total_effect_magnitude=magnitude,
        effect_above_floor=magnitude >= minimum_effect_magnitude,
    )
