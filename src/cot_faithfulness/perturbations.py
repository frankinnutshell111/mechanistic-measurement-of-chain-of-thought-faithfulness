"""Centralized construction and application of activation perturbations.

Hooks deliberately delegate all intervention arithmetic to this module. Changing
the perturbation rule therefore does not require changing hook lifecycle code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import MechanisticConfig
from .utils import stable_seed


class PerturbationError(RuntimeError):
    """Raised when a requested direction cannot be constructed safely."""


@dataclass(frozen=True)
class DirectionMetadata:
    direction_index: int
    direction_type: str
    seed: int | None
    direction_norm: float
    relative_direction_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_index": self.direction_index,
            "direction_type": self.direction_type,
            "seed": self.seed,
            "direction_norm": self.direction_norm,
            "relative_direction_norm": self.relative_direction_norm,
        }


@dataclass(frozen=True)
class PerturbationDirections:
    """Structured and orthogonal random directions for one layer/block."""

    values: Any
    metadata: tuple[DirectionMetadata, ...]
    activation_norm: float
    structured_direction_norm: float
    detectable: bool
    exclusion_reason: str | None

    @property
    def num_directions(self) -> int:
        return len(self.metadata)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "activation_norm": self.activation_norm,
            "structured_direction_norm": self.structured_direction_norm,
            "detectable": self.detectable,
            "exclusion_reason": self.exclusion_reason,
            "directions": [item.to_dict() for item in self.metadata],
        }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise RuntimeError("Install torch before constructing activation perturbations") from exc
    return torch


def _validate_activation_block(value: Any, name: str) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be a torch tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [block_length, hidden_size]")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} dimensions must be nonzero")


def frobenius_norm(value: Any) -> float:
    """Calculate a stable Frobenius norm in float32."""

    _validate_activation_block(value, "activation block")
    torch = _torch()
    return float(torch.linalg.vector_norm(value.detach().float()).cpu())


def structured_direction(neutral_block: Any, hinted_block: Any) -> Any:
    """Return U = H_hint - H_control in float32 for stable later arithmetic."""

    _validate_activation_block(neutral_block, "neutral_block")
    _validate_activation_block(hinted_block, "hinted_block")
    if tuple(neutral_block.shape) != tuple(hinted_block.shape):
        raise ValueError("Neutral and hinted activation blocks must have identical shapes")
    result = hinted_block.detach().float() - neutral_block.detach().float()
    if not bool(_torch().isfinite(result).all()):
        raise PerturbationError("Structured direction contains non-finite values")
    return result


def _sample_noise(
    shape: tuple[int, ...],
    *,
    distribution: str,
    generator: Any,
) -> Any:
    torch = _torch()
    if distribution == "rademacher":
        return (
            torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
            .float()
            .mul_(2)
            .sub_(1)
        )
    if distribution == "gaussian":
        return torch.randn(shape, generator=generator, dtype=torch.float32)
    raise ValueError(f"Unsupported random direction distribution: {distribution}")


def _orthogonalize(candidate: Any, basis: list[Any]) -> Any:
    """Use two-pass modified Gram-Schmidt to reduce float32 residual error."""

    result = candidate.flatten().float()
    for _ in range(2):
        for unit_vector in basis:
            result = result - torch_dot(result, unit_vector) * unit_vector
    return result


def torch_dot(left: Any, right: Any) -> Any:
    """Tiny indirection that keeps torch import lazy at module import time."""

    return _torch().dot(left, right)


def build_perturbation_directions(
    neutral_block: Any,
    hinted_block: Any,
    *,
    question_id: str,
    hint_type: str,
    layer_idx: int,
    block_idx: int,
    base_seed: int,
    config: MechanisticConfig,
) -> PerturbationDirections:
    """Construct the structured direction and norm-matched random controls."""

    torch = _torch()
    structured = structured_direction(neutral_block, hinted_block).cpu()
    activation_norm = frobenius_norm(neutral_block)
    structured_norm = float(torch.linalg.vector_norm(structured).cpu())
    denominator = activation_norm + config.direction_norm_epsilon
    structured_metadata = DirectionMetadata(
        direction_index=0,
        direction_type="structured",
        seed=None,
        direction_norm=structured_norm,
        relative_direction_norm=structured_norm / denominator,
    )
    if structured_norm <= config.direction_norm_epsilon:
        return PerturbationDirections(
            values=structured.unsqueeze(0),
            metadata=(structured_metadata,),
            activation_norm=activation_norm,
            structured_direction_norm=structured_norm,
            detectable=False,
            exclusion_reason="structured_direction_norm_below_threshold",
        )

    unit_structured = structured.flatten() / structured_norm
    basis = [unit_structured]
    directions = [structured]
    metadata = [structured_metadata]
    for direction_index in range(1, config.num_random_directions + 1):
        seed = stable_seed(
            base_seed,
            "activation_perturbation",
            question_id,
            hint_type,
            layer_idx,
            block_idx,
            direction_index,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        accepted = None
        for _ in range(config.gram_schmidt_max_attempts):
            candidate = _sample_noise(
                tuple(structured.shape),
                distribution=config.random_distribution,
                generator=generator,
            )
            candidate_norm = float(torch.linalg.vector_norm(candidate).cpu())
            orthogonal = _orthogonalize(candidate, basis)
            orthogonal_norm = float(torch.linalg.vector_norm(orthogonal).cpu())
            if orthogonal_norm > config.gram_schmidt_tolerance * max(candidate_norm, 1.0):
                accepted = orthogonal / orthogonal_norm
                break
        if accepted is None:
            raise PerturbationError(
                "Could not construct the requested number of numerically independent "
                "random directions"
            )
        basis.append(accepted)
        random_direction = accepted.reshape_as(structured) * structured_norm
        random_norm = float(torch.linalg.vector_norm(random_direction).cpu())
        directions.append(random_direction)
        metadata.append(
            DirectionMetadata(
                direction_index=direction_index,
                direction_type="random",
                seed=seed,
                direction_norm=random_norm,
                relative_direction_norm=random_norm / denominator,
            )
        )
    return PerturbationDirections(
        values=torch.stack(directions),
        metadata=tuple(metadata),
        activation_norm=activation_norm,
        structured_direction_norm=structured_norm,
        detectable=True,
        exclusion_reason=None,
    )


def apply_activation_perturbation(
    baseline_block: Any,
    perturbation: Any,
    *,
    alpha: float,
    operation: str = "additive",
) -> Any:
    """Apply the configured block calculation for hooks.

    ``additive`` interprets ``perturbation`` as U and returns H + alpha*U.
    ``replacement`` interprets it as a target H* and interpolates H + alpha*(H*-H).
    """

    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    if not hasattr(baseline_block, "shape") or baseline_block.ndim != 3:
        raise ValueError("baseline_block must have shape [batch, block_length, hidden_size]")
    if not hasattr(perturbation, "shape") or perturbation.ndim not in {2, 3}:
        raise ValueError("perturbation must have shape [block, hidden] or [batch, block, hidden]")
    values = perturbation
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if tuple(values.shape[1:]) != tuple(baseline_block.shape[1:]):
        raise ValueError("Perturbation block and target block shapes do not match")
    if values.shape[0] not in {1, baseline_block.shape[0]}:
        raise ValueError("Perturbation batch must be one or match the hidden-state batch")
    values = values.to(device=baseline_block.device, dtype=baseline_block.dtype)
    if values.shape[0] == 1 and baseline_block.shape[0] != 1:
        values = values.expand(baseline_block.shape[0], -1, -1)
    if float(alpha) == 0.0:
        return baseline_block.clone()
    if operation == "additive":
        return baseline_block + float(alpha) * values
    if operation == "replacement":
        return baseline_block + float(alpha) * (values - baseline_block)
    raise ValueError(f"Unknown perturbation operation: {operation}")
