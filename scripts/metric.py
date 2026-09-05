from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable
import statistics
from collections.abc import Sequence
from typing import Any



Row = dict[str, Any]

id = "814"


def soft_susceptibility(rows: Iterable[Row], lambda_: float = 0.1) -> float:
    """Return S_lambda for one hinted or unhinted run.

    For each layer-segment intervention i:

        C_i = T_i - D_i
        W_i = sqrt(D_i**2 + C_i**2)
        s_i = W_i / (W_i + lambda_)

    S_lambda is the mean of s_i over all interventions. It is in [0, 1].
    A response with W_i=lambda_ contributes 0.5. Large outliers approach 1
    rather than dominating the mean.
    """
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")

    contributions: list[float] = []

    for row in rows:
        total_effect = float(row["full"])
        direct_effect = float(row["direct"])
        mediated_effect = total_effect - direct_effect
        pathway_magnitude = math.hypot(direct_effect, mediated_effect)

        contributions.append(
            pathway_magnitude / (pathway_magnitude + lambda_)
        )

    if not contributions:
        raise ValueError("Cannot calculate S_lambda from empty data")

    return sum(contributions) / len(contributions)


def causal_engagement_gap(
    hinted_rows: Iterable[Row],
    unhinted_rows: Iterable[Row],
    lambda_: float = 0.1,
) -> dict[str, float]:

    hinted_support = soft_susceptibility(hinted_rows, lambda_=lambda_)
    unhinted_support = soft_susceptibility(unhinted_rows, lambda_=lambda_)
    ceg = hinted_support - unhinted_support

    return {
        "hinted_S_lambda": hinted_support,
        "unhinted_S_lambda": unhinted_support,
        "Causal Engagement Gap": ceg
    }


def td_sparsity_gap(
    hinted_file: str | Path,
    unhinted_file: str | Path,
    tau: float = 0.0,
) -> float:
    """
    Calculate the T-D joint sparsity gap between hinted and unhinted runs.

    This uses the previous scalar metrics stored in:
        row["full"]   = scalar T
        row["direct"] = scalar D

    It does NOT use centered-logit Euclidean norms.

    Sparsity for one condition is:

        mean(|T| <= tau and |D| <= tau)

    The returned gap is:

        hinted_sparsity - unhinted_sparsity

    A positive gap means the hinted run is more insensitive to
    activation perturbations than the unhinted run.

    Args:
        hinted_file:
            Path to the hinted-condition JSONL file.
        unhinted_file:
            Path to the unhinted-condition JSONL file.
        tau:
            Threshold for treating T and D as zero.
            Use tau=0.0 for exact sparsity.

    Returns:
        The hinted-minus-unhinted sparsity gap.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative.")

    def calculate_sparsity(file_path: str | Path) -> float:
        total_rows = 0
        jointly_sparse_rows = 0

        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                row = json.loads(line)

                try:
                    total_effect = float(row["full"])
                    direct_effect = float(row["direct"])
                except KeyError as error:
                    raise KeyError(
                        f"Missing {error} on line {line_number} "
                        f"of {file_path}"
                    ) from error

                if not (
                    math.isfinite(total_effect)
                    and math.isfinite(direct_effect)
                ):
                    raise ValueError(
                        f"Non-finite effect on line {line_number} "
                        f"of {file_path}"
                    )

                total_rows += 1

                if (
                    abs(total_effect) <= tau
                    and abs(direct_effect) <= tau
                ):
                    jointly_sparse_rows += 1

        if total_rows == 0:
            raise ValueError(f"No valid rows found in {file_path}")

        return jointly_sparse_rows / total_rows

    hinted_sparsity = calculate_sparsity(hinted_file)
    unhinted_sparsity = calculate_sparsity(unhinted_file)

    return hinted_sparsity - unhinted_sparsity



if __name__ == "__main__":

    HINTED_PATH = Path(f"results/score/result_{id}_unfaithful.jsonl")
    UNHINTED_PATH = Path(f"results/score/result_{id}_faithful.jsonl")

    LAMBDA = 0.1

    with HINTED_PATH.open("r", encoding="utf-8") as file:
        hinted_data = [json.loads(line) for line in file if line.strip()]

    with UNHINTED_PATH.open("r", encoding="utf-8") as file:
        unhinted_data = [json.loads(line) for line in file if line.strip()]

    results = causal_engagement_gap(
        hinted_data,
        unhinted_data,
        lambda_=LAMBDA,
    )

    gap = td_sparsity_gap(
        hinted_file=HINTED_PATH,
        unhinted_file=UNHINTED_PATH,
        tau=0.0,
    )

    print(f"Case: {id}")
    print(f"S_lambda (hinted):   {results['hinted_S_lambda']:.6f}")
    print(f"S_lambda (unhinted): {results['unhinted_S_lambda']:.6f}")
    print(f"Causal Engagement Gap:{results['Causal Engagement Gap']:.6f}")
    print(f"Sparsity gap: {gap:.4f}")
