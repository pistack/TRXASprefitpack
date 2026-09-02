"""Pure Python helpers for fit_tscan_qt confidence interval scans."""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from TRXASprefitpack.driver.anal_fit import confidence_interval


SIGMA_LEVELS = (1.0, 2.0)


@dataclass(frozen=True)
class CIErrorRow:
    parameter_index: int
    parameter_name: str
    value: float
    minus_1sigma: float
    plus_1sigma: float
    minus_2sigma: float
    plus_2sigma: float


def sigma_to_alpha(sigma: float) -> float:
    """Convert a two-sided normal sigma level to significance alpha."""
    sigma = float(sigma)

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a finite positive number.")

    return float(2.0 * norm.sf(sigma))


def validate_parameter_indices(
    result,
    parameter_indices: Sequence[int],
) -> tuple[int, ...]:
    indices = np.asarray(parameter_indices)

    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(
            "At least one parameter must be selected."
        )

    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(
            "Parameter indices must be integers."
        )

    indices = indices.astype(int)
    num_parameters = len(result["x"])

    if np.any(indices < 0) or np.any(indices >= num_parameters):
        raise IndexError(
            "A selected parameter index is out of range."
        )

    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Selected parameter indices must be unique."
        )

    for index in indices:
        lower, upper = result["bounds"][index]

        if lower == upper:
            raise ValueError(
                f"Fixed parameter cannot be scanned: "
                f"{result['param_name'][index]}"
            )

    return tuple(int(index) for index in indices)


def run_selected_ci_scans(
    result,
    parameter_indices: Sequence[int],
    ci_runner: Callable = confidence_interval,
) -> dict[float, object]:
    """Calculate 1-sigma and 2-sigma profile confidence intervals."""
    indices = validate_parameter_indices(
        result,
        parameter_indices,
    )

    return {
        sigma: ci_runner(
            result,
            sigma_to_alpha(sigma),
            parameter_indices=indices,
        )
        for sigma in SIGMA_LEVELS
    }


def _interval_errors(interval) -> tuple[float, float]:
    lower_delta, upper_delta = interval

    if np.isnan(lower_delta) or np.isnan(upper_delta):
        return np.nan, np.nan

    return abs(float(lower_delta)), abs(float(upper_delta))


def ci_results_to_error_rows(
    result,
    parameter_indices: Sequence[int],
    ci_results: Mapping[float, object],
) -> list[CIErrorRow]:
    indices = validate_parameter_indices(
        result,
        parameter_indices,
    )

    ci_1sigma = ci_results[1.0]["ci"]
    ci_2sigma = ci_results[2.0]["ci"]

    rows = []

    for index in indices:
        minus_1sigma, plus_1sigma = _interval_errors(
            ci_1sigma[index]
        )
        minus_2sigma, plus_2sigma = _interval_errors(
            ci_2sigma[index]
        )

        rows.append(
            CIErrorRow(
                parameter_index=index,
                parameter_name=str(
                    result["param_name"][index]
                ),
                value=float(result["x"][index]),
                minus_1sigma=minus_1sigma,
                plus_1sigma=plus_1sigma,
                minus_2sigma=minus_2sigma,
                plus_2sigma=plus_2sigma,
            )
        )

    return rows


def ci_error_rows_to_report(
    rows: Sequence[CIErrorRow],
) -> str:
    lines = [
        "Profile confidence interval scan",
        "",
        "1 sigma: 68.268949% confidence",
        "2 sigma: 95.449974% confidence",
        "",
    ]

    for row in rows:
        if np.isnan(row.minus_1sigma):
            one_sigma = "not found"
        else:
            one_sigma = (
                f"{row.value:.8g} "
                f"-{row.minus_1sigma:.8g}/"
                f"+{row.plus_1sigma:.8g}"
            )

        if np.isnan(row.minus_2sigma):
            two_sigma = "not found"
        else:
            two_sigma = (
                f"{row.value:.8g} "
                f"-{row.minus_2sigma:.8g}/"
                f"+{row.plus_2sigma:.8g}"
            )

        lines.extend(
            [
                row.parameter_name,
                f"  1 sigma: {one_sigma}",
                f"  2 sigma: {two_sigma}",
            ]
        )

    return "\n".join(lines)