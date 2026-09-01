"""
GUI-independent result-view helpers.

These functions convert TransientResult and ADSResult objects into plain
rows, tables, strings, and NumPy plot arrays. PyQt5 widgets should use these
helpers instead of inspecting numerical result objects directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..driver.transient_result import TransientResult
from .ads_config import ADSResult


__all__ = [
    "transient_result_to_parameter_rows",
    "transient_result_to_fit_plot_arrays",
    "transient_result_to_residual_plot_arrays",
    "transient_result_to_report_text",
    "ads_result_to_spectra_table",
    "ads_result_to_plot_arrays",
    "ads_result_to_summary_rows",
    "ads_result_to_report_text",
]


def transient_result_to_parameter_rows(
    result: TransientResult,
) -> list[dict[str, Any]]:
    """Return fitted parameters as table rows.

    Each row contains:

    - name
    - value
    - error
    - lower_bound
    - upper_bound
    - fixed
    """
    _require_transient_result(result)
    _require_transient_keys(
        result,
        ("param_name", "x", "x_eps", "bounds"),
    )

    names = np.asarray(result["param_name"], dtype=object)
    values = np.asarray(result["x"], dtype=float)
    errors = np.asarray(result["x_eps"], dtype=float)
    bounds = np.asarray(result["bounds"], dtype=float)

    if names.ndim != 1:
        raise ValueError("param_name must be a 1D array.")

    if values.shape != names.shape:
        raise ValueError(
            "x and param_name must have the same shape."
        )

    if errors.shape != names.shape:
        raise ValueError(
            "x_eps and param_name must have the same shape."
        )

    expected_bounds_shape = (names.size, 2)

    if bounds.shape != expected_bounds_shape:
        raise ValueError(
            "bounds must have shape "
            f"{expected_bounds_shape}; got {bounds.shape}."
        )

    rows: list[dict[str, Any]] = []

    for name, value, error, bound in zip(
        names,
        values,
        errors,
        bounds,
    ):
        lower = float(bound[0])
        upper = float(bound[1])

        rows.append(
            {
                "name": str(name),
                "value": float(value),
                "error": float(error),
                "lower_bound": lower,
                "upper_bound": upper,
                "fixed": bool(lower == upper),
            }
        )

    return rows


def transient_result_to_fit_plot_arrays(
    result: TransientResult,
) -> list[dict[str, Any]]:
    """Return one fit-plot entry for each time-scan trace.

    Each returned item contains:

    - dataset_index
    - dataset_name
    - trace_index
    - trace_name
    - time
    - intensity
    - eps
    - fit
    """
    _require_transient_result(result)

    datasets = _transient_dataset_arrays(
        result,
        include_residual=False,
    )

    plot_entries: list[dict[str, Any]] = []

    for dataset in datasets:
        for trace_index in range(
            dataset["intensity"].shape[1]
        ):
            plot_entries.append(
                {
                    "dataset_index": dataset["dataset_index"],
                    "dataset_name": dataset["dataset_name"],
                    "trace_index": trace_index,
                    "trace_name": f"trace_{trace_index + 1}",
                    "time": dataset["time"].copy(),
                    "intensity": dataset["intensity"][
                        :, trace_index
                    ].copy(),
                    "eps": dataset["eps"][
                        :, trace_index
                    ].copy(),
                    "fit": dataset["fit"][
                        :, trace_index
                    ].copy(),
                }
            )

    return plot_entries


def transient_result_to_residual_plot_arrays(
    result: TransientResult,
) -> list[dict[str, Any]]:
    """Return one residual-plot entry for each time-scan trace.

    Both raw and standardized residuals are returned. Standardized residuals
    follow ``residual / eps``.
    """
    _require_transient_result(result)

    datasets = _transient_dataset_arrays(
        result,
        include_residual=True,
    )

    plot_entries: list[dict[str, Any]] = []

    for dataset in datasets:
        residual_matrix = dataset["residual"]

        for trace_index in range(
            residual_matrix.shape[1]
        ):
            eps = dataset["eps"][:, trace_index]
            residual = residual_matrix[:, trace_index]

            plot_entries.append(
                {
                    "dataset_index": dataset["dataset_index"],
                    "dataset_name": dataset["dataset_name"],
                    "trace_index": trace_index,
                    "trace_name": f"trace_{trace_index + 1}",
                    "time": dataset["time"].copy(),
                    "residual": residual.copy(),
                    "eps": eps.copy(),
                    "standardized_residual": (
                        residual / eps
                    ).copy(),
                }
            )

    return plot_entries


def transient_result_to_report_text(
    result: TransientResult,
) -> str:
    """Return the numerical driver's existing text report."""
    _require_transient_result(result)
    return str(result)


def ads_result_to_spectra_table(
    result: ADSResult,
) -> dict[str, Any]:
    """Return associated spectra as a column-oriented table description.

    The return value contains:

    ``columns``
        Tuple of stable column names.

    ``rows``
        List of tuples, one tuple per energy point.

    If spectra errors are available, each spectrum column is immediately
    followed by its ``_eps`` column.
    """
    _require_ads_result(result)

    columns: list[str] = ["energy"]

    for spectrum_name in result.spectrum_names:
        columns.append(str(spectrum_name))

        if result.spectra_eps is not None:
            columns.append(f"{spectrum_name}_eps")

    rows: list[tuple[float, ...]] = []

    for energy_index, energy in enumerate(result.energy):
        row: list[float] = [float(energy)]

        for component_index in range(result.n_component):
            row.append(
                float(
                    result.spectra[
                        energy_index,
                        component_index,
                    ]
                )
            )

            if result.spectra_eps is not None:
                row.append(
                    float(
                        result.spectra_eps[
                            energy_index,
                            component_index,
                        ]
                    )
                )

        rows.append(tuple(row))

    return {
        "columns": tuple(columns),
        "rows": rows,
    }


def ads_result_to_plot_arrays(
    result: ADSResult,
) -> dict[str, list[dict[str, Any]]]:
    """Return associated spectra and energy-scan fit plot arrays.

    The return value has two lists:

    ``spectra``
        One item per associated spectrum.

    ``fits``
        One item per measured time delay.
    """
    _require_ads_result(result)

    spectra_entries: list[dict[str, Any]] = []

    for component_index, spectrum_name in enumerate(
        result.spectrum_names
    ):
        spectrum_eps = None

        if result.spectra_eps is not None:
            spectrum_eps = result.spectra_eps[
                :,
                component_index,
            ].copy()

        spectra_entries.append(
            {
                "component_index": component_index,
                "name": str(spectrum_name),
                "energy": result.energy.copy(),
                "spectrum": result.spectra[
                    :,
                    component_index,
                ].copy(),
                "spectrum_eps": spectrum_eps,
            }
        )

    fit_entries: list[dict[str, Any]] = []

    for time_index, time_value in enumerate(result.time):
        intensity = result.intensity[:, time_index]
        fit = result.fit[:, time_index]

        fit_entries.append(
            {
                "time_index": time_index,
                "time": float(time_value),
                "energy": result.energy.copy(),
                "intensity": intensity.copy(),
                "eps": result.eps[:, time_index].copy(),
                "fit": fit.copy(),
                "residual": (intensity - fit).copy(),
            }
        )

    return {
        "spectra": spectra_entries,
        "fits": fit_entries,
    }


def ads_result_to_summary_rows(
    result: ADSResult,
) -> list[dict[str, Any]]:
    """Return stable key/value rows summarizing an ADS result."""
    _require_ads_result(result)

    rows: list[dict[str, Any]] = [
        {"name": "mode", "value": result.mode},
        {"name": "n_energy", "value": result.n_energy},
        {"name": "n_time", "value": result.n_time},
        {"name": "n_component", "value": result.n_component},
        {"name": "has_svd", "value": result.has_svd},
    ]

    metadata = result.model_metadata or {}

    metadata_keys = (
        "dataset_name",
        "rate_model_kind",
        "t0",
        "irf",
        "fwhm",
        "eta",
        "base",
        "cond_num",
        "exclude",
    )

    for key in metadata_keys:
        if key in metadata and metadata[key] is not None:
            rows.append(
                {
                    "name": key,
                    "value": _plain_summary_value(
                        metadata[key]
                    ),
                }
            )

    if result.has_svd:
        rows.append(
            {
                "name": "n_svd_component",
                "value": int(result.svd_s.size),
            }
        )

    rows.append(
        {
            "name": "spectrum_names",
            "value": tuple(result.spectrum_names),
        }
    )

    return rows


def ads_result_to_report_text(
    result: ADSResult,
) -> str:
    """Return a compact human-readable ADS report."""
    _require_ads_result(result)

    summary_rows = ads_result_to_summary_rows(result)

    lines = ["[ADS Result]"]

    for row in summary_rows:
        lines.append(
            f"{row['name']}: "
            f"{_format_report_value(row['value'])}"
        )

    if result.spectra_eps is None:
        lines.append("spectra_errors: unavailable")
    else:
        lines.append("spectra_errors: available")

    return "\n".join(lines)


def _transient_dataset_arrays(
    result: TransientResult,
    *,
    include_residual: bool,
) -> list[dict[str, Any]]:
    required_keys = [
        "name_of_dset",
        "t",
        "intensity",
        "eps",
        "fit",
    ]

    if include_residual:
        required_keys.append("res")

    _require_transient_keys(
        result,
        tuple(required_keys),
    )

    names = tuple(
        str(name)
        for name in result["name_of_dset"]
    )
    time_arrays = tuple(result["t"])
    intensity_arrays = tuple(result["intensity"])
    eps_arrays = tuple(result["eps"])
    fit_arrays = tuple(result["fit"])

    sequences: list[tuple | Sequence] = [
        time_arrays,
        intensity_arrays,
        eps_arrays,
        fit_arrays,
    ]

    if include_residual:
        residual_arrays = tuple(result["res"])
        sequences.append(residual_arrays)
    else:
        residual_arrays = ()

    n_dataset = len(names)

    for sequence in sequences:
        if len(sequence) != n_dataset:
            raise ValueError(
                "Transient result dataset sequences must "
                "have the same length."
            )

    datasets: list[dict[str, Any]] = []

    for dataset_index, dataset_name in enumerate(names):
        time = np.asarray(
            time_arrays[dataset_index],
            dtype=float,
        )
        intensity = np.asarray(
            intensity_arrays[dataset_index],
            dtype=float,
        )
        eps = np.asarray(
            eps_arrays[dataset_index],
            dtype=float,
        )
        fit = np.asarray(
            fit_arrays[dataset_index],
            dtype=float,
        )

        if time.ndim != 1:
            raise ValueError(
                f"t[{dataset_index}] must be a 1D array."
            )

        if intensity.ndim != 2:
            raise ValueError(
                f"intensity[{dataset_index}] must be a 2D array."
            )

        expected_shape = (
            time.size,
            intensity.shape[1],
        )

        if intensity.shape != expected_shape:
            raise ValueError(
                f"intensity[{dataset_index}] first dimension "
                "must match its time axis."
            )

        if eps.shape != intensity.shape:
            raise ValueError(
                f"eps[{dataset_index}] shape must match intensity."
            )

        if fit.shape != intensity.shape:
            raise ValueError(
                f"fit[{dataset_index}] shape must match intensity."
            )

        if np.any(eps <= 0):
            raise ValueError(
                f"eps[{dataset_index}] must contain "
                "only positive values."
            )

        dataset = {
            "dataset_index": dataset_index,
            "dataset_name": dataset_name,
            "time": time,
            "intensity": intensity,
            "eps": eps,
            "fit": fit,
        }

        if include_residual:
            residual = np.asarray(
                residual_arrays[dataset_index],
                dtype=float,
            )

            if residual.shape != intensity.shape:
                raise ValueError(
                    f"res[{dataset_index}] shape must "
                    "match intensity."
                )

            dataset["residual"] = residual

        datasets.append(dataset)

    return datasets


def _require_transient_result(
    result: TransientResult,
) -> None:
    if not isinstance(result, TransientResult):
        raise TypeError(
            "result must be a TransientResult."
        )


def _require_transient_keys(
    result: TransientResult,
    keys: tuple[str, ...],
) -> None:
    missing = [
        key
        for key in keys
        if key not in result
    ]

    if missing:
        raise ValueError(
            "TransientResult is missing required key(s): "
            + ", ".join(missing)
        )


def _require_ads_result(
    result: ADSResult,
) -> None:
    if not isinstance(result, ADSResult):
        raise TypeError("result must be an ADSResult.")


def _plain_summary_value(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return tuple(value.tolist())

    if isinstance(value, list):
        return tuple(value)

    return value


def _format_report_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"

    if isinstance(value, tuple):
        return ", ".join(
            _format_report_value(item)
            for item in value
        )

    return str(value)