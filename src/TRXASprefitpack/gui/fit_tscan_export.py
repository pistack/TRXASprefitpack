"""Export helpers for fit_tscan_qt results."""

from __future__ import annotations

import csv
from pathlib import Path

from ..driver.transient_result import TransientResult
from .result_views import (
    transient_result_to_fit_plot_arrays,
    transient_result_to_parameter_rows,
    transient_result_to_report_text,
    transient_result_to_residual_plot_arrays,
)


def export_parameter_csv(
    result: TransientResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    path = _prepare_path(path, overwrite)
    rows = transient_result_to_parameter_rows(result)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "name",
                "value",
                "error",
                "lower_bound",
                "upper_bound",
                "fixed",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def export_fit_csv(
    result: TransientResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    path = _prepare_path(path, overwrite)
    entries = transient_result_to_fit_plot_arrays(result)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "dataset",
                "trace",
                "time",
                "intensity",
                "eps",
                "fit",
            )
        )

        for entry in entries:
            for values in zip(
                entry["time"],
                entry["intensity"],
                entry["eps"],
                entry["fit"],
            ):
                writer.writerow(
                    (
                        entry["dataset_name"],
                        entry["trace_name"],
                        *values,
                    )
                )

    return path


def export_residual_csv(
    result: TransientResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    path = _prepare_path(path, overwrite)
    entries = transient_result_to_residual_plot_arrays(result)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "dataset",
                "trace",
                "time",
                "residual",
                "eps",
                "standardized_residual",
            )
        )

        for entry in entries:
            for values in zip(
                entry["time"],
                entry["residual"],
                entry["eps"],
                entry["standardized_residual"],
            ):
                writer.writerow(
                    (
                        entry["dataset_name"],
                        entry["trace_name"],
                        *values,
                    )
                )

    return path


def export_report_txt(
    result: TransientResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    path = _prepare_path(path, overwrite)

    path.write_text(
        transient_result_to_report_text(result),
        encoding="utf-8",
    )
    return path


def _prepare_path(path, overwrite: bool) -> Path:
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return path