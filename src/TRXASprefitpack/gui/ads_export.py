"""Export helpers for calc_dads_qt results."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .ads_config import ADSResult
from .result_views import (
    ads_result_to_report_text,
    ads_result_to_summary_rows,
)


def export_ads_spectra_csv(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export energy and associated spectra without uncertainty columns."""
    _require_result(result)
    rows = zip(result.energy, *result.spectra.T)
    return _write_csv(
        path,
        ("energy", *result.spectrum_names),
        rows,
        overwrite=overwrite,
    )


def export_ads_spectra_error_csv(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export associated-spectrum uncertainties when available."""
    _require_result(result)
    if result.spectra_eps is None:
        raise ValueError("Associated-spectrum errors are unavailable.")
    rows = zip(result.energy, *result.spectra_eps.T)
    return _write_csv(
        path,
        (
            "energy",
            *(f"{name}_eps" for name in result.spectrum_names),
        ),
        rows,
        overwrite=overwrite,
    )


def export_ads_fit_csv(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export the reconstructed energy-by-time matrix."""
    _require_result(result)
    return _export_energy_time_matrix(
        result.energy,
        result.time,
        result.fit,
        path,
        overwrite=overwrite,
    )


def export_ads_residual_csv(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export the raw residual matrix, intensity minus reconstruction."""
    _require_result(result)
    return _export_energy_time_matrix(
        result.energy,
        result.time,
        result.intensity - result.fit,
        path,
        overwrite=overwrite,
    )


def export_ads_svd_csv(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export U, singular values, and Vh as one tidy CSV table."""
    _require_result(result)
    if not result.has_svd:
        raise ValueError("SVD arrays are unavailable.")
    assert result.svd_u is not None
    assert result.svd_s is not None
    assert result.svd_vh is not None

    rows: list[tuple[Any, ...]] = []
    for energy_index, energy in enumerate(result.energy):
        for component in range(result.svd_s.size):
            rows.append(
                (
                    "U",
                    energy_index,
                    component,
                    energy,
                    component + 1,
                    result.svd_u[energy_index, component],
                )
            )
    for component, singular_value in enumerate(result.svd_s):
        rows.append(
            (
                "S",
                component,
                "",
                component + 1,
                "",
                singular_value,
            )
        )
    for component in range(result.svd_s.size):
        for time_index, time in enumerate(result.time):
            rows.append(
                (
                    "Vh",
                    component,
                    time_index,
                    component + 1,
                    time,
                    result.svd_vh[component, time_index],
                )
            )

    return _write_csv(
        path,
        ("array", "row", "column", "axis_0", "axis_1", "value"),
        rows,
        overwrite=overwrite,
    )


def export_ads_summary_csv(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export the result summary table."""
    _require_result(result)
    rows = (
        (row["name"], _csv_value(row["value"]))
        for row in ads_result_to_summary_rows(result)
    )
    return _write_csv(
        path,
        ("name", "value"),
        rows,
        overwrite=overwrite,
    )


def export_ads_report_txt(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export the human-readable result report."""
    _require_result(result)
    output = _prepare_path(path, overwrite)
    output.write_text(
        ads_result_to_report_text(result) + "\n",
        encoding="utf-8",
    )
    return output


def export_ads_figure(
    figure,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export a Matplotlib-compatible figure as PNG or PDF."""
    if not hasattr(figure, "savefig"):
        raise TypeError("figure must provide a savefig() method.")
    output = Path(path)
    if output.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("Figure path must end in .png or .pdf.")
    output = _prepare_path(output, overwrite)
    figure.savefig(output)
    return output


def export_ads_rate_model_json(
    result: ADSResult,
    path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export the original custom rate-model specification."""
    _require_result(result)
    metadata = result.model_metadata or {}
    model = metadata.get("rate_model")
    if metadata.get("rate_model_kind") != "custom" or not isinstance(
        model, Mapping
    ):
        raise ValueError("A custom rate model is unavailable.")
    output = _prepare_path(path, overwrite)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(model, stream, indent=2)
        stream.write("\n")
    return output


def export_ads_bundle(
    result: ADSResult,
    directory,
    *,
    stem: str = "ads_result",
    figure=None,
    figure_formats: Iterable[str] = ("png",),
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export every result artifact applicable to one ADSResult."""
    _require_result(result)
    if (
        not stem
        or stem in {".", ".."}
        or "/" in stem
        or "\\" in stem
        or Path(stem).name != stem
    ):
        raise ValueError("stem must be a non-empty file-name stem.")
    target = Path(directory)
    formats = tuple(str(item).lower().lstrip(".") for item in figure_formats)
    if any(item not in {"png", "pdf"} for item in formats):
        raise ValueError("figure_formats may contain only 'png' or 'pdf'.")
    if figure is None and formats:
        formats = ()

    paths = {
        "spectra": target / f"{stem}_spectra.csv",
        "fit": target / f"{stem}_fit.csv",
        "residual": target / f"{stem}_residual.csv",
        "summary": target / f"{stem}_summary.csv",
        "report": target / f"{stem}_report.txt",
    }
    if result.spectra_eps is not None:
        paths["spectra_errors"] = target / f"{stem}_spectra_errors.csv"
    if result.has_svd:
        paths["svd"] = target / f"{stem}_svd.csv"
    metadata = result.model_metadata or {}
    if (
        metadata.get("rate_model_kind") == "custom"
        and isinstance(metadata.get("rate_model"), Mapping)
    ):
        paths["rate_model"] = target / f"{stem}_rate_model.json"
    for image_format in formats:
        paths[f"figure_{image_format}"] = (
            target / f"{stem}_figure.{image_format}"
        )

    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"File already exists: {existing[0]}")

    target.mkdir(parents=True, exist_ok=True)
    export_ads_spectra_csv(result, paths["spectra"], overwrite=True)
    export_ads_fit_csv(result, paths["fit"], overwrite=True)
    export_ads_residual_csv(result, paths["residual"], overwrite=True)
    export_ads_summary_csv(result, paths["summary"], overwrite=True)
    export_ads_report_txt(result, paths["report"], overwrite=True)
    if "spectra_errors" in paths:
        export_ads_spectra_error_csv(
            result, paths["spectra_errors"], overwrite=True
        )
    if "svd" in paths:
        export_ads_svd_csv(result, paths["svd"], overwrite=True)
    if "rate_model" in paths:
        export_ads_rate_model_json(
            result, paths["rate_model"], overwrite=True
        )
    for image_format in formats:
        export_ads_figure(
            figure,
            paths[f"figure_{image_format}"],
            overwrite=True,
        )
    return paths


def _export_energy_time_matrix(
    energy: np.ndarray,
    time: np.ndarray,
    matrix: np.ndarray,
    path,
    *,
    overwrite: bool,
) -> Path:
    rows = zip(energy, *matrix.T)
    return _write_csv(
        path,
        ("energy", *(f"time={value:.12g}" for value in time)),
        rows,
        overwrite=overwrite,
    )


def _write_csv(path, header, rows, *, overwrite: bool) -> Path:
    output = _prepare_path(path, overwrite)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return output


def _prepare_path(path, overwrite: bool) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _require_result(result: ADSResult) -> None:
    if not isinstance(result, ADSResult):
        raise TypeError("result must be an ADSResult.")


def _csv_value(value) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list, dict, np.ndarray)):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return json.dumps(value)
    return value
