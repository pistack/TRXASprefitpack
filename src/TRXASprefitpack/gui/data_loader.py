"""
File-loading utilities for GUI-side data models.

These functions are GUI-framework independent. They should not import PyQt5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .models import EScanDataset, TScanDataset, TScanTrace


def read_tscan_trace(path: str | Path, *, name: str | None = None) -> TScanTrace:
    """Read one time-delay trace from a text file.

    Expected columns:
        column 0: time
        column 1: intensity
        column 2: eps

    Extra columns are ignored.
    """
    path = Path(path)
    data = np.genfromtxt(path)

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"{path.name} must contain at least three columns: "
            "time, intensity, eps."
        )

    trace_name = name if name is not None else path.stem

    return TScanTrace(
        path=path,
        name=trace_name,
        t=data[:, 0],
        intensity=data[:, 1],
        eps=data[:, 2],
    )


def make_tscan_dataset_from_files(
    name: str,
    paths: Sequence[str | Path],
) -> TScanDataset:
    """Create one TScanDataset from multiple trace files."""
    if len(paths) == 0:
        raise ValueError("At least one trace file is required.")

    traces = tuple(read_tscan_trace(path) for path in paths)
    return TScanDataset(name=name, traces=traces)


def read_escan_dataset(
    intensity_matrix_path: str | Path,
    time_path: str | Path,
    eps_matrix_path: str | Path,
    *,
    name: str | None = None,
) -> EScanDataset:
    """Read one energy-scan matrix dataset for DADS/SADS workflows.

    Expected intensity matrix format:
        column 0: energy
        column 1..N: intensity at each time delay

    Expected time file:
        1D time-delay array, shape (n_time,)

    Expected eps matrix:
        2D array, shape (n_energy, n_time)
    """
    intensity_matrix_path = Path(intensity_matrix_path)
    time_path = Path(time_path)
    eps_matrix_path = Path(eps_matrix_path)

    raw = np.genfromtxt(intensity_matrix_path)
    time = np.genfromtxt(time_path)
    eps = np.genfromtxt(eps_matrix_path)

    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(
            f"{intensity_matrix_path.name} must contain energy in the first "
            "column and at least one intensity column."
        )

    energy = raw[:, 0]
    intensity = raw[:, 1:]

    dataset_name = name if name is not None else intensity_matrix_path.stem

    return EScanDataset(
        name=dataset_name,
        energy=energy,
        time=time,
        intensity=intensity,
        eps=eps,
        intensity_path=intensity_matrix_path,
        eps_path=eps_matrix_path,
        time_path=time_path,
    )