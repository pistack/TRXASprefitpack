"""
GUI-side data models for time-scan and energy-scan workflows.

These models are intentionally GUI-framework independent. They should not
import PyQt5. PyQt widgets can use these objects as validated data containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def _as_path_or_none(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _validate_positive_eps(eps: np.ndarray, name: str = "eps") -> None:
    if np.any(eps <= 0):
        raise ValueError(f"{name} must contain only positive values.")


@dataclass(frozen=True)
class TScanTrace:
    """Single time-delay trace loaded from one file.

    A trace is the smallest GUI-side unit for transient fitting.

    Parameters
    ----------
    path : str or Path
        Source file path.
    name : str
        Display name for this trace.
    t : array-like, shape (n_time,)
        Time-delay axis.
    intensity : array-like, shape (n_time,)
        Intensity values for this trace.
    eps : array-like, shape (n_time,)
        Estimated errors for this trace.
    """

    path: str | Path
    name: str
    t: np.ndarray
    intensity: np.ndarray
    eps: np.ndarray

    def __post_init__(self) -> None:
        path = Path(self.path)
        t = np.asarray(self.t, dtype=float)
        intensity = np.asarray(self.intensity, dtype=float)
        eps = np.asarray(self.eps, dtype=float)

        if not self.name:
            raise ValueError("name must not be empty.")

        if t.ndim != 1:
            raise ValueError(f"t must be a 1D array; got ndim={t.ndim}.")

        if intensity.ndim != 1:
            raise ValueError(
                f"intensity must be a 1D array for TScanTrace; "
                f"got ndim={intensity.ndim}."
            )

        if eps.ndim != 1:
            raise ValueError(
                f"eps must be a 1D array for TScanTrace; got ndim={eps.ndim}."
            )

        if intensity.shape != eps.shape:
            raise ValueError(
                "intensity and eps must have the same shape; "
                f"got {intensity.shape} and {eps.shape}."
            )

        if intensity.size != t.size:
            raise ValueError(
                "intensity length must match t length; "
                f"got {intensity.size} and {t.size}."
            )

        _validate_positive_eps(eps)

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "eps", eps)

    @property
    def n_time(self) -> int:
        return self.t.size


@dataclass(frozen=True)
class TScanDataset:
    """Collection of time traces sharing the same time axis.

    This is converted to the transient fitting driver convention:

    - t: shape (n_time,)
    - intensity: shape (n_time, n_trace)
    - eps: shape (n_time, n_trace)
    """

    name: str
    traces: tuple[TScanTrace, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        traces = tuple(self.traces)

        if not self.name:
            raise ValueError("name must not be empty.")

        if len(traces) == 0:
            raise ValueError("TScanDataset must contain at least one trace.")

        t0 = traces[0].t

        for trace in traces[1:]:
            if trace.t.shape != t0.shape or not np.allclose(trace.t, t0):
                raise ValueError(
                    "All traces in one TScanDataset must share the same time axis."
                )

        object.__setattr__(self, "traces", traces)

    @property
    def n_time(self) -> int:
        return self.traces[0].n_time

    @property
    def n_trace(self) -> int:
        return len(self.traces)

    @property
    def trace_names(self) -> tuple[str, ...]:
        return tuple(trace.name for trace in self.traces)

    @property
    def time_range(self) -> tuple[float, float]:
        t = self.traces[0].t
        return float(np.min(t)), float(np.max(t))

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return driver-compatible arrays.

        Returns
        -------
        t : np.ndarray, shape (n_time,)
        intensity : np.ndarray, shape (n_time, n_trace)
        eps : np.ndarray, shape (n_time, n_trace)
        """
        t = self.traces[0].t.copy()
        intensity = np.column_stack([trace.intensity for trace in self.traces])
        eps = np.column_stack([trace.eps for trace in self.traces])

        return t, intensity, eps


@dataclass(frozen=True)
class EScanDataset:
    """Energy-scan matrix dataset for DADS/SADS workflows.

    The matrix convention follows the existing calc_dads GUI and ADS drivers:

    - energy: shape (n_energy,)
    - time: shape (n_time,)
    - intensity: shape (n_energy, n_time)
    - eps: shape (n_energy, n_time)
    """

    name: str
    energy: np.ndarray
    time: np.ndarray
    intensity: np.ndarray
    eps: np.ndarray
    intensity_path: str | Path | None = None
    eps_path: str | Path | None = None
    time_path: str | Path | None = None

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy, dtype=float)
        time = np.asarray(self.time, dtype=float)
        intensity = np.asarray(self.intensity, dtype=float)
        eps = np.asarray(self.eps, dtype=float)

        if not self.name:
            raise ValueError("name must not be empty.")

        if energy.ndim != 1:
            raise ValueError(
                f"energy must be a 1D array; got ndim={energy.ndim}."
            )

        if time.ndim != 1:
            raise ValueError(f"time must be a 1D array; got ndim={time.ndim}.")

        if intensity.ndim != 2:
            raise ValueError(
                f"intensity must be a 2D array; got ndim={intensity.ndim}."
            )

        if eps.ndim != 2:
            raise ValueError(f"eps must be a 2D array; got ndim={eps.ndim}.")

        if intensity.shape != eps.shape:
            raise ValueError(
                "intensity and eps must have the same shape; "
                f"got {intensity.shape} and {eps.shape}."
            )

        expected_shape = (energy.size, time.size)
        if intensity.shape != expected_shape:
            raise ValueError(
                "intensity shape must be (n_energy, n_time); "
                f"got {intensity.shape}, expected {expected_shape}."
            )

        _validate_positive_eps(eps)

        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(
            self, "intensity_path", _as_path_or_none(self.intensity_path)
        )
        object.__setattr__(self, "eps_path", _as_path_or_none(self.eps_path))
        object.__setattr__(self, "time_path", _as_path_or_none(self.time_path))

    @property
    def n_energy(self) -> int:
        return self.energy.size

    @property
    def n_time(self) -> int:
        return self.time.size

    @property
    def energy_range(self) -> tuple[float, float]:
        return float(np.min(self.energy)), float(np.max(self.energy))

    @property
    def time_range(self) -> tuple[float, float]:
        return float(np.min(self.time)), float(np.max(self.time))


def tscan_datasets_to_driver_inputs(
    datasets: Sequence[TScanDataset],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Convert GUI TScanDataset objects to transient fitting driver inputs."""
    if len(datasets) == 0:
        raise ValueError("At least one TScanDataset is required.")

    t: list[np.ndarray] = []
    intensity: list[np.ndarray] = []
    eps: list[np.ndarray] = []
    names: list[str] = []

    for dataset in datasets:
        ti, yi, ei = dataset.to_arrays()
        t.append(ti)
        intensity.append(yi)
        eps.append(ei)
        names.append(dataset.name)

    return t, intensity, eps, np.asarray(names, dtype=object)