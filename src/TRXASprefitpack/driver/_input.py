"""
Input normalization and validation utilities for transient fitting drivers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_dataset_sequence(values, name: str) -> list[np.ndarray]:
    """Convert a single array or a sequence of arrays into a dataset list."""
    if values is None:
        raise ValueError(f"{name} must not be None.")

    if isinstance(values, np.ndarray):
        return [np.asarray(values)]

    try:
        return [np.asarray(v) for v in values]
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an array or a sequence of arrays."
        ) from exc


def normalize_tscan_inputs(t, intensity, eps):
    """Normalize transient scan inputs to driver/residual conventions.

    The residual functions expect each dataset to have the form:

    - t[i]:          shape (n_time,)
    - intensity[i]:  shape (n_time, n_trace)
    - eps[i]:        shape (n_time, n_trace)

    A single 1D trace is accepted and converted to shape (n_time, 1).
    A single dataset array is accepted and converted to a one-element list.

    Parameters
    ----------
    t : array-like or sequence of array-like
        Time axes.
    intensity : array-like or sequence of array-like
        Intensity data. Each item may be 1D or 2D.
    eps : array-like or sequence of array-like
        Estimated errors. Each item may be 1D or 2D.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]
        Normalized ``t``, ``intensity``, and ``eps`` lists.
    """
    t_list = _as_dataset_sequence(t, "t")
    intensity_list = _as_dataset_sequence(intensity, "intensity")
    eps_list = _as_dataset_sequence(eps, "eps")

    if not (len(t_list) == len(intensity_list) == len(eps_list)):
        raise ValueError(
            "t, intensity, and eps must contain the same number of datasets; "
            f"got {len(t_list)}, {len(intensity_list)}, and {len(eps_list)}."
        )

    normalized_t = []
    normalized_intensity = []
    normalized_eps = []

    for idx, (ti, yi, ei) in enumerate(zip(t_list, intensity_list, eps_list)):
        ti = np.asarray(ti, dtype=float)
        yi = np.asarray(yi, dtype=float)
        ei = np.asarray(ei, dtype=float)

        if ti.ndim != 1:
            raise ValueError(f"t[{idx}] must be a 1D array.")

        if yi.ndim == 1:
            yi = yi.reshape(-1, 1)
        elif yi.ndim != 2:
            raise ValueError(
                f"intensity[{idx}] must be a 1D or 2D array; "
                f"got ndim={yi.ndim}."
            )

        if ei.ndim == 1:
            ei = ei.reshape(-1, 1)
        elif ei.ndim != 2:
            raise ValueError(
                f"eps[{idx}] must be a 1D or 2D array; "
                f"got ndim={ei.ndim}."
            )

        if yi.shape != ei.shape:
            raise ValueError(
                f"intensity[{idx}] and eps[{idx}] must have the same shape; "
                f"got {yi.shape} and {ei.shape}."
            )

        if yi.shape[0] != ti.size:
            raise ValueError(
                f"intensity[{idx}].shape[0] must match t[{idx}].size; "
                f"got {yi.shape[0]} and {ti.size}."
            )

        if np.any(ei <= 0):
            raise ValueError(f"eps[{idx}] must contain only positive values.")

        normalized_t.append(ti)
        normalized_intensity.append(yi)
        normalized_eps.append(ei)

    return normalized_t, normalized_intensity, normalized_eps


def expected_t0_count(intensity: Sequence[np.ndarray], same_t0: bool) -> int:
    """Return the required number of t0 parameters."""
    if same_t0:
        return len(intensity)

    return sum(np.asarray(data).shape[1] for data in intensity)


def validate_t0_count(t0_init, intensity: Sequence[np.ndarray], same_t0: bool) -> None:
    """Validate the number of initial t0 parameters."""
    t0_init = np.atleast_1d(t0_init)
    expected = expected_t0_count(intensity, same_t0)

    if t0_init.size != expected:
        raise ValueError(
            f"t0_init must contain {expected} value(s) for same_t0={same_t0}, "
            f"got {t0_init.size}."
        )