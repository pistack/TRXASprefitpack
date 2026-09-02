"""
Validation helpers for GUI-side configuration objects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import TScanDataset


def expected_t0_count_for_tscan(
    datasets: Sequence[TScanDataset],
    same_t0: bool,
) -> int:
    """Return expected t0 count for time-scan fitting."""
    if len(datasets) == 0:
        raise ValueError("At least one TScanDataset is required.")

    if same_t0:
        return len(datasets)

    return sum(dataset.n_trace for dataset in datasets)


def validate_t0_count_for_tscan(
    datasets: Sequence[TScanDataset],
    t0_init,
    same_t0: bool,
) -> None:
    """Validate t0_init length for GUI time-scan datasets."""
    t0_init = np.atleast_1d(np.asarray(t0_init, dtype=float))
    expected = expected_t0_count_for_tscan(datasets, same_t0)

    if t0_init.size != expected:
        raise ValueError(
            f"t0_init must contain {expected} value(s) for same_t0={same_t0}; "
            f"got {t0_init.size}."
        )


def validate_tau_array(tau, *, allow_none: bool = False) -> np.ndarray | None:
    """Validate lifetime array."""
    if tau is None:
        if allow_none:
            return None
        raise ValueError("tau must not be None.")

    tau = np.atleast_1d(np.asarray(tau, dtype=float))

    if tau.ndim != 1:
        raise ValueError("tau must be a 1D array.")

    if tau.size == 0:
        if allow_none:
            return None
        raise ValueError("tau must contain at least one value.")

    if not np.all(np.isfinite(tau)):
        raise ValueError("tau must contain only finite values.")

    if np.any(tau <= 0):
        raise ValueError("tau must contain only positive values.")

    return tau


def validate_bounds(
    init_values,
    bounds: Sequence[tuple[float, float]] | None,
    name: str,
) -> None:
    """Validate bounds length and ordering."""
    init_values = np.atleast_1d(np.asarray(init_values, dtype=float))

    if bounds is None:
        return

    if len(bounds) != init_values.size:
        raise ValueError(
            f"{name} bounds must have length {init_values.size}; "
            f"got {len(bounds)}."
        )

    for idx, (lower, upper) in enumerate(bounds):
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"{name} bounds[{idx}] must be finite.")

        if lower > upper:
            raise ValueError(
                f"{name} bounds[{idx}] lower value must be <= upper value."
            )

        value = init_values[idx]
        if value < lower or value > upper:
            raise ValueError(
                f"{name}[{idx}]={value} is outside bounds ({lower}, {upper})."
            )


def validate_tau_mask(
    datasets: Sequence[TScanDataset],
    tau_mask,
    n_tau: int,
    base: bool = False
) -> list[np.ndarray] | None:
    """Validate tau_mask for transient decay fitting.

    Expected convention:
        one boolean array per dataset, each with shape (n_tau,)
    """
    if tau_mask is None:
        return None

    if n_tau < 0:
        raise ValueError("n_tau must be non-negative.")

    if len(tau_mask) != len(datasets):
        raise ValueError(
            f"tau_mask must contain one mask per dataset; "
            f"got {len(tau_mask)} masks for {len(datasets)} datasets."
        )

    expected_size = n_tau + int(base)
    out: list[np.ndarray] = []

    for idx, mask in enumerate(tau_mask):
        mask = np.asarray(mask, dtype=bool)

        if mask.ndim != 1:
            raise ValueError(f"tau_mask[{idx}] must be a 1D boolean array.")

        if mask.size != expected_size:
            raise ValueError(
                f"tau_mask[{idx}] must have length {expected_size}; got {mask.size}."
            )

        out.append(mask)

    return out