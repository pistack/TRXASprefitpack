"""
Parser utilities for GUI text inputs.

These functions are GUI-framework independent. They convert user-facing
string inputs into validated Python/NumPy objects.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..mathfun import calc_eta, calc_fwhm


VALID_IRF = {"g", "c", "pv"}


def parse_float(text: str, name: str) -> float:
    """Parse one float value from GUI text."""
    if text is None or str(text).strip() == "":
        raise ValueError(f"{name} must not be empty.")

    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a float value.") from exc

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def parse_positive_float(text: str, name: str) -> float:
    """Parse one positive float value."""
    value = parse_float(text, name)

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def parse_nonnegative_float(text: str, name: str) -> float:
    """Parse one non-negative float value."""
    value = parse_float(text, name)

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")

    return value


def parse_float_array(
    text: str,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray | None:
    """Parse comma-separated float values.

    Examples
    --------
    "1, 2, 3" -> np.array([1.0, 2.0, 3.0])
    "1e-3, 2.5E+1" -> np.array([1e-3, 25.0])
    """
    if text is None or str(text).strip() == "":
        if allow_empty:
            return None
        raise ValueError(f"{name} must not be empty.")

    fields = [field.strip() for field in str(text).split(",")]

    if any(field == "" for field in fields):
        raise ValueError(f"{name} contains an empty field.")

    try:
        values = np.asarray([float(field) for field in fields], dtype=float)
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain comma-separated float values."
        ) from exc

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")

    return values


def parse_positive_float_array(
    text: str,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray | None:
    """Parse comma-separated positive float values."""
    values = parse_float_array(text, name, allow_empty=allow_empty)

    if values is None:
        return None

    if np.any(values <= 0):
        raise ValueError(f"{name} must contain only positive values.")

    return values


def parse_irf(text: str) -> str:
    """Parse IRF identifier."""
    if text is None:
        raise ValueError("irf must not be empty.")

    irf = str(text).strip().lower()

    if irf not in VALID_IRF:
        raise ValueError("irf must be one of {'g', 'c', 'pv'}.")

    return irf


def parse_fwhm_eta(
    irf: str,
    fwhm_g_text: str,
    fwhm_l_text: str,
) -> tuple[float | np.ndarray, float | None]:
    """Parse IRF width inputs and return driver-compatible fwhm and eta.

    For Gaussian IRF, use fwhm_G.
    For Cauchy IRF, use fwhm_L.
    For pseudo-Voigt IRF, parse both fwhm_G and fwhm_L and convert them
    to effective fwhm and eta.
    """
    irf = parse_irf(irf)

    if irf == "g":
        return parse_positive_float(fwhm_g_text, "fwhm_G"), None

    if irf == "c":
        return parse_positive_float(fwhm_l_text, "fwhm_L"), None

    fwhm_g = parse_positive_float(fwhm_g_text, "fwhm_G")
    fwhm_l = parse_positive_float(fwhm_l_text, "fwhm_L")

    fwhm = calc_fwhm(fwhm_g, fwhm_l)
    eta = calc_eta(fwhm_g, fwhm_l)

    return fwhm, eta


def parse_bounds(
    lower_text: str,
    upper_text: str,
    init_values: Sequence[float] | np.ndarray,
    name: str,
) -> list[tuple[float, float]]:
    """Parse lower/upper bounds for parameter arrays.

    Empty lower or upper text means default unbounded side is not accepted here.
    For GUI fixed/default-bound workflows, pass explicit lower and upper values.

    A scalar lower/upper value is broadcast to all init_values.
    A comma-separated array must either have length 1 or len(init_values).
    """
    init_values = np.atleast_1d(np.asarray(init_values, dtype=float))

    lower = parse_float_array(lower_text, f"{name} lower bound")
    upper = parse_float_array(upper_text, f"{name} upper bound")

    assert lower is not None
    assert upper is not None

    lower = _broadcast_bound_values(lower, init_values.size, f"{name} lower bound")
    upper = _broadcast_bound_values(upper, init_values.size, f"{name} upper bound")

    if np.any(lower > upper):
        raise ValueError(f"{name} lower bounds must be <= upper bounds.")

    return [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]


def _broadcast_bound_values(
    values: np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    """Broadcast scalar bound arrays to expected size."""
    if values.size == size:
        return values

    if values.size == 1:
        return np.full(size, values[0], dtype=float)

    raise ValueError(f"{name} must have length 1 or {size}; got {values.size}.")