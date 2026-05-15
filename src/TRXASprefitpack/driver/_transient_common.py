"""
Common utilities for transient fitting driver routines.

This module contains behavior-preserving helper functions for driver-level
bookkeeping. It should not contain model-specific residual, gradient, Hessian,
or A-matrix construction logic.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from scipy.optimize import basinhopping

from ._ampgo import ampgo


GLBSOLVER = {"basinhopping": basinhopping, "ampgo": ampgo}


def validate_transient_driver_options(
    method_glb: Optional[str],
    method_lsq: str,
    irf: str,
) -> None:
    """Validate common transient fitting driver options."""
    if method_glb is not None and method_glb not in GLBSOLVER:
        raise ValueError(
            "Unsupported global optimization method. "
            "Supported methods are None, 'ampgo', and 'basinhopping'."
        )

    if method_lsq not in {"trf", "lm", "dogbox"}:
        raise ValueError(
            "Invalid local least-squares solver. "
            "method_lsq must be one of {'trf', 'lm', 'dogbox'}."
        )

    if irf not in {"g", "c", "pv"}:
        raise ValueError(
            "Unsupported instrumental response function. "
            "irf must be one of {'g', 'c', 'pv'}."
        )


def get_num_irf(irf: str) -> int:
    """Return the number of IRF parameters for a given IRF model."""
    if irf in {"g", "c"}:
        return 1
    if irf == "pv":
        return 2
    raise ValueError("irf must be one of {'g', 'c', 'pv'}.")


def make_fixed_mask(bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Return a boolean mask indicating fixed parameters."""
    return np.asarray([lower == upper for lower, upper in bounds], dtype=bool)


def make_lsq_bounds(
    bounds: Sequence[Tuple[float, float]],
    *,
    fixed_eps: float = 1e-8,
    fixed_abs_eps: float = 1e-16,
) -> tuple[list[float], list[float]]:
    """Convert parameter bounds to scipy.optimize.least_squares bounds.

    scipy.optimize.least_squares does not accept exactly identical lower and
    upper bounds. For fixed parameters, slightly expand the upper bound while
    keeping the lower bound unchanged. This preserves the existing driver
    behavior.
    """
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for lower, upper in bounds:
        lower_bounds.append(lower)

        if lower == upper:
            if upper > 0:
                upper = upper * (1.0 + fixed_eps) + fixed_abs_eps
            else:
                upper = upper * (1.0 - fixed_eps) + fixed_abs_eps

        upper_bounds.append(upper)

    return lower_bounds, upper_bounds


def prepare_global_kwargs(
    kwargs_glb: Optional[dict[str, Any]],
    *,
    args: tuple[Any, ...],
    bounds: Sequence[Tuple[float, float]],
) -> dict[str, Any]:
    """Prepare kwargs for global optimization without mutating user input."""
    prepared = dict(kwargs_glb) if kwargs_glb is not None else {}

    min_go_kwargs = {
        "args": args,
        "jac": True,
        "bounds": bounds,
    }

    minimizer_kwargs = prepared.get("minimizer_kwargs")

    if minimizer_kwargs is None:
        prepared["minimizer_kwargs"] = min_go_kwargs
    else:
        minimizer_kwargs = dict(minimizer_kwargs)
        minimizer_kwargs["args"] = min_go_kwargs["args"]
        minimizer_kwargs["jac"] = min_go_kwargs["jac"]
        minimizer_kwargs["bounds"] = min_go_kwargs["bounds"]
        prepared["minimizer_kwargs"] = minimizer_kwargs

    return prepared


def run_global_optimization(
    method_glb: Optional[str],
    param: np.ndarray,
    *,
    args: tuple[Any, ...],
    bounds: Sequence[Tuple[float, float]],
    grad_func: Callable,
    grad_func_same_t0: Callable,
    same_t0: bool,
    kwargs_glb: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run optional global optimization and return an optimizer-like result.

    If method_glb is None, return a minimal result dict compatible with the
    existing driver code.
    """
    if method_glb is None:
        return {
            "x": param,
            "message": None,
            "nfev": 0,
        }

    prepared_kwargs = prepare_global_kwargs(
        kwargs_glb,
        args=args,
        bounds=bounds,
    )

    grad = grad_func_same_t0 if same_t0 else grad_func
    return GLBSOLVER[method_glb](grad, param, **prepared_kwargs)


def prepare_lsq_kwargs(
    kwargs_lsq: Optional[dict[str, Any]],
    *,
    args: tuple[Any, ...],
) -> dict[str, Any]:
    """Prepare kwargs for scipy.optimize.least_squares.

    The input dictionary is copied to avoid mutating user-provided kwargs.
    Existing 'args' and 'kwargs' entries are intentionally overwritten to
    preserve the current driver behavior.
    """
    prepared = dict(kwargs_lsq) if kwargs_lsq is not None else {}
    prepared.pop("args", None)
    prepared.pop("kwargs", None)
    prepared["args"] = args
    return prepared


def calc_individual_chi2(
    chi: np.ndarray,
    intensity: Sequence[np.ndarray],
    *,
    num_param_ind: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-dataset/per-trace chi2 and reduced chi2."""
    start = 0
    chi2_ind = np.empty(len(intensity), dtype=object)
    red_chi2_ind = np.empty(len(intensity), dtype=object)

    for i, data in enumerate(intensity):
        step = data.shape[0]
        n_trace = data.shape[1]

        chi2_aux = np.empty(n_trace, dtype=float)

        for j in range(n_trace):
            end = start + step
            chi2_aux[j] = np.sum(chi[start:end] ** 2)
            start = end

        chi2_ind[i] = chi2_aux
        red_chi2_ind[i] = chi2_aux / (step - num_param_ind)

    return chi2_ind, red_chi2_ind


def calc_covariance_from_hessian(
    hessian: np.ndarray,
    fixed_mask: np.ndarray,
    red_chi2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate covariance, scaled covariance, correlation, and parameter error.

    Parameters fixed by identical lower/upper bounds are excluded from the
    matrix inversion and left as zero in the covariance matrix.
    """
    cov = np.zeros_like(hessian, dtype=float)

    free_mask = ~fixed_mask
    n_free_param = np.sum(free_mask)

    if n_free_param == 0:
        cov_scaled = red_chi2 * cov
        param_eps = np.sqrt(np.diag(cov_scaled))
        corr = cov_scaled.copy()
        return cov, cov_scaled, corr, param_eps

    mask_2d = np.einsum("i,j->ij", free_mask, free_mask)

    hessian_free = hessian[mask_2d].reshape((n_free_param, n_free_param))
    cov_free = np.linalg.inv(hessian_free)

    cov[mask_2d] = cov_free.flatten()

    cov_scaled = red_chi2 * cov
    param_eps = np.sqrt(np.diag(cov_scaled))

    corr = cov_scaled.copy()
    weight = np.einsum("i,j->ij", param_eps, param_eps)

    valid = mask_2d & (weight != 0)
    corr[valid] = corr[valid] / weight[valid]

    return cov, cov_scaled, corr, param_eps


def default_dataset_names(n_dataset: int) -> np.ndarray:
    """Create default dataset names."""
    names = np.empty(n_dataset, dtype=object)
    for i in range(n_dataset):
        names[i] = f"dataset_{i + 1}"
    return names


def count_total_scans(intensity: Sequence[np.ndarray]) -> int:
    """Return the total number of traces/scans across all datasets."""
    return sum(data.shape[1] for data in intensity)