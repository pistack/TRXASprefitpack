"""
Configuration objects for GUI-driven transient fitting.

These config objects are GUI-framework independent. They validate user-facing
inputs before they are passed to fitting job runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .validators import validate_bounds, validate_tau_array


TransientModel = Literal["decay"]
IRFModel = Literal["g", "c", "pv"]
GlobalOptimizer = Literal["ampgo", "basinhopping"]
LeastSquaresMethod = Literal["trf", "dogbox", "lm"]


def _normalize_tau_mask_shape_only(
    tau_mask: Sequence[np.ndarray] | None,
    n_tau: int,
) -> list[np.ndarray] | None:
    """Validate tau_mask shape except dataset count.

    Dataset-count validation is performed later in the job runner because the
    config object does not know the selected datasets.
    """
    if tau_mask is None:
        return None

    out: list[np.ndarray] = []

    for idx, mask in enumerate(tau_mask):
        mask = np.asarray(mask, dtype=bool)

        if mask.ndim != 1:
            raise ValueError(f"tau_mask[{idx}] must be a 1D boolean array.")

        if mask.size != n_tau:
            raise ValueError(
                f"tau_mask[{idx}] must have length {n_tau}; got {mask.size}."
            )

        out.append(mask)

    return out


@dataclass(frozen=True)
class FitTransientExpConfig:
    """Configuration for fit_transient_exp GUI workflow.

    This config intentionally covers the standard GUI-exposed workflow only.
    Advanced optimizer kwargs are not represented here and should remain a
    Python API feature.
    """

    irf: IRFModel
    fwhm_init: float | np.ndarray
    t0_init: np.ndarray
    tau_init: np.ndarray | None
    base: bool
    method_glb: GlobalOptimizer | None = None
    method_lsq: LeastSquaresMethod = "trf"
    bound_fwhm: Sequence[tuple[float, float]] | None = None
    bound_t0: Sequence[tuple[float, float]] | None = None
    bound_tau: Sequence[tuple[float, float]] | None = None
    same_t0: bool = False
    tau_mask: Sequence[np.ndarray] | None = None

    def __post_init__(self) -> None:
        irf = str(self.irf).strip().lower()

        if irf not in {"g", "c", "pv"}:
            raise ValueError("irf must be one of {'g', 'c', 'pv'}.")

        if self.method_glb is not None and self.method_glb not in {
            "ampgo",
            "basinhopping",
        }:
            raise ValueError(
                "method_glb must be None, 'ampgo', or 'basinhopping'."
            )

        if self.method_lsq not in {"trf", "dogbox", "lm"}:
            raise ValueError("method_lsq must be one of {'trf', 'dogbox', 'lm'}.")

        fwhm_init = np.atleast_1d(np.asarray(self.fwhm_init, dtype=float))
        t0_init = np.atleast_1d(np.asarray(self.t0_init, dtype=float))

        if fwhm_init.ndim != 1:
            raise ValueError("fwhm_init must be a scalar or 1D array.")

        if t0_init.ndim != 1:
            raise ValueError("t0_init must be a scalar or 1D array.")

        expected_fwhm_size = 2 if irf == "pv" else 1
        if fwhm_init.size != expected_fwhm_size:
            raise ValueError(
                f"fwhm_init must contain {expected_fwhm_size} value(s) "
                f"for irf='{irf}', got {fwhm_init.size}."
            )

        if not np.all(np.isfinite(fwhm_init)):
            raise ValueError("fwhm_init must contain only finite values.")

        if np.any(fwhm_init <= 0):
            raise ValueError("fwhm_init must contain only positive values.")

        if not np.all(np.isfinite(t0_init)):
            raise ValueError("t0_init must contain only finite values.")

        tau_init = validate_tau_array(self.tau_init, allow_none=True)

        validate_bounds(fwhm_init, self.bound_fwhm, "fwhm")
        validate_bounds(t0_init, self.bound_t0, "t0")

        if tau_init is not None:
            validate_bounds(tau_init, self.bound_tau, "tau")
            tau_mask = _normalize_tau_mask_shape_only(
                self.tau_mask,
                tau_init.size,
            )
        else:
            if self.bound_tau is not None:
                raise ValueError("bound_tau must be None when tau_init is None.")
            if self.tau_mask is not None:
                raise ValueError("tau_mask must be None when tau_init is None.")
            tau_mask = None

        object.__setattr__(self, "irf", irf)
        object.__setattr__(self, "fwhm_init", fwhm_init)
        object.__setattr__(self, "t0_init", t0_init)
        object.__setattr__(self, "tau_init", tau_init)
        object.__setattr__(self, "base", bool(self.base))
        object.__setattr__(self, "same_t0", bool(self.same_t0))

        if self.bound_fwhm is not None:
            object.__setattr__(self, "bound_fwhm", list(self.bound_fwhm))
        if self.bound_t0 is not None:
            object.__setattr__(self, "bound_t0", list(self.bound_t0))
        if self.bound_tau is not None:
            object.__setattr__(self, "bound_tau", list(self.bound_tau))
        object.__setattr__(self, "tau_mask", tau_mask)


@dataclass(frozen=True)
class FitConfigBundle:
    """Transient fit config plus GUI-level metadata."""

    name: str
    config: FitTransientExpConfig

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty.")