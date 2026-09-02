"""
Job runners for GUI-driven transient fitting.

These functions are GUI-framework independent. They adapt validated GUI-side
data models and config objects to the numerical fitting drivers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..driver import fit_transient_exp
from ..driver.transient_result import TransientResult
from .fit_config import FitTransientExpConfig
from .models import TScanDataset, tscan_datasets_to_driver_inputs
from .validators import validate_t0_count_for_tscan, validate_tau_mask


def run_fit_transient_exp_config(
    config: FitTransientExpConfig,
    datasets: Sequence[TScanDataset],
) -> TransientResult:
    """Run fit_transient_exp from GUI config and datasets.

    Parameters
    ----------
    config
        Validated GUI-side configuration for fit_transient_exp.
    datasets
        One or more grouped time-scan datasets.

    Returns
    -------
    TransientResult
        Fitting result returned by fit_transient_exp.
    """
    datasets = list(datasets)

    if len(datasets) == 0:
        raise ValueError("At least one TScanDataset is required.")

    validate_t0_count_for_tscan(
        datasets,
        config.t0_init,
        same_t0=config.same_t0,
    )

    tau_mask = None
    if config.tau_init is not None:
        tau_mask = validate_tau_mask(
            datasets,
            config.tau_mask,
            n_tau=config.tau_init.size,
            base=config.base
        )
    elif config.tau_mask is not None:
        raise ValueError("tau_mask must be None when tau_init is None.")

    t, intensity, eps, name_of_dset = tscan_datasets_to_driver_inputs(datasets)

    result = fit_transient_exp(
        irf=config.irf,
        fwhm_init=_driver_fwhm_init(config),
        t0_init=config.t0_init,
        tau_init=config.tau_init,
        base=config.base,
        method_glb=config.method_glb,
        method_lsq=config.method_lsq,
        bound_fwhm=config.bound_fwhm,
        bound_t0=config.bound_t0,
        bound_tau=config.bound_tau,
        same_t0=config.same_t0,
        tau_mask=tau_mask,
        name_of_dset=name_of_dset,
        t=t,
        intensity=intensity,
        eps=eps,
    )

    return result


def _driver_fwhm_init(config: FitTransientExpConfig):
    """Return fwhm_init in the shape expected by fit_transient_exp.

    FitTransientExpConfig stores fwhm_init as a 1D NumPy array. The fitting
    driver accepts either a scalar for Gaussian/Cauchy IRF or a 2-value array
    for pseudo-Voigt IRF.
    """
    if config.irf in {"g", "c"}:
        return float(config.fwhm_init[0])

    return np.asarray(config.fwhm_init, dtype=float)