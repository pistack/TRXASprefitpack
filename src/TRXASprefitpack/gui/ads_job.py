"""
Job runner for GUI-driven DADS and SADS calculations.

This module is independent of PyQt5. It adapts validated GUI configuration
and energy-scan datasets to the existing numerical ADS drivers.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from ..driver import dads, dads_svd, sads, sads_svd
from ..mathfun import solve_seq_model
from .ads_config import ADSConfig, ADSResult
from .models import EScanDataset
from .rate_model import (
    build_rate_matrix,
    solve_rate_model_real,
)


__all__ = ["run_ads_config"]


def run_ads_config(
    config: ADSConfig,
    dataset: EScanDataset,
) -> ADSResult:
    """Run one ADS calculation from validated GUI-side objects."""
    if not isinstance(config, ADSConfig):
        raise TypeError("config must be an ADSConfig.")

    if not isinstance(dataset, EScanDataset):
        raise TypeError("dataset must be an EScanDataset.")

    model_time = np.asarray(
        dataset.time - config.t0,
        dtype=float,
    )

    if config.mode == "dads":
        return _run_dads(
            config,
            dataset,
            model_time,
            use_svd=False,
        )

    if config.mode == "dads_svd":
        return _run_dads(
            config,
            dataset,
            model_time,
            use_svd=True,
        )

    if config.mode == "sads":
        return _run_standard_sads(
            config,
            dataset,
            model_time,
            use_svd=False,
        )

    if config.mode == "sads_svd":
        return _run_standard_sads(
            config,
            dataset,
            model_time,
            use_svd=True,
        )

    if config.mode == "custom_sads":
        return _run_custom_sads(
            config,
            dataset,
            model_time,
            use_svd=False,
        )

    if config.mode == "custom_sads_svd":
        return _run_custom_sads(
            config,
            dataset,
            model_time,
            use_svd=True,
        )

    raise ValueError(
        f"Unsupported ADS mode: {config.mode!r}."
    )


def _run_dads(
    config: ADSConfig,
    dataset: EScanDataset,
    model_time: np.ndarray,
    *,
    use_svd: bool,
) -> ADSResult:
    """Run DADS or DADS-SVD."""
    assert config.tau is not None

    spectrum_names = _dads_spectrum_names(
        config.tau.size,
        base=config.base,
    )

    if use_svd:
        spectra, fit = dads_svd(
            escan_time=model_time,
            fwhm=config.fwhm,
            tau=config.tau,
            base=config.base,
            irf=config.irf,
            eta=config.eta,
            intensity=dataset.intensity,
            cond_num=config.cond_num,
        )

        spectra = np.asarray(spectra, dtype=float)
        fit = np.asarray(fit, dtype=float)
        spectra_eps = None

        svd_u, svd_s, svd_vh = _truncated_data_svd(
            dataset.intensity,
            config.cond_num,
        )
    else:
        spectra_raw, spectra_eps_raw, fit = dads(
            escan_time=model_time,
            fwhm=config.fwhm,
            tau=config.tau,
            base=config.base,
            irf=config.irf,
            eta=config.eta,
            intensity=dataset.intensity,
            eps=dataset.eps,
        )

        # dads returns (n_component, n_energy).
        spectra = np.asarray(
            spectra_raw,
            dtype=float,
        ).T
        spectra_eps = np.asarray(
            spectra_eps_raw,
            dtype=float,
        ).T
        fit = np.asarray(fit, dtype=float)

        svd_u = None
        svd_s = None
        svd_vh = None

    metadata = {
        "dataset_name": dataset.name,
        "model_time": model_time.copy(),
        "t0": config.t0,
        "irf": config.irf,
        "fwhm": config.fwhm,
        "eta": config.eta,
        "tau": config.tau.copy(),
        "base": config.base,
        "cond_num": config.cond_num if use_svd else None,
    }

    return _make_ads_result(
        config=config,
        dataset=dataset,
        spectra=spectra,
        spectra_eps=spectra_eps,
        fit=fit,
        spectrum_names=spectrum_names,
        svd_u=svd_u,
        svd_s=svd_s,
        svd_vh=svd_vh,
        model_metadata=metadata,
    )


def _run_standard_sads(
    config: ADSConfig,
    dataset: EScanDataset,
    model_time: np.ndarray,
    *,
    use_svd: bool,
) -> ADSResult:
    """Run sequential-model SADS or SADS-SVD."""
    assert config.tau is not None
    assert config.y0 is not None

    eigval, eigenvectors, coefficients = solve_seq_model(
        tau=config.tau,
        y0=config.y0,
    )

    eigval, eigenvectors, coefficients = (
        _normalize_rate_solution(
            eigval=eigval,
            eigenvectors=eigenvectors,
            coefficients=coefficients,
            y0=config.y0,
            n_species=config.tau.size + 1,
        )
    )

    all_names = tuple(
        f"species_{index + 1}"
        for index in range(config.tau.size + 1)
    )
    spectrum_names = _kept_spectrum_names(
        all_names,
        config.exclude,
    )

    return _run_sads_driver(
        config=config,
        dataset=dataset,
        model_time=model_time,
        eigval=eigval,
        eigenvectors=eigenvectors,
        coefficients=coefficients,
        spectrum_names=spectrum_names,
        use_svd=use_svd,
        rate_matrix=None,
        rate_model_kind="sequential",
    )


def _run_custom_sads(
    config: ADSConfig,
    dataset: EScanDataset,
    model_time: np.ndarray,
    *,
    use_svd: bool,
) -> ADSResult:
    """Run custom-rate SADS or custom-rate SADS-SVD."""
    assert config.rate_model is not None

    rate_matrix = build_rate_matrix(
        config.rate_model
    )

    eigval, eigenvectors, coefficients = (
        solve_rate_model_real(
            rate_matrix,
            config.rate_model.y0,
        )
    )

    eigval, eigenvectors, coefficients = (
        _normalize_rate_solution(
            eigval=eigval,
            eigenvectors=eigenvectors,
            coefficients=coefficients,
            y0=config.rate_model.y0,
            n_species=len(config.rate_model.species),
        )
    )

    spectrum_names = _kept_spectrum_names(
        config.rate_model.species,
        config.exclude,
    )

    return _run_sads_driver(
        config=config,
        dataset=dataset,
        model_time=model_time,
        eigval=eigval,
        eigenvectors=eigenvectors,
        coefficients=coefficients,
        spectrum_names=spectrum_names,
        use_svd=use_svd,
        rate_matrix=rate_matrix,
        rate_model_kind="custom",
    )


def _run_sads_driver(
    *,
    config: ADSConfig,
    dataset: EScanDataset,
    model_time: np.ndarray,
    eigval: np.ndarray,
    eigenvectors: np.ndarray,
    coefficients: np.ndarray,
    spectrum_names: tuple[str, ...],
    use_svd: bool,
    rate_matrix: np.ndarray | None,
    rate_model_kind: str,
) -> ADSResult:
    """Call sads or sads_svd and wrap the result."""
    if use_svd:
        spectra, fit = sads_svd(
            escan_time=model_time,
            fwhm=config.fwhm,
            eigval=eigval,
            V=eigenvectors,
            c=coefficients,
            exclude=config.exclude,
            irf=config.irf,
            eta=config.eta,
            intensity=dataset.intensity,
            cond_num=config.cond_num,
        )

        # sads_svd already returns (n_energy, n_component).
        spectra = np.asarray(spectra, dtype=float)
        fit = np.asarray(fit, dtype=float)
        spectra_eps = None

        svd_u, svd_s, svd_vh = _truncated_data_svd(
            dataset.intensity,
            config.cond_num,
        )
    else:
        spectra_raw, spectra_eps_raw, fit = sads(
            escan_time=model_time,
            fwhm=config.fwhm,
            eigval=eigval,
            V=eigenvectors,
            c=coefficients,
            exclude=config.exclude,
            irf=config.irf,
            eta=config.eta,
            intensity=dataset.intensity,
            eps=dataset.eps,
        )

        # sads returns (n_component, n_energy).
        spectra = np.asarray(
            spectra_raw,
            dtype=float,
        ).T
        spectra_eps = np.asarray(
            spectra_eps_raw,
            dtype=float,
        ).T
        fit = np.asarray(fit, dtype=float)

        svd_u = None
        svd_s = None
        svd_vh = None

    metadata = {
        "dataset_name": dataset.name,
        "model_time": model_time.copy(),
        "t0": config.t0,
        "irf": config.irf,
        "fwhm": config.fwhm,
        "eta": config.eta,
        "tau": (
            None
            if config.tau is None
            else config.tau.copy()
        ),
        "y0": (
            config.rate_model.y0.copy()
            if config.rate_model is not None
            else config.y0.copy()
        ),
        "exclude": config.exclude,
        "eigval": eigval.copy(),
        "eigenvectors": eigenvectors.copy(),
        "coefficients": coefficients.copy(),
        "rate_matrix": (
            None
            if rate_matrix is None
            else rate_matrix.copy()
        ),
        "rate_model_kind": rate_model_kind,
        "cond_num": config.cond_num if use_svd else None,
    }

    return _make_ads_result(
        config=config,
        dataset=dataset,
        spectra=spectra,
        spectra_eps=spectra_eps,
        fit=fit,
        spectrum_names=spectrum_names,
        svd_u=svd_u,
        svd_s=svd_s,
        svd_vh=svd_vh,
        model_metadata=metadata,
    )


def _make_ads_result(
    *,
    config: ADSConfig,
    dataset: EScanDataset,
    spectra: np.ndarray,
    spectra_eps: np.ndarray | None,
    fit: np.ndarray,
    spectrum_names: tuple[str, ...],
    svd_u: np.ndarray | None,
    svd_s: np.ndarray | None,
    svd_vh: np.ndarray | None,
    model_metadata: dict,
) -> ADSResult:
    """Construct the common ADSResult representation."""
    return ADSResult(
        mode=config.mode,
        energy=dataset.energy,
        time=dataset.time,
        intensity=dataset.intensity,
        eps=dataset.eps,
        spectra=spectra,
        spectra_eps=spectra_eps,
        fit=fit,
        spectrum_names=spectrum_names,
        svd_u=svd_u,
        svd_s=svd_s,
        svd_vh=svd_vh,
        model_metadata=model_metadata,
    )


def _dads_spectrum_names(
    n_tau: int,
    *,
    base: bool,
) -> tuple[str, ...]:
    names = [
        f"decay_{index + 1}"
        for index in range(n_tau)
    ]

    if base:
        names.append("base")

    return tuple(names)


def _kept_spectrum_names(
    species_names: tuple[str, ...],
    exclude: tuple[int, ...] | None,
) -> tuple[str, ...]:
    """Return species names not excluded from SADS calculation."""
    if exclude is None:
        kept = tuple(species_names)
    else:
        excluded = set(exclude)
        kept = tuple(
            name
            for index, name in enumerate(species_names)
            if index not in excluded
        )

    if len(kept) == 0:
        raise ValueError(
            "At least one species must remain after exclusion."
        )

    return kept


def _truncated_data_svd(
    intensity: np.ndarray,
    cond_num: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the data SVD retained by the ADS-SVD cutoff.

    This mirrors the current numerical drivers:

    ``s > cond_num * s[0]``
    """
    intensity = np.asarray(intensity, dtype=float)

    if intensity.ndim != 2:
        raise ValueError(
            "intensity must be a 2D array for SVD."
        )

    svd_u, svd_s, svd_vh = linalg.svd(
        intensity,
        full_matrices=False,
    )

    if svd_s.size == 0:
        return svd_u, svd_s, svd_vh

    n_survived = int(
        np.sum(svd_s > cond_num * svd_s[0])
    )

    return (
        svd_u[:, :n_survived],
        svd_s[:n_survived],
        svd_vh[:n_survived, :],
    )


def _normalize_rate_solution(
    *,
    eigval,
    eigenvectors,
    coefficients,
    y0,
    n_species: int,
    imag_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a real eigensystem returned by a rate-model solver."""
    eigval = _real_finite_array(
        eigval,
        "eigval",
        imag_tol,
    )
    eigenvectors = _real_finite_array(
        eigenvectors,
        "eigenvectors",
        imag_tol,
    )
    coefficients = _real_finite_array(
        coefficients,
        "coefficients",
        imag_tol,
    )
    y0 = _real_finite_array(
        y0,
        "y0",
        imag_tol,
    )

    if eigval.shape != (n_species,):
        raise ValueError(
            f"eigval must have shape ({n_species},); "
            f"got {eigval.shape}."
        )

    if eigenvectors.shape != (n_species, n_species):
        raise ValueError(
            "eigenvectors must have shape "
            f"({n_species}, {n_species}); "
            f"got {eigenvectors.shape}."
        )

    if coefficients.shape != (n_species,):
        raise ValueError(
            f"coefficients must have shape ({n_species},); "
            f"got {coefficients.shape}."
        )

    if y0.shape != (n_species,):
        raise ValueError(
            f"y0 must have shape ({n_species},); "
            f"got {y0.shape}."
        )

    if not np.allclose(
        eigenvectors @ coefficients,
        y0,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            "Rate-model eigenmodes do not reconstruct y0."
        )

    return eigval, eigenvectors, coefficients


def _real_finite_array(
    value,
    name: str,
    imag_tol: float,
) -> np.ndarray:
    """Return a finite real array, rejecting complex modes."""
    try:
        array = np.asarray(value, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a numeric array."
        ) from exc

    if (
        not np.all(np.isfinite(array.real))
        or not np.all(np.isfinite(array.imag))
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    scale = 1.0

    if array.size:
        scale = max(
            scale,
            float(np.max(np.abs(array.real))),
        )

    if np.any(
        np.abs(array.imag) > imag_tol * scale
    ):
        raise ValueError(
            f"{name} contains a complex mode; "
            "only real modes are supported."
        )

    return np.asarray(array.real, dtype=float)