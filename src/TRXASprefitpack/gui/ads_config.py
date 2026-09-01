"""
Configuration and result objects for GUI-driven ADS/DADS/SADS workflows.

These objects are GUI-framework independent. They validate user-facing
configuration before being passed to ADS job runners.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Literal, Any

import numpy as np
from .rate_model import RateModelSpec, validate_rate_model_spec


ADSMode = Literal[
    "dads",
    "dads_svd",
    "sads",
    "sads_svd",
    "custom_sads",
    "custom_sads_svd",
]
IRFModel = Literal["g", "c", "pv"]

VALID_ADS_MODES = {
    "dads",
    "dads_svd",
    "sads",
    "sads_svd",
    "custom_sads",
    "custom_sads_svd",
}

VALID_IRF = {"g", "c", "pv"}


@dataclass(frozen=True)
class ADSConfig:
    """Configuration for DADS/SADS GUI workflows.

    The Qt GUI intentionally supports DADS, DADS-SVD, SADS, SADS-SVD,
    custom-rate SADS, and custom-rate SADS-SVD only.

    Oscillation-associated ADS workflows such as dads_osc and sads_osc are
    intentionally out of scope for the GUI.
    """

    mode: ADSMode
    irf: IRFModel
    fwhm: float
    eta: float | None
    t0: float
    tau: np.ndarray | None
    base: bool = True
    cond_num: float = 0.0
    rate_model: RateModelSpec | None = None
    y0: np.ndarray | None = None
    exclude: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        irf = str(self.irf).strip().lower()

        if mode not in VALID_ADS_MODES:
            raise ValueError(
                "mode must be one of "
                "{'dads', 'dads_svd', 'sads', 'sads_svd', "
                "'custom_sads', 'custom_sads_svd'}."
            )

        if irf not in VALID_IRF:
            raise ValueError("irf must be one of {'g', 'c', 'pv'}.")

        fwhm = float(self.fwhm)
        t0 = float(self.t0)
        cond_num = float(self.cond_num)

        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError("fwhm must be a finite positive value.")

        if not np.isfinite(t0):
            raise ValueError("t0 must be finite.")

        if not np.isfinite(cond_num) or cond_num < 0:
            raise ValueError("cond_num must be a finite non-negative value.")

        eta = self.eta
        if irf == "pv":
            if eta is None:
                raise ValueError("eta must be provided for pseudo-Voigt IRF.")
            eta = float(eta)
            if not np.isfinite(eta):
                raise ValueError("eta must be finite.")
        else:
            if eta is not None:
                eta = float(eta)
                if not np.isfinite(eta):
                    raise ValueError("eta must be finite.")

        tau = _normalize_tau_for_mode(mode, self.tau)
        rate_model = _validate_rate_model_for_mode(mode, self.rate_model)
        y0, exclude = _normalize_sads_inputs(
            mode=mode,
            tau=tau,
            rate_model=rate_model,
            y0=self.y0,
            exclude=self.exclude
        ) 

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "irf", irf)
        object.__setattr__(self, "fwhm", fwhm)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "base", bool(self.base))
        object.__setattr__(self, "cond_num", cond_num)
        object.__setattr__(self, "rate_model", rate_model)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "exclude", exclude)


def _normalize_tau_for_mode(mode: str, tau) -> np.ndarray | None:
    """Validate tau depending on ADS mode."""
    uses_tau = mode in {"dads", "dads_svd", "sads", "sads_svd"}
    uses_custom_rate = mode in {"custom_sads", "custom_sads_svd"}

    if uses_custom_rate:
        if tau is not None:
            raise ValueError("tau must be None for custom rate model modes.")
        return None

    if uses_tau:
        if tau is None:
            raise ValueError(f"tau must be provided for mode='{mode}'.")

        tau = np.atleast_1d(np.asarray(tau, dtype=float))

        if tau.ndim != 1:
            raise ValueError("tau must be a 1D array.")

        if tau.size == 0:
            raise ValueError("tau must contain at least one value.")

        if not np.all(np.isfinite(tau)):
            raise ValueError("tau must contain only finite values.")

        if np.any(tau <= 0):
            raise ValueError("tau must contain only positive values.")

        return tau

    raise ValueError(f"Unsupported ADS mode: {mode}")


def _validate_rate_model_for_mode(mode: str, rate_model):
    """Validate presence/absence of custom rate model object.

    The actual RateModelSpec type is introduced in a later PR. For now this
    function only checks whether a custom model is required or forbidden.
    """
    uses_custom_rate = mode in {"custom_sads", "custom_sads_svd"}

    if uses_custom_rate and rate_model is None:
        raise ValueError(f"rate_model must be provided for mode='{mode}'.")

    if not uses_custom_rate and rate_model is not None:
        raise ValueError("rate_model must be None for non-custom ADS modes.")

    if uses_custom_rate:
        rate_model = validate_rate_model_spec(rate_model)

    return rate_model

def _normalize_sads_inputs(
        mode: str,
        tau: np.ndarray | None,
        rate_model: RateModelSpec | None,
        y0,
        exclude,
) -> tuple[np.ndarray | None, tuple[int,...]|None]:
    """Normalize mode-dependent initial populatons and exclude species"""

    is_dads = mode in {"dads", "dads_svd"}
    is_standard_sads = mode in {"sads", "sads_svd"}
    is_custom_sads = mode in {"custom_sads", "custom_sads_svd"}

    if is_dads:
        if y0 is not None:
            raise ValueError(
                f"y0 must be None for mode='{mode}'."
            )

        if exclude is not None:
            raise ValueError(
                f"exclude must be None for mode='{mode}'."
            )

        return None, None

    if is_standard_sads:
        assert tau is not None

        normalized_y0 = _normalized_y0(
            y0,
            n_species=tau.size + 1
        )

        normalized_exclude = _normalized_exclude(
            exclude,
            n_species=tau.size+1
        )

        return normalized_y0, normalized_exclude

    if is_custom_sads:
        assert rate_model is not None

        if y0 is not None:
            raise ValueError(
                "Top-level y0 must be None for custom rate model modes;"
                "use RateModelSpec.y0."
            )

        normalized_exclude = _normalized_exclude(
            exclude,
            n_species=len(rate_model.species),
        )

        return None, normalized_exclude

    raise ValueError(f"Unsupported ADS model: {mode}")

def _normalized_y0(y0, n_species: int) -> np.ndarray:
    """Validate initial populations for a standard sequential SADS model."""
    if y0 is None:
        raise ValueError(
            """y0 must be provided for standard SADS model."""
        )

    try:
        normalized = np.asarray(y0, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "y0 must be a numeric array."
        ) from exc

    if normalized.ndim != 1:
        raise ValueError("y0 must be a 1D array.")

    expected_shape = (n_species, )

    if normalized.shape != expected_shape:
        raise ValueError(
            f"y0 must have shape {expected_shape};"
            f"got {normalized.shape}."
        )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "y0 must contain only finite values."
        )
    return normalized.copy()

def _normalized_exclude(exclude, n_species: int) -> tuple[int, ...] | None:
    """Normalize exclude spcies indices to non-negative indices.

    Python-style negative indices are accepted. For example, ``-1`` is
    normalized to ``n_species - 1``.
    """

    if exclude is None:
        return None

    if isinstance(exclude, (str, bytes)):
        raise ValueError(
            "exclude must be a sequence of integer indices."
        )

    try:
        raw_indices = tuple(exclude)
    except TypeError as exc:
        raise ValueError(
            "exclude must be a sequence of integr indices"
        ) from exc

    if len(raw_indices) == 0:
        return None

    normalized: list[int] = []

    for raw_index in raw_indices:
        if isinstance(raw_index, (bool, np.bool_)):
            raise ValueError(
                "exclude must contain only integer indices."
            )

        try:
            index = operator.index(raw_index)
        except TypeError as exc:
            raise ValueError(
                "exclude must contain only integer indices."
            ) from exc

        if index < 0:
            index += n_species

        if index < 0 or index >= n_species:
            raise ValueError(
                f"exclude index {raw_index!r} is out of range "
                f"for {n_species} species. "
            )

        normalized.append(index)

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "exclude must contain unique species indices."
            )

    return tuple(normalized)


@dataclass(frozen=True)
class ADSResult:
    """Result object for GUI-driven ADS/DADS/SADS workflows.

    Shapes
    ------
    energy : (n_energy,)
    time : (n_time,)
    intensity : (n_energy, n_time)
    eps : (n_energy, n_time)
    spectra : (n_energy, n_component)
    fit : (n_energy, n_time)
    """

    mode: str
    energy: np.ndarray
    time: np.ndarray
    intensity: np.ndarray
    eps: np.ndarray
    spectra: np.ndarray
    fit: np.ndarray
    spectrum_names: tuple[str, ...]
    spectra_eps: np.ndarray | None = None
    svd_u: np.ndarray | None = None
    svd_s: np.ndarray | None = None
    svd_vh: np.ndarray | None = None
    model_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()

        if mode not in VALID_ADS_MODES:
            raise ValueError(
                "mode must be one of "
                "{'dads', 'dads_svd', 'sads', 'sads_svd', "
                "'custom_sads', 'custom_sads_svd'}."
            )

        energy = np.asarray(self.energy, dtype=float)
        time = np.asarray(self.time, dtype=float)
        intensity = np.asarray(self.intensity, dtype=float)
        eps = np.asarray(self.eps, dtype=float)
        spectra = np.asarray(self.spectra, dtype=float)
        fit = np.asarray(self.fit, dtype=float)

        if energy.ndim != 1:
            raise ValueError("energy must be a 1D array.")

        if time.ndim != 1:
            raise ValueError("time must be a 1D array.")

        if intensity.ndim != 2:
            raise ValueError("intensity must be a 2D array.")

        if eps.ndim != 2:
            raise ValueError("eps must be a 2D array.")

        expected_matrix_shape = (energy.size, time.size)
        if intensity.shape != expected_matrix_shape:
            raise ValueError(
                "intensity shape must be (n_energy, n_time); "
                f"got {intensity.shape}, expected {expected_matrix_shape}."
            )

        if eps.shape != intensity.shape:
            raise ValueError(
                "eps and intensity must have the same shape; "
                f"got {eps.shape} and {intensity.shape}."
            )

        if np.any(eps <= 0):
            raise ValueError("eps must contain only positive values.")

        if spectra.ndim != 2:
            raise ValueError("spectra must be a 2D array.")

        if spectra.shape[0] != energy.size:
            raise ValueError(
                "spectra.shape[0] must match energy size; "
                f"got {spectra.shape[0]} and {energy.size}."
            )

        if len(self.spectrum_names) != spectra.shape[1]:
            raise ValueError(
                "spectrum_names length must match spectra.shape[1]; "
                f"got {len(self.spectrum_names)} and {spectra.shape[1]}."
            )

        spectra_eps = _optional_array(
            self.spectra_eps,
            "spectra_eps",
        )

        if spectra_eps is not None:
            if spectra_eps.shape != spectra.shape:
                raise ValueError(
                    "spectra_eps shape must match spectra shape; "
                    f"got {spectra_eps.shape} and {spectra.shape}."
                )

            if np.any(spectra_eps<0):
                raise ValueError(
                    "Spectra_eps must contain only non-negative values."
                )

        if fit.shape != intensity.shape:
            raise ValueError(
                "fit shape must match intensity shape; "
                f"got {fit.shape} and {intensity.shape}."
            )

        svd_u = _optional_array(self.svd_u, "svd_u")
        svd_s = _optional_array(self.svd_s, "svd_s")
        svd_vh = _optional_array(self.svd_vh, "svd_vh")

        if svd_s is not None and svd_s.ndim != 1:
            raise ValueError("svd_s must be a 1D array.")

        if svd_u is not None and svd_u.ndim != 2:
            raise ValueError("svd_u must be a 2D array.")

        if svd_vh is not None and svd_vh.ndim != 2:
            raise ValueError("svd_vh must be a 2D array.")

        if (svd_u is None) != (svd_s is None) or (svd_s is None) != (svd_vh is None):
            raise ValueError("svd_u, svd_s, and svd_vh must be provided together.")

        if svd_u is not None:
            n_component = svd_s.size
            if svd_u.shape != (energy.size, n_component):
                raise ValueError(
                    "svd_u shape must be (n_energy, n_component); "
                    f"got {svd_u.shape}, expected {(energy.size, n_component)}."
                )
            if svd_vh.shape != (n_component, time.size):
                raise ValueError(
                    "svd_vh shape must be (n_component, n_time); "
                    f"got {svd_vh.shape}, expected {(n_component, time.size)}."
                )

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "spectra", spectra)
        object.__setattr__(self, "fit", fit)
        object.__setattr__(self, "spectrum_names", tuple(self.spectrum_names))
        object.__setattr__(self, "spectra_eps", spectra_eps)
        object.__setattr__(self, "svd_u", svd_u)
        object.__setattr__(self, "svd_s", svd_s)
        object.__setattr__(self, "svd_vh", svd_vh)

        if self.model_metadata is not None:
            object.__setattr__(self, "model_metadata", dict(self.model_metadata))

    @property
    def n_energy(self) -> int:
        return self.energy.size

    @property
    def n_time(self) -> int:
        return self.time.size

    @property
    def n_component(self) -> int:
        return self.spectra.shape[1]

    @property
    def has_svd(self) -> bool:
        return self.svd_u is not None


def _optional_array(value, name: str) -> np.ndarray | None:
    if value is None:
        return None

    arr = np.asarray(value, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")

    return arr