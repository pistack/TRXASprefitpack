"""
Pure-Python specification and validation for custom kinetic rate models.

This module is independent of PyQt5. It supports only real, diagonalizable
eigenmode systems suitable for the initial custom-SADS GUI scope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg


__all__ = [
    "RateEdge",
    "RateModelSpec",
    "validate_rate_model_spec",
    "build_rate_matrix",
    "solve_rate_model_real",
    "rate_model_to_dict",
    "rate_model_from_dict",
]


@dataclass(frozen=True)
class RateEdge:
    """One directed first-order transition.

    For ``source -> target`` with rate ``k``:

    ``K[source, source] -= k``
    ``K[target, source] += k``
    """

    source: str
    target: str
    rate: float

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        target = str(self.target).strip()

        if not source:
            raise ValueError("source must not be empty.")

        if not target:
            raise ValueError("target must not be empty.")

        if source == target:
            raise ValueError("source and target must be different.")

        try:
            rate = float(self.rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rate must be a finite positive value."
            ) from exc

        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be a finite positive value.")

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "rate", rate)


@dataclass(frozen=True)
class RateModelSpec:
    """Validated custom first-order kinetic model."""

    species: tuple[str, ...]
    edges: tuple[RateEdge, ...]
    y0: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.species, (str, bytes)):
            raise ValueError("species must be a sequence of names.")

        species = tuple(str(name).strip() for name in self.species)

        if len(species) == 0:
            raise ValueError(
                "species must contain at least one name."
            )

        if any(not name for name in species):
            raise ValueError("species names must not be empty.")

        if len(set(species)) != len(species):
            raise ValueError("species names must be unique.")

        edges = tuple(self.edges)

        if any(not isinstance(edge, RateEdge) for edge in edges):
            raise TypeError(
                "edges must contain only RateEdge objects."
            )

        known_species = set(species)

        for edge in edges:
            if edge.source not in known_species:
                raise ValueError(
                    f"Unknown source species: {edge.source!r}."
                )

            if edge.target not in known_species:
                raise ValueError(
                    f"Unknown target species: {edge.target!r}."
                )

        raw_y0 = np.asarray(self.y0)

        if np.iscomplexobj(raw_y0):
            if np.any(np.asarray(raw_y0).imag != 0):
                raise ValueError("y0 must contain only real values.")
            raw_y0 = raw_y0.real

        try:
            y0 = np.asarray(raw_y0, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "y0 must be a numeric array."
            ) from exc

        if y0.ndim != 1:
            raise ValueError("y0 must be a 1D array.")

        expected_shape = (len(species),)

        if y0.shape != expected_shape:
            raise ValueError(
                f"y0 must have shape {expected_shape}; "
                f"got {y0.shape}."
            )

        if not np.all(np.isfinite(y0)):
            raise ValueError(
                "y0 must contain only finite values."
            )

        y0 = y0.copy()
        y0.setflags(write=False)

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "y0", y0)


def validate_rate_model_spec(
    spec: RateModelSpec,
) -> RateModelSpec:
    """Validate and return a RateModelSpec.

    RateModelSpec performs structural validation during construction. This
    function provides an explicit validation entry point for GUI code and
    job runners.
    """
    if not isinstance(spec, RateModelSpec):
        raise TypeError("spec must be a RateModelSpec.")

    return spec


def build_rate_matrix(
    spec: RateModelSpec,
) -> np.ndarray:
    """Construct the column-vector rate matrix K.

    The convention is ``dy/dt = K y``. For an edge ``A -> B`` with rate
    ``k``, the source-column entries are modified as follows:

    ``K[A, A] -= k``
    ``K[B, A] += k``
    """
    spec = validate_rate_model_spec(spec)

    n_species = len(spec.species)
    species_index = {
        name: index
        for index, name in enumerate(spec.species)
    }

    rate_matrix = np.zeros(
        (n_species, n_species),
        dtype=float,
    )

    for edge in spec.edges:
        source_index = species_index[edge.source]
        target_index = species_index[edge.target]

        rate_matrix[source_index, source_index] -= edge.rate
        rate_matrix[target_index, source_index] += edge.rate

    column_sum = np.sum(rate_matrix, axis=0)

    if not np.allclose(
        column_sum,
        0.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "rate matrix columns must sum to zero."
        )

    return rate_matrix


def solve_rate_model_real(
    rate_matrix: np.ndarray,
    y0: np.ndarray,
    *,
    imag_tol: float = 1e-10,
    condition_max: float = 1e12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a rate model using real eigenmodes only.

    Returns
    -------
    eigval
        Real eigenvalues of the rate matrix.
    eigenvectors
        Real right-eigenvector matrix V.
    coefficients
        Real coefficients c satisfying ``y0 = V @ c``.

    Raises
    ------
    ValueError
        If the matrix produces a complex eigenmode, does not have a stable
        real eigenvector basis, or is incompatible with y0.

    Notes
    -----
    This function intentionally does not rely on
    ``TRXASprefitpack.mathfun.solve_model`` to detect complex eigenvalues,
    because that function currently returns ``eigval.real``.
    """
    imag_tol = float(imag_tol)
    condition_max = float(condition_max)

    if not np.isfinite(imag_tol) or imag_tol < 0:
        raise ValueError(
            "imag_tol must be finite and non-negative."
        )

    if not np.isfinite(condition_max) or condition_max <= 0:
        raise ValueError(
            "condition_max must be finite and positive."
        )

    matrix = _as_nearly_real_array(
        rate_matrix,
        "rate_matrix",
        imag_tol,
    )
    y0_array = _as_nearly_real_array(
        y0,
        "y0",
        imag_tol,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(
            "rate_matrix must be a square 2D array."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "rate_matrix must contain at least one species."
        )

    expected_y0_shape = (matrix.shape[0],)

    if (
        y0_array.ndim != 1
        or y0_array.shape != expected_y0_shape
    ):
        raise ValueError(
            f"y0 must have shape {expected_y0_shape}; "
            f"got {y0_array.shape}."
        )

    eigval_raw, eigenvectors_raw = linalg.eig(matrix)

    eigval = _as_nearly_real_array(
        eigval_raw,
        "eigenvalues",
        imag_tol,
    )
    eigenvectors = _as_nearly_real_array(
        eigenvectors_raw,
        "eigenvectors",
        imag_tol,
    )

    condition_number = float(
        np.linalg.cond(eigenvectors)
    )

    if (
        not np.isfinite(condition_number)
        or condition_number > condition_max
    ):
        raise ValueError(
            "rate matrix has a singular, defective, or "
            "ill-conditioned real eigenvector basis."
        )

    try:
        coefficients = linalg.solve(
            eigenvectors,
            y0_array,
            assume_a="gen",
            check_finite=True,
        )
    except linalg.LinAlgError as exc:
        raise ValueError(
            "Could not solve y0 = V c for a real "
            "eigenmode basis."
        ) from exc

    coefficients = _as_nearly_real_array(
        coefficients,
        "mode coefficients",
        imag_tol,
    )

    if not np.allclose(
        eigenvectors @ coefficients,
        y0_array,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            "Real eigenmode coefficients do not reconstruct y0."
        )

    return eigval, eigenvectors, coefficients


def rate_model_to_dict(
    spec: RateModelSpec,
) -> dict[str, Any]:
    """Convert a RateModelSpec into a JSON-compatible dictionary."""
    spec = validate_rate_model_spec(spec)

    return {
        "species": list(spec.species),
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "rate": edge.rate,
            }
            for edge in spec.edges
        ],
        "y0": spec.y0.tolist(),
    }


def rate_model_from_dict(
    data: Mapping[str, Any],
) -> RateModelSpec:
    """Construct and validate a RateModelSpec from a dictionary."""
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping.")

    try:
        species_data = data["species"]
        edges_data = data["edges"]
        y0_data = data["y0"]
    except KeyError as exc:
        raise ValueError(
            f"Missing rate model field: {exc.args[0]!r}."
        ) from exc

    if isinstance(species_data, (str, bytes)):
        raise ValueError(
            "species must be a sequence of names."
        )

    if not isinstance(species_data, Sequence):
        raise ValueError(
            "species must be a sequence of names."
        )

    if (
        isinstance(edges_data, (str, bytes))
        or not isinstance(edges_data, Sequence)
    ):
        raise ValueError(
            "edges must be a sequence of mappings."
        )

    edges: list[RateEdge] = []

    for index, item in enumerate(edges_data):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"edges[{index}] must be a mapping."
            )

        try:
            edge = RateEdge(
                source=item["source"],
                target=item["target"],
                rate=item["rate"],
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing field {exc.args[0]!r} "
                f"in edges[{index}]."
            ) from exc

        edges.append(edge)

    return RateModelSpec(
        species=tuple(species_data),
        edges=tuple(edges),
        y0=y0_data,
    )


def _as_nearly_real_array(
    value,
    name: str,
    imag_tol: float,
) -> np.ndarray:
    """Convert an array to float after rejecting complex values."""
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

    real_scale = 0.0

    if array.size:
        real_scale = float(
            np.max(np.abs(array.real))
        )

    scale = max(1.0, real_scale)

    if np.any(
        np.abs(array.imag) > imag_tol * scale
    ):
        raise ValueError(
            f"{name} contains a complex mode; "
            "only real modes are supported."
        )

    return np.asarray(array.real, dtype=float)