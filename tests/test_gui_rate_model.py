import json
import os
import sys

import numpy as np
import pytest


path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.rate_model import (
    RateEdge,
    RateModelSpec,
    build_rate_matrix,
    rate_model_from_dict,
    rate_model_to_dict,
    solve_rate_model_real,
    validate_rate_model_spec,
)


def make_sequential_spec():
    return RateModelSpec(
        species=("A", "B", "C"),
        edges=(
            RateEdge("A", "B", 2.0),
            RateEdge("B", "C", 0.5),
        ),
        y0=np.array([1.0, 0.0, 0.0]),
    )


def test_rate_edge_accepts_valid_edge():
    edge = RateEdge("A", "B", 2.5)

    assert edge.source == "A"
    assert edge.target == "B"
    assert edge.rate == pytest.approx(2.5)


def test_rate_edge_normalizes_names():
    edge = RateEdge(" A ", " B ", 1.0)

    assert edge.source == "A"
    assert edge.target == "B"


@pytest.mark.parametrize(
    "source,target",
    [
        ("", "B"),
        ("A", ""),
        ("A", "A"),
        (" A ", "A"),
    ],
)
def test_rate_edge_rejects_invalid_species(source, target):
    with pytest.raises(ValueError):
        RateEdge(source, target, 1.0)


@pytest.mark.parametrize(
    "rate",
    [0.0, -1.0, np.nan, np.inf, -np.inf],
)
def test_rate_edge_rejects_invalid_rate(rate):
    with pytest.raises(ValueError, match="rate"):
        RateEdge("A", "B", rate)


def test_rate_model_spec_accepts_valid_model():
    spec = make_sequential_spec()

    assert spec.species == ("A", "B", "C")
    assert len(spec.edges) == 2

    np.testing.assert_allclose(
        spec.y0,
        np.array([1.0, 0.0, 0.0]),
    )


def test_rate_model_spec_normalizes_species_names():
    spec = RateModelSpec(
        species=(" A ", " B "),
        edges=(RateEdge("A", "B", 1.0),),
        y0=np.array([1.0, 0.0]),
    )

    assert spec.species == ("A", "B")


def test_rate_model_spec_rejects_empty_species():
    with pytest.raises(ValueError, match="species"):
        RateModelSpec(
            species=(),
            edges=(),
            y0=np.array([]),
        )


def test_rate_model_spec_rejects_species_string():
    with pytest.raises(ValueError, match="sequence"):
        RateModelSpec(
            species="AB",
            edges=(),
            y0=np.array([1.0, 0.0]),
        )


def test_rate_model_spec_rejects_duplicate_species():
    with pytest.raises(ValueError, match="unique"):
        RateModelSpec(
            species=("A", "A"),
            edges=(),
            y0=np.array([1.0, 0.0]),
        )


def test_rate_model_spec_rejects_duplicate_after_normalization():
    with pytest.raises(ValueError, match="unique"):
        RateModelSpec(
            species=("A", " A "),
            edges=(),
            y0=np.array([1.0, 0.0]),
        )


def test_rate_model_spec_rejects_unknown_source():
    with pytest.raises(ValueError, match="source"):
        RateModelSpec(
            species=("A", "B"),
            edges=(RateEdge("X", "B", 1.0),),
            y0=np.array([1.0, 0.0]),
        )


def test_rate_model_spec_rejects_unknown_target():
    with pytest.raises(ValueError, match="target"):
        RateModelSpec(
            species=("A", "B"),
            edges=(RateEdge("A", "X", 1.0),),
            y0=np.array([1.0, 0.0]),
        )


def test_rate_model_spec_rejects_non_edge_object():
    with pytest.raises(TypeError, match="RateEdge"):
        RateModelSpec(
            species=("A", "B"),
            edges=(("A", "B", 1.0),),
            y0=np.array([1.0, 0.0]),
        )


def test_rate_model_spec_rejects_non_1d_y0():
    with pytest.raises(ValueError, match="1D"):
        RateModelSpec(
            species=("A", "B"),
            edges=(),
            y0=np.array([[1.0, 0.0]]),
        )


def test_rate_model_spec_rejects_wrong_y0_length():
    with pytest.raises(ValueError, match="shape"):
        RateModelSpec(
            species=("A", "B"),
            edges=(),
            y0=np.array([1.0]),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rate_model_spec_rejects_nonfinite_y0(bad_value):
    with pytest.raises(ValueError, match="finite"):
        RateModelSpec(
            species=("A", "B"),
            edges=(),
            y0=np.array([1.0, bad_value]),
        )


def test_rate_model_spec_rejects_complex_y0():
    with pytest.raises(ValueError, match="real"):
        RateModelSpec(
            species=("A", "B"),
            edges=(),
            y0=np.array([1.0 + 1.0j, 0.0]),
        )


def test_validate_rate_model_spec_returns_same_object():
    spec = make_sequential_spec()

    assert validate_rate_model_spec(spec) is spec


def test_validate_rate_model_spec_rejects_wrong_type():
    with pytest.raises(TypeError, match="RateModelSpec"):
        validate_rate_model_spec({"species": ["A"]})


def test_build_rate_matrix_single_edge():
    spec = RateModelSpec(
        species=("A", "B"),
        edges=(RateEdge("A", "B", 2.0),),
        y0=np.array([1.0, 0.0]),
    )

    rate_matrix = build_rate_matrix(spec)

    expected = np.array(
        [
            [-2.0, 0.0],
            [2.0, 0.0],
        ]
    )

    np.testing.assert_allclose(rate_matrix, expected)


def test_build_rate_matrix_sequential_model():
    rate_matrix = build_rate_matrix(
        make_sequential_spec()
    )

    expected = np.array(
        [
            [-2.0, 0.0, 0.0],
            [2.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )

    np.testing.assert_allclose(rate_matrix, expected)


def test_build_rate_matrix_parallel_edges_are_added():
    spec = RateModelSpec(
        species=("A", "B"),
        edges=(
            RateEdge("A", "B", 1.0),
            RateEdge("A", "B", 2.0),
        ),
        y0=np.array([1.0, 0.0]),
    )

    expected = np.array(
        [
            [-3.0, 0.0],
            [3.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        build_rate_matrix(spec),
        expected,
    )


def test_build_rate_matrix_columns_sum_to_zero():
    rate_matrix = build_rate_matrix(
        make_sequential_spec()
    )

    np.testing.assert_allclose(
        np.sum(rate_matrix, axis=0),
        np.zeros(3),
    )


def test_solve_rate_model_real_solves_real_model():
    spec = make_sequential_spec()
    rate_matrix = build_rate_matrix(spec)

    eigval, eigenvectors, coefficients = (
        solve_rate_model_real(
            rate_matrix,
            spec.y0,
        )
    )

    np.testing.assert_allclose(
        np.sort(eigval),
        np.array([-2.0, -0.5, 0.0]),
    )

    np.testing.assert_allclose(
        rate_matrix @ eigenvectors,
        eigenvectors * eigval,
    )

    np.testing.assert_allclose(
        eigenvectors @ coefficients,
        spec.y0,
        atol=1e-7,
        rtol=1e-7,
    )

    assert np.isrealobj(eigval)
    assert np.isrealobj(eigenvectors)
    assert np.isrealobj(coefficients)


def test_solve_rate_model_real_rejects_complex_eigenmode():
    cycle_spec = RateModelSpec(
        species=("A", "B", "C"),
        edges=(
            RateEdge("A", "B", 1.0),
            RateEdge("B", "C", 1.0),
            RateEdge("C", "A", 1.0),
        ),
        y0=np.array([1.0, 0.0, 0.0]),
    )

    rate_matrix = build_rate_matrix(cycle_spec)

    with pytest.raises(
        ValueError,
        match="complex mode",
    ):
        solve_rate_model_real(
            rate_matrix,
            cycle_spec.y0,
        )


def test_solve_rate_model_real_rejects_defective_basis():
    defective_spec = RateModelSpec(
        species=("A", "B", "C"),
        edges=(
            RateEdge("A", "B", 1.0),
            RateEdge("B", "C", 1.0),
        ),
        y0=np.array([1.0, 0.0, 0.0]),
    )

    rate_matrix = build_rate_matrix(defective_spec)

    with pytest.raises(
        ValueError,
        match="defective",
    ):
        solve_rate_model_real(
            rate_matrix,
            defective_spec.y0,
        )


def test_solve_rate_model_real_accepts_tiny_imaginary_noise():
    rate_matrix = np.array(
        [
            [-1.0 + 1e-14j, 0.0],
            [1.0 - 1e-14j, 0.0],
        ]
    )

    eigval, eigenvectors, coefficients = (
        solve_rate_model_real(
            rate_matrix,
            np.array([1.0, 0.0]),
        )
    )

    assert np.isrealobj(eigval)
    assert np.isrealobj(eigenvectors)
    assert np.isrealobj(coefficients)


def test_solve_rate_model_real_rejects_complex_input():
    rate_matrix = np.array(
        [
            [-1.0 + 1e-3j, 0.0],
            [1.0 - 1e-3j, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="complex mode",
    ):
        solve_rate_model_real(
            rate_matrix,
            np.array([1.0, 0.0]),
        )


def test_solve_rate_model_real_rejects_nonsquare_matrix():
    with pytest.raises(ValueError, match="square"):
        solve_rate_model_real(
            np.ones((2, 3)),
            np.ones(2),
        )


def test_solve_rate_model_real_rejects_empty_matrix():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        solve_rate_model_real(
            np.empty((0, 0)),
            np.empty(0),
        )


def test_solve_rate_model_real_rejects_wrong_y0_shape():
    with pytest.raises(ValueError, match="shape"):
        solve_rate_model_real(
            np.eye(2),
            np.ones(3),
        )


@pytest.mark.parametrize(
    "imag_tol",
    [-1.0, np.nan, np.inf],
)
def test_solve_rate_model_real_rejects_invalid_tolerance(
    imag_tol,
):
    with pytest.raises(ValueError, match="imag_tol"):
        solve_rate_model_real(
            np.eye(2),
            np.ones(2),
            imag_tol=imag_tol,
        )


def test_rate_model_dict_roundtrip():
    original = make_sequential_spec()

    data = rate_model_to_dict(original)
    restored = rate_model_from_dict(data)

    assert restored.species == original.species
    assert restored.edges == original.edges

    np.testing.assert_allclose(
        restored.y0,
        original.y0,
    )


def test_rate_model_to_dict_is_json_serializable():
    data = rate_model_to_dict(
        make_sequential_spec()
    )

    encoded = json.dumps(data)
    decoded = json.loads(encoded)

    assert decoded["species"] == ["A", "B", "C"]
    assert decoded["edges"][0]["source"] == "A"
    assert decoded["edges"][0]["target"] == "B"
    assert decoded["edges"][0]["rate"] == pytest.approx(2.0)
    assert decoded["y0"] == [1.0, 0.0, 0.0]


def test_rate_model_from_dict_rejects_missing_field():
    with pytest.raises(ValueError, match="Missing"):
        rate_model_from_dict(
            {
                "species": ["A", "B"],
                "edges": [],
            }
        )


def test_rate_model_from_dict_rejects_species_string():
    with pytest.raises(ValueError, match="sequence"):
        rate_model_from_dict(
            {
                "species": "AB",
                "edges": [],
                "y0": [1.0, 0.0],
            }
        )


def test_rate_model_from_dict_rejects_invalid_edge_item():
    with pytest.raises(ValueError, match=r"edges\[0\]"):
        rate_model_from_dict(
            {
                "species": ["A", "B"],
                "edges": ["A -> B"],
                "y0": [1.0, 0.0],
            }
        )


def test_rate_model_from_dict_rejects_missing_edge_field():
    with pytest.raises(ValueError, match="Missing field"):
        rate_model_from_dict(
            {
                "species": ["A", "B"],
                "edges": [
                    {
                        "source": "A",
                        "target": "B",
                    }
                ],
                "y0": [1.0, 0.0],
            }
        )