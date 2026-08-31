import numpy as np
import pytest

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.gui.models import TScanDataset, TScanTrace
from TRXASprefitpack.gui.validators import (
    expected_t0_count_for_tscan,
    validate_bounds,
    validate_t0_count_for_tscan,
    validate_tau_array,
    validate_tau_mask,
)

def make_trace(name="trace", t=None, intensity=None, eps=None):
    if t is None:
        t = np.array([0.0, 1.0, 2.0])
    if intensity is None:
        intensity = np.array([1.0, 2.0, 3.0])
    if eps is None:
        eps = np.array([0.1, 0.1, 0.1])

    return TScanTrace(
        path=f"{name}.dat",
        name=name,
        t=t,
        intensity=intensity,
        eps=eps,
    )


def make_dataset(name="dataset", n_trace=1):
    traces = tuple(
        make_trace(
            name=f"{name}_{idx}",
            intensity=np.array([1.0, 2.0, 3.0]) + idx,
        )
        for idx in range(n_trace)
    )
    return TScanDataset(name=name, traces=traces)


def test_expected_t0_count_for_tscan_scanwise():
    datasets = [make_dataset("a", n_trace=2), make_dataset("b", n_trace=3)]

    assert expected_t0_count_for_tscan(datasets, same_t0=False) == 5


def test_expected_t0_count_for_tscan_same_t0():
    datasets = [make_dataset("a", n_trace=2), make_dataset("b", n_trace=3)]

    assert expected_t0_count_for_tscan(datasets, same_t0=True) == 2


def test_expected_t0_count_for_tscan_rejects_empty_datasets():
    with pytest.raises(ValueError, match="At least one"):
        expected_t0_count_for_tscan([], same_t0=False)


def test_validate_t0_count_for_tscan_passes_scanwise():
    datasets = [make_dataset("a", n_trace=2), make_dataset("b", n_trace=3)]

    validate_t0_count_for_tscan(
        datasets,
        np.zeros(5),
        same_t0=False,
    )


def test_validate_t0_count_for_tscan_passes_same_t0():
    datasets = [make_dataset("a", n_trace=2), make_dataset("b", n_trace=3)]

    validate_t0_count_for_tscan(
        datasets,
        np.zeros(2),
        same_t0=True,
    )


def test_validate_t0_count_for_tscan_rejects_wrong_count():
    datasets = [make_dataset("a", n_trace=2), make_dataset("b", n_trace=3)]

    with pytest.raises(ValueError, match="t0_init must contain 5"):
        validate_t0_count_for_tscan(
            datasets,
            np.zeros(4),
            same_t0=False,
        )


def test_validate_tau_array():
    tau = validate_tau_array(np.array([1.0, 2.0]), allow_none=False)

    np.testing.assert_allclose(tau, np.array([1.0, 2.0]))


def test_validate_tau_array_allows_none():
    assert validate_tau_array(None, allow_none=True) is None


def test_validate_tau_array_rejects_none_when_not_allowed():
    with pytest.raises(ValueError, match="must not be None"):
        validate_tau_array(None, allow_none=False)


@pytest.mark.parametrize(
    "tau",
    [
        np.array([]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, 0.0]),
        np.array([1.0, -1.0]),
    ],
)
def test_validate_tau_array_rejects_invalid_values(tau):
    with pytest.raises(ValueError):
        validate_tau_array(tau, allow_none=False)


def test_validate_bounds_accepts_valid_bounds():
    validate_bounds(
        np.array([1.0, 2.0]),
        [(0.0, 2.0), (1.0, 3.0)],
        "tau",
    )


def test_validate_bounds_allows_none():
    validate_bounds(np.array([1.0, 2.0]), None, "tau")


def test_validate_bounds_rejects_wrong_length():
    with pytest.raises(ValueError, match="length"):
        validate_bounds(np.array([1.0, 2.0]), [(0.0, 2.0)], "tau")


def test_validate_bounds_rejects_nonfinite_bounds():
    with pytest.raises(ValueError, match="finite"):
        validate_bounds(np.array([1.0]), [(0.0, np.inf)], "tau")


def test_validate_bounds_rejects_lower_greater_than_upper():
    with pytest.raises(ValueError, match="<="):
        validate_bounds(np.array([1.0]), [(2.0, 1.0)], "tau")


def test_validate_bounds_rejects_init_outside_bounds():
    with pytest.raises(ValueError, match="outside bounds"):
        validate_bounds(np.array([3.0]), [(0.0, 2.0)], "tau")


def test_validate_tau_mask_accepts_valid_mask():
    datasets = [make_dataset("a", n_trace=2), make_dataset("b", n_trace=1)]
    tau_mask = [
        np.array([True, False, True]),
        np.array([False, True, True]),
    ]

    out = validate_tau_mask(datasets, tau_mask, n_tau=3)

    assert len(out) == 2
    np.testing.assert_array_equal(out[0], tau_mask[0])
    np.testing.assert_array_equal(out[1], tau_mask[1])


def test_validate_tau_mask_allows_none():
    datasets = [make_dataset("a", n_trace=2)]

    assert validate_tau_mask(datasets, None, n_tau=3) is None


def test_validate_tau_mask_rejects_wrong_dataset_count():
    datasets = [make_dataset("a", n_trace=2), make_dataset("b", n_trace=1)]
    tau_mask = [np.array([True, False, True])]

    with pytest.raises(ValueError, match="one mask per dataset"):
        validate_tau_mask(datasets, tau_mask, n_tau=3)


def test_validate_tau_mask_rejects_non_1d_mask():
    datasets = [make_dataset("a", n_trace=2)]
    tau_mask = [np.array([[True, False, True]])]

    with pytest.raises(ValueError, match="1D"):
        validate_tau_mask(datasets, tau_mask, n_tau=3)


def test_validate_tau_mask_rejects_wrong_mask_length():
    datasets = [make_dataset("a", n_trace=2)]
    tau_mask = [np.array([True, False])]

    with pytest.raises(ValueError, match="length 3"):
        validate_tau_mask(datasets, tau_mask, n_tau=3)