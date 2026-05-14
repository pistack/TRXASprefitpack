import os
import sys
import numpy as np
import pytest

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.driver._input import (
    normalize_tscan_inputs,
    expected_t0_count,
    validate_t0_count,
)


def test_normalize_single_1d_trace():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    eps = np.array([0.1, 0.1, 0.1])

    t_list, y_list, eps_list = normalize_tscan_inputs(t, y, eps)

    assert len(t_list) == 1
    assert len(y_list) == 1
    assert len(eps_list) == 1

    np.testing.assert_allclose(t_list[0], t)
    assert y_list[0].shape == (3, 1)
    assert eps_list[0].shape == (3, 1)
    np.testing.assert_allclose(y_list[0][:, 0], y)
    np.testing.assert_allclose(eps_list[0][:, 0], eps)


def test_normalize_single_2d_dataset():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    eps = np.full_like(y, 0.1)

    t_list, y_list, eps_list = normalize_tscan_inputs(t, y, eps)

    assert len(t_list) == 1
    assert y_list[0].shape == (3, 2)
    assert eps_list[0].shape == (3, 2)
    np.testing.assert_allclose(y_list[0], y)
    np.testing.assert_allclose(eps_list[0], eps)


def test_normalize_multiple_datasets_with_1d_traces():
    t = [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.5, 1.0, 1.5]),
    ]
    y = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 1.5, 2.0, 2.5]),
    ]
    eps = [
        np.array([0.1, 0.1, 0.1]),
        np.array([0.2, 0.2, 0.2, 0.2]),
    ]

    t_list, y_list, eps_list = normalize_tscan_inputs(t, y, eps)

    assert len(t_list) == 2
    assert y_list[0].shape == (3, 1)
    assert y_list[1].shape == (4, 1)
    assert eps_list[0].shape == (3, 1)
    assert eps_list[1].shape == (4, 1)


def test_reject_mismatched_dataset_count():
    t = [np.array([0.0, 1.0])]
    y = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    eps = [np.array([0.1, 0.1])]

    with pytest.raises(ValueError, match="same number of datasets"):
        normalize_tscan_inputs(t, y, eps)


def test_reject_non_1d_time_axis():
    t = np.array([[0.0, 1.0]])
    y = np.array([1.0, 2.0])
    eps = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match=r"t\[0\] must be a 1D array"):
        normalize_tscan_inputs(t, y, eps)


def test_reject_mismatched_intensity_eps_shape():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    eps = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])

    with pytest.raises(ValueError, match="must have the same shape"):
        normalize_tscan_inputs(t, y, eps)


def test_reject_mismatched_time_length():
    t = np.array([0.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    eps = np.array([0.1, 0.1, 0.1])

    with pytest.raises(ValueError, match=r"shape\[0\] must match"):
        normalize_tscan_inputs(t, y, eps)


def test_reject_nonpositive_eps():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    eps = np.array([0.1, 0.0, 0.1])

    with pytest.raises(ValueError, match="positive"):
        normalize_tscan_inputs(t, y, eps)


def test_expected_t0_count_scanwise():
    intensity = [
        np.zeros((10, 2)),
        np.zeros((8, 3)),
    ]

    assert expected_t0_count(intensity, same_t0=False) == 5


def test_expected_t0_count_same_t0():
    intensity = [
        np.zeros((10, 2)),
        np.zeros((8, 3)),
    ]

    assert expected_t0_count(intensity, same_t0=True) == 2


def test_validate_t0_count_scanwise_passes():
    intensity = [
        np.zeros((10, 2)),
        np.zeros((8, 3)),
    ]
    t0_init = np.zeros(5)

    validate_t0_count(t0_init, intensity, same_t0=False)


def test_validate_t0_count_same_t0_passes():
    intensity = [
        np.zeros((10, 2)),
        np.zeros((8, 3)),
    ]
    t0_init = np.zeros(2)

    validate_t0_count(t0_init, intensity, same_t0=True)


def test_validate_t0_count_raises():
    intensity = [
        np.zeros((10, 2)),
        np.zeros((8, 3)),
    ]
    t0_init = np.zeros(4)

    with pytest.raises(ValueError, match="t0_init must contain 5"):
        validate_t0_count(t0_init, intensity, same_t0=False)