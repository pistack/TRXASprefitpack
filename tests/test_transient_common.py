import numpy as np
import pytest
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.driver._transient_common import (
    calc_covariance_from_hessian,
    calc_individual_chi2,
    count_total_scans,
    default_dataset_names,
    get_num_irf,
    make_fixed_mask,
    make_lsq_bounds,
    prepare_lsq_kwargs,
    validate_transient_driver_options,
)


def test_validate_transient_driver_options_accepts_valid_options():
    validate_transient_driver_options(None, "trf", "g")
    validate_transient_driver_options("ampgo", "dogbox", "c")
    validate_transient_driver_options("basinhopping", "lm", "pv")


def test_validate_transient_driver_options_rejects_invalid_global_method():
    with pytest.raises(ValueError, match="Unsupported global optimization"):
        validate_transient_driver_options("bad", "trf", "g")


def test_validate_transient_driver_options_rejects_invalid_lsq_method():
    with pytest.raises(ValueError, match="Invalid local least-squares"):
        validate_transient_driver_options(None, "bad", "g")


def test_validate_transient_driver_options_rejects_invalid_irf():
    with pytest.raises(ValueError, match="Unsupported instrumental response"):
        validate_transient_driver_options(None, "trf", "bad")


def test_get_num_irf():
    assert get_num_irf("g") == 1
    assert get_num_irf("c") == 1
    assert get_num_irf("pv") == 2


def test_get_num_irf_rejects_invalid_irf():
    with pytest.raises(ValueError):
        get_num_irf("bad")


def test_make_fixed_mask():
    bounds = [(0.0, 1.0), (2.0, 2.0), (-1.0, 3.0)]
    fixed = make_fixed_mask(bounds)

    np.testing.assert_array_equal(fixed, np.array([False, True, False]))


def test_make_lsq_bounds_expands_fixed_positive_bound():
    bounds = [(1.0, 1.0)]
    lower, upper = make_lsq_bounds(bounds)

    assert lower == [1.0]
    assert upper[0] > 1.0


def test_make_lsq_bounds_expands_fixed_negative_bound():
    bounds = [(-1.0, -1.0)]
    lower, upper = make_lsq_bounds(bounds)

    assert lower == [-1.0]
    assert upper[0] > -1.0


def test_make_lsq_bounds_preserves_unfixed_bounds():
    bounds = [(0.0, 2.0), (-1.0, 1.0)]
    lower, upper = make_lsq_bounds(bounds)

    assert lower == [0.0, -1.0]
    assert upper == [2.0, 1.0]


def test_prepare_lsq_kwargs_does_not_mutate_user_input():
    kwargs = {"args": ("old",), "kwargs": {"old": True}, "max_nfev": 10}
    prepared = prepare_lsq_kwargs(kwargs, args=("new",))

    assert prepared["args"] == ("new",)
    assert "kwargs" not in prepared
    assert prepared["max_nfev"] == 10

    assert kwargs["args"] == ("old",)
    assert kwargs["kwargs"] == {"old": True}


def test_calc_individual_chi2():
    intensity = [
        np.zeros((3, 2)),
        np.zeros((2, 1)),
    ]

    # dataset 1 trace 1: [1,2,3] -> 14
    # dataset 1 trace 2: [4,5,6] -> 77
    # dataset 2 trace 1: [7,8]   -> 113
    chi = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

    chi2_ind, red_chi2_ind = calc_individual_chi2(
        chi,
        intensity,
        num_param_ind=1,
    )

    np.testing.assert_allclose(chi2_ind[0], np.array([14.0, 77.0]))
    np.testing.assert_allclose(chi2_ind[1], np.array([113.0]))

    np.testing.assert_allclose(red_chi2_ind[0], np.array([7.0, 38.5]))
    np.testing.assert_allclose(red_chi2_ind[1], np.array([113.0]))


def test_calc_covariance_from_hessian_with_one_fixed_parameter():
    hessian = np.diag([2.0, 4.0, 8.0])
    fixed_mask = np.array([False, True, False])
    red_chi2 = 2.0

    cov, cov_scaled, corr, param_eps = calc_covariance_from_hessian(
        hessian,
        fixed_mask,
        red_chi2,
    )

    expected_cov = np.zeros((3, 3))
    expected_cov[0, 0] = 1 / 2.0
    expected_cov[2, 2] = 1 / 8.0

    np.testing.assert_allclose(cov, expected_cov)
    np.testing.assert_allclose(cov_scaled, red_chi2 * expected_cov)
    np.testing.assert_allclose(param_eps, np.sqrt(np.diag(cov_scaled)))

    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[2, 2] == pytest.approx(1.0)
    assert corr[1, 1] == pytest.approx(0.0)


def test_default_dataset_names():
    names = default_dataset_names(3)
    assert list(names) == ["dataset_1", "dataset_2", "dataset_3"]


def test_count_total_scans():
    intensity = [
        np.zeros((10, 2)),
        np.zeros((8, 3)),
    ]

    assert count_total_scans(intensity) == 5