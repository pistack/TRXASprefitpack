import numpy as np
import pytest

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.gui.fit_config import (
    FitConfigBundle,
    FitTransientExpConfig,
    _normalize_tau_mask_shape_only,
)


def make_config(**kwargs):
    base = {
        "irf": "g",
        "fwhm_init": 0.12,
        "t0_init": np.array([0.0]),
        "tau_init": np.array([1.0, 10.0]),
        "base": True,
    }
    base.update(kwargs)
    return FitTransientExpConfig(**base)


def test_fit_transient_exp_config_accepts_gaussian_model():
    config = make_config()

    assert config.irf == "g"
    np.testing.assert_allclose(config.fwhm_init, np.array([0.12]))
    np.testing.assert_allclose(config.t0_init, np.array([0.0]))
    np.testing.assert_allclose(config.tau_init, np.array([1.0, 10.0]))
    assert config.base is True
    assert config.method_glb is None
    assert config.method_lsq == "trf"
    assert config.same_t0 is False


def test_fit_transient_exp_config_normalizes_irf_case():
    config = make_config(irf="PV", fwhm_init=np.array([0.1, 0.2]))

    assert config.irf == "pv"


def test_fit_transient_exp_config_accepts_pseudo_voigt_fwhm_pair():
    config = make_config(irf="pv", fwhm_init=np.array([0.1, 0.2]))

    np.testing.assert_allclose(config.fwhm_init, np.array([0.1, 0.2]))


def test_fit_transient_exp_config_accepts_cauchy_model():
    config = make_config(irf="c", fwhm_init=0.2)

    assert config.irf == "c"
    np.testing.assert_allclose(config.fwhm_init, np.array([0.2]))


def test_fit_transient_exp_config_accepts_optimizer_methods():
    config = make_config(method_glb="ampgo", method_lsq="dogbox")

    assert config.method_glb == "ampgo"
    assert config.method_lsq == "dogbox"


def test_fit_transient_exp_config_accepts_bounds():
    config = make_config(
        bound_fwhm=[(0.05, 0.2)],
        bound_t0=[(-1.0, 1.0)],
        bound_tau=[(0.1, 2.0), (2.0, 20.0)],
    )

    assert config.bound_fwhm == [(0.05, 0.2)]
    assert config.bound_t0 == [(-1.0, 1.0)]
    assert config.bound_tau == [(0.1, 2.0), (2.0, 20.0)]


def test_fit_transient_exp_config_accepts_tau_none_for_baseline_only():
    config = make_config(tau_init=None, base=True)

    assert config.tau_init is None


def test_fit_transient_exp_config_rejects_invalid_irf():
    with pytest.raises(ValueError, match="irf"):
        make_config(irf="bad")


def test_fit_transient_exp_config_rejects_invalid_global_optimizer():
    with pytest.raises(ValueError, match="method_glb"):
        make_config(method_glb="bad")


def test_fit_transient_exp_config_rejects_invalid_lsq_method():
    with pytest.raises(ValueError, match="method_lsq"):
        make_config(method_lsq="bad")


def test_fit_transient_exp_config_rejects_wrong_fwhm_count_for_gaussian():
    with pytest.raises(ValueError, match="fwhm_init must contain 1"):
        make_config(irf="g", fwhm_init=np.array([0.1, 0.2]))


def test_fit_transient_exp_config_rejects_wrong_fwhm_count_for_pv():
    with pytest.raises(ValueError, match="fwhm_init must contain 2"):
        make_config(irf="pv", fwhm_init=0.1)


@pytest.mark.parametrize("fwhm_init", [0.0, -0.1, np.nan, np.inf])
def test_fit_transient_exp_config_rejects_invalid_fwhm(fwhm_init):
    with pytest.raises(ValueError):
        make_config(fwhm_init=fwhm_init)


def test_fit_transient_exp_config_rejects_nonfinite_t0():
    with pytest.raises(ValueError, match="t0_init"):
        make_config(t0_init=np.array([0.0, np.nan]))


@pytest.mark.parametrize(
    "tau_init",
    [
        np.array([1.0, 0.0]),
        np.array([1.0, -1.0]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_fit_transient_exp_config_rejects_invalid_tau(tau_init):
    with pytest.raises(ValueError):
        make_config(tau_init=tau_init)


def test_fit_transient_exp_config_rejects_bound_tau_when_tau_none():
    with pytest.raises(ValueError, match="bound_tau"):
        make_config(tau_init=None, bound_tau=[(0.0, 1.0)])


def test_fit_transient_exp_config_rejects_tau_mask_when_tau_none():
    with pytest.raises(ValueError, match="tau_mask"):
        make_config(tau_init=None, tau_mask=[np.array([True, False])])


def test_fit_transient_exp_config_accepts_tau_mask_shape_only():
    config = make_config(
        tau_init=np.array([1.0, 10.0]),
        tau_mask=[
            np.array([True, False, True]),
            np.array([False, True, True]),
        ],
    )

    assert len(config.tau_mask) == 2
    np.testing.assert_array_equal(config.tau_mask[0], np.array([True, False, True]))
    np.testing.assert_array_equal(config.tau_mask[1], np.array([False, True, True]))


def test_fit_transient_exp_config_rejects_tau_mask_wrong_length():
    with pytest.raises(ValueError, match="length 3"):
        make_config(
            tau_init=np.array([1.0, 10.0]),
            tau_mask=[np.array([True, False])],
        )


def test_fit_transient_exp_config_rejects_tau_mask_non_1d():
    with pytest.raises(ValueError, match="1D"):
        make_config(
            tau_init=np.array([1.0, 10.0]),
            tau_mask=[np.array([[True, False]])],
        )


def test_normalize_tau_mask_shape_only_allows_none():
    assert _normalize_tau_mask_shape_only(None, n_tau=2) is None


def test_fit_transient_exp_config_rejects_init_outside_bounds():
    with pytest.raises(ValueError, match="outside bounds"):
        make_config(bound_tau=[(0.1, 2.0), (2.0, 5.0)])


def test_fit_config_bundle_accepts_config():
    config = make_config()
    bundle = FitConfigBundle(name="fit 1", config=config)

    assert bundle.name == "fit 1"
    assert bundle.config is config


def test_fit_config_bundle_rejects_empty_name():
    config = make_config()

    with pytest.raises(ValueError, match="name"):
        FitConfigBundle(name="", config=config)