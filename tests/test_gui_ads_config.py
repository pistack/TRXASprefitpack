import numpy as np
import pytest
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.gui.ads_config import ADSConfig, ADSResult
from TRXASprefitpack.gui.rate_model import RateEdge, RateModelSpec


def make_rate_model():
    return RateModelSpec(
        species=("A", "B"),
        edges=(RateEdge("A", "B", 1.0),),
        y0 = np.array([1.0, 0.0]),
    )


def make_ads_config(**kwargs):
    base = {
        "mode": "dads",
        "irf": "g",
        "fwhm": 0.12,
        "eta": None,
        "t0": 0.0,
        "tau": np.array([1.0, 10.0]),
        "base": True,
        "cond_num": 0.0,
        "y0": None,
        "exclude": None
    }
    base.update(kwargs)
    return ADSConfig(**base)


def make_ads_result(**kwargs):
    energy = np.array([100.0, 101.0, 102.0])
    time = np.array([0.0, 1.0])
    intensity = np.array(
        [
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ]
    )
    eps = np.full_like(intensity, 0.1)
    spectra = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
    )
    fit = intensity.copy()

    base = {
        "mode": "dads",
        "energy": energy,
        "time": time,
        "intensity": intensity,
        "eps": eps,
        "spectra": spectra,
        "fit": fit,
        "spectrum_names": ("comp_1", "comp_2"),
    }
    base.update(kwargs)
    return ADSResult(**base)


def test_ads_config_accepts_dads_mode():
    config = make_ads_config()

    assert config.mode == "dads"
    assert config.irf == "g"
    assert config.fwhm == pytest.approx(0.12)
    assert config.eta is None
    assert config.t0 == pytest.approx(0.0)
    np.testing.assert_allclose(config.tau, np.array([1.0, 10.0]))
    assert config.base is True
    assert config.cond_num == pytest.approx(0.0)
    assert config.rate_model is None


@pytest.mark.parametrize("mode", ["dads", "dads_svd"])
def test_ads_config_accepts_dads_modes(mode):
    config = make_ads_config(mode=mode)

    assert config.mode == mode
    np.testing.assert_allclose(config.tau, np.array([1.0, 10.0]))
    assert config.y0 is None
    assert config.exclude is None

@pytest.mark.parametrize("mode", ["sads", "sads_svd"])
def test_ads_config_accepts_sads_modes(mode):
    config = make_ads_config(mode=mode, y0=np.array([1.0, 0.0, 0.0]),
                             exclude=(-1,),)

    assert config.mode == mode
    np.testing.assert_allclose(config.tau, np.array([1.0, 10.0]))
    np.testing.assert_allclose(config.y0, np.array([1.0, 0.0, 0.0]))
    assert config.exclude == (2,)


@pytest.mark.parametrize("mode", ["custom_sads", "custom_sads_svd"])
def test_ads_config_accepts_custom_rate_modes(mode):
    model = make_rate_model()

    config = make_ads_config(
        mode=mode,
        tau=None,
        rate_model=model,
    )

    assert config.mode == mode
    assert config.tau is None
    assert config.rate_model is model


def test_ads_config_normalizes_case():
    config = make_ads_config(mode="DADS", irf="G")

    assert config.mode == "dads"
    assert config.irf == "g"


def test_ads_config_accepts_pseudo_voigt_with_eta():
    config = make_ads_config(irf="pv", eta=0.4)

    assert config.irf == "pv"
    assert config.eta == pytest.approx(0.4)


def test_ads_config_rejects_pseudo_voigt_without_eta():
    with pytest.raises(ValueError, match="eta"):
        make_ads_config(irf="pv", eta=None)


def test_ads_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        make_ads_config(mode="dads_osc")


def test_ads_config_rejects_invalid_irf():
    with pytest.raises(ValueError, match="irf"):
        make_ads_config(irf="bad")


@pytest.mark.parametrize("fwhm", [0.0, -0.1, np.nan, np.inf])
def test_ads_config_rejects_invalid_fwhm(fwhm):
    with pytest.raises(ValueError, match="fwhm"):
        make_ads_config(fwhm=fwhm)


@pytest.mark.parametrize("t0", [np.nan, np.inf])
def test_ads_config_rejects_invalid_t0(t0):
    with pytest.raises(ValueError, match="t0"):
        make_ads_config(t0=t0)


@pytest.mark.parametrize("cond_num", [-1.0, np.nan, np.inf])
def test_ads_config_rejects_invalid_cond_num(cond_num):
    with pytest.raises(ValueError, match="cond_num"):
        make_ads_config(cond_num=cond_num)


@pytest.mark.parametrize(
    "tau",
    [
        None,
        np.array([]),
        np.array([1.0, 0.0]),
        np.array([1.0, -1.0]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_ads_config_rejects_invalid_tau_for_standard_modes(tau):
    with pytest.raises(ValueError, match="tau"):
        make_ads_config(mode="dads", tau=tau)


def test_ads_config_rejects_tau_for_custom_mode():
    with pytest.raises(ValueError, match="tau"):
        make_ads_config(
            mode="custom_sads",
            tau=np.array([1.0]),
            rate_model=make_rate_model(),
        )


def test_ads_config_rejects_missing_rate_model_for_custom_mode():
    with pytest.raises(ValueError, match="rate_model"):
        make_ads_config(
            mode="custom_sads",
            tau=None,
            rate_model=None,
        )


def test_ads_config_rejects_rate_model_for_standard_mode():
    with pytest.raises(ValueError, match="rate_model"):
        make_ads_config(
            mode="dads",
            rate_model=make_rate_model(),
        )


def test_ads_result_accepts_valid_arrays():
    result = make_ads_result()

    assert result.mode == "dads"
    assert result.n_energy == 3
    assert result.n_time == 2
    assert result.n_component == 2
    assert result.has_svd is False
    assert result.spectrum_names == ("comp_1", "comp_2")


def test_ads_result_accepts_svd_arrays():
    svd_u = np.ones((3, 2))
    svd_s = np.array([10.0, 1.0])
    svd_vh = np.ones((2, 2))

    result = make_ads_result(
        svd_u=svd_u,
        svd_s=svd_s,
        svd_vh=svd_vh,
    )

    assert result.has_svd is True
    np.testing.assert_allclose(result.svd_u, svd_u)
    np.testing.assert_allclose(result.svd_s, svd_s)
    np.testing.assert_allclose(result.svd_vh, svd_vh)


def test_ads_result_accepts_metadata_copy():
    metadata = {"source": "test"}
    result = make_ads_result(model_metadata=metadata)

    assert result.model_metadata == metadata
    assert result.model_metadata is not metadata


def test_ads_result_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        make_ads_result(mode="dads_osc")


def test_ads_result_rejects_non_1d_energy():
    with pytest.raises(ValueError, match="energy"):
        make_ads_result(energy=np.array([[100.0, 101.0, 102.0]]))


def test_ads_result_rejects_non_1d_time():
    with pytest.raises(ValueError, match="time"):
        make_ads_result(time=np.array([[0.0, 1.0]]))


def test_ads_result_rejects_non_2d_intensity():
    with pytest.raises(ValueError, match="intensity"):
        make_ads_result(intensity=np.array([1.0, 2.0, 3.0]))


def test_ads_result_rejects_non_2d_eps():
    with pytest.raises(ValueError, match="eps"):
        make_ads_result(eps=np.array([0.1, 0.1, 0.1]))


def test_ads_result_rejects_wrong_intensity_shape():
    with pytest.raises(ValueError, match=r"\(n_energy, n_time\)"):
        make_ads_result(intensity=np.ones((2, 3)), eps=np.ones((2, 3)))


def test_ads_result_rejects_mismatched_eps_shape():
    with pytest.raises(ValueError, match="same shape"):
        make_ads_result(eps=np.ones((3, 3)))


def test_ads_result_rejects_nonpositive_eps():
    eps = np.array(
        [
            [0.1, 0.1],
            [0.1, 0.0],
            [0.1, 0.1],
        ]
    )

    with pytest.raises(ValueError, match="positive"):
        make_ads_result(eps=eps)


def test_ads_result_rejects_non_2d_spectra():
    with pytest.raises(ValueError, match="spectra"):
        make_ads_result(spectra=np.array([0.1, 0.2, 0.3]))


def test_ads_result_rejects_wrong_spectra_energy_length():
    spectra = np.ones((2, 2))

    with pytest.raises(ValueError, match="spectra.shape"):
        make_ads_result(spectra=spectra)


def test_ads_result_rejects_wrong_spectrum_names_length():
    with pytest.raises(ValueError, match="spectrum_names"):
        make_ads_result(spectrum_names=("only_one",))


def test_ads_result_rejects_wrong_fit_shape():
    with pytest.raises(ValueError, match="fit shape"):
        make_ads_result(fit=np.ones((3, 3)))


def test_ads_result_rejects_partial_svd_arrays():
    with pytest.raises(ValueError, match="provided together"):
        make_ads_result(
            svd_u=np.ones((3, 2)),
            svd_s=np.array([1.0, 0.5]),
            svd_vh=None,
        )


def test_ads_result_rejects_wrong_svd_u_shape():
    with pytest.raises(ValueError, match="svd_u shape"):
        make_ads_result(
            svd_u=np.ones((2, 2)),
            svd_s=np.array([1.0, 0.5]),
            svd_vh=np.ones((2, 2)),
        )


def test_ads_result_rejects_wrong_svd_vh_shape():
    with pytest.raises(ValueError, match="svd_vh shape"):
        make_ads_result(
            svd_u=np.ones((3, 2)),
            svd_s=np.array([1.0, 0.5]),
            svd_vh=np.ones((2, 3)),
        )

@pytest.mark.parametrize("mode", ["sads", "sads_svd"])
def test_standard_sads_requires_y0(mode):
    with pytest.raises(ValueError, match="y0"):
        make_ads_config(
            mode=mode,
            y0=None,
        )


def test_standard_sads_rejects_wrong_y0_length():
    with pytest.raises(ValueError, match="shape"):
        make_ads_config(
            mode="sads",
            y0=np.array([1.0, 0.0]),
        )


def test_standard_sads_rejects_non_1d_y0():
    with pytest.raises(ValueError, match="1D"):
        make_ads_config(
            mode="sads",
            y0=np.array([[1.0, 0.0, 0.0]]),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_standard_sads_rejects_nonfinite_y0(bad_value):
    with pytest.raises(ValueError, match="finite"):
        make_ads_config(
            mode="sads",
            y0=np.array([1.0, bad_value, 0.0]),
        )

def test_dads_rejects_y0():
    with pytest.raises(ValueError, match="y0"):
        make_ads_config(
            mode="dads",
            y0=np.array([1.0, 0.0, 0.0]),
        )


def test_dads_rejects_exclude():
    with pytest.raises(ValueError, match="exclude"):
        make_ads_config(
            mode="dads",
            exclude=(-1,),
        )


def test_standard_sads_normalizes_negative_exclude():
    config = make_ads_config(
        mode="sads",
        y0=np.array([1.0, 0.0, 0.0]),
        exclude=(0, -1),
    )

    assert config.exclude == (0, 2)


def test_standard_sads_normalizes_empty_exclude_to_none():
    config = make_ads_config(
        mode="sads",
        y0=np.array([1.0, 0.0, 0.0]),
        exclude=(),
    )

    assert config.exclude is None


@pytest.mark.parametrize(
    "exclude",
    [
        (3,),
        (-4,),
        (0, 0),
        (-1, 2),
        (0.5,),
        (True,),
    ],
)
def test_standard_sads_rejects_invalid_exclude(exclude):
    with pytest.raises(ValueError, match="exclude"):
        make_ads_config(
            mode="sads",
            y0=np.array([1.0, 0.0, 0.0]),
            exclude=exclude,
        )


def test_custom_sads_uses_rate_model_y0():
    model = make_rate_model()

    config = make_ads_config(
        mode="custom_sads",
        tau=None,
        rate_model=model,
        exclude=(-1,),
    )

    assert config.rate_model is model
    assert config.y0 is None
    assert config.exclude == (1,)

    np.testing.assert_allclose(
        config.rate_model.y0,
        np.array([1.0, 0.0]),
    )


def test_custom_sads_rejects_top_level_y0():
    with pytest.raises(ValueError, match="Top-level y0"):
        make_ads_config(
            mode="custom_sads",
            tau=None,
            rate_model=make_rate_model(),
            y0=np.array([1.0, 0.0]),
        )


def test_custom_sads_requires_rate_model_spec():
    with pytest.raises(TypeError, match="RateModelSpec"):
        make_ads_config(
            mode="custom_sads",
            tau=None,
            rate_model=object(),
        )


def test_ads_result_accepts_spectra_eps():
    spectra_eps = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
            [0.05, 0.06],
        ]
    )

    result = make_ads_result(
        spectra_eps=spectra_eps,
    )

    np.testing.assert_allclose(
        result.spectra_eps,
        spectra_eps,
    )


def test_ads_result_accepts_zero_spectra_eps():
    result = make_ads_result(
        spectra_eps=np.zeros((3, 2)),
    )

    np.testing.assert_allclose(
        result.spectra_eps,
        np.zeros((3, 2)),
    )


def test_ads_result_rejects_wrong_spectra_eps_shape():
    with pytest.raises(ValueError, match="spectra_eps shape"):
        make_ads_result(
            spectra_eps=np.ones((2, 3)),
        )


def test_ads_result_rejects_negative_spectra_eps():
    spectra_eps = np.array(
        [
            [0.01, 0.02],
            [0.03, -0.01],
            [0.05, 0.06],
        ]
    )

    with pytest.raises(ValueError, match="non-negative"):
        make_ads_result(
            spectra_eps=spectra_eps,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_ads_result_rejects_nonfinite_spectra_eps(bad_value):
    spectra_eps = np.full((3, 2), 0.1)
    spectra_eps[1, 1] = bad_value

    with pytest.raises(ValueError, match="finite"):
        make_ads_result(
            spectra_eps=spectra_eps,
        )