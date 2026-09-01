import os
import sys

import numpy as np
import pytest


path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.ads_config import ADSConfig, ADSResult
from TRXASprefitpack.gui.models import EScanDataset
from TRXASprefitpack.gui.rate_model import RateEdge, RateModelSpec
import TRXASprefitpack.gui.ads_job as ads_job


def make_dataset():
    energy = np.array([100.0, 101.0, 102.0])
    time = np.array([-0.5, 0.0, 1.0, 5.0])
    intensity = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ]
    )
    eps = np.full_like(intensity, 0.1)

    return EScanDataset(
        name="sample",
        energy=energy,
        time=time,
        intensity=intensity,
        eps=eps,
    )


def make_rate_model():
    return RateModelSpec(
        species=("A", "B"),
        edges=(RateEdge("A", "B", 1.0),),
        y0=np.array([1.0, 0.0]),
    )


def make_config(mode, **kwargs):
    values = {
        "mode": mode,
        "irf": "g",
        "fwhm": 0.12,
        "eta": None,
        "t0": 0.25,
        "tau": np.array([2.0]),
        "base": True,
        "cond_num": 0.1,
        "rate_model": None,
        "y0": None,
        "exclude": None,
    }
    values.update(kwargs)

    return ADSConfig(**values)


def test_run_ads_config_rejects_wrong_config_type():
    with pytest.raises(TypeError, match="ADSConfig"):
        ads_job.run_ads_config(
            object(),
            make_dataset(),
        )


def test_run_ads_config_rejects_wrong_dataset_type():
    config = make_config("dads")

    with pytest.raises(TypeError, match="EScanDataset"):
        ads_job.run_ads_config(
            config,
            object(),
        )


def test_run_dads_routes_arguments_and_transposes(monkeypatch):
    dataset = make_dataset()
    config = make_config("dads")
    received = {}

    spectra_raw = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    spectra_eps_raw = np.full((2, 3), 0.01)
    fit = dataset.intensity.copy()

    def fake_dads(**kwargs):
        received.update(kwargs)
        return spectra_raw, spectra_eps_raw, fit

    monkeypatch.setattr(ads_job, "dads", fake_dads)

    result = ads_job.run_ads_config(
        config,
        dataset,
    )

    assert isinstance(result, ADSResult)
    assert result.mode == "dads"
    assert result.spectrum_names == ("decay_1", "base")
    assert result.has_svd is False

    np.testing.assert_allclose(
        received["escan_time"],
        dataset.time - config.t0,
    )
    np.testing.assert_allclose(
        received["tau"],
        config.tau,
    )
    np.testing.assert_allclose(
        received["intensity"],
        dataset.intensity,
    )
    np.testing.assert_allclose(
        received["eps"],
        dataset.eps,
    )

    assert received["fwhm"] == pytest.approx(config.fwhm)
    assert received["base"] is True
    assert received["irf"] == "g"
    assert received["eta"] is None

    np.testing.assert_allclose(
        result.spectra,
        spectra_raw.T,
    )
    np.testing.assert_allclose(
        result.spectra_eps,
        spectra_eps_raw.T,
    )
    np.testing.assert_allclose(
        result.fit,
        fit,
    )
    np.testing.assert_allclose(
        result.time,
        dataset.time,
    )
    np.testing.assert_allclose(
        result.model_metadata["model_time"],
        dataset.time - config.t0,
    )


def test_run_dads_without_base_names_only_decays(monkeypatch):
    dataset = make_dataset()
    config = make_config(
        "dads",
        base=False,
    )

    def fake_dads(**kwargs):
        spectra = np.array([[0.1, 0.2, 0.3]])
        spectra_eps = np.full((1, 3), 0.01)
        return spectra, spectra_eps, dataset.intensity.copy()

    monkeypatch.setattr(ads_job, "dads", fake_dads)

    result = ads_job.run_ads_config(
        config,
        dataset,
    )

    assert result.spectrum_names == ("decay_1",)


def test_run_dads_svd_routes_and_preserves_orientation(monkeypatch):
    dataset = make_dataset()
    config = make_config("dads_svd")
    received = {}

    spectra = np.array(
        [
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6],
        ]
    )
    fit = dataset.intensity.copy()

    def fake_dads_svd(**kwargs):
        received.update(kwargs)
        return spectra, fit

    svd_u = np.ones((3, 2))
    svd_s = np.array([10.0, 1.0])
    svd_vh = np.ones((2, 4))

    def fake_svd(intensity, cond_num):
        np.testing.assert_allclose(
            intensity,
            dataset.intensity,
        )
        assert cond_num == pytest.approx(config.cond_num)
        return svd_u, svd_s, svd_vh

    monkeypatch.setattr(
        ads_job,
        "dads_svd",
        fake_dads_svd,
    )
    monkeypatch.setattr(
        ads_job,
        "_truncated_data_svd",
        fake_svd,
    )

    result = ads_job.run_ads_config(
        config,
        dataset,
    )

    np.testing.assert_allclose(
        received["escan_time"],
        dataset.time - config.t0,
    )

    assert received["cond_num"] == pytest.approx(0.1)
    assert "eps" not in received

    np.testing.assert_allclose(
        result.spectra,
        spectra,
    )
    assert result.spectra_eps is None
    assert result.has_svd is True

    np.testing.assert_allclose(result.svd_u, svd_u)
    np.testing.assert_allclose(result.svd_s, svd_s)
    np.testing.assert_allclose(result.svd_vh, svd_vh)


def test_run_standard_sads_routes_solver_and_driver(monkeypatch):
    dataset = make_dataset()
    config = make_config(
        "sads",
        y0=np.array([1.0, 0.0]),
        exclude=(-1,),
    )
    solver_received = {}
    driver_received = {}

    eigval = np.array([-0.5, 0.0])
    eigenvectors = np.eye(2)
    coefficients = np.array([1.0, 0.0])

    def fake_solve_seq_model(*, tau, y0):
        solver_received["tau"] = tau
        solver_received["y0"] = y0
        return eigval, eigenvectors, coefficients

    spectra_raw = np.array([[0.1, 0.2, 0.3]])
    spectra_eps_raw = np.full((1, 3), 0.01)
    fit = dataset.intensity.copy()

    def fake_sads(**kwargs):
        driver_received.update(kwargs)
        return spectra_raw, spectra_eps_raw, fit

    monkeypatch.setattr(
        ads_job,
        "solve_seq_model",
        fake_solve_seq_model,
    )
    monkeypatch.setattr(ads_job, "sads", fake_sads)

    result = ads_job.run_ads_config(
        config,
        dataset,
    )

    np.testing.assert_allclose(
        solver_received["tau"],
        config.tau,
    )
    np.testing.assert_allclose(
        solver_received["y0"],
        config.y0,
    )

    assert driver_received["exclude"] == (1,)

    np.testing.assert_allclose(
        driver_received["eigval"],
        eigval,
    )
    np.testing.assert_allclose(
        driver_received["V"],
        eigenvectors,
    )
    np.testing.assert_allclose(
        driver_received["c"],
        coefficients,
    )
    np.testing.assert_allclose(
        driver_received["escan_time"],
        dataset.time - config.t0,
    )

    assert result.spectrum_names == ("species_1",)

    np.testing.assert_allclose(
        result.spectra,
        spectra_raw.T,
    )
    np.testing.assert_allclose(
        result.spectra_eps,
        spectra_eps_raw.T,
    )

    assert (
        result.model_metadata["rate_model_kind"]
        == "sequential"
    )


def test_run_standard_sads_svd_routes_driver(monkeypatch):
    dataset = make_dataset()
    config = make_config(
        "sads_svd",
        y0=np.array([1.0, 0.0]),
        exclude=None,
    )
    received = {}

    monkeypatch.setattr(
        ads_job,
        "solve_seq_model",
        lambda **kwargs: (
            np.array([-0.5, 0.0]),
            np.eye(2),
            np.array([1.0, 0.0]),
        ),
    )

    spectra = np.ones((3, 2))
    fit = dataset.intensity.copy()

    def fake_sads_svd(**kwargs):
        received.update(kwargs)
        return spectra, fit

    monkeypatch.setattr(
        ads_job,
        "sads_svd",
        fake_sads_svd,
    )
    monkeypatch.setattr(
        ads_job,
        "_truncated_data_svd",
        lambda intensity, cond_num: (
            np.ones((3, 1)),
            np.array([1.0]),
            np.ones((1, 4)),
        ),
    )

    result = ads_job.run_ads_config(
        config,
        dataset,
    )

    assert received["exclude"] is None
    assert received["cond_num"] == pytest.approx(0.1)
    assert "eps" not in received

    assert result.spectrum_names == (
        "species_1",
        "species_2",
    )
    assert result.spectra_eps is None
    assert result.has_svd is True


def test_run_custom_sads_routes_rate_model(monkeypatch):
    dataset = make_dataset()
    rate_model = make_rate_model()
    config = make_config(
        "custom_sads",
        tau=None,
        rate_model=rate_model,
        exclude=(-1,),
    )

    rate_matrix = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    eigval = np.array([-1.0, 0.0])
    eigenvectors = np.eye(2)
    coefficients = np.array([1.0, 0.0])

    received = {}

    def fake_build(spec):
        assert spec is rate_model
        return rate_matrix

    def fake_solve(matrix, y0):
        np.testing.assert_allclose(
            matrix,
            rate_matrix,
        )
        np.testing.assert_allclose(
            y0,
            rate_model.y0,
        )
        return eigval, eigenvectors, coefficients

    def fake_sads(**kwargs):
        received.update(kwargs)
        return (
            np.array([[0.1, 0.2, 0.3]]),
            np.full((1, 3), 0.01),
            dataset.intensity.copy(),
        )

    monkeypatch.setattr(
        ads_job,
        "build_rate_matrix",
        fake_build,
    )
    monkeypatch.setattr(
        ads_job,
        "solve_rate_model_real",
        fake_solve,
    )
    monkeypatch.setattr(ads_job, "sads", fake_sads)

    result = ads_job.run_ads_config(
        config,
        dataset,
    )

    assert received["exclude"] == (1,)
    assert result.spectrum_names == ("A",)

    np.testing.assert_allclose(
        result.model_metadata["rate_matrix"],
        rate_matrix,
    )
    assert (
        result.model_metadata["rate_model_kind"]
        == "custom"
    )


def test_run_custom_sads_svd_routes_driver(monkeypatch):
    dataset = make_dataset()
    rate_model = make_rate_model()
    config = make_config(
        "custom_sads_svd",
        tau=None,
        rate_model=rate_model,
    )

    monkeypatch.setattr(
        ads_job,
        "build_rate_matrix",
        lambda spec: np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )
    monkeypatch.setattr(
        ads_job,
        "solve_rate_model_real",
        lambda matrix, y0: (
            np.array([-1.0, 0.0]),
            np.eye(2),
            np.array([1.0, 0.0]),
        ),
    )

    received = {}

    def fake_sads_svd(**kwargs):
        received.update(kwargs)
        return (
            np.ones((3, 2)),
            dataset.intensity.copy(),
        )

    monkeypatch.setattr(
        ads_job,
        "sads_svd",
        fake_sads_svd,
    )
    monkeypatch.setattr(
        ads_job,
        "_truncated_data_svd",
        lambda intensity, cond_num: (
            np.ones((3, 1)),
            np.array([1.0]),
            np.ones((1, 4)),
        ),
    )

    result = ads_job.run_ads_config(
        config,
        dataset,
    )

    assert received["exclude"] is None
    assert result.spectrum_names == ("A", "B")
    assert result.spectra_eps is None
    assert result.has_svd is True


def test_run_sads_rejects_excluding_all_species(
    monkeypatch,
):
    dataset = make_dataset()
    config = make_config(
        "sads",
        y0=np.array([1.0, 0.0]),
        exclude=(0, 1),
    )

    monkeypatch.setattr(
        ads_job,
        "solve_seq_model",
        lambda **kwargs: (
            np.array([-0.5, 0.0]),
            np.eye(2),
            np.array([1.0, 0.0]),
        ),
    )

    with pytest.raises(
        ValueError,
        match="At least one species",
    ):
        ads_job.run_ads_config(
            config,
            dataset,
        )


def test_run_sads_rejects_nonfinite_solver_output(
    monkeypatch,
):
    dataset = make_dataset()
    config = make_config(
        "sads",
        y0=np.array([1.0, 0.0]),
    )

    monkeypatch.setattr(
        ads_job,
        "solve_seq_model",
        lambda **kwargs: (
            np.array([-0.5, 0.0]),
            np.array(
                [
                    [1.0, np.nan],
                    [0.0, 1.0],
                ]
            ),
            np.array([1.0, 0.0]),
        ),
    )

    with pytest.raises(ValueError, match="finite"):
        ads_job.run_ads_config(
            config,
            dataset,
        )


def test_run_sads_rejects_complex_solver_output(
    monkeypatch,
):
    dataset = make_dataset()
    config = make_config(
        "sads",
        y0=np.array([1.0, 0.0]),
    )

    monkeypatch.setattr(
        ads_job,
        "solve_seq_model",
        lambda **kwargs: (
            np.array([-0.5 + 0.2j, 0.0]),
            np.eye(2),
            np.array([1.0, 0.0]),
        ),
    )

    with pytest.raises(ValueError, match="complex mode"):
        ads_job.run_ads_config(
            config,
            dataset,
        )


def test_run_sads_rejects_solution_not_matching_y0(
    monkeypatch,
):
    dataset = make_dataset()
    config = make_config(
        "sads",
        y0=np.array([1.0, 0.0]),
    )

    monkeypatch.setattr(
        ads_job,
        "solve_seq_model",
        lambda **kwargs: (
            np.array([-0.5, 0.0]),
            np.eye(2),
            np.array([0.0, 1.0]),
        ),
    )

    with pytest.raises(
        ValueError,
        match="reconstruct y0",
    ):
        ads_job.run_ads_config(
            config,
            dataset,
        )


def test_truncated_data_svd_uses_relative_cutoff():
    intensity = np.diag([4.0, 1.0])

    svd_u, svd_s, svd_vh = (
        ads_job._truncated_data_svd(
            intensity,
            cond_num=0.5,
        )
    )

    assert svd_u.shape == (2, 1)
    assert svd_s.shape == (1,)
    assert svd_vh.shape == (1, 2)

    np.testing.assert_allclose(
        svd_s,
        np.array([4.0]),
    )


def test_truncated_data_svd_keeps_full_reconstruction():
    intensity = np.array(
        [
            [3.0, 1.0],
            [1.0, 2.0],
        ]
    )

    svd_u, svd_s, svd_vh = (
        ads_job._truncated_data_svd(
            intensity,
            cond_num=0.0,
        )
    )

    reconstructed = (
        svd_u
        @ np.diag(svd_s)
        @ svd_vh
    )

    np.testing.assert_allclose(
        reconstructed,
        intensity,
    )