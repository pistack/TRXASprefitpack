import numpy as np
import pytest

import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.gui.fit_config import FitTransientExpConfig
from TRXASprefitpack.gui.fit_job import (
    _driver_fwhm_init,
    run_fit_transient_exp_config,
)
from TRXASprefitpack.gui.models import TScanDataset, TScanTrace


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


def test_driver_fwhm_init_returns_scalar_for_gaussian():
    config = make_config(irf="g", fwhm_init=0.12)

    assert _driver_fwhm_init(config) == pytest.approx(0.12)


def test_driver_fwhm_init_returns_scalar_for_cauchy():
    config = make_config(irf="c", fwhm_init=0.2)

    assert _driver_fwhm_init(config) == pytest.approx(0.2)


def test_driver_fwhm_init_returns_array_for_pseudo_voigt():
    config = make_config(irf="pv", fwhm_init=np.array([0.1, 0.2]))

    np.testing.assert_allclose(
        _driver_fwhm_init(config),
        np.array([0.1, 0.2]),
    )


def test_run_fit_transient_exp_config_calls_driver(monkeypatch):
    import TRXASprefitpack.gui.fit_job as fit_job

    dataset = make_dataset("dset1", n_trace=1)
    config = make_config(
        irf="g",
        fwhm_init=0.12,
        t0_init=np.array([0.0]),
        tau_init=np.array([1.0, 10.0]),
        base=True,
        method_glb=None,
        method_lsq="trf",
        same_t0=False,
    )

    captured = {}

    def fake_fit_transient_exp(**kwargs):
        captured.update(kwargs)
        return {"model": "decay", "ok": True}

    monkeypatch.setattr(fit_job, "fit_transient_exp", fake_fit_transient_exp)

    result = run_fit_transient_exp_config(config, [dataset])

    assert result == {"model": "decay", "ok": True}

    assert captured["irf"] == "g"
    assert captured["fwhm_init"] == pytest.approx(0.12)
    np.testing.assert_allclose(captured["t0_init"], np.array([0.0]))
    np.testing.assert_allclose(captured["tau_init"], np.array([1.0, 10.0]))
    assert captured["base"] is True
    assert captured["method_glb"] is None
    assert captured["method_lsq"] == "trf"
    assert captured["same_t0"] is False

    assert len(captured["t"]) == 1
    assert len(captured["intensity"]) == 1
    assert len(captured["eps"]) == 1

    np.testing.assert_allclose(captured["t"][0], np.array([0.0, 1.0, 2.0]))
    assert captured["intensity"][0].shape == (3, 1)
    assert captured["eps"][0].shape == (3, 1)
    assert list(captured["name_of_dset"]) == ["dset1"]


def test_run_fit_transient_exp_config_validates_t0_count_scanwise(monkeypatch):
    dataset = make_dataset("dset1", n_trace=2)

    config = make_config(
        t0_init=np.array([0.0]),  # should be 2 for same_t0=False
        same_t0=False,
    )

    with pytest.raises(ValueError, match="t0_init must contain 2"):
        run_fit_transient_exp_config(config, [dataset])


def test_run_fit_transient_exp_config_validates_t0_count_same_t0(monkeypatch):
    datasets = [
        make_dataset("dset1", n_trace=2),
        make_dataset("dset2", n_trace=3),
    ]

    config = make_config(
        t0_init=np.array([0.0]),  # should be 2 for same_t0=True
        same_t0=True,
    )

    with pytest.raises(ValueError, match="t0_init must contain 2"):
        run_fit_transient_exp_config(config, datasets)


def test_run_fit_transient_exp_config_accepts_same_t0(monkeypatch):
    import TRXASprefitpack.gui.fit_job as fit_job

    datasets = [
        make_dataset("dset1", n_trace=2),
        make_dataset("dset2", n_trace=3),
    ]

    config = make_config(
        t0_init=np.array([0.0, 0.1]),
        same_t0=True,
    )

    captured = {}

    def fake_fit_transient_exp(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(fit_job, "fit_transient_exp", fake_fit_transient_exp)

    result = run_fit_transient_exp_config(config, datasets)

    assert result == {"ok": True}
    assert captured["same_t0"] is True
    np.testing.assert_allclose(captured["t0_init"], np.array([0.0, 0.1]))
    assert list(captured["name_of_dset"]) == ["dset1", "dset2"]


def test_run_fit_transient_exp_config_validates_tau_mask_dataset_count():
    dataset = make_dataset("dset1", n_trace=1)

    config = make_config(
        tau_init=np.array([1.0, 10.0]),
        tau_mask=[
            np.array([True, False]),
            np.array([False, True]),
        ],
    )

    with pytest.raises(ValueError, match="one mask per dataset"):
        run_fit_transient_exp_config(config, [dataset])


def test_run_fit_transient_exp_config_passes_tau_mask(monkeypatch):
    import TRXASprefitpack.gui.fit_job as fit_job

    dataset = make_dataset("dset1", n_trace=1)

    tau_mask = [np.array([True, False])]
    config = make_config(
        tau_init=np.array([1.0, 10.0]),
        tau_mask=tau_mask,
    )

    captured = {}

    def fake_fit_transient_exp(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(fit_job, "fit_transient_exp", fake_fit_transient_exp)

    result = run_fit_transient_exp_config(config, [dataset])

    assert result == {"ok": True}
    assert len(captured["tau_mask"]) == 1
    np.testing.assert_array_equal(captured["tau_mask"][0], tau_mask[0])


def test_run_fit_transient_exp_config_rejects_empty_dataset_list():
    config = make_config()

    with pytest.raises(ValueError, match="At least one"):
        run_fit_transient_exp_config(config, [])