import os
import sys

import numpy as np
import pytest


path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.driver.transient_result import TransientResult
from TRXASprefitpack.gui.ads_config import ADSResult
from TRXASprefitpack.gui.result_views import (
    ads_result_to_plot_arrays,
    ads_result_to_report_text,
    ads_result_to_spectra_table,
    ads_result_to_summary_rows,
    transient_result_to_fit_plot_arrays,
    transient_result_to_parameter_rows,
    transient_result_to_report_text,
    transient_result_to_residual_plot_arrays,
)


def make_transient_result():
    result = TransientResult()

    time_1 = np.array([0.0, 1.0, 2.0])
    time_2 = np.array([0.0, 2.0])

    intensity_1 = np.array(
        [
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
        ]
    )
    intensity_2 = np.array(
        [
            [4.0],
            [5.0],
        ]
    )

    fit_1 = intensity_1 - 0.1
    fit_2 = intensity_2 - 0.2

    eps_1 = np.full_like(intensity_1, 0.2)
    eps_2 = np.full_like(intensity_2, 0.5)

    result["model"] = "decay"
    result["same_t0"] = False
    result["name_of_dset"] = np.array(
        ["dataset_A", "dataset_B"],
        dtype=object,
    )
    result["t"] = [time_1, time_2]
    result["intensity"] = [intensity_1, intensity_2]
    result["eps"] = [eps_1, eps_2]
    result["fit"] = [fit_1, fit_2]
    result["res"] = [
        intensity_1 - fit_1,
        intensity_2 - fit_2,
    ]

    result["irf"] = "g"
    result["fwhm"] = 0.12
    result["eta"] = 0.0
    result["base"] = True

    result["param_name"] = np.array(
        ["fwhm_G", "tau_1"],
        dtype=object,
    )
    result["x"] = np.array([0.12, 2.0])
    result["x_eps"] = np.array([0.01, 0.2])
    result["bounds"] = [
        (0.01, 1.0),
        (2.0, 2.0),
    ]

    result["c"] = [
        np.array(
            [
                [1.0, 2.0],
                [0.1, 0.2],
            ]
        ),
        np.array(
            [
                [3.0],
                [0.3],
            ]
        ),
    ]

    result["chi2"] = 10.0
    result["chi2_ind"] = np.array(
        [
            np.array([2.0, 3.0]),
            np.array([5.0]),
        ],
        dtype=object,
    )
    result["aic"] = 15.0
    result["bic"] = 16.0
    result["red_chi2"] = 1.25
    result["red_chi2_ind"] = np.array(
        [
            np.array([1.0, 1.5]),
            np.array([2.5]),
        ],
        dtype=object,
    )

    result["nfev"] = 20
    result["n_param"] = 2
    result["n_param_ind"] = 2
    result["num_pts"] = 8
    result["corr"] = np.array(
        [
            [1.0, 0.05],
            [0.05, 1.0],
        ]
    )

    result["method_glb"] = None
    result["message_glb"] = None
    result["method_lsq"] = "trf"
    result["message_lsq"] = "success"
    result["success_lsq"] = True
    result["status"] = 0

    result["n_decay"] = 1
    result["n_osc"] = 0

    return result


def make_ads_result(
    *,
    spectra_eps=True,
    with_svd=False,
):
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
    fit = intensity - 0.05

    spectra = np.array(
        [
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6],
        ]
    )

    if spectra_eps:
        spectrum_errors = np.full_like(
            spectra,
            0.01,
        )
    else:
        spectrum_errors = None

    kwargs = {}

    if with_svd:
        kwargs.update(
            {
                "svd_u": np.ones((3, 2)),
                "svd_s": np.array([10.0, 1.0]),
                "svd_vh": np.ones((2, 2)),
            }
        )

    return ADSResult(
        mode="dads",
        energy=energy,
        time=time,
        intensity=intensity,
        eps=eps,
        spectra=spectra,
        spectra_eps=spectrum_errors,
        fit=fit,
        spectrum_names=("decay_1", "base"),
        model_metadata={
            "dataset_name": "sample",
            "t0": 0.25,
            "irf": "g",
            "fwhm": 0.12,
            "eta": None,
            "base": True,
            "cond_num": 0.1 if with_svd else None,
        },
        **kwargs,
    )


def test_transient_parameter_rows():
    result = make_transient_result()

    rows = transient_result_to_parameter_rows(result)

    assert rows == [
        {
            "name": "fwhm_G",
            "value": 0.12,
            "error": 0.01,
            "lower_bound": 0.01,
            "upper_bound": 1.0,
            "fixed": False,
        },
        {
            "name": "tau_1",
            "value": 2.0,
            "error": 0.2,
            "lower_bound": 2.0,
            "upper_bound": 2.0,
            "fixed": True,
        },
    ]


def test_transient_parameter_rows_reject_wrong_bounds():
    result = make_transient_result()
    result["bounds"] = [(0.0, 1.0)]

    with pytest.raises(ValueError, match="bounds"):
        transient_result_to_parameter_rows(result)


def test_transient_parameter_rows_reject_missing_key():
    result = make_transient_result()
    del result["x_eps"]

    with pytest.raises(ValueError, match="x_eps"):
        transient_result_to_parameter_rows(result)


def test_transient_fit_plot_arrays_flatten_traces():
    result = make_transient_result()

    entries = transient_result_to_fit_plot_arrays(
        result
    )

    assert len(entries) == 3

    first = entries[0]

    assert first["dataset_index"] == 0
    assert first["dataset_name"] == "dataset_A"
    assert first["trace_index"] == 0
    assert first["trace_name"] == "trace_1"

    np.testing.assert_allclose(
        first["time"],
        np.array([0.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(
        first["intensity"],
        np.array([1.0, 1.5, 2.0]),
    )
    np.testing.assert_allclose(
        first["fit"],
        np.array([0.9, 1.4, 1.9]),
    )

    assert entries[2]["dataset_name"] == "dataset_B"
    assert entries[2]["trace_name"] == "trace_1"


def test_transient_fit_plot_arrays_are_copies():
    result = make_transient_result()

    entries = transient_result_to_fit_plot_arrays(
        result
    )

    assert not np.shares_memory(
        entries[0]["intensity"],
        result["intensity"][0],
    )
    assert not np.shares_memory(
        entries[0]["fit"],
        result["fit"][0],
    )


def test_transient_residual_plot_arrays():
    result = make_transient_result()

    entries = transient_result_to_residual_plot_arrays(
        result
    )

    assert len(entries) == 3

    np.testing.assert_allclose(
        entries[0]["residual"],
        np.array([0.1, 0.1, 0.1]),
    )
    np.testing.assert_allclose(
        entries[0]["standardized_residual"],
        np.array([0.5, 0.5, 0.5]),
    )


def test_transient_plot_rejects_mismatched_fit_shape():
    result = make_transient_result()
    result["fit"][0] = np.ones((2, 2))

    with pytest.raises(ValueError, match=r"fit\[0\]"):
        transient_result_to_fit_plot_arrays(result)


def test_transient_report_uses_existing_report():
    result = make_transient_result()

    report = transient_result_to_report_text(result)

    assert "[Model information]" in report
    assert "[Optimization Results]" in report
    assert "tau_1" in report


@pytest.mark.parametrize(
    "function",
    [
        transient_result_to_parameter_rows,
        transient_result_to_fit_plot_arrays,
        transient_result_to_residual_plot_arrays,
        transient_result_to_report_text,
    ],
)
def test_transient_helpers_reject_wrong_type(function):
    with pytest.raises(TypeError, match="TransientResult"):
        function({})


def test_ads_spectra_table_with_errors():
    result = make_ads_result()

    table = ads_result_to_spectra_table(result)

    assert table["columns"] == (
        "energy",
        "decay_1",
        "decay_1_eps",
        "base",
        "base_eps",
    )

    assert table["rows"][0] == (
        100.0,
        0.1,
        0.01,
        0.4,
        0.01,
    )


def test_ads_spectra_table_without_errors():
    result = make_ads_result(
        spectra_eps=False,
    )

    table = ads_result_to_spectra_table(result)

    assert table["columns"] == (
        "energy",
        "decay_1",
        "base",
    )
    assert table["rows"][0] == (
        100.0,
        0.1,
        0.4,
    )


def test_ads_plot_arrays():
    result = make_ads_result()

    plot_arrays = ads_result_to_plot_arrays(result)

    assert len(plot_arrays["spectra"]) == 2
    assert len(plot_arrays["fits"]) == 2

    first_spectrum = plot_arrays["spectra"][0]

    assert first_spectrum["name"] == "decay_1"

    np.testing.assert_allclose(
        first_spectrum["energy"],
        result.energy,
    )
    np.testing.assert_allclose(
        first_spectrum["spectrum"],
        np.array([0.1, 0.2, 0.3]),
    )
    np.testing.assert_allclose(
        first_spectrum["spectrum_eps"],
        np.array([0.01, 0.01, 0.01]),
    )

    first_fit = plot_arrays["fits"][0]

    assert first_fit["time"] == pytest.approx(0.0)

    np.testing.assert_allclose(
        first_fit["intensity"],
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(
        first_fit["fit"],
        np.array([0.95, 1.95, 2.95]),
    )
    np.testing.assert_allclose(
        first_fit["residual"],
        np.array([0.05, 0.05, 0.05]),
    )


def test_ads_plot_arrays_are_copies():
    result = make_ads_result()

    plot_arrays = ads_result_to_plot_arrays(result)

    assert not np.shares_memory(
        plot_arrays["spectra"][0]["spectrum"],
        result.spectra,
    )
    assert not np.shares_memory(
        plot_arrays["fits"][0]["intensity"],
        result.intensity,
    )


def test_ads_summary_rows_without_svd():
    result = make_ads_result()

    rows = ads_result_to_summary_rows(result)
    summary = {
        row["name"]: row["value"]
        for row in rows
    }

    assert summary["mode"] == "dads"
    assert summary["n_energy"] == 3
    assert summary["n_time"] == 2
    assert summary["n_component"] == 2
    assert summary["has_svd"] is False
    assert summary["dataset_name"] == "sample"
    assert summary["t0"] == pytest.approx(0.25)
    assert summary["spectrum_names"] == (
        "decay_1",
        "base",
    )
    assert "n_svd_component" not in summary


def test_ads_summary_rows_with_svd():
    result = make_ads_result(
        spectra_eps=False,
        with_svd=True,
    )

    rows = ads_result_to_summary_rows(result)
    summary = {
        row["name"]: row["value"]
        for row in rows
    }

    assert summary["has_svd"] is True
    assert summary["n_svd_component"] == 2
    assert summary["cond_num"] == pytest.approx(0.1)


def test_ads_report_text():
    result = make_ads_result()

    report = ads_result_to_report_text(result)

    assert report.startswith("[ADS Result]")
    assert "mode: dads" in report
    assert "dataset_name: sample" in report
    assert "n_component: 2" in report
    assert "spectrum_names: decay_1, base" in report
    assert "spectra_errors: available" in report


def test_ads_report_text_without_spectra_errors():
    result = make_ads_result(
        spectra_eps=False,
    )

    report = ads_result_to_report_text(result)

    assert "spectra_errors: unavailable" in report


@pytest.mark.parametrize(
    "function",
    [
        ads_result_to_spectra_table,
        ads_result_to_plot_arrays,
        ads_result_to_summary_rows,
        ads_result_to_report_text,
    ],
)
def test_ads_helpers_reject_wrong_type(function):
    with pytest.raises(TypeError, match="ADSResult"):
        function({})