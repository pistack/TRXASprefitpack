import csv
import os
from pathlib import Path
import sys

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.driver.transient_result import TransientResult
from TRXASprefitpack.gui.fit_tscan_data_tab import FitTScanDataTab
from TRXASprefitpack.gui.fit_tscan_export import (
    export_fit_csv,
    export_parameter_csv,
    export_report_txt,
    export_residual_csv,
)
from TRXASprefitpack.gui.fit_tscan_parameter_tabs import (
    FitTScanParameterTabs,
)
from TRXASprefitpack.gui.fit_tscan_worker import FitTScanWorker
from TRXASprefitpack.gui.fit_tscan_window import FitTScanWindow
from TRXASprefitpack.gui.models import TScanDataset, TScanTrace
from matplotlib.container import ErrorbarContainer


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(
        ["test_fit_tscan_workflow"]
    )

@pytest.fixture
def fitted_result():
    time = np.array(
        [-1.0, -0.2, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=float,
    )

    fit = np.array(
        [
            [0.00, 0.00],
            [0.01, -0.01],
            [1.00, 0.80],
            [0.98, 0.78],
            [0.90, 0.70],
            [0.40, 0.25],
            [0.05, 0.02],
        ],
        dtype=float,
    )

    raw_residual = np.array(
        [
            [0.002, -0.004],
            [-0.003, 0.002],
            [0.005, -0.006],
            [-0.004, 0.003],
            [0.003, -0.002],
            [-0.002, 0.004],
            [0.001, -0.003],
        ],
        dtype=float,
    )

    eps = np.array(
        [
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
        ],
        dtype=float,
    )

    intensity = fit + raw_residual

    weighted_residual = raw_residual / eps

    result = TransientResult()

    result.update(
        {
            "model": "decay",
            "irf": "g",
            "same_t0": False,
            "name_of_dset": np.array(["fixture_dataset"]),
            "t": [time],
            "intensity": [intensity],
            "eps": [eps],
            "fit": [fit],
            "res": [weighted_residual],
            "fwhm": 0.12,
            "eta": 0.0,
            "base": True,
            "n_decay": 2,
            "n_osc": 0,
            "tau_mask": [
                np.array(
                    [True, True],
                    dtype=bool,
                )
            ],
            # 행: decay 1, decay 2, baseline
            # 열: trace 1, trace 2
            "c": [
                np.array(
                    [
                        [1.00, 0.80],
                        [0.40, 0.25],
                        [0.01, -0.01],
                    ],
                    dtype=float,
                )
            ],
            "param_name": np.array(
                [
                    "fwhm",
                    "t_0_1_1",
                    "t_0_1_2",
                    "tau_1",
                    "tau_2",
                ]
            ),
            "x": np.array(
                [0.12, 0.01, -0.02, 1.0, 10.0],
                dtype=float,
            ),
            "x_eps": np.array(
                [0.005, 0.002, 0.002, 0.05, 0.50],
                dtype=float,
            ),
            "bounds": np.array(
                [
                    [0.05, 0.30],
                    [-0.50, 0.50],
                    [-0.50, 0.50],
                    [0.10, 5.00],
                    [5.00, 50.00],
                ],
                dtype=float,
            ),
            "chi2": 1.25,
            "red_chi2": 0.14,
            "chi2_ind": [np.array([0.55, 0.70])],
            "red_chi2_ind": [np.array([0.12, 0.16])],
            "aic": 8.50,
            "bic": 9.25,
            "nfev": 12,
            "n_param": 5,
            "n_param_ind": 4,
            "num_pts": int(intensity.size),
            "jac": np.zeros((intensity.size, 5)),
            "cov": np.eye(5),
            "cov_scaled": np.eye(5) * 0.14,
            "corr": np.eye(5),
            "method_glb": None,
            "message_glb": None,
            "method_lsq": "trf",
            "success_lsq": True,
            "message_lsq": "Fixture fit converged.",
            "status": 0,
        }
    )

    return result


def make_trace(name="trace_1"):
    return TScanTrace(
        path=f"{name}.txt",
        name=name,
        t=np.array([0.0, 1.0, 2.0]),
        intensity=np.array([1.0, 1.5, 2.0]),
        eps=np.array([0.1, 0.1, 0.1]),
    )


def make_dataset():
    return TScanDataset(
        name="dataset_1",
        traces=(make_trace(),),
    )


def make_result():
    result = TransientResult()

    intensity = np.array(
        [
            [1.0],
            [1.5],
            [2.0],
        ]
    )
    fit = intensity - 0.1

    result.update(
        {
            "model": "decay",
            "same_t0": False,
            "name_of_dset": np.array(
                ["dataset_1"],
                dtype=object,
            ),
            "t": [np.array([0.0, 1.0, 2.0])],
            "intensity": [intensity],
            "eps": [np.full_like(intensity, 0.1)],
            "fit": [fit],
            "res": [intensity - fit],
            "irf": "g",
            "fwhm": 0.1,
            "eta": 0.0,
            "base": True,
            "param_name": np.array(
                ["fwhm_G", "tau_1"],
                dtype=object,
            ),
            "x": np.array([0.1, 1.0]),
            "x_eps": np.array([0.01, 0.1]),
            "bounds": [(0.01, 1.0), (0.1, 10.0)],
            "c": [np.array([[1.0], [0.1]])],
            "chi2": 1.0,
            "chi2_ind": np.array(
                [np.array([1.0])],
                dtype=object,
            ),
            "aic": 2.0,
            "bic": 3.0,
            "red_chi2": 1.0,
            "red_chi2_ind": np.array(
                [np.array([1.0])],
                dtype=object,
            ),
            "nfev": 5,
            "n_param": 2,
            "n_param_ind": 2,
            "num_pts": 3,
            "corr": np.eye(2),
            "method_glb": None,
            "message_glb": None,
            "method_lsq": "trf",
            "message_lsq": "success",
            "success_lsq": True,
            "status": 0,
            "n_decay": 1,
            "n_osc": 0,
        }
    )

    return result


def test_data_tab_builds_dataset(qapp, monkeypatch):
    tab = FitTScanDataTab()

    monkeypatch.setattr(
        "TRXASprefitpack.gui.fit_tscan_data_tab."
        "read_tscan_trace",
        lambda path: make_trace(Path(path).stem),
    )

    tab.load_trace_files(["one.txt", "two.txt"])

    datasets = tab.datasets()

    assert len(datasets) == 1
    assert datasets[0].n_trace == 2
    assert tab.trace_table.rowCount() == 2


def test_data_tab_rejects_empty_dataset(qapp):
    tab = FitTScanDataTab()

    with pytest.raises(ValueError, match="no traces"):
        tab.datasets()


def test_parameter_tab_builds_config(qapp):
    tab = FitTScanParameterTabs()
    dataset = make_dataset()

    tab.set_datasets([dataset])
    tab.t0_edit.setText("0.0")
    tab.tau_edit.setText("1.0, 10.0")

    config = tab.build_config([dataset])

    assert config.irf == "g"
    assert config.same_t0 is False

    np.testing.assert_allclose(
        config.t0_init,
        np.array([0.0]),
    )
    np.testing.assert_allclose(
        config.tau_init,
        np.array([1.0, 10.0]),
    )

    assert len(config.tau_mask) == 1
    np.testing.assert_array_equal(
        config.tau_mask[0],
        np.array([True, True]),
    )


def test_parameter_tab_pseudo_voigt(qapp):
    tab = FitTScanParameterTabs()
    dataset = make_dataset()

    tab.set_datasets([dataset])
    tab.irf_combo.setCurrentIndex(
        tab.irf_combo.findData("pv")
    )
    tab.fwhm_g_edit.setText("0.1")
    tab.fwhm_l_edit.setText("0.2")

    config = tab.build_config([dataset])

    assert config.irf == "pv"

    np.testing.assert_allclose(
        config.fwhm_init,
        np.array([0.1, 0.2]),
    )


def test_worker_returns_result(qapp):
    expected = make_result()
    received = []
    finished = []

    worker = FitTScanWorker(
        object(),
        [make_dataset()],
        job_runner=lambda config, datasets: expected,
    )
    worker.result_ready.connect(received.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert received == [expected]
    assert finished == [True]


def test_worker_emits_exception(qapp):
    errors = []
    finished = []

    def failing_runner(config, datasets):
        raise RuntimeError("failure")

    worker = FitTScanWorker(
        object(),
        [make_dataset()],
        job_runner=failing_runner,
    )
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert finished == [True]


def test_exports(tmp_path):
    result = make_result()

    parameter_path = export_parameter_csv(
        result,
        tmp_path / "parameters.csv",
    )
    fit_path = export_fit_csv(
        result,
        tmp_path / "fit.csv",
    )
    residual_path = export_residual_csv(
        result,
        tmp_path / "residual.csv",
    )
    report_path = export_report_txt(
        result,
        tmp_path / "report.txt",
    )

    assert parameter_path.exists()
    assert fit_path.exists()
    assert residual_path.exists()
    assert report_path.exists()

    with parameter_path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.reader(stream))

    assert rows[0] == [
        "name",
        "value",
        "error",
        "lower_bound",
        "upper_bound",
        "fixed",
    ]
    assert rows[1][0] == "fwhm_G"

    assert "dataset_1" in fit_path.read_text(
        encoding="utf-8"
    )
    assert "standardized_residual" in (
        residual_path.read_text(encoding="utf-8")
    )
    assert "[Model information]" in (
        report_path.read_text(encoding="utf-8")
    )


def test_export_refuses_overwrite(tmp_path):
    path = tmp_path / "existing.csv"
    path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_parameter_csv(
            make_result(),
            path,
        )


def test_complete_window_has_workflow_tabs(qapp):
    window = FitTScanWindow()

    assert window.tab_widget.count() == 3
    assert window.tab_widget.tabText(0) == "Data"
    assert (
        window.tab_widget.tabText(1)
        == "Model and Parameters"
    )
    assert window.tab_widget.tabText(2) == "Results"

    assert window.data_tab is not None
    assert window.parameter_tab is not None
    assert window.result_tab is not None

    window.close()


def test_result_tab_displays_result(qapp):
    window = FitTScanWindow()
    result = make_result()

    window.result_tab.set_result(result)

    assert window.result_tab.result is result
    assert window.result_tab.parameter_table.rowCount() == 2
    assert window.result_tab.trace_selector.count() == 1
    assert "[Model information]" in (
        window.result_tab.report_view.toPlainText()
    )
    assert window.result_tab.export_fit_button.isEnabled()

    window.close()

def test_result_plot_axes_are_synchronized(qtbot, fitted_result):
    window = FitTScanWindow()
    qtbot.addWidget(window)

    window.result_tab.set_result(fitted_result)

    fit_axis = window.result_tab.fit_axis
    residual_axis = window.result_tab.residual_axis

    assert residual_axis.get_shared_x_axes().joined(
        fit_axis,
        residual_axis,
    )

    np.testing.assert_allclose(
        fit_axis.get_xlim(),
        residual_axis.get_xlim(),
    )


def test_result_plot_has_navigation_toolbar(qtbot):
    window = FitTScanWindow()
    qtbot.addWidget(window)

    assert window.result_tab.navigation_toolbar is not None


def test_residual_plot_has_errorbar(qtbot, fitted_result):
    window = FitTScanWindow()
    qtbot.addWidget(window)

    window.result_tab.set_result(fitted_result)

    assert any(
        isinstance(container, ErrorbarContainer)
        for container in window.result_tab.residual_axis.containers
    )


def test_result_plot_supports_symlog(qtbot, fitted_result):
    window = FitTScanWindow()
    qtbot.addWidget(window)

    window.result_tab.set_result(fitted_result)

    index = window.result_tab.xscale_combo.findData("symlog")
    window.result_tab.xscale_combo.setCurrentIndex(index)
    window.result_tab.linthresh_spin.setValue(0.25)

    assert window.result_tab.fit_axis.get_xscale() == "symlog"
    assert window.result_tab.residual_axis.get_xscale() == "symlog"

    np.testing.assert_allclose(
        window.result_tab.fit_axis.get_xlim(),
        window.result_tab.residual_axis.get_xlim(),
    )