import os
import sys

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.ads_config import ADSResult
from TRXASprefitpack.gui.calc_dads_calculation_tab import (
    CalcDADSCalculationTab,
)
from TRXASprefitpack.gui.calc_dads_data_tab import CalcDADSDataTab
from TRXASprefitpack.gui.calc_dads_result_tab import CalcDADSResultTab
from TRXASprefitpack.gui.calc_dads_svd_tab import CalcDADSSVDTab
from TRXASprefitpack.gui.calc_dads_window import CalcDADSWindow
from TRXASprefitpack.gui.calc_dads_worker import CalcDADSWorker
from TRXASprefitpack.gui.models import EScanDataset


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance()
    if application is None:
        application = QApplication(["test_calc_dads_workflow"])
    return application


@pytest.fixture
def escan_dataset():
    return EScanDataset(
        name="sample",
        energy=np.array([100.0, 101.0, 102.0]),
        time=np.array([-0.5, 0.0, 1.0, 5.0]),
        intensity=np.array(
            [
                [1.0, 0.8, 0.4, 0.1],
                [2.0, 1.5, 0.7, 0.2],
                [1.5, 1.0, 0.5, 0.15],
            ]
        ),
        eps=np.full((3, 4), 0.05),
    )


@pytest.fixture
def ads_result(escan_dataset):
    spectra = np.array(
        [
            [1.0, 0.1],
            [2.0, 0.2],
            [1.5, 0.15],
        ]
    )
    return ADSResult(
        mode="dads",
        energy=escan_dataset.energy,
        time=escan_dataset.time,
        intensity=escan_dataset.intensity,
        eps=escan_dataset.eps,
        spectra=spectra,
        spectra_eps=np.full_like(spectra, 0.01),
        fit=escan_dataset.intensity - 0.01,
        spectrum_names=("decay_1", "base"),
        model_metadata={"dataset_name": escan_dataset.name},
    )


def test_data_tab_accepts_and_clears_dataset(qapp, escan_dataset):
    tab = CalcDADSDataTab()
    received = []
    tab.dataset_changed.connect(received.append)

    tab.set_dataset(escan_dataset)

    assert tab.dataset() is escan_dataset
    assert tab.preview_table.rowCount() == escan_dataset.n_energy
    assert tab.preview_table.columnCount() == escan_dataset.n_time + 1
    assert received[-1] is escan_dataset

    tab.clear_dataset()
    assert received[-1] is None
    with pytest.raises(ValueError, match="Load"):
        tab.dataset()


def test_data_tab_loads_dataset_files(qapp, tmp_path, escan_dataset):
    intensity_path = tmp_path / "intensity.txt"
    time_path = tmp_path / "time.txt"
    eps_path = tmp_path / "eps.txt"

    np.savetxt(
        intensity_path,
        np.column_stack(
            (escan_dataset.energy, escan_dataset.intensity)
        ),
    )
    np.savetxt(time_path, escan_dataset.time)
    np.savetxt(eps_path, escan_dataset.eps)

    tab = CalcDADSDataTab()
    dataset = tab.load_dataset(
        intensity_path,
        time_path,
        eps_path,
        name="loaded",
    )

    assert dataset.name == "loaded"
    np.testing.assert_allclose(dataset.energy, escan_dataset.energy)
    np.testing.assert_allclose(dataset.time, escan_dataset.time)
    np.testing.assert_allclose(dataset.intensity, escan_dataset.intensity)
    np.testing.assert_allclose(dataset.eps, escan_dataset.eps)


def test_data_tab_rejects_wrong_dataset_type(qapp):
    tab = CalcDADSDataTab()
    with pytest.raises(TypeError, match="EScanDataset"):
        tab.set_dataset(object())


def test_svd_tab_displays_components(qapp, escan_dataset):
    tab = CalcDADSSVDTab()
    tab.set_dataset(escan_dataset)

    expected_count = min(escan_dataset.n_energy, escan_dataset.n_time)
    assert tab.singular_table.rowCount() == expected_count
    assert tab.component_combo.count() == expected_count
    assert tab.navigation_toolbar is not None

    tab.cutoff_spin.setValue(0.5)
    assert tab.cond_num == pytest.approx(0.5)
    assert tab.retained_label.text().startswith("Retained:")


def test_svd_tab_clears_dataset(qapp, escan_dataset):
    tab = CalcDADSSVDTab()
    tab.set_dataset(escan_dataset)
    tab.set_dataset(None)

    assert tab.singular_table.rowCount() == 0
    assert tab.component_combo.count() == 0
    assert tab.retained_label.text() == "No dataset loaded."


def test_calculation_tab_builds_dads_config(qapp):
    tab = CalcDADSCalculationTab()
    tab.tau_edit.setText("1.0, 10.0")

    config = tab.build_config()

    assert config.mode == "dads"
    assert config.irf == "g"
    assert config.base is True
    np.testing.assert_allclose(config.tau, np.array([1.0, 10.0]))
    assert config.y0 is None


def test_calculation_tab_builds_sads_svd_config(qapp):
    tab = CalcDADSCalculationTab()
    tab.mode_combo.setCurrentIndex(
        tab.mode_combo.findData("sads_svd")
    )
    tab.tau_edit.setText("1.0, 10.0")
    tab.y0_edit.setText("1.0, 0.0, 0.0")
    tab.exclude_edit.setText("2")
    tab.set_cond_num(0.01)

    config = tab.build_config()

    assert config.mode == "sads_svd"
    assert config.cond_num == pytest.approx(0.01)
    np.testing.assert_allclose(config.y0, np.array([1.0, 0.0, 0.0]))
    assert config.exclude == (2,)


def test_calculation_tab_exposes_supported_non_osc_modes(qapp):
    tab = CalcDADSCalculationTab()
    modes = {
        tab.mode_combo.itemData(index)
        for index in range(tab.mode_combo.count())
    }
    assert modes == {
        "dads", "dads_svd", 
        "sads", "sads_svd",
        "custom_sads", "custom_sads_svd"
        }


def test_worker_emits_result(qapp, escan_dataset, ads_result):
    config = CalcDADSCalculationTab().build_config()
    worker = CalcDADSWorker(
        config,
        escan_dataset,
        job_runner=lambda _config, _dataset: ads_result,
    )
    received = []
    finished = []
    worker.result_ready.connect(received.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert received == [ads_result]
    assert finished == [True]


def test_worker_emits_error(qapp, escan_dataset):
    config = CalcDADSCalculationTab().build_config()
    expected = RuntimeError("calculation failed")

    def fail(_config, _dataset):
        raise expected

    worker = CalcDADSWorker(config, escan_dataset, job_runner=fail)
    errors = []
    worker.error.connect(errors.append)

    worker.run()

    assert errors == [expected]


def test_result_tab_displays_spectra_and_reconstruction(
    qapp,
    ads_result,
):
    tab = CalcDADSResultTab()
    tab.set_result(ads_result)

    assert tab.summary_table.rowCount() > 0
    assert tab.spectra_table.rowCount() == ads_result.n_energy
    assert tab.time_combo.count() == ads_result.n_time
    assert tab.navigation_toolbar is not None
    assert tab.residual_axis.get_shared_x_axes().joined(
        tab.fit_axis,
        tab.residual_axis,
    )
    assert "[ADS Result]" in tab.report_view.toPlainText()


def test_result_tab_rejects_wrong_result_type(qapp):
    tab = CalcDADSResultTab()
    with pytest.raises(TypeError, match="ADSResult"):
        tab.set_result(object())


def test_complete_window_has_connected_workflow_tabs(qapp):
    window = CalcDADSWindow()
    try:
        assert window.tab_widget.count() == 4
        assert window.tab_widget.tabText(0) == "Data"
        assert window.tab_widget.tabText(1) == "SVD"
        assert window.tab_widget.tabText(2) == "Calculation"
        assert window.tab_widget.tabText(3) == "Results"
        assert isinstance(window.data_tab, CalcDADSDataTab)
        assert isinstance(window.svd_tab, CalcDADSSVDTab)
        assert isinstance(
            window.calculation_tab,
            CalcDADSCalculationTab,
        )
        assert isinstance(window.result_tab, CalcDADSResultTab)
    finally:
        window.close()


def test_window_runs_calculation_in_worker(
    qtbot,
    escan_dataset,
    ads_result,
):
    calls = []

    def fake_runner(config, dataset):
        calls.append((config, dataset))
        return ads_result

    window = CalcDADSWindow(job_runner=fake_runner)
    qtbot.addWidget(window)
    window.data_tab.set_dataset(escan_dataset)

    window.run_calculation()

    qtbot.waitUntil(
        lambda: window.result_tab.result is ads_result,
        timeout=3000,
    )
    qtbot.waitUntil(
        lambda: window._calculation_thread is None,
        timeout=3000,
    )

    assert len(calls) == 1
    assert calls[0][1] is escan_dataset
    assert window.tab_widget.currentWidget() is window.result_tab
    assert window.statusBar().currentMessage() == "Calculation completed"
    assert window.calculation_tab.run_button.isEnabled()


def test_window_reports_configuration_error_without_dataset(qapp):
    window = CalcDADSWindow()
    try:
        window.run_calculation()
        assert window.statusBar().currentMessage() == "Configuration error"
        assert "Load an energy-scan" in (
            window.calculation_tab.validation_label.text()
        )
        assert window._calculation_thread is None
    finally:
        window.close()
