import os
import sys

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.ads_config import ADSResult
from TRXASprefitpack.gui.calc_dads_calculation_tab import (
    CalcDADSCalculationTab,
)
from TRXASprefitpack.gui.calc_dads_custom_sads import (
    CalcDADSCustomSADSPanel,
)
from TRXASprefitpack.gui.calc_dads_window import CalcDADSWindow
from TRXASprefitpack.gui.models import EScanDataset
from TRXASprefitpack.gui.rate_model import RateEdge, RateModelSpec


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance()
    if application is None:
        application = QApplication(["test_calc_dads_custom_sads"])
    return application


@pytest.fixture
def rate_model():
    return RateModelSpec(
        species=("MLCT", "MC", "GS"),
        edges=(
            RateEdge("MLCT", "MC", 2.0),
            RateEdge("MC", "GS", 0.1),
        ),
        y0=np.array([1.0, 0.0, 0.0]),
    )


@pytest.fixture
def escan_dataset():
    return EScanDataset(
        name="custom-sads",
        energy=np.array([100.0, 101.0, 102.0]),
        time=np.array([-0.2, 0.0, 1.0, 10.0]),
        intensity=np.array(
            [
                [0.0, 0.5, 0.3, 0.1],
                [0.0, 1.0, 0.6, 0.2],
                [0.0, 0.8, 0.4, 0.1],
            ]
        ),
        eps=np.full((3, 4), 0.05),
    )


def make_result(dataset, mode="custom_sads"):
    return ADSResult(
        mode=mode,
        energy=dataset.energy,
        time=dataset.time,
        intensity=dataset.intensity,
        eps=dataset.eps,
        spectra=np.ones((dataset.n_energy, 3)),
        spectra_eps=(
            None
            if mode.endswith("_svd")
            else np.full((dataset.n_energy, 3), 0.01)
        ),
        fit=dataset.intensity.copy(),
        spectrum_names=("MLCT", "MC", "GS"),
        model_metadata={
            "dataset_name": dataset.name,
            "rate_model_kind": "custom",
            "rate_matrix": np.array(
                [
                    [-2.0, 0.0, 0.0],
                    [2.0, -0.1, 0.0],
                    [0.0, 0.1, 0.0],
                ]
            ),
        },
    )


def select_mode(tab, mode):
    index = tab.mode_combo.findData(mode)
    assert index >= 0
    tab.mode_combo.setCurrentIndex(index)


def test_custom_panel_shows_matrix_and_real_modes(qapp, rate_model):
    panel = CalcDADSCustomSADSPanel()
    panel.set_model(rate_model)

    model = panel.build_rate_model()

    assert model.species == rate_model.species
    assert panel.editor.matrix_table.rowCount() == 3
    assert panel.mode_summary_table.rowCount() == 3
    assert "real modes only" in panel.mode_summary_label.text()


def test_custom_panel_clears_stale_summary_after_edit(qapp, rate_model):
    panel = CalcDADSCustomSADSPanel()
    panel.set_model(rate_model)

    panel.editor.edge_table.item(0, 2).setText("1.5")

    assert panel.mode_summary_table.rowCount() == 0
    assert panel.editor.validated_model is None


def test_custom_panel_rejects_invalid_model_before_run(qapp):
    panel = CalcDADSCustomSADSPanel()
    panel.editor.edge_table.item(0, 2).setText("1 + 2")

    with pytest.raises(ValueError, match="Invalid custom rate model"):
        panel.build_rate_model()


@pytest.mark.parametrize(
    ("mode", "expected_cond_num"),
    [
        ("custom_sads", 0.0),
        ("custom_sads_svd", 0.02),
    ],
)
def test_calculation_tab_builds_custom_config(
    qapp,
    rate_model,
    mode,
    expected_cond_num,
):
    tab = CalcDADSCalculationTab()
    select_mode(tab, mode)
    tab.custom_sads_panel.set_model(rate_model)
    tab.exclude_edit.setText("2")
    tab.cond_num_edit.setText("0.02")

    config = tab.build_config()

    assert config.mode == mode
    assert config.tau is None
    assert config.rate_model is not None
    assert config.rate_model.species == rate_model.species
    assert config.y0 is None
    assert config.exclude == (2,)
    assert config.cond_num == pytest.approx(expected_cond_num)


def test_custom_mode_control_visibility(qapp):
    tab = CalcDADSCalculationTab()
    assert tab.custom_sads_panel.isHidden()

    select_mode(tab, "custom_sads")
    assert not tab.custom_sads_panel.isHidden()
    assert not tab.tau_edit.isEnabled()
    assert not tab.y0_edit.isEnabled()
    assert tab.exclude_edit.isEnabled()
    assert not tab.cond_num_edit.isEnabled()

    select_mode(tab, "custom_sads_svd")
    assert tab.cond_num_edit.isEnabled()


def test_window_runs_custom_sads_and_displays_result(
    qtbot,
    escan_dataset,
    rate_model,
):
    expected = make_result(escan_dataset)
    calls = []

    def fake_runner(config, dataset):
        calls.append((config, dataset))
        return expected

    window = CalcDADSWindow(job_runner=fake_runner)
    qtbot.addWidget(window)
    window.data_tab.set_dataset(escan_dataset)
    select_mode(window.calculation_tab, "custom_sads")
    window.calculation_tab.custom_sads_panel.set_model(rate_model)

    window.run_calculation()

    qtbot.waitUntil(
        lambda: window.result_tab.result is expected,
        timeout=3000,
    )
    qtbot.waitUntil(
        lambda: window._calculation_thread is None,
        timeout=3000,
    )

    assert len(calls) == 1
    config, dataset = calls[0]
    assert config.mode == "custom_sads"
    assert config.rate_model is not None
    assert dataset is escan_dataset
    assert window.result_tab.result is expected
    assert window.tab_widget.currentWidget() is window.result_tab


def test_window_rejects_invalid_custom_model_before_worker(
    qapp,
    escan_dataset,
):
    window = CalcDADSWindow(job_runner=lambda *_args: pytest.fail())
    try:
        window.data_tab.set_dataset(escan_dataset)
        select_mode(window.calculation_tab, "custom_sads")
        editor = window.calculation_tab.custom_sads_panel.editor
        editor.edge_table.item(0, 2).setText("invalid")

        window.run_calculation()

        assert window._calculation_thread is None
        assert window.statusBar().currentMessage() == "Configuration error"
        assert "Invalid custom rate model" in (
            window.calculation_tab.validation_label.text()
        )
    finally:
        window.close()
