import csv
import json
import os
import sys

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QFileDialog

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.ads_config import ADSResult
from TRXASprefitpack.gui.ads_export import (
    export_ads_bundle,
    export_ads_figure,
    export_ads_fit_csv,
    export_ads_rate_model_json,
    export_ads_report_txt,
    export_ads_residual_csv,
    export_ads_spectra_csv,
    export_ads_spectra_error_csv,
    export_ads_summary_csv,
    export_ads_svd_csv,
)
from TRXASprefitpack.gui.calc_dads_result_tab import CalcDADSResultTab


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance()
    if application is None:
        application = QApplication(["test_ads_export"])
    return application


@pytest.fixture
def ads_result():
    energy = np.array([100.0, 101.0, 102.0])
    time = np.array([0.0, 1.0])
    intensity = np.array(
        [
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
        ]
    )
    return ADSResult(
        mode="custom_sads_svd",
        energy=energy,
        time=time,
        intensity=intensity,
        eps=np.full((3, 2), 0.1),
        spectra=np.array(
            [
                [1.0, 0.1],
                [2.0, 0.2],
                [3.0, 0.3],
            ]
        ),
        spectra_eps=np.full((3, 2), 0.01),
        fit=intensity - 0.05,
        spectrum_names=("MLCT", "MC"),
        svd_u=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]
        ),
        svd_s=np.array([3.0, 1.0]),
        svd_vh=np.array(
            [
                [1.0, 0.5],
                [0.2, 0.1],
            ]
        ),
        model_metadata={
            "dataset_name": "sample",
            "rate_model_kind": "custom",
            "rate_model": {
                "species": ["MLCT", "MC"],
                "edges": [
                    {"source": "MLCT", "target": "MC", "rate": 2.0}
                ],
                "y0": [1.0, 0.0],
            },
        },
    )


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))


def without_optional_arrays(result):
    return ADSResult(
        mode="dads",
        energy=result.energy,
        time=result.time,
        intensity=result.intensity,
        eps=result.eps,
        spectra=result.spectra,
        fit=result.fit,
        spectrum_names=result.spectrum_names,
        model_metadata={"dataset_name": "sample"},
    )


def test_export_spectra_and_errors(ads_result, tmp_path):
    spectra_path = export_ads_spectra_csv(
        ads_result, tmp_path / "spectra.csv"
    )
    errors_path = export_ads_spectra_error_csv(
        ads_result, tmp_path / "spectra-errors.csv"
    )

    spectra = read_csv(spectra_path)
    errors = read_csv(errors_path)
    assert spectra[0] == ["energy", "MLCT", "MC"]
    assert spectra[1] == ["100.0", "1.0", "0.1"]
    assert errors[0] == ["energy", "MLCT_eps", "MC_eps"]
    assert errors[1] == ["100.0", "0.01", "0.01"]


def test_export_fit_and_residual_matrices(ads_result, tmp_path):
    fit_path = export_ads_fit_csv(ads_result, tmp_path / "fit.csv")
    residual_path = export_ads_residual_csv(
        ads_result, tmp_path / "residual.csv"
    )

    fit = read_csv(fit_path)
    residual = read_csv(residual_path)
    assert fit[0] == ["energy", "time=0", "time=1"]
    assert fit[1] == ["100.0", "0.95", "0.45"]
    assert residual[1][0] == "100.0"
    np.testing.assert_allclose(
        np.asarray(residual[1][1:], dtype=float),
        np.array([0.05, 0.05]),
    )


def test_export_svd_as_tidy_csv(ads_result, tmp_path):
    output = export_ads_svd_csv(ads_result, tmp_path / "svd.csv")
    rows = read_csv(output)

    assert rows[0] == [
        "array",
        "row",
        "column",
        "axis_0",
        "axis_1",
        "value",
    ]
    assert {row[0] for row in rows[1:]} == {"U", "S", "Vh"}
    assert len(rows) == 1 + 3 * 2 + 2 + 2 * 2


def test_export_summary_report_and_custom_model(ads_result, tmp_path):
    summary_path = export_ads_summary_csv(
        ads_result, tmp_path / "summary.csv"
    )
    report_path = export_ads_report_txt(
        ads_result, tmp_path / "report.txt"
    )
    model_path = export_ads_rate_model_json(
        ads_result, tmp_path / "model.json"
    )

    summary = dict(read_csv(summary_path)[1:])
    assert summary["mode"] == "custom_sads_svd"
    assert json.loads(summary["spectrum_names"]) == ["MLCT", "MC"]
    assert report_path.read_text(encoding="utf-8").startswith("[ADS Result]")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model["edges"][0]["rate"] == pytest.approx(2.0)


def test_optional_exporters_reject_unavailable_data(ads_result, tmp_path):
    result = without_optional_arrays(ads_result)
    with pytest.raises(ValueError, match="errors are unavailable"):
        export_ads_spectra_error_csv(result, tmp_path / "errors.csv")
    with pytest.raises(ValueError, match="SVD arrays are unavailable"):
        export_ads_svd_csv(result, tmp_path / "svd.csv")
    with pytest.raises(ValueError, match="custom rate model"):
        export_ads_rate_model_json(result, tmp_path / "model.json")


class DummyFigure:
    def savefig(self, path):
        path.write_bytes(b"figure")


def test_export_bundle_writes_all_applicable_artifacts(ads_result, tmp_path):
    paths = export_ads_bundle(
        ads_result,
        tmp_path / "bundle",
        stem="sample",
        figure=DummyFigure(),
        figure_formats=("png", "pdf"),
    )

    assert set(paths) == {
        "spectra",
        "spectra_errors",
        "fit",
        "residual",
        "svd",
        "summary",
        "report",
        "rate_model",
        "figure_png",
        "figure_pdf",
    }
    assert all(path.exists() for path in paths.values())


def test_bundle_skips_optional_artifacts(ads_result, tmp_path):
    result = without_optional_arrays(ads_result)
    paths = export_ads_bundle(result, tmp_path / "minimal", figure_formats=())

    assert set(paths) == {"spectra", "fit", "residual", "summary", "report"}


def test_bundle_preflights_collisions(ads_result, tmp_path):
    directory = tmp_path / "collision"
    directory.mkdir()
    existing = directory / "ads_result_summary.csv"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="summary"):
        export_ads_bundle(ads_result, directory, figure_formats=())

    assert existing.read_text(encoding="utf-8") == "keep"
    assert not (directory / "ads_result_spectra.csv").exists()


@pytest.mark.parametrize(
    "stem",
    ["", ".", "..", "nested/result", "../result", "nested\\result"],
)
def test_bundle_rejects_unsafe_stem(ads_result, tmp_path, stem):
    with pytest.raises(ValueError, match="file-name stem"):
        export_ads_bundle(ads_result, tmp_path, stem=stem)


def test_export_validation_and_overwrite(ads_result, tmp_path):
    path = export_ads_report_txt(ads_result, tmp_path / "report.txt")
    with pytest.raises(FileExistsError):
        export_ads_report_txt(ads_result, path)
    export_ads_report_txt(ads_result, path, overwrite=True)

    with pytest.raises(TypeError, match="ADSResult"):
        export_ads_spectra_csv(object(), tmp_path / "bad.csv")
    with pytest.raises(TypeError, match="savefig"):
        export_ads_figure(object(), tmp_path / "bad.png")
    with pytest.raises(ValueError, match=".png or .pdf"):
        export_ads_figure(DummyFigure(), tmp_path / "bad.svg")
    with pytest.raises(ValueError, match="png.*pdf"):
        export_ads_bundle(
            ads_result,
            tmp_path / "bad-format",
            figure=DummyFigure(),
            figure_formats=("svg",),
        )


def test_result_tab_exports_bundle(qapp, ads_result, tmp_path, monkeypatch):
    tab = CalcDADSResultTab()
    assert not tab.export_button.isEnabled()
    tab.set_result(ads_result)
    assert tab.export_button.isEnabled()

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args: str(tmp_path),
    )
    tab.export_results_dialog()

    assert (tmp_path / "ads_result_spectra.csv").exists()
    assert (tmp_path / "ads_result_figure.png").exists()
    assert (tmp_path / "ads_result_figure.pdf").exists()

    tab.clear_result()
    assert not tab.export_button.isEnabled()


def test_result_tab_export_cancel_and_overwrite_choices(
    qapp, ads_result, tmp_path, monkeypatch
):
    from PyQt5.QtWidgets import QMessageBox

    tab = CalcDADSResultTab()
    tab.set_result(ads_result)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *args: ""
    )
    tab.export_results_dialog()

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args: str(tmp_path),
    )
    export_ads_bundle(
        ads_result,
        tmp_path,
        figure=tab.figure,
        figure_formats=("png", "pdf"),
    )
    report = tmp_path / "ads_result_report.txt"
    original = report.read_text(encoding="utf-8")

    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.No
    )
    tab.export_results_dialog()
    assert report.read_text(encoding="utf-8") == original

    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.Yes
    )
    tab.export_results_dialog()
    assert report.read_text(encoding="utf-8") == original


def test_result_tab_reports_export_error(
    qapp, ads_result, tmp_path, monkeypatch
):
    import TRXASprefitpack.gui.calc_dads_result_tab as result_tab_module

    tab = CalcDADSResultTab()
    tab.set_result(ads_result)
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args: str(tmp_path),
    )
    monkeypatch.setattr(
        result_tab_module,
        "export_ads_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("export failed")
        ),
    )
    errors = []
    monkeypatch.setattr(
        result_tab_module.QMessageBox,
        "critical",
        lambda *args: errors.append(args[-1]),
    )

    tab.export_results_dialog()

    assert errors == ["export failed"]
