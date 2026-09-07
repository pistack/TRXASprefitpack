import json
import os
import sys

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QTableWidgetItem,
)

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.rate_model import RateEdge, RateModelSpec
from TRXASprefitpack.gui.rate_model_editor import RateModelEditor


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance()
    if application is None:
        application = QApplication(["test_rate_model_editor"])
    return application


@pytest.fixture
def sequential_model():
    return RateModelSpec(
        species=("S1", "S2", "S3"),
        edges=(
            RateEdge("S1", "S2", 0.5),
            RateEdge("S2", "S3", 0.1),
        ),
        y0=np.array([1.0, 0.0, 0.0]),
    )


def test_editor_starts_with_valid_real_model(qapp):
    editor = RateModelEditor()

    model = editor.validated_model

    assert model is not None
    assert model.species == ("A", "B")
    assert editor.matrix_table.rowCount() == 2
    assert editor.matrix_table.item(0, 0).text() == "-1"
    assert editor.matrix_table.item(1, 0).text() == "1"


def test_editor_builds_and_validates_model(qapp, sequential_model):
    editor = RateModelEditor()
    received = []
    editor.model_validated.connect(received.append)
    editor.set_model(sequential_model)

    model = editor.validate_model()

    assert model is not None
    assert model.species == sequential_model.species
    assert model.edges == sequential_model.edges
    np.testing.assert_allclose(model.y0, sequential_model.y0)
    assert received[-1] is model
    assert editor.matrix_table.rowCount() == 3


def test_editor_adds_and_removes_species(qapp):
    editor = RateModelEditor()

    editor.add_species("C", 0.0)
    assert editor.species_names() == ("A", "B", "C")

    editor.species_table.selectRow(2)
    editor.remove_species()
    assert editor.species_names() == ("A", "B")
    assert editor.validated_model is None


def test_editor_adds_and_removes_edge(qapp):
    editor = RateModelEditor()
    initial_count = editor.edge_table.rowCount()

    editor.add_edge(source="B", target="A", rate=0.25)

    assert editor.edge_table.rowCount() == initial_count + 1
    source = editor.edge_table.cellWidget(initial_count, 0)
    target = editor.edge_table.cellWidget(initial_count, 1)
    assert isinstance(source, QComboBox)
    assert isinstance(target, QComboBox)
    assert source.currentText() == "B"
    assert target.currentText() == "A"

    editor.edge_table.selectRow(initial_count)
    editor.remove_edge()
    assert editor.edge_table.rowCount() == initial_count


def test_species_rename_updates_edge_choices(qapp):
    editor = RateModelEditor()
    editor.species_table.item(1, 0).setText("Product")

    target = editor.edge_table.cellWidget(0, 1)

    assert isinstance(target, QComboBox)
    assert target.findText("Product") >= 0
    assert target.findText("B") == -1
    assert target.currentText() == "Product"


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "1 + 2"])
def test_editor_rejects_invalid_rate_literals(qapp, rate):
    editor = RateModelEditor()
    editor.edge_table.setItem(0, 2, QTableWidgetItem(rate))

    assert editor.validate_model() is None
    assert editor.validated_model is None
    assert editor.matrix_table.rowCount() == 0


def test_editor_rejects_duplicate_species(qapp):
    editor = RateModelEditor()
    editor.species_table.item(1, 0).setText("A")

    assert editor.validate_model() is None
    assert "unique" in editor.validation_label.text()


def test_editor_rejects_empty_and_blank_species(qapp):
    editor = RateModelEditor()
    editor.species_table.setRowCount(0)
    editor.edge_table.setRowCount(0)

    with pytest.raises(ValueError, match="at least one"):
        editor.build_model()

    editor.add_species(" ", 1.0)
    with pytest.raises(ValueError, match="must not be empty"):
        editor.build_model()


def test_editor_rejects_missing_edge_controls(qapp):
    editor = RateModelEditor()
    editor.edge_table.removeCellWidget(0, 0)
    with pytest.raises(ValueError, match="source is missing"):
        editor.build_model()

    editor = RateModelEditor()
    editor.edge_table.removeCellWidget(0, 1)
    with pytest.raises(ValueError, match="target is missing"):
        editor.build_model()


def test_editor_rejects_complex_eigenmode(qapp):
    editor = RateModelEditor()
    cycle = RateModelSpec(
        species=("A", "B", "C"),
        edges=(
            RateEdge("A", "B", 1.0),
            RateEdge("B", "C", 1.0),
            RateEdge("C", "A", 1.0),
        ),
        y0=np.array([1.0, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="complex mode"):
        editor.set_model(cycle)


def test_editor_json_roundtrip(qapp, tmp_path, sequential_model):
    editor = RateModelEditor()
    editor.set_model(sequential_model)
    json_path = tmp_path / "rate_model.json"

    editor.save_model(json_path)
    loaded_editor = RateModelEditor()
    restored = loaded_editor.load_model(json_path)

    assert restored.species == sequential_model.species
    assert restored.edges == sequential_model.edges
    np.testing.assert_allclose(restored.y0, sequential_model.y0)
    assert loaded_editor.validated_model is restored


def test_editor_refuses_to_save_invalid_model(qapp, tmp_path):
    editor = RateModelEditor()
    editor.edge_table.item(0, 2).setText("not-a-number")

    with pytest.raises(ValueError, match="numeric literal"):
        editor.save_model(tmp_path / "invalid.json")


def test_editor_file_dialog_roundtrip(
    qapp, tmp_path, sequential_model, monkeypatch
):
    path = tmp_path / "dialog-model.json"
    editor = RateModelEditor()
    editor.set_model(sequential_model)
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args: (str(path), "JSON files (*.json)"),
    )
    editor.save_model_dialog()
    assert path.exists()

    restored_editor = RateModelEditor()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: (str(path), "JSON files (*.json)"),
    )
    restored_editor.load_model_dialog()
    assert restored_editor.validated_model is not None
    assert restored_editor.validated_model.species == sequential_model.species


def test_editor_dialog_cancel_and_load_error(qapp, tmp_path, monkeypatch):
    editor = RateModelEditor()
    original = editor.validated_model

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: ("", "")
    )
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args: ("", "")
    )
    editor.save_model_dialog()
    editor.load_model_dialog()
    assert editor.validated_model is original

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: (str(invalid_path), "JSON files (*.json)"),
    )
    editor.load_model_dialog()
    assert editor.validated_model is original
    assert editor.validation_label.text()


def test_editor_load_rejects_arbitrary_expression(qapp, tmp_path):
    path = tmp_path / "unsafe.json"
    path.write_text(
        json.dumps(
            {
                "species": ["A", "B"],
                "edges": [
                    {"source": "A", "target": "B", "rate": "1 + 2"}
                ],
                "y0": [1.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    editor = RateModelEditor()

    with pytest.raises(ValueError, match="finite positive"):
        editor.load_model(path)
