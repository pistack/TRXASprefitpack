"""Qt editor for safe, first-order custom rate models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .rate_model import (
    RateEdge,
    RateModelSpec,
    build_rate_matrix,
    rate_model_from_dict,
    rate_model_to_dict,
    solve_rate_model_real,
)


class RateModelEditor(QWidget):
    """Edit and validate rate models without evaluating Python code."""

    model_validated = pyqtSignal(object)
    model_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._validated_model: RateModelSpec | None = None
        self._species_snapshot: tuple[str, ...] = ()
        self._create_ui()
        self.set_model(
            RateModelSpec(
                species=("A", "B"),
                edges=(RateEdge("A", "B", 1.0),),
                y0=np.array([1.0, 0.0]),
            )
        )

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)

        species_group = QGroupBox("Species and initial populations", self)
        species_layout = QVBoxLayout(species_group)
        self.species_table = QTableWidget(0, 2, species_group)
        self.species_table.setHorizontalHeaderLabels(("Species", "y0"))
        self.species_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.species_table.itemChanged.connect(self._handle_species_edit)
        species_layout.addWidget(self.species_table)

        species_buttons = QHBoxLayout()
        self.add_species_button = QPushButton("Add species", species_group)
        self.remove_species_button = QPushButton(
            "Remove species", species_group
        )
        self.add_species_button.clicked.connect(self.add_species)
        self.remove_species_button.clicked.connect(self.remove_species)
        species_buttons.addWidget(self.add_species_button)
        species_buttons.addWidget(self.remove_species_button)
        species_buttons.addStretch()
        species_layout.addLayout(species_buttons)
        root.addWidget(species_group)

        edge_group = QGroupBox("First-order transitions", self)
        edge_layout = QVBoxLayout(edge_group)
        self.edge_table = QTableWidget(0, 3, edge_group)
        self.edge_table.setHorizontalHeaderLabels(
            ("Source", "Target", "Rate")
        )
        self.edge_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.edge_table.itemChanged.connect(self._invalidate)
        edge_layout.addWidget(self.edge_table)

        edge_buttons = QHBoxLayout()
        self.add_edge_button = QPushButton("Add edge", edge_group)
        self.remove_edge_button = QPushButton("Remove edge", edge_group)
        self.add_edge_button.clicked.connect(self.add_edge)
        self.remove_edge_button.clicked.connect(self.remove_edge)
        edge_buttons.addWidget(self.add_edge_button)
        edge_buttons.addWidget(self.remove_edge_button)
        edge_buttons.addStretch()
        edge_layout.addLayout(edge_buttons)
        root.addWidget(edge_group)

        matrix_group = QGroupBox("Rate matrix K", self)
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix_table = QTableWidget(0, 0, matrix_group)
        self.matrix_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        matrix_layout.addWidget(self.matrix_table)
        root.addWidget(matrix_group)

        action_row = QHBoxLayout()
        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        action_row.addWidget(self.validation_label, 1)

        self.validate_button = QPushButton("Validate model", self)
        self.load_button = QPushButton("Load JSON", self)
        self.save_button = QPushButton("Save JSON", self)
        self.validate_button.clicked.connect(self.validate_model)
        self.load_button.clicked.connect(self.load_model_dialog)
        self.save_button.clicked.connect(self.save_model_dialog)
        action_row.addWidget(self.validate_button)
        action_row.addWidget(self.load_button)
        action_row.addWidget(self.save_button)
        root.addLayout(action_row)

    @property
    def validated_model(self) -> RateModelSpec | None:
        """Return the most recently validated, unmodified model."""
        return self._validated_model

    def species_names(self) -> tuple[str, ...]:
        return tuple(
            self._item_text(self.species_table, row, 0).strip()
            for row in range(self.species_table.rowCount())
        )

    def build_model(self) -> RateModelSpec:
        """Build a model using float literals from the editor tables."""
        species = self.species_names()
        if not species:
            raise ValueError("species must contain at least one name.")
        if any(not name for name in species):
            raise ValueError("species names must not be empty.")
        if len(set(species)) != len(species):
            raise ValueError("species names must be unique.")

        y0 = np.array(
            [
                self._parse_float_literal(
                    self._item_text(self.species_table, row, 1),
                    f"y0[{row}]",
                )
                for row in range(self.species_table.rowCount())
            ],
            dtype=float,
        )

        edges = []
        for row in range(self.edge_table.rowCount()):
            source_widget = self.edge_table.cellWidget(row, 0)
            target_widget = self.edge_table.cellWidget(row, 1)
            if not isinstance(source_widget, QComboBox):
                raise ValueError(f"edge[{row}] source is missing.")
            if not isinstance(target_widget, QComboBox):
                raise ValueError(f"edge[{row}] target is missing.")
            rate = self._parse_float_literal(
                self._item_text(self.edge_table, row, 2),
                f"edge[{row}] rate",
            )
            edges.append(
                RateEdge(
                    source_widget.currentText(),
                    target_widget.currentText(),
                    rate,
                )
            )

        return RateModelSpec(
            species=species,
            edges=tuple(edges),
            y0=y0,
        )

    def validate_model(self) -> RateModelSpec | None:
        """Validate structure and confirm a real eigenmode solution."""
        try:
            model = self.build_model()
            matrix = build_rate_matrix(model)
            solve_rate_model_real(matrix, model.y0)
        except (TypeError, ValueError) as exc:
            self._validated_model = None
            self.validation_label.setText(str(exc))
            self._clear_matrix()
            return None

        self._validated_model = model
        self.validation_label.setText("Valid real-eigenmode rate model.")
        self._show_matrix(model, matrix)
        self.model_validated.emit(model)
        return model

    def set_model(self, model: RateModelSpec) -> None:
        """Replace the editor contents with a validated model."""
        matrix = build_rate_matrix(model)
        solve_rate_model_real(matrix, model.y0)

        self.species_table.blockSignals(True)
        self.edge_table.blockSignals(True)
        try:
            self.species_table.setRowCount(0)
            for name, population in zip(model.species, model.y0):
                row = self.species_table.rowCount()
                self.species_table.insertRow(row)
                self.species_table.setItem(row, 0, QTableWidgetItem(name))
                self.species_table.setItem(
                    row, 1, QTableWidgetItem(f"{population:.12g}")
                )

            self._species_snapshot = self.species_names()
            self.edge_table.setRowCount(0)
            for edge in model.edges:
                self.add_edge(
                    source=edge.source,
                    target=edge.target,
                    rate=edge.rate,
                )
        finally:
            self.edge_table.blockSignals(False)
            self.species_table.blockSignals(False)

        self._validated_model = model
        self.validation_label.setText("Valid real-eigenmode rate model.")
        self._show_matrix(model, matrix)

    def add_species(
        self,
        name: str | None = None,
        population: float = 0.0,
    ) -> None:
        row = self.species_table.rowCount()
        if name is None:
            existing = set(self.species_names())
            number = row + 1
            name = f"species_{number}"
            while name in existing:
                number += 1
                name = f"species_{number}"

        self.species_table.insertRow(row)
        self.species_table.setItem(row, 0, QTableWidgetItem(str(name)))
        self.species_table.setItem(
            row, 1, QTableWidgetItem(f"{float(population):.12g}")
        )
        self._refresh_edge_species()
        self._species_snapshot = self.species_names()
        self._invalidate()

    def remove_species(self) -> None:
        old_species = self.species_names()
        rows = self._selected_rows(self.species_table)
        if not rows and self.species_table.currentRow() >= 0:
            rows = [self.species_table.currentRow()]
        for row in sorted(rows, reverse=True):
            self.species_table.removeRow(row)
        self._refresh_edge_species(old_species=old_species)
        self._species_snapshot = self.species_names()
        self._invalidate()

    def add_edge(
        self,
        source: str | None = None,
        target: str | None = None,
        rate: float = 1.0,
    ) -> None:
        species = self.species_names()
        row = self.edge_table.rowCount()
        self.edge_table.insertRow(row)

        source_combo = self._make_species_combo(species, source)
        default_target = target
        if default_target is None and len(species) > 1:
            default_target = species[1]
        target_combo = self._make_species_combo(species, default_target)
        source_combo.currentIndexChanged.connect(self._invalidate)
        target_combo.currentIndexChanged.connect(self._invalidate)

        self.edge_table.setCellWidget(row, 0, source_combo)
        self.edge_table.setCellWidget(row, 1, target_combo)
        self.edge_table.setItem(
            row, 2, QTableWidgetItem(f"{float(rate):.12g}")
        )
        self._invalidate()

    def remove_edge(self) -> None:
        rows = self._selected_rows(self.edge_table)
        if not rows and self.edge_table.currentRow() >= 0:
            rows = [self.edge_table.currentRow()]
        for row in sorted(rows, reverse=True):
            self.edge_table.removeRow(row)
        self._invalidate()

    def save_model(self, path: str | Path) -> RateModelSpec:
        """Validate and save a model as JSON."""
        model = self.validate_model()
        if model is None:
            raise ValueError(self.validation_label.text())
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(rate_model_to_dict(model), stream, indent=2)
            stream.write("\n")
        return model

    def load_model(self, path: str | Path) -> RateModelSpec:
        """Load JSON through the pure-layer schema validator."""
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        model = rate_model_from_dict(data)
        self.set_model(model)
        self.model_validated.emit(model)
        return model

    def save_model_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save rate model", "", "JSON files (*.json)"
        )
        if path:
            try:
                self.save_model(path)
            except (OSError, TypeError, ValueError) as exc:
                self.validation_label.setText(str(exc))

    def load_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load rate model", "", "JSON files (*.json)"
        )
        if path:
            try:
                self.load_model(path)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                self.validation_label.setText(str(exc))

    def _handle_species_edit(self) -> None:
        old_species = self._species_snapshot
        new_species = self.species_names()
        self._refresh_edge_species(old_species=old_species)
        self._species_snapshot = new_species
        self._invalidate()

    def _refresh_edge_species(
        self,
        *,
        old_species: tuple[str, ...] | None = None,
    ) -> None:
        species = self.species_names()
        for row in range(self.edge_table.rowCount()):
            for column in (0, 1):
                combo = self.edge_table.cellWidget(row, column)
                if not isinstance(combo, QComboBox):
                    continue
                selected = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(species)
                index = combo.findText(selected)
                if (
                    index < 0
                    and old_species is not None
                    and selected in old_species
                ):
                    old_index = old_species.index(selected)
                    if old_index < len(species):
                        index = old_index
                if index >= 0:
                    combo.setCurrentIndex(index)
                combo.blockSignals(False)

    def _show_matrix(
        self,
        model: RateModelSpec,
        matrix: np.ndarray,
    ) -> None:
        size = len(model.species)
        self.matrix_table.setRowCount(size)
        self.matrix_table.setColumnCount(size)
        self.matrix_table.setHorizontalHeaderLabels(model.species)
        self.matrix_table.setVerticalHeaderLabels(model.species)
        for row in range(size):
            for column in range(size):
                self.matrix_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(f"{matrix[row, column]:.8g}"),
                )

    def _clear_matrix(self) -> None:
        self.matrix_table.setRowCount(0)
        self.matrix_table.setColumnCount(0)

    def _invalidate(self, *args) -> None:
        del args
        self._validated_model = None
        self.validation_label.setText("Model has unvalidated changes.")
        self._clear_matrix()
        self.model_changed.emit()

    @staticmethod
    def _make_species_combo(
        species: tuple[str, ...],
        selected: str | None,
    ) -> QComboBox:
        combo = QComboBox()
        combo.addItems(species)
        if selected is not None:
            index = combo.findText(selected)
            if index >= 0:
                combo.setCurrentIndex(index)
        return combo

    @staticmethod
    def _item_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return "" if item is None else item.text()

    @staticmethod
    def _selected_rows(table: QTableWidget) -> list[int]:
        return sorted({index.row() for index in table.selectedIndexes()})

    @staticmethod
    def _parse_float_literal(text: str, name: str) -> float:
        try:
            value = float(text.strip())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric literal.") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value
