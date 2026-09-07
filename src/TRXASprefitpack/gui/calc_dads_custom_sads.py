"""Custom-rate SADS controls for calc_dads_qt."""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .rate_model import (
    RateModelSpec,
    build_rate_matrix,
    solve_rate_model_real,
)
from .rate_model_editor import RateModelEditor


class CalcDADSCustomSADSPanel(QWidget):
    """Edit, validate, and summarize a custom real-mode rate model."""

    model_validated = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._create_ui()
        self.editor.model_changed.connect(self.clear_mode_summary)
        self.editor.model_validated.connect(self._show_mode_summary)
        initial_model = self.editor.validated_model
        if initial_model is not None:
            self._show_mode_summary(initial_model)

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        editor_group = QGroupBox("Custom rate model", self)
        editor_layout = QVBoxLayout(editor_group)
        self.editor = RateModelEditor(editor_group)
        editor_layout.addWidget(self.editor)
        root.addWidget(editor_group)

        summary_group = QGroupBox("Real eigenmode summary", self)
        summary_layout = QVBoxLayout(summary_group)
        self.mode_summary_table = QTableWidget(0, 3, summary_group)
        self.mode_summary_table.setHorizontalHeaderLabels(
            ("Mode", "Eigenvalue", "Initial coefficient")
        )
        self.mode_summary_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )
        summary_layout.addWidget(self.mode_summary_table)
        self.mode_summary_label = QLabel(summary_group)
        self.mode_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.mode_summary_label)
        root.addWidget(summary_group)

    def build_rate_model(self) -> RateModelSpec:
        """Validate the currently edited model immediately before a run."""
        model = self.editor.validate_model()
        if model is None:
            message = self.editor.validation_label.text()
            raise ValueError(f"Invalid custom rate model: {message}")
        return model

    def set_model(self, model: RateModelSpec) -> None:
        self.editor.set_model(model)
        self._show_mode_summary(model)

    def clear_mode_summary(self) -> None:
        self.mode_summary_table.setRowCount(0)
        self.mode_summary_label.setText(
            "Validate the model to inspect its real eigenmodes."
        )

    def _show_mode_summary(self, model: RateModelSpec) -> None:
        matrix = build_rate_matrix(model)
        eigval, _eigenvectors, coefficients = solve_rate_model_real(
            matrix,
            model.y0,
        )

        order = np.argsort(eigval)
        self.mode_summary_table.setRowCount(len(order))
        for row, mode_index in enumerate(order):
            values = (
                str(int(mode_index) + 1),
                f"{eigval[mode_index]:.8g}",
                f"{coefficients[mode_index]:.8g}",
            )
            for column, value in enumerate(values):
                self.mode_summary_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )

        self.mode_summary_label.setText(
            f"{len(model.species)} species; "
            f"{len(model.edges)} transitions; real modes only."
        )
        self.model_validated.emit(model)
