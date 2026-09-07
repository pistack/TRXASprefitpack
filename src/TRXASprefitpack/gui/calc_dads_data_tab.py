"""Energy-scan dataset loading tab for calc_dads_qt."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .data_loader import read_escan_dataset
from .models import EScanDataset


class CalcDADSDataTab(QWidget):
    """Load and preview one energy-scan matrix dataset."""

    dataset_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dataset: EScanDataset | None = None
        self._create_ui()
        self._refresh_preview()

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit("dataset_1", self)
        form.addRow("Dataset name", self.name_edit)

        self.intensity_edit = QLineEdit(self)
        form.addRow(
            "Intensity matrix",
            self._path_row(
                self.intensity_edit,
                "Select intensity matrix",
            ),
        )

        self.time_edit = QLineEdit(self)
        form.addRow(
            "Time delays",
            self._path_row(
                self.time_edit,
                "Select time-delay file",
            ),
        )

        self.eps_edit = QLineEdit(self)
        form.addRow(
            "Error matrix",
            self._path_row(
                self.eps_edit,
                "Select error matrix",
            ),
        )

        root.addLayout(form)

        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset", self)
        self.load_button.clicked.connect(self.load_from_fields)
        button_row.addWidget(self.load_button)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_dataset)
        button_row.addWidget(self.clear_button)
        button_row.addStretch()
        root.addLayout(button_row)

        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        self.preview_table = QTableWidget(0, 0, self)
        self.preview_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )
        root.addWidget(self.preview_table)

        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        root.addWidget(self.validation_label)

    def _path_row(self, edit: QLineEdit, title: str) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)

        button = QPushButton("Browse...", widget)
        button.clicked.connect(
            lambda: self._select_path(edit, title)
        )
        layout.addWidget(button)
        return widget

    def _select_path(self, edit: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Text data (*.txt *.dat *.csv);;All files (*)",
        )
        if path:
            edit.setText(path)

    def load_from_fields(self) -> None:
        try:
            self.load_dataset(
                self.intensity_edit.text(),
                self.time_edit.text(),
                self.eps_edit.text(),
                name=self.name_edit.text().strip() or None,
            )
        except Exception as exc:
            self.validation_label.setText(str(exc))
            QMessageBox.critical(
                self,
                "Could not load dataset",
                str(exc),
            )

    def load_dataset(
        self,
        intensity_path: str | Path,
        time_path: str | Path,
        eps_path: str | Path,
        *,
        name: str | None = None,
    ) -> EScanDataset:
        dataset = read_escan_dataset(
            intensity_path,
            time_path,
            eps_path,
            name=name,
        )
        self.set_dataset(dataset)
        return dataset

    def set_dataset(self, dataset: EScanDataset) -> None:
        if not isinstance(dataset, EScanDataset):
            raise TypeError("dataset must be an EScanDataset.")

        self._dataset = dataset
        self.name_edit.setText(dataset.name)

        if dataset.intensity_path is not None:
            self.intensity_edit.setText(str(dataset.intensity_path))
        if dataset.time_path is not None:
            self.time_edit.setText(str(dataset.time_path))
        if dataset.eps_path is not None:
            self.eps_edit.setText(str(dataset.eps_path))

        self.validation_label.setText("")
        self._refresh_preview()
        self.dataset_changed.emit(dataset)

    def dataset(self) -> EScanDataset:
        if self._dataset is None:
            raise ValueError("Load an energy-scan dataset first.")
        return self._dataset

    def clear_dataset(self) -> None:
        self._dataset = None
        self._refresh_preview()
        self.validation_label.setText("")
        self.dataset_changed.emit(None)

    def _refresh_preview(self) -> None:
        self.preview_table.clear()

        if self._dataset is None:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            self.summary_label.setText("No dataset loaded.")
            return

        dataset = self._dataset
        self.summary_label.setText(
            f"{dataset.name}: {dataset.n_energy} energy points × "
            f"{dataset.n_time} time delays"
        )

        self.preview_table.setRowCount(dataset.n_energy)
        self.preview_table.setColumnCount(dataset.n_time + 1)
        self.preview_table.setHorizontalHeaderLabels(
            ["energy"]
            + [f"t={value:g}" for value in dataset.time]
        )

        for row, energy in enumerate(dataset.energy):
            self.preview_table.setItem(
                row,
                0,
                QTableWidgetItem(f"{energy:.8g}"),
            )
            for column in range(dataset.n_time):
                self.preview_table.setItem(
                    row,
                    column + 1,
                    QTableWidgetItem(
                        f"{dataset.intensity[row, column]:.8g}"
                    ),
                )
