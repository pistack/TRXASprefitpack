"""SVD inspection tab for calc_dads_qt."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from .models import EScanDataset


class CalcDADSSVDTab(QWidget):
    """Inspect the data SVD and choose its relative cutoff."""

    cutoff_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dataset: EScanDataset | None = None
        self._u = np.empty((0, 0))
        self._s = np.empty(0)
        self._vh = np.empty((0, 0))
        self._create_ui()
        self.clear_dataset()

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Relative SVD cutoff", self))
        self.cutoff_spin = QDoubleSpinBox(self)
        self.cutoff_spin.setDecimals(8)
        self.cutoff_spin.setRange(0.0, 1.0)
        self.cutoff_spin.setSingleStep(0.001)
        self.cutoff_spin.setValue(0.0)
        self.cutoff_spin.valueChanged.connect(self._update_cutoff)
        controls.addWidget(self.cutoff_spin)

        controls.addWidget(QLabel("Component", self))
        self.component_combo = QComboBox(self)
        self.component_combo.currentIndexChanged.connect(
            self._update_plot
        )
        controls.addWidget(self.component_combo)

        self.retained_label = QLabel(self)
        controls.addWidget(self.retained_label)
        controls.addStretch()
        root.addLayout(controls)

        self.singular_table = QTableWidget(0, 3, self)
        self.singular_table.setHorizontalHeaderLabels(
            ("Component", "Singular value", "Relative")
        )
        self.singular_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )
        root.addWidget(self.singular_table)

        self.figure = Figure(figsize=(8, 5))
        self.value_axis = self.figure.add_subplot(221)
        self.energy_axis = self.figure.add_subplot(223)
        self.time_axis = self.figure.add_subplot(224)
        self.canvas = FigureCanvas(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.canvas, self)
        root.addWidget(self.navigation_toolbar)
        root.addWidget(self.canvas)

    @property
    def cond_num(self) -> float:
        return float(self.cutoff_spin.value())

    def set_dataset(self, dataset: EScanDataset | None) -> None:
        if dataset is None:
            self.clear_dataset()
            return
        if not isinstance(dataset, EScanDataset):
            raise TypeError("dataset must be an EScanDataset.")

        self._dataset = dataset
        self._u, self._s, self._vh = linalg.svd(
            dataset.intensity,
            full_matrices=False,
        )
        self._populate_table()
        self._populate_components()
        self._update_cutoff()

    def clear_dataset(self) -> None:
        self._dataset = None
        self._u = np.empty((0, 0))
        self._s = np.empty(0)
        self._vh = np.empty((0, 0))
        self.singular_table.setRowCount(0)
        self.component_combo.clear()
        self.retained_label.setText("No dataset loaded.")
        self._update_plot()

    def _populate_table(self) -> None:
        self.singular_table.setRowCount(self._s.size)
        scale = self._s[0] if self._s.size and self._s[0] else 1.0
        for row, value in enumerate(self._s):
            values = (str(row + 1), f"{value:.8g}", f"{value / scale:.8g}")
            for column, text in enumerate(values):
                self.singular_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(text),
                )

    def _populate_components(self) -> None:
        self.component_combo.clear()
        for index in range(self._s.size):
            self.component_combo.addItem(
                f"Component {index + 1}",
                index,
            )

    def _update_cutoff(self) -> None:
        if self._s.size:
            retained = int(
                np.sum(self._s > self.cond_num * self._s[0])
            )
            self.retained_label.setText(f"Retained: {retained}")
        else:
            self.retained_label.setText("No dataset loaded.")
        self.cutoff_changed.emit(self.cond_num)
        self._update_plot()

    def _update_plot(self) -> None:
        self.value_axis.clear()
        self.energy_axis.clear()
        self.time_axis.clear()

        self.value_axis.set_title("Singular values")
        self.energy_axis.set_title("Left singular vector")
        self.time_axis.set_title("Right singular vector")

        if self._dataset is not None and self._s.size:
            indices = np.arange(1, self._s.size + 1)
            self.value_axis.semilogy(indices, self._s, "o-")
            self.value_axis.axhline(
                self.cond_num * self._s[0],
                color="tab:red",
                linestyle="--",
            )

            component = self.component_combo.currentData()
            if component is not None:
                component = int(component)
                self.energy_axis.plot(
                    self._dataset.energy,
                    self._u[:, component],
                )
                self.time_axis.plot(
                    self._dataset.time,
                    self._vh[component, :],
                )

        self.value_axis.set_xlabel("Component")
        self.energy_axis.set_xlabel("Energy")
        self.time_axis.set_xlabel("Time delay")
        self.figure.tight_layout()
        self.canvas.draw_idle()
