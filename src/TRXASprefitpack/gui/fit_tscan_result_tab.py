"""Transient fitting result table, plot, report, and export UI."""

from __future__ import annotations

import os

import numpy as np

from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
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

from ..driver.transient_result import TransientResult
from .fit_tscan_export import (
    export_fit_csv,
    export_parameter_csv,
    export_report_txt,
    export_residual_csv,
)
from .result_views import (
    transient_result_to_fit_plot_arrays,
    transient_result_to_parameter_rows,
    transient_result_to_report_text,
    transient_result_to_residual_plot_arrays,
)


class FitTScanResultTab(QWidget):
    """Display and export a TransientResult."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.result: TransientResult | None = None
        self._fit_entries = []
        self._residual_entries = []

        self._create_ui()
        self.clear_result()

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)

        self.parameter_table = QTableWidget(0, 6, self)
        self.parameter_table.setHorizontalHeaderLabels(
            (
                "Parameter",
                "Value",
                "Error",
                "Lower",
                "Upper",
                "Fixed",
            )
        )
        self.parameter_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )
        root.addWidget(self.parameter_table)

        selector_row = QHBoxLayout()
        self.trace_selector = QComboBox(self)
        self.trace_selector.currentIndexChanged.connect(
            self._update_plot
        )
        selector_row.addWidget(self.trace_selector)

        self.copy_report_button = QPushButton(
            "Copy Report",
            self,
        )
        self.copy_report_button.clicked.connect(
            self.copy_report
        )
        selector_row.addWidget(self.copy_report_button)

        self.trace_combo = QComboBox()
        self.trace_combo.currentIndexChanged.connect(self._update_plot)
        selector_row.addWidget(self.trace_combo)

        selector_row.addSpacing(12)
        selector_row.addWidget(QLabel("Time scale"))

        self.xscale_combo = QComboBox()
        self.xscale_combo.addItem("Linear", "linear")
        self.xscale_combo.addItem("SymLog", "symlog")
        self.xscale_combo.currentIndexChanged.connect(self._update_xscale)
        selector_row.addWidget(self.xscale_combo)

        selector_row.addWidget(QLabel("linthresh"))

        self.linthresh_spin = QDoubleSpinBox()
        self.linthresh_spin.setDecimals(8)
        self.linthresh_spin.setRange(1.0e-8, 1.0e12)
        self.linthresh_spin.setValue(0.1)
        self.linthresh_spin.setSingleStep(0.1)
        self.linthresh_spin.setEnabled(False)
        self.linthresh_spin.valueChanged.connect(self._update_xscale)
        selector_row.addWidget(self.linthresh_spin)

        selector_row.addStretch()

        root.addLayout(selector_row)

        self.figure = Figure(figsize=(8, 5))
        self.fit_axis = self.figure.add_subplot(211)
        self.residual_axis = self.figure.add_subplot(212, sharex=self.fit_axis)
        self.canvas = FigureCanvas(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.canvas, self)

        root.addWidget(self.navigation_toolbar)
        root.addWidget(self.canvas)

        self._update_xscale()

        self.report_view = QPlainTextEdit(self)
        self.report_view.setReadOnly(True)
        root.addWidget(self.report_view)

        export_row = QHBoxLayout()

        self.export_parameter_button = QPushButton(
            "Export Parameters",
            self,
        )
        self.export_parameter_button.clicked.connect(
            lambda: self._select_export(
                "parameters.csv",
                export_parameter_csv,
            )
        )
        export_row.addWidget(self.export_parameter_button)

        self.export_fit_button = QPushButton(
            "Export Fits",
            self,
        )
        self.export_fit_button.clicked.connect(
            lambda: self._select_export(
                "fits.csv",
                export_fit_csv,
            )
        )
        export_row.addWidget(self.export_fit_button)

        self.export_residual_button = QPushButton(
            "Export Residuals",
            self,
        )
        self.export_residual_button.clicked.connect(
            lambda: self._select_export(
                "residuals.csv",
                export_residual_csv,
            )
        )
        export_row.addWidget(self.export_residual_button)

        self.export_report_button = QPushButton(
            "Export Report",
            self,
        )
        self.export_report_button.clicked.connect(
            lambda: self._select_export(
                "report.txt",
                export_report_txt,
            )
        )
        export_row.addWidget(self.export_report_button)

        root.addLayout(export_row)

        self._export_buttons = (
            self.export_parameter_button,
            self.export_fit_button,
            self.export_residual_button,
            self.export_report_button,
        )

    def set_result(
        self,
        result: TransientResult,
    ) -> None:
        self.result = result

        parameter_rows = (
            transient_result_to_parameter_rows(result)
        )
        self.parameter_table.setRowCount(
            len(parameter_rows)
        )

        for row_index, row in enumerate(parameter_rows):
            values = (
                row["name"],
                f"{row['value']:.8g}",
                f"{row['error']:.8g}",
                f"{row['lower_bound']:.8g}",
                f"{row['upper_bound']:.8g}",
                str(row["fixed"]),
            )

            for column, value in enumerate(values):
                self.parameter_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(value),
                )

        self._fit_entries = (
            transient_result_to_fit_plot_arrays(result)
        )
        self._residual_entries = (
            transient_result_to_residual_plot_arrays(result)
        )

        self.trace_selector.clear()

        for entry in self._fit_entries:
            self.trace_selector.addItem(
                f"{entry['dataset_name']} / "
                f"{entry['trace_name']}"
            )

        self.report_view.setPlainText(
            transient_result_to_report_text(result)
        )

        self.copy_report_button.setEnabled(True)

        for button in self._export_buttons:
            button.setEnabled(True)

        if self._fit_entries:
            self.trace_selector.setCurrentIndex(0)
            self._update_plot(0)

    def clear_result(self) -> None:
        self.result = None
        self._fit_entries = []
        self._residual_entries = []

        self.parameter_table.setRowCount(0)
        self.trace_selector.clear()
        self.report_view.clear()

        self.fit_axis.clear()
        self.residual_axis.clear()
        self.canvas.draw_idle()

        self.copy_report_button.setEnabled(False)

        for button in self._export_buttons:
            button.setEnabled(False)

    def copy_report(self) -> None:
        if self.result is None:
            return

        QApplication.clipboard().setText(
            self.report_view.toPlainText()
        )

    def _update_plot(self, index: int) -> None:
        if index < 0 or index >= len(self._fit_entries) or index >= len(self._residual_entries):
            return
        

        fit_entry = self._fit_entries[index]
        residual_entry = self._residual_entries[index]

        time = np.asarray(fit_entry["time"], dtype=float)
        intensity = np.asarray(fit_entry["intensity"], dtype=float)
        eps = np.asarray(fit_entry["eps"], dtype=float)
        fit = np.asarray(fit_entry["fit"], dtype=float)

        residual_time = np.asarray(residual_entry["time"], dtype=float)
        residual = np.asarray(residual_entry["residual"], dtype=float)
        residual_eps = np.asarray(residual_entry["eps"], dtype=float)

        if time.shape != residual_time.shape or not np.allclose(
            time, residual_time, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                "Fit trace and residual trace have different time axes."
                )

        if not (time.size == intensity.size
                == eps.size == fit.size
                == residual.size == residual_eps.size):
            raise ValueError(
                "Fit result arrays must have identical lengths."
            )

        self.fit_axis.clear()
        self.residual_axis.clear()

        self.fit_axis.errorbar(time, intensity, yerr=eps, fmt="o",
                               markersize=4, capsize=2, linestyle=None,
                               label="Data")

        self.fit_axis.plot(time, fit, linewidth=1.5, label="Fit")

        self.residual_axis.axhline(0.0, color="black", linewidth=0.8,)

        self.residual_axis.errorbar(time, residual, yerr=residual_eps,
                                    fmt="o", linestyle="none", markersize=4,
                                    capsize=2, elinewidth=0.8, label="Residual")

        title = fit_entry.get("label", f"trace_{index + 1}")
        self.fit_axis.set_title(title)
        self.fit_axis.set_ylabel("Intensity")
        self.fit_axis.legend()

        self.residual_axis.set_ylabel("Residual")
        self.residual_axis.set_xlabel("Time delay")

        self._apply_xscale()
        self._set_shared_time_limits(time)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _select_export(
        self,
        default_name: str,
        exporter,
    ) -> None:
        if self.result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export result",
            default_name,
            "All files (*)",
        )

        if not path:
            return

        path = Path(path)

        if path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite file?",
                f"{path.name} already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if answer != QMessageBox.Yes:
                return

        try:
            exporter(
                self.result,
                path,
                overwrite=True,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                str(exc),
            )

    def _update_xscale(self, *_args):
        scale = self.xscale_combo.currentData()
        self.linthresh_spin.setEnabled(scale == "symlog")

        if not hasattr(self, "fit_axis"):
            return

        self._apply_xscale()
        self.canvas.draw_idle()

    def _apply_xscale(self):
        scale = self.xscale_combo.currentData()

        if scale == "symlog":
            linthresh = self.linthresh_spin.value()
            self.fit_axis.set_xscale("symlog", linthresh=linthresh)
            self.residual_axis.set_xscale("symlog", linthresh=linthresh)
        else:
            self.fit_axis.set_xscale("linear")
            self.residual_axis.set_xscale("linear")

    def _set_shared_time_limits(self, time):
        finite_time = np.asarray(time, dtype=float)
        finite_time = finite_time[np.isfinite(finite_time)]

        if finite_time.size == 0:
            return

        time_min = float(np.min(finite_time))
        time_max = float(np.max(finite_time))

        if time_min == time_max:
            padding = max(abs(time_min), 1.0) * 0.05
        else:
            padding = (time_max - time_min) * 0.03

        self.fit_axis.set_xlim(time_min - padding, time_max + padding)