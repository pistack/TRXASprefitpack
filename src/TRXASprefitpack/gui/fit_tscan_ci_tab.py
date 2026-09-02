"""Confidence interval scan tab for fit_tscan_qt."""

import numpy as np

from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .fit_tscan_ci import (
    ci_error_rows_to_report,
    ci_results_to_error_rows,
    run_selected_ci_scans,
)
from .fit_tscan_ci_worker import FitTScanCIWorker


class FitTScanCITab(QWidget):
    def __init__(
        self,
        ci_runner=run_selected_ci_scans,
        parent=None,
    ):
        super().__init__(parent)

        self._result = None
        self._ci_runner = ci_runner
        self._thread = None
        self._worker = None

        self._create_ui()
        self.clear_result()

    def _create_ui(self):
        root = QVBoxLayout(self)

        root.addWidget(
            QLabel(
                "Select parameters for profile CI scanning. "
                "Both 1σ and 2σ intervals will be calculated."
            )
        )

        self.parameter_table = QTableWidget(0, 4)
        self.parameter_table.setHorizontalHeaderLabels(
            [
                "Scan",
                "Parameter",
                "Best value",
                "Current covariance error",
            ]
        )
        root.addWidget(self.parameter_table)

        button_row = QHBoxLayout()

        self.select_all_button = QPushButton("Select all")
        self.select_all_button.clicked.connect(
            self._select_all_parameters
        )
        button_row.addWidget(self.select_all_button)

        self.clear_selection_button = QPushButton("Clear selection")
        self.clear_selection_button.clicked.connect(
            self._clear_parameter_selection
        )
        button_row.addWidget(self.clear_selection_button)

        button_row.addStretch()

        self.run_button = QPushButton("Estimate 1σ and 2σ errors")
        self.run_button.clicked.connect(self._start_scan)
        button_row.addWidget(self.run_button)

        root.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.status_label = QLabel()
        root.addWidget(self.status_label)

        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            [
                "Parameter",
                "Best value",
                "−1σ",
                "+1σ",
                "−2σ",
                "+2σ",
            ]
        )
        root.addWidget(self.result_table)

        self.report_edit = QTextEdit()
        self.report_edit.setReadOnly(True)
        root.addWidget(self.report_edit)

        report_row = QHBoxLayout()
        report_row.addStretch()

        self.copy_button = QPushButton("Copy report")
        self.copy_button.clicked.connect(self._copy_report)
        report_row.addWidget(self.copy_button)

        root.addLayout(report_row)

    def set_result(self, result):
        self._result = result
        self.parameter_table.setRowCount(len(result["x"]))

        for row, (name, value, error, bounds) in enumerate(
            zip(
                result["param_name"],
                result["x"],
                result["x_eps"],
                result["bounds"],
            )
        ):
            fixed = bounds[0] == bounds[1]

            check_item = QTableWidgetItem()
            check_item.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsUserCheckable
            )

            if fixed:
                check_item.setCheckState(Qt.Unchecked)
                check_item.setFlags(Qt.NoItemFlags)
                check_item.setToolTip(
                    "Fixed parameters cannot be scanned."
                )
            else:
                check_item.setCheckState(Qt.Checked)

            self.parameter_table.setItem(row, 0, check_item)
            self.parameter_table.setItem(
                row,
                1,
                QTableWidgetItem(str(name)),
            )
            self.parameter_table.setItem(
                row,
                2,
                QTableWidgetItem(f"{float(value):.8g}"),
            )
            self.parameter_table.setItem(
                row,
                3,
                QTableWidgetItem(f"{float(error):.8g}"),
            )

        self.result_table.setRowCount(0)
        self.report_edit.clear()
        self.status_label.setText("Ready")
        self._set_controls_enabled(True)

    def clear_result(self):
        self._result = None
        self.parameter_table.setRowCount(0)
        self.result_table.setRowCount(0)
        self.report_edit.clear()
        self.status_label.setText(
            "Run a fit before calculating confidence intervals."
        )
        self._set_controls_enabled(False)

    def selected_parameter_indices(self):
        indices = []

        for row in range(self.parameter_table.rowCount()):
            item = self.parameter_table.item(row, 0)

            if (
                item is not None
                and item.flags() != Qt.NoItemFlags
                and item.checkState() == Qt.Checked
            ):
                indices.append(row)

        return indices

    def _select_all_parameters(self):
        for row in range(self.parameter_table.rowCount()):
            item = self.parameter_table.item(row, 0)

            if item is not None and item.flags() != Qt.NoItemFlags:
                item.setCheckState(Qt.Checked)

    def _clear_parameter_selection(self):
        for row in range(self.parameter_table.rowCount()):
            item = self.parameter_table.item(row, 0)

            if item is not None and item.flags() != Qt.NoItemFlags:
                item.setCheckState(Qt.Unchecked)

    def _start_scan(self):
        if self._result is None:
            return

        selected_indices = self.selected_parameter_indices()

        if not selected_indices:
            self.status_label.setText(
                "Select at least one parameter."
            )
            return

        if self._thread is not None:
            return

        self._set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText(
            "Calculating 1σ and 2σ profile intervals..."
        )

        self._thread = QThread(self)
        self._worker = FitTScanCIWorker(
            result=self._result,
            parameter_indices=selected_indices,
            ci_runner=self._ci_runner,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)

        self._worker.finished.connect(
            lambda results: self._scan_finished(
                selected_indices,
                results,
            )
        )
        self._worker.failed.connect(self._scan_failed)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)

        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    def _scan_finished(
        self,
        selected_indices,
        ci_results,
    ):
        rows = ci_results_to_error_rows(
            self._result,
            selected_indices,
            ci_results,
        )

        self._display_rows(rows)
        self.report_edit.setPlainText(
            ci_error_rows_to_report(rows)
        )
        self.status_label.setText("CI scan completed.")
        self.copy_button.setEnabled(True)

    def _scan_failed(self, exception):
        self.status_label.setText(
            f"CI scan failed: {exception}"
        )

    def _thread_finished(self):
        thread = self._thread

        self._thread = None
        self._worker = None

        self.progress_bar.setVisible(False)
        self._set_controls_enabled(self._result is not None)

        if thread is not None:
            thread.deleteLater()

    def _display_rows(self, rows):
        self.result_table.setRowCount(len(rows))

        for row_index, row in enumerate(rows):
            values = [
                row.parameter_name,
                self._format_number(row.value),
                self._format_number(row.minus_1sigma),
                self._format_number(row.plus_1sigma),
                self._format_number(row.minus_2sigma),
                self._format_number(row.plus_2sigma),
            ]

            for column, value in enumerate(values):
                self.result_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(value),
                )

    @staticmethod
    def _format_number(value):
        if np.isnan(value):
            return "not found"

        return f"{value:.8g}"

    def _copy_report(self):
        QApplication.clipboard().setText(
            self.report_edit.toPlainText()
        )

    def _set_controls_enabled(self, enabled):
        self.parameter_table.setEnabled(enabled)
        self.select_all_button.setEnabled(enabled)
        self.clear_selection_button.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.copy_button.setEnabled(
            enabled and bool(self.report_edit.toPlainText())
        )