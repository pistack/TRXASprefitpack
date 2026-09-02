"""Complete PyQt5 window for fit_tscan_qt."""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QThread
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from .fit_job import run_fit_transient_exp_config
from .fit_tscan_data_tab import FitTScanDataTab
from .fit_tscan_parameter_tabs import FitTScanParameterTabs
from .fit_tscan_result_tab import FitTScanResultTab
from .fit_tscan_ci_tab import FitTScanCITab
from .fit_tscan_worker import FitTScanWorker


class FitTScanWindow(QMainWindow):
    """Complete basic time-scan transient fitting workflow."""

    WINDOW_TITLE = "TRXASprefitpack - Time-Scan Fitting"

    def __init__(
        self,
        parent=None,
        *,
        job_runner: Callable = run_fit_transient_exp_config,
    ) -> None:
        super().__init__(parent)

        self.job_runner = job_runner
        self._fit_thread = None
        self._fit_worker = None

        self.setObjectName("fit_tscan_window")
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1100, 800)

        self._create_actions()
        self._create_menu_bar()
        self._create_tabs()
        self.statusBar().showMessage("Ready")

    def _create_actions(self) -> None:
        self.exit_action = QAction("Exit", self)
        self.exit_action.setObjectName("exit_action")
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)

        self.about_action = QAction("About", self)
        self.about_action.setObjectName("about_action")
        self.about_action.triggered.connect(
            self.show_about_dialog
        )

    def _create_menu_bar(self) -> None:
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.exit_action)

        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction(self.about_action)

    def _create_tabs(self) -> None:
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setObjectName(
            "fit_tscan_tab_widget"
        )

        self.data_tab = FitTScanDataTab(self.tab_widget)
        self.parameter_tab = FitTScanParameterTabs(
            self.tab_widget
        )
        self.result_tab = FitTScanResultTab(
            self.tab_widget
        )

        self.tab_widget.addTab(self.data_tab, "Data")
        self.tab_widget.addTab(
            self.parameter_tab,
            "Model and Parameters",
        )
        self.tab_widget.addTab(self.result_tab, "Results")

        self.setCentralWidget(self.tab_widget)

        self.data_tab.datasets_changed.connect(
            self._synchronize_parameter_datasets
        )
        self.parameter_tab.run_button.clicked.connect(
            self.run_fit
        )

        self.ci_tab = FitTScanCITab(parent=self)

        self.tab_widget.addTab(
            self.ci_tab,
            "CI scan",
        )

    def run_fit(self) -> None:
        if self._fit_thread is not None:
            return

        try:
            datasets = self.data_tab.datasets()
            self.parameter_tab.set_datasets(datasets)
            config = self.parameter_tab.build_config(
                datasets
            )
        except Exception as exc:
            self.parameter_tab.validation_label.setText(
                str(exc)
            )
            self.statusBar().showMessage(
                "Configuration error"
            )
            return

        self.result_tab.clear_result()
        self.parameter_tab.set_running(True)
        self.statusBar().showMessage("Fitting...")

        thread = QThread(self)
        worker = FitTScanWorker(
            config,
            datasets,
            job_runner=self.job_runner,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.result_ready.connect(
            self._handle_fit_result
        )
        worker.error.connect(self._handle_fit_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._handle_thread_finished)

        self._fit_thread = thread
        self._fit_worker = worker

        thread.start()

    def _handle_fit_result(self, result) -> None:
        self.result_tab.set_result(result)
        self.ci_tab.set_result(result)
        self.tab_widget.setCurrentWidget(
            self.result_tab
        )
        self.statusBar().showMessage("Fit completed")

    def _handle_fit_error(self, error: Exception) -> None:
        self.statusBar().showMessage("Fit failed")
        QMessageBox.critical(
            self,
            "Fit failed",
            str(error),
        )

    def _handle_thread_finished(self) -> None:
        self.parameter_tab.set_running(False)
        self._fit_thread = None
        self._fit_worker = None

    def _synchronize_parameter_datasets(self) -> None:
        try:
            datasets = self.data_tab.datasets()
        except ValueError:
            datasets = []

        self.parameter_tab.set_datasets(datasets)

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About fit_tscan_qt",
            "TRXASprefitpack time-scan transient fitting GUI.",
        )

    def closeEvent(self, event) -> None:
        if (
            self._fit_thread is not None
            and self._fit_thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Fit running",
                "Wait for the current fit to finish before closing.",
            )
            event.ignore()
            return

        event.accept()