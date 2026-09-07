"""Complete PyQt5 window for calc_dads_qt."""

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

from .ads_job import run_ads_config
from .calc_dads_calculation_tab import CalcDADSCalculationTab
from .calc_dads_data_tab import CalcDADSDataTab
from .calc_dads_result_tab import CalcDADSResultTab
from .calc_dads_svd_tab import CalcDADSSVDTab
from .calc_dads_worker import CalcDADSWorker


class CalcDADSWindow(QMainWindow):
    """Main window for the DADS/SADS calculation GUI."""

    WINDOW_TITLE = "TRXASprefitpack - DADS/SADS Calculation"

    def __init__(
        self,
        parent=None,
        *,
        job_runner: Callable = run_ads_config,
    ) -> None:
        super().__init__(parent)

        self.job_runner = job_runner
        self._calculation_thread = None
        self._calculation_worker = None

        self.setObjectName("calc_dads_window")
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1000, 700)

        self._create_actions()
        self._create_menu_bar()
        self._create_central_tabs()
        self._create_status_bar()

    def _create_actions(self) -> None:
        self.exit_action = QAction("Exit", self)
        self.exit_action.setObjectName("exit_action")
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.setStatusTip("Close the application")
        self.exit_action.triggered.connect(self.close)

        self.about_action = QAction("About", self)
        self.about_action.setObjectName("about_action")
        self.about_action.setStatusTip(
            "Show information about calc_dads_qt"
        )
        self.about_action.triggered.connect(
            self.show_about_dialog
        )

    def _create_menu_bar(self) -> None:
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.setObjectName("file_menu")
        self.file_menu.addAction(self.exit_action)

        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.setObjectName("help_menu")
        self.help_menu.addAction(self.about_action)

    def _create_central_tabs(self) -> None:
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setObjectName(
            "calc_dads_tab_widget"
        )
        self.tab_widget.setDocumentMode(True)

        self.data_tab = CalcDADSDataTab(self.tab_widget)
        self.svd_tab = CalcDADSSVDTab(self.tab_widget)
        self.calculation_tab = CalcDADSCalculationTab(self.tab_widget)
        self.result_tab = CalcDADSResultTab(self.tab_widget)

        self.tab_widget.addTab(self.data_tab, "Data")
        self.tab_widget.addTab(self.svd_tab, "SVD")
        self.tab_widget.addTab(self.calculation_tab, "Calculation")
        self.tab_widget.addTab(self.result_tab, "Results")

        self.setCentralWidget(self.tab_widget)

        self.data_tab.dataset_changed.connect(
            self._handle_dataset_changed
        )
        self.svd_tab.cutoff_changed.connect(
            self.calculation_tab.set_cond_num
        )
        self.calculation_tab.run_button.clicked.connect(
            self.run_calculation
        )

    def _create_status_bar(self) -> None:
        self.statusBar().setObjectName(
            "calc_dads_status_bar"
        )
        self.statusBar().showMessage("Ready")

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About calc_dads_qt",
            "TRXASprefitpack DADS/SADS calculation GUI.",
        )

    def run_calculation(self) -> None:
        if self._calculation_thread is not None:
            return

        try:
            dataset = self.data_tab.dataset()
            config = self.calculation_tab.build_config()
        except Exception as exc:
            self.calculation_tab.validation_label.setText(str(exc))
            self.statusBar().showMessage("Configuration error")
            return

        self.result_tab.clear_result()
        self.calculation_tab.set_running(True)
        self.statusBar().showMessage("Calculating...")

        thread = QThread(self)
        worker = CalcDADSWorker(
            config,
            dataset,
            job_runner=self.job_runner,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.result_ready.connect(self._handle_result)
        worker.error.connect(self._handle_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._handle_thread_finished)

        self._calculation_thread = thread
        self._calculation_worker = worker
        thread.start()

    def _handle_dataset_changed(self, dataset) -> None:
        self.svd_tab.set_dataset(dataset)
        self.result_tab.clear_result()

    def _handle_result(self, result) -> None:
        self.result_tab.set_result(result)
        self.tab_widget.setCurrentWidget(self.result_tab)
        self.statusBar().showMessage("Calculation completed")

    def _handle_error(self, error: Exception) -> None:
        self.statusBar().showMessage("Calculation failed")
        QMessageBox.critical(
            self,
            "Calculation failed",
            str(error),
        )

    def _handle_thread_finished(self) -> None:
        self.calculation_tab.set_running(False)
        self._calculation_thread = None
        self._calculation_worker = None

    def closeEvent(self, event) -> None:
        if (
            self._calculation_thread is not None
            and self._calculation_thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Calculation running",
                "Wait for the current calculation to finish before closing.",
            )
            event.ignore()
            return
        event.accept()
