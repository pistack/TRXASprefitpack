"""
PyQt5 main-window skeleton for calc_dads_qt.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class CalcDADSWindow(QMainWindow):
    """Main window for the DADS/SADS calculation GUI."""

    WINDOW_TITLE = "TRXASprefitpack - DADS/SADS Calculation"

    TAB_NAMES = (
        "Data",
        "SVD",
        "Calculation",
        "Results",
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

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

        descriptions = (
            "Load and preview an energy-scan matrix dataset.",
            "Inspect singular values and singular vectors.",
            "Configure and run a DADS or SADS calculation.",
            "Display associated spectra and reconstructed data.",
        )

        for tab_name, description in zip(
            self.TAB_NAMES,
            descriptions,
        ):
            self.tab_widget.addTab(
                _make_placeholder_widget(
                    description,
                    parent=self.tab_widget,
                ),
                tab_name,
            )

        self.setCentralWidget(self.tab_widget)

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


def _make_placeholder_widget(
    text: str,
    *,
    parent=None,
) -> QWidget:
    widget = QWidget(parent)
    layout = QVBoxLayout(widget)

    label = QLabel(text, widget)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignCenter)

    layout.addWidget(label)
    return widget