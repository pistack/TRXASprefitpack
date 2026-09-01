import configparser
import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QTabWidget

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.app_calc_dads import (
    create_calc_dads_application,
)
from TRXASprefitpack.gui.app_fit_tscan import (
    create_fit_tscan_application,
)
from TRXASprefitpack.gui.calc_dads_window import (
    CalcDADSWindow,
)
from TRXASprefitpack.gui.fit_tscan_window import (
    FitTScanWindow,
)
import TRXASprefitpack.gui.app_calc_dads as app_calc_dads
import TRXASprefitpack.gui.app_fit_tscan as app_fit_tscan


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance()

    if application is None:
        application = QApplication(
            ["test_gui_qt_skeleton"]
        )

    return application


def tab_names(tab_widget):
    return tuple(
        tab_widget.tabText(index)
        for index in range(tab_widget.count())
    )


def menu_action_texts(menu):
    return tuple(
        action.text().replace("&", "")
        for action in menu.actions()
    )


def test_fit_tscan_window_skeleton(qapp):
    window = FitTScanWindow()

    assert window.objectName() == "fit_tscan_window"
    assert window.windowTitle() == (
        "TRXASprefitpack - Time-Scan Fitting"
    )

    assert isinstance(
        window.centralWidget(),
        QTabWidget,
    )
    assert window.centralWidget() is window.tab_widget

    assert tab_names(window.tab_widget) == (
        "Data",
        "Model and Parameters",
        "Results",
    )

    assert window.statusBar().currentMessage() == "Ready"
    assert menu_action_texts(window.file_menu) == ("Exit",)
    assert menu_action_texts(window.help_menu) == ("About",)

    assert window.exit_action.objectName() == "exit_action"
    assert window.about_action.objectName() == "about_action"

    window.close()


def test_calc_dads_window_skeleton(qapp):
    window = CalcDADSWindow()

    assert window.objectName() == "calc_dads_window"
    assert window.windowTitle() == (
        "TRXASprefitpack - DADS/SADS Calculation"
    )

    assert isinstance(
        window.centralWidget(),
        QTabWidget,
    )
    assert window.centralWidget() is window.tab_widget

    assert tab_names(window.tab_widget) == (
        "Data",
        "SVD",
        "Calculation",
        "Results",
    )

    assert window.statusBar().currentMessage() == "Ready"
    assert menu_action_texts(window.file_menu) == ("Exit",)
    assert menu_action_texts(window.help_menu) == ("About",)

    window.close()


def test_create_fit_tscan_application_reuses_qapp(qapp):
    application, window = (
        create_fit_tscan_application(
            ["fit_tscan_qt"]
        )
    )

    assert application is qapp
    assert isinstance(window, FitTScanWindow)

    window.close()


def test_create_calc_dads_application_reuses_qapp(qapp):
    application, window = (
        create_calc_dads_application(
            ["calc_dads_qt"]
        )
    )

    assert application is qapp
    assert isinstance(window, CalcDADSWindow)

    window.close()


@pytest.mark.parametrize(
    "module,creator_name",
    [
        (
            app_fit_tscan,
            "create_fit_tscan_application",
        ),
        (
            app_calc_dads,
            "create_calc_dads_application",
        ),
    ],
)
def test_application_main_shows_window_and_runs_event_loop(
    monkeypatch,
    module,
    creator_name,
):
    class FakeApplication:
        def __init__(self):
            self.executed = False

        def exec_(self):
            self.executed = True
            return 17

    class FakeWindow:
        def __init__(self):
            self.shown = False

        def show(self):
            self.shown = True

    application = FakeApplication()
    window = FakeWindow()
    received = {}

    def fake_creator(argv):
        received["argv"] = argv
        return application, window

    monkeypatch.setattr(
        module,
        creator_name,
        fake_creator,
    )

    exit_code = module.main(["program", "--test"])

    assert received["argv"] == ["program", "--test"]
    assert window.shown is True
    assert application.executed is True
    assert exit_code == 17


def test_setup_cfg_has_qt_optional_dependency():
    setup_path = (
        Path(__file__).resolve().parents[1]
        / "setup.cfg"
    )

    parser = configparser.ConfigParser()
    parser.read(setup_path)

    assert parser.has_section("options.extras_require")
    assert "qt" in parser["options.extras_require"]
    assert (
        "PyQt5"
        in parser["options.extras_require"]["qt"]
    )


def test_setup_cfg_has_new_qt_entry_points():
    setup_path = (
        Path(__file__).resolve().parents[1]
        / "setup.cfg"
    )

    parser = configparser.ConfigParser()
    parser.read(setup_path)

    console_scripts = parser[
        "options.entry_points"
    ]["console_scripts"]

    assert (
        "fit_tscan_qt = "
        "TRXASprefitpack.gui.app_fit_tscan:main"
        in console_scripts
    )
    assert (
        "calc_dads_qt = "
        "TRXASprefitpack.gui.app_calc_dads:main"
        in console_scripts
    )

    # Legacy Tkinter entry points must remain available.
    assert "fit_tscan_gui" in console_scripts
    assert "calc_dads_gui" in console_scripts