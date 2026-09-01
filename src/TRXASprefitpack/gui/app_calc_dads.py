"""
Application entry point for calc_dads_qt.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PyQt5.QtWidgets import QApplication

from .calc_dads_window import CalcDADSWindow


def create_calc_dads_application(
    argv: Sequence[str] | None = None,
) -> tuple[QApplication, CalcDADSWindow]:
    """Create or reuse QApplication and construct the main window."""
    application = QApplication.instance()

    if application is None:
        arguments = _application_arguments(
            argv,
            program_name="calc_dads_qt",
        )
        application = QApplication(arguments)

    window = CalcDADSWindow()
    return application, window


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the calc_dads_qt application."""
    application, window = (
        create_calc_dads_application(argv)
    )

    window.show()
    return int(application.exec_())


def _application_arguments(
    argv: Sequence[str] | None,
    *,
    program_name: str,
) -> list[str]:
    if argv is None:
        arguments = list(sys.argv)
    else:
        arguments = [str(value) for value in argv]

    if len(arguments) == 0:
        arguments = [program_name]

    return arguments


if __name__ == "__main__":
    raise SystemExit(main())