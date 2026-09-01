"""
Application entry point for fit_tscan_qt.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PyQt5.QtWidgets import QApplication

from .fit_tscan_window import FitTScanWindow


def create_fit_tscan_application(
    argv: Sequence[str] | None = None,
) -> tuple[QApplication, FitTScanWindow]:
    """Create or reuse QApplication and construct the main window."""
    application = QApplication.instance()

    if application is None:
        arguments = _application_arguments(
            argv,
            program_name="fit_tscan_qt",
        )
        application = QApplication(arguments)

    window = FitTScanWindow()
    return application, window


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the fit_tscan_qt application."""
    application, window = (
        create_fit_tscan_application(argv)
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