"""Qt worker for profile confidence interval scans."""

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .fit_tscan_ci import run_selected_ci_scans


class FitTScanCIWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(
        self,
        result,
        parameter_indices,
        ci_runner=run_selected_ci_scans,
        parent=None,
    ):
        super().__init__(parent)

        self._result = result
        self._parameter_indices = tuple(parameter_indices)
        self._ci_runner = ci_runner

    @pyqtSlot()
    def run(self):
        try:
            ci_results = self._ci_runner(
                self._result,
                self._parameter_indices,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.failed.emit(exc)
            return

        self.finished.emit(ci_results)