"""Background worker for transient fitting."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ..driver.transient_result import TransientResult
from .fit_config import FitTransientExpConfig
from .fit_job import run_fit_transient_exp_config
from .models import TScanDataset


class FitTScanWorker(QObject):
    """Execute a fit job outside the GUI thread."""

    result_ready = pyqtSignal(object)
    error = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        config: FitTransientExpConfig,
        datasets: Sequence[TScanDataset],
        *,
        job_runner: Callable = run_fit_transient_exp_config,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.config = config
        self.datasets = tuple(datasets)
        self.job_runner = job_runner

    @pyqtSlot()
    def run(self) -> None:
        try:
            result: TransientResult = self.job_runner(
                self.config,
                self.datasets,
            )
        except Exception as exc:
            self.error.emit(exc)
        else:
            self.result_ready.emit(result)
        finally:
            self.finished.emit()