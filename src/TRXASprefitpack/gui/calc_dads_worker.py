"""Background worker for calc_dads_qt calculations."""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .ads_config import ADSConfig, ADSResult
from .ads_job import run_ads_config
from .models import EScanDataset


class CalcDADSWorker(QObject):
    """Execute an ADS job outside the GUI thread."""

    result_ready = pyqtSignal(object)
    error = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        config: ADSConfig,
        dataset: EScanDataset,
        *,
        job_runner: Callable = run_ads_config,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.dataset = dataset
        self.job_runner = job_runner

    @pyqtSlot()
    def run(self) -> None:
        try:
            result: ADSResult = self.job_runner(
                self.config,
                self.dataset,
            )
        except Exception as exc:
            self.error.emit(exc)
        else:
            self.result_ready.emit(result)
        finally:
            self.finished.emit()
