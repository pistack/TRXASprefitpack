"""Time-scan dataset loading tab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .data_loader import read_tscan_trace
from .models import TScanDataset, TScanTrace


@dataclass
class _DatasetDraft:
    name: str
    traces: list[TScanTrace] = field(default_factory=list)


class FitTScanDataTab(QWidget):
    """Load and organize one or more time-scan datasets."""

    datasets_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._drafts: list[_DatasetDraft] = [
            _DatasetDraft("dataset_1")
        ]

        self._create_ui()
        self._refresh_dataset_selector()
        self._refresh_trace_table()

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)

        dataset_row = QHBoxLayout()

        dataset_row.addWidget(QLabel("Dataset:", self))

        self.dataset_selector = QComboBox(self)
        self.dataset_selector.currentIndexChanged.connect(
            self._on_dataset_selected
        )
        dataset_row.addWidget(self.dataset_selector)

        self.add_dataset_button = QPushButton(
            "Add Dataset",
            self,
        )
        self.add_dataset_button.clicked.connect(
            self.add_dataset
        )
        dataset_row.addWidget(self.add_dataset_button)

        self.remove_dataset_button = QPushButton(
            "Remove Dataset",
            self,
        )
        self.remove_dataset_button.clicked.connect(
            self.remove_current_dataset
        )
        dataset_row.addWidget(self.remove_dataset_button)

        root.addLayout(dataset_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Dataset name:", self))

        self.dataset_name_edit = QLineEdit(self)
        self.dataset_name_edit.editingFinished.connect(
            self._update_current_name
        )
        name_row.addWidget(self.dataset_name_edit)

        root.addLayout(name_row)

        file_row = QHBoxLayout()

        self.add_trace_button = QPushButton(
            "Add Trace Files",
            self,
        )
        self.add_trace_button.clicked.connect(
            self.select_trace_files
        )
        file_row.addWidget(self.add_trace_button)

        self.remove_trace_button = QPushButton(
            "Remove Selected Trace",
            self,
        )
        self.remove_trace_button.clicked.connect(
            self.remove_selected_traces
        )
        file_row.addWidget(self.remove_trace_button)

        self.clear_button = QPushButton(
            "Clear Dataset",
            self,
        )
        self.clear_button.clicked.connect(
            self.clear_current_dataset
        )
        file_row.addWidget(self.clear_button)

        file_row.addStretch()
        root.addLayout(file_row)

        self.trace_table = QTableWidget(0, 5, self)
        self.trace_table.setHorizontalHeaderLabels(
            (
                "Trace",
                "File",
                "Points",
                "Time min",
                "Time max",
            )
        )
        self.trace_table.setSelectionBehavior(
            QTableWidget.SelectRows
        )
        self.trace_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )
        root.addWidget(self.trace_table)

        self.preview_label = QLabel(self)
        root.addWidget(self.preview_label)

        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        root.addWidget(self.validation_label)

    @property
    def current_index(self) -> int:
        return self.dataset_selector.currentIndex()

    def add_dataset(self, name: str | None = None) -> None:
        if isinstance(name, bool):
            name = None

        if name is None:
            name = f"dataset_{len(self._drafts) + 1}"

        self._drafts.append(_DatasetDraft(str(name)))
        self._refresh_dataset_selector(
            selected=len(self._drafts) - 1
        )
        self._emit_changed()

    def remove_current_dataset(self) -> None:
        index = self.current_index

        if index < 0:
            return

        if len(self._drafts) == 1:
            self._drafts[0] = _DatasetDraft("dataset_1")
            self._refresh_dataset_selector(selected=0)
        else:
            del self._drafts[index]
            self._refresh_dataset_selector(
                selected=min(index, len(self._drafts) - 1)
            )

        self._refresh_trace_table()
        self._emit_changed()

    def select_trace_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select time-scan trace files",
            "",
            "Text data (*.txt *.dat *.csv);;All files (*)",
        )

        if not paths:
            return

        try:
            self.load_trace_files(paths)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Could not load traces",
                str(exc),
            )

    def load_trace_files(
        self,
        paths: Sequence[str | Path],
    ) -> None:
        draft = self._current_draft()

        loaded = [
            read_tscan_trace(path)
            for path in paths
        ]
        combined = draft.traces + loaded

        # Validate the shared time-axis convention before mutation.
        TScanDataset(
            name=draft.name,
            traces=tuple(combined),
        )

        draft.traces.extend(loaded)
        self.validation_label.setText("")
        self._refresh_trace_table()
        self._emit_changed()

    def remove_selected_traces(self) -> None:
        rows = sorted(
            {
                item.row()
                for item in self.trace_table.selectedItems()
            },
            reverse=True,
        )

        draft = self._current_draft()

        for row in rows:
            del draft.traces[row]

        self._refresh_trace_table()
        self._emit_changed()

    def clear_current_dataset(self) -> None:
        self._current_draft().traces.clear()
        self._refresh_trace_table()
        self._emit_changed()

    def datasets(self) -> list[TScanDataset]:
        datasets: list[TScanDataset] = []

        for index, draft in enumerate(self._drafts):
            name = draft.name.strip()

            if not name:
                raise ValueError(
                    f"Dataset {index + 1} name must not be empty."
                )

            if not draft.traces:
                raise ValueError(
                    f"Dataset {name!r} contains no traces."
                )

            datasets.append(
                TScanDataset(
                    name=name,
                    traces=tuple(draft.traces),
                )
            )

        return datasets

    def _current_draft(self) -> _DatasetDraft:
        index = self.current_index

        if index < 0:
            raise ValueError("No dataset is selected.")

        return self._drafts[index]

    def _update_current_name(self) -> None:
        if self.current_index < 0:
            return

        self._current_draft().name = (
            self.dataset_name_edit.text().strip()
        )
        self._refresh_dataset_selector(
            selected=self.current_index
        )
        self._emit_changed()

    def _on_dataset_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._drafts):
            return

        self.dataset_name_edit.setText(
            self._drafts[index].name
        )
        self._refresh_trace_table()

    def _refresh_dataset_selector(
        self,
        *,
        selected: int | None = None,
    ) -> None:
        if selected is None:
            selected = max(self.current_index, 0)

        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()

        for draft in self._drafts:
            self.dataset_selector.addItem(draft.name)

        selected = min(selected, len(self._drafts) - 1)
        self.dataset_selector.setCurrentIndex(selected)
        self.dataset_selector.blockSignals(False)

        self._on_dataset_selected(selected)

    def _refresh_trace_table(self) -> None:
        if self.current_index < 0:
            self.trace_table.setRowCount(0)
            return

        draft = self._current_draft()
        self.trace_table.setRowCount(len(draft.traces))

        for row, trace in enumerate(draft.traces):
            values = (
                trace.name,
                str(trace.path),
                str(trace.n_time),
                f"{trace.t.min():.8g}",
                f"{trace.t.max():.8g}",
            )

            for column, value in enumerate(values):
                self.trace_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )

        if draft.traces:
            n_time = draft.traces[0].n_time
            self.preview_label.setText(
                f"{len(draft.traces)} trace(s), "
                f"{n_time} time points"
            )
        else:
            self.preview_label.setText("No traces loaded")

    def _emit_changed(self) -> None:
        self.datasets_changed.emit()