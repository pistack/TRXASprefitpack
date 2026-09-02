"""Model, parameter, bound, and tau-mask controls."""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .fit_config import FitTransientExpConfig
from .models import TScanDataset
from .parsers import (
    parse_bounds,
    parse_float_array,
    parse_fwhm_eta,
    parse_irf,
    parse_positive_float,
    parse_positive_float_array,
)
from .validators import validate_t0_count_for_tscan


class FitTScanParameterTabs(QWidget):
    """Controls used to construct FitTransientExpConfig."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tau_mask_state: dict[tuple[str, str], bool] = {}
        self._create_ui()

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)

        model_group = QGroupBox("Model", self)
        model_form = QFormLayout(model_group)

        self.irf_combo = QComboBox(model_group)
        self.irf_combo.addItem("Gaussian", "g")
        self.irf_combo.addItem("Cauchy", "c")
        self.irf_combo.addItem("Pseudo-Voigt", "pv")
        self.irf_combo.currentIndexChanged.connect(
            self._update_irf_state
        )
        model_form.addRow("IRF", self.irf_combo)

        self.fwhm_g_edit = QLineEdit("0.1", model_group)
        self.fwhm_l_edit = QLineEdit("0.1", model_group)
        model_form.addRow("FWHM G", self.fwhm_g_edit)
        model_form.addRow("FWHM L", self.fwhm_l_edit)

        self.t0_edit = QLineEdit("0.0", model_group)
        model_form.addRow("t0 initial", self.t0_edit)

        self.tau_edit = QLineEdit("1.0", model_group)
        self.tau_edit.editingFinished.connect(
            self._refresh_tau_mask_from_text
        )
        model_form.addRow("Tau initial", self.tau_edit)

        self.base_checkbox = QCheckBox("Include baseline", model_group)
        self.base_checkbox.setChecked(True)
        self.base_checkbox.toggled.connect(self._refresh_tau_mask_from_text)
        model_form.addRow(self.base_checkbox)

        self.same_t0_checkbox = QCheckBox(
            "Use one t0 per dataset",
            model_group,
        )
        model_form.addRow(self.same_t0_checkbox)

        self.global_method_combo = QComboBox(model_group)
        self.global_method_combo.addItem("None", None)
        self.global_method_combo.addItem("AMPGO", "ampgo")
        self.global_method_combo.addItem(
            "Basin hopping",
            "basinhopping",
        )
        model_form.addRow(
            "Global optimizer",
            self.global_method_combo,
        )

        self.lsq_method_combo = QComboBox(model_group)
        self.lsq_method_combo.addItems(("trf", "dogbox", "lm"))
        model_form.addRow(
            "Least-squares method",
            self.lsq_method_combo,
        )

        root.addWidget(model_group)

        bounds_group = QGroupBox("Bounds", self)
        bounds_form = QFormLayout(bounds_group)

        self.fwhm_lower_edit = QLineEdit("1e-6", bounds_group)
        self.fwhm_upper_edit = QLineEdit("1e3", bounds_group)
        bounds_form.addRow(
            "FWHM lower",
            self.fwhm_lower_edit,
        )
        bounds_form.addRow(
            "FWHM upper",
            self.fwhm_upper_edit,
        )

        self.t0_lower_edit = QLineEdit("-1e3", bounds_group)
        self.t0_upper_edit = QLineEdit("1e3", bounds_group)
        bounds_form.addRow("t0 lower", self.t0_lower_edit)
        bounds_form.addRow("t0 upper", self.t0_upper_edit)

        self.tau_lower_edit = QLineEdit("1e-6", bounds_group)
        self.tau_upper_edit = QLineEdit("1e9", bounds_group)
        bounds_form.addRow("Tau lower", self.tau_lower_edit)
        bounds_form.addRow("Tau upper", self.tau_upper_edit)

        root.addWidget(bounds_group)

        mask_group = QGroupBox("Tau mask by dataset", self)
        mask_layout = QVBoxLayout(mask_group)

        self.tau_mask_table = QTableWidget(0, 0, mask_group)
        mask_layout.addWidget(self.tau_mask_table)

        root.addWidget(mask_group)

        run_row = QHBoxLayout()

        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        run_row.addWidget(self.validation_label, 1)

        self.run_button = QPushButton("Run Fit", self)
        run_row.addWidget(self.run_button)

        root.addLayout(run_row)
        root.addStretch()

        self._datasets: list[TScanDataset] = []
        self._update_irf_state()

    def set_datasets(
        self,
        datasets: list[TScanDataset],
    ) -> None:
        self._datasets = list(datasets)
        self._refresh_tau_mask_from_text()

    def build_config(
        self,
        datasets: list[TScanDataset],
    ) -> FitTransientExpConfig:
        irf = parse_irf(
            self.irf_combo.currentData()
        )

        # Validate the IRF input using the shared parser.
        parse_fwhm_eta(
            irf,
            self.fwhm_g_edit.text(),
            self.fwhm_l_edit.text(),
        )

        if irf == "g":
            fwhm_init = np.array(
                [
                    parse_positive_float(
                        self.fwhm_g_edit.text(),
                        "fwhm_G",
                    )
                ]
            )
        elif irf == "c":
            fwhm_init = np.array(
                [
                    parse_positive_float(
                        self.fwhm_l_edit.text(),
                        "fwhm_L",
                    )
                ]
            )
        else:
            fwhm_init = np.array(
                [
                    parse_positive_float(
                        self.fwhm_g_edit.text(),
                        "fwhm_G",
                    ),
                    parse_positive_float(
                        self.fwhm_l_edit.text(),
                        "fwhm_L",
                    ),
                ]
            )

        t0_init = parse_float_array(
            self.t0_edit.text(),
            "t0",
        )
        tau_init = parse_positive_float_array(
            self.tau_edit.text(),
            "tau",
        )

        assert t0_init is not None
        assert tau_init is not None

        validate_t0_count_for_tscan(
            datasets,
            t0_init,
            same_t0=self.same_t0_checkbox.isChecked(),
        )

        bound_fwhm = parse_bounds(
            self.fwhm_lower_edit.text(),
            self.fwhm_upper_edit.text(),
            fwhm_init,
            "fwhm",
        )
        bound_t0 = parse_bounds(
            self.t0_lower_edit.text(),
            self.t0_upper_edit.text(),
            t0_init,
            "t0",
        )
        bound_tau = parse_bounds(
            self.tau_lower_edit.text(),
            self.tau_upper_edit.text(),
            tau_init,
            "tau",
        )

        self._synchronize_tau_mask(
            datasets,
            tau_init.size,
        )

        tau_mask: list[np.ndarray] = []

        for row in range(self.tau_mask_table.rowCount()):
            tau_mask.append(
                np.array(
                    [
                        self.tau_mask_table.item(
                            row,
                            column,
                        ).checkState()
                        == Qt.Checked
                        for column in range(
                            self.tau_mask_table.columnCount()
                        )
                    ],
                    dtype=bool,
                )
            )

        config = FitTransientExpConfig(
            irf=irf,
            fwhm_init=fwhm_init,
            t0_init=t0_init,
            tau_init=tau_init,
            base=self.base_checkbox.isChecked(),
            method_glb=self.global_method_combo.currentData(),
            method_lsq=self.lsq_method_combo.currentText(),
            bound_fwhm=bound_fwhm,
            bound_t0=bound_t0,
            bound_tau=bound_tau,
            same_t0=self.same_t0_checkbox.isChecked(),
            tau_mask=tau_mask,
        )

        self.validation_label.setText("")
        return config

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)

    def _update_irf_state(self) -> None:
        irf = self.irf_combo.currentData()

        self.fwhm_g_edit.setEnabled(
            irf in {"g", "pv"}
        )
        self.fwhm_l_edit.setEnabled(
            irf in {"c", "pv"}
        )

    def _refresh_tau_mask_from_text(self) -> None:
        try:
            tau = parse_positive_float_array(
                self.tau_edit.text(),
                "tau",
            )
        except ValueError:
            return

        if tau is not None:
            self._synchronize_tau_mask(
                self._datasets,
                tau.size,
            )

    def _synchronize_tau_mask(
        self,
        datasets: list[TScanDataset],
        n_tau: int,
    ) -> None:
        names = [dataset.name for dataset in datasets]

        labels = [f"tau_{index+1}" for index in range(n_tau)]

        if self.base_checkbox.isChecked():
            labels.append("baseline")

        for row in range(self.tau_mask_table.rowCount()):
            row_header = self.tau_mask_table.verticalHeaderItem(row)

            if row_header is None:
                continue

            for column in range(self.tau_mask_table.columnCount()):
                column_header = self.tau_mask_table.horizontalHeaderItem(column)
                item = self.tau_mask_table.item(row, column)

                if column_header is not None and item is not None:
                    self._tau_mask_state[
                        (row_header.text(), column_header.text())] = item.checkState() == Qt.Checked

        current_names = [
            self.tau_mask_table.verticalHeaderItem(row).text()
            for row in range(self.tau_mask_table.rowCount())
            if self.tau_mask_table.verticalHeaderItem(row)
        ]

        current_labels = [
            self.tau_mask_table.horizontalHeaderItem(column).text()
            for column in range(self.tau_mask_table.columnCount())
            if self.tau_mask_table.horizontalHeaderItem(column)
        ]

        if (
            self.tau_mask_table.rowCount() == len(datasets)
            and self.tau_mask_table.columnCount() == len(labels)
            and current_names == names
            and current_labels == labels
        ):
            return

        self.tau_mask_table.clear()
        self.tau_mask_table.setRowCount(len(datasets))
        self.tau_mask_table.setColumnCount(len(labels))
        self.tau_mask_table.setHorizontalHeaderLabels(labels)
        self.tau_mask_table.setVerticalHeaderLabels(names)

        for row, name in enumerate(names):
            for column, label in enumerate(labels):
                item = QTableWidgetItem()
                item.setFlags(
                    Qt.ItemIsEnabled
                    | Qt.ItemIsUserCheckable
                )
                item.setCheckState(
                    Qt.Checked
                    if self._tau_mask_state.get((name, label), True)
                    else Qt.Unchecked
                )
                self.tau_mask_table.setItem(
                    row,
                    column,
                    item,
                )
