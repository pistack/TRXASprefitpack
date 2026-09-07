"""DADS/SADS calculation controls for calc_dads_qt."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .ads_config import ADSConfig
from .calc_dads_custom_sads import CalcDADSCustomSADSPanel

from .parsers import (
    parse_float_array,
    parse_fwhm_eta,
    parse_nonnegative_float,
    parse_positive_float_array,
    parse_float,
)


class CalcDADSCalculationTab(QWidget):
    """Build a validated ADSConfig for standard DADS/SADS modes."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._create_ui()
        self._update_mode_state()
        self._update_irf_state()

    def _create_ui(self) -> None:
        root = QVBoxLayout(self)

        model_group = QGroupBox("Calculation", self)
        form = QFormLayout(model_group)

        self.mode_combo = QComboBox(model_group)
        self.mode_combo.addItem("DADS", "dads")
        self.mode_combo.addItem("DADS (SVD)", "dads_svd")
        self.mode_combo.addItem("SADS", "sads")
        self.mode_combo.addItem("SADS (SVD)", "sads_svd")
        self.mode_combo.addItem("Custom SADS", "custom_sads")
        self.mode_combo.addItem("Custom SADS (SVD)", "custom_sads_svd")
        self.mode_combo.currentIndexChanged.connect(
            self._update_mode_state
        )
        form.addRow("Mode", self.mode_combo)

        self.irf_combo = QComboBox(model_group)
        self.irf_combo.addItem("Gaussian", "g")
        self.irf_combo.addItem("Cauchy", "c")
        self.irf_combo.addItem("Pseudo-Voigt", "pv")
        self.irf_combo.currentIndexChanged.connect(
            self._update_irf_state
        )
        form.addRow("IRF", self.irf_combo)

        self.fwhm_g_edit = QLineEdit("0.1", model_group)
        self.fwhm_l_edit = QLineEdit("0.1", model_group)
        form.addRow("FWHM G", self.fwhm_g_edit)
        form.addRow("FWHM L", self.fwhm_l_edit)

        self.t0_edit = QLineEdit("0.0", model_group)
        self.tau_edit = QLineEdit("1.0, 10.0", model_group)
        form.addRow("t0", self.t0_edit)
        form.addRow("Tau", self.tau_edit)

        self.base_checkbox = QCheckBox("Include baseline", model_group)
        self.base_checkbox.setChecked(True)
        form.addRow(self.base_checkbox)

        self.cond_num_edit = QLineEdit("0.0", model_group)
        form.addRow("Relative SVD cutoff", self.cond_num_edit)

        self.y0_edit = QLineEdit("1.0, 0.0, 0.0", model_group)
        form.addRow("Initial populations", self.y0_edit)

        self.exclude_edit = QLineEdit("", model_group)
        self.exclude_edit.setPlaceholderText("0-based indices, e.g. 1, 2")
        form.addRow("Exclude species", self.exclude_edit)

        root.addWidget(model_group)

        self.custom_sads_panel = CalcDADSCustomSADSPanel(self)
        root.addWidget(self.custom_sads_panel)

        run_row = QHBoxLayout()
        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        run_row.addWidget(self.validation_label, 1)

        self.run_button = QPushButton("Run Calculation", self)
        run_row.addWidget(self.run_button)
        root.addLayout(run_row)
        root.addStretch()

    def build_config(self) -> ADSConfig:
        mode = str(self.mode_combo.currentData())
        irf = str(self.irf_combo.currentData())
        fwhm, eta = parse_fwhm_eta(
            irf,
            self.fwhm_g_edit.text(),
            self.fwhm_l_edit.text(),
        )
        t0 = parse_float(self.t0_edit.text(), "t0")

        is_custom = mode in {"custom_sads", "custom_sads_svd"}
        if is_custom:
            tau = None
            rate_model = self.custom_sads_panel.build_rate_model()
        else:
            tau = parse_positive_float_array(
                self.tau_edit.text(),
                "tau",
                )
            assert tau is not None
            rate_model = None

        use_svd = mode.endswith("_svd")
        cond_num = (
            parse_nonnegative_float(
                self.cond_num_edit.text(),
                "SVD cutoff",
            )
            if use_svd
            else 0.0
        )

        if mode in {"sads", "sads_svd"}:
            y0 = parse_float_array(
                self.y0_edit.text(),
                "initial populations",
            )
            exclude = self._parse_exclude()
        elif is_custom:
            y0 = None
            exclude = self._parse_exclude()
        else:
            y0 = None
            exclude = None

        config = ADSConfig(
            mode=mode,
            irf=irf,
            fwhm=float(fwhm),
            eta=eta,
            t0=t0,
            tau=tau,
            base=self.base_checkbox.isChecked(),
            cond_num=cond_num,
            rate_model=rate_model,
            y0=y0,
            exclude=exclude,
        )
        self.validation_label.setText("")
        return config

    def set_cond_num(self, value: float) -> None:
        self.cond_num_edit.setText(f"{float(value):.8g}")

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.custom_sads_panel.setEnabled(not running)

    def _parse_exclude(self) -> tuple[int, ...] | None:
        text = self.exclude_edit.text().strip()
        if not text:
            return None
        try:
            values = tuple(
                int(field.strip())
                for field in text.split(",")
            )
        except ValueError as exc:
            raise ValueError(
                "exclude must contain comma-separated integer indices."
            ) from exc
        return values

    def _update_mode_state(self) -> None:
        mode = self.mode_combo.currentData()
        is_standard_sads = mode in {"sads", "sads_svd"}
        is_custom = mode in {"custom_sads", "custom_sads_svd"}
        use_svd = mode in {"dads_svd", "sads_svd", "custom_sads_svd"}
        self.tau_edit.setEnabled(not is_custom)
        self.y0_edit.setEnabled(is_standard_sads)
        self.exclude_edit.setEnabled(is_standard_sads or is_custom)
        self.cond_num_edit.setEnabled(use_svd)
        self.custom_sads_panel.setVisible(is_custom)

    def _update_irf_state(self) -> None:
        irf = self.irf_combo.currentData()
        self.fwhm_g_edit.setEnabled(irf in {"g", "pv"})
        self.fwhm_l_edit.setEnabled(irf in {"c", "pv"})
