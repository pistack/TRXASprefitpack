import pytest
import os
import sys
import numpy as np

pytest.importorskip("PyQt5")

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

from TRXASprefitpack.gui.fit_tscan_ci_worker import (
    FitTScanCIWorker,
)
from TRXASprefitpack import TransientResult

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + "/../src/")

@pytest.fixture
def fitted_result():
    time = np.array(
        [-1.0, -0.2, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=float,
    )

    fit = np.array(
        [
            [0.00, 0.00],
            [0.01, -0.01],
            [1.00, 0.80],
            [0.98, 0.78],
            [0.90, 0.70],
            [0.40, 0.25],
            [0.05, 0.02],
        ],
        dtype=float,
    )

    raw_residual = np.array(
        [
            [0.002, -0.004],
            [-0.003, 0.002],
            [0.005, -0.006],
            [-0.004, 0.003],
            [0.003, -0.002],
            [-0.002, 0.004],
            [0.001, -0.003],
        ],
        dtype=float,
    )

    eps = np.array(
        [
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
            [0.010, 0.020],
        ],
        dtype=float,
    )

    intensity = fit + raw_residual

    weighted_residual = raw_residual / eps

    result = TransientResult()

    result.update(
        {
            "model": "decay",
            "irf": "g",
            "same_t0": False,
            "name_of_dset": np.array(["fixture_dataset"]),
            "t": [time],
            "intensity": [intensity],
            "eps": [eps],
            "fit": [fit],
            "res": [weighted_residual],
            "fwhm": 0.12,
            "eta": 0.0,
            "base": True,
            "n_decay": 2,
            "n_osc": 0,
            "tau_mask": [
                np.array(
                    [True, True, True],
                    dtype=bool,
                )
            ],
            "c": [
                np.array(
                    [
                        [1.00, 0.80],
                        [0.40, 0.25],
                        [0.01, -0.01],
                    ],
                    dtype=float,
                )
            ],
            "param_name": np.array(
                [
                    "fwhm",
                    "t_0_1_1",
                    "t_0_1_2",
                    "tau_1",
                    "tau_2",
                ]
            ),
            "x": np.array(
                [0.12, 0.01, -0.02, 1.0, 10.0],
                dtype=float,
            ),
            "x_eps": np.array(
                [0.005, 0.002, 0.002, 0.05, 0.50],
                dtype=float,
            ),
            "bounds": np.array(
                [
                    [0.05, 0.30],
                    [-0.50, 0.50],
                    [-0.50, 0.50],
                    [0.10, 5.00],
                    [5.00, 50.00],
                ],
                dtype=float,
            ),
            "chi2": 1.25,
            "red_chi2": 0.14,
            "chi2_ind": [np.array([0.55, 0.70])],
            "red_chi2_ind": [np.array([0.12, 0.16])],
            "aic": 8.50,
            "bic": 9.25,
            "nfev": 12,
            "n_param": 5,
            "n_param_ind": 4,
            "num_pts": int(intensity.size),
            "jac": np.zeros((intensity.size, 5)),
            "cov": np.eye(5),
            "cov_scaled": np.eye(5) * 0.14,
            "corr": np.eye(5),
            "method_glb": None,
            "message_glb": None,
            "method_lsq": "trf",
            "success_lsq": True,
            "message_lsq": "Fixture fit converged.",
            "status": 0,
        }
    )

    return result


def test_ci_worker_emits_finished(fitted_result):
    expected = {
        1.0: {"ci": []},
        2.0: {"ci": []},
    }

    def fake_runner(result, parameter_indices):
        assert result is fitted_result
        assert parameter_indices == (3, 4)
        return expected

    worker = FitTScanCIWorker(
        fitted_result,
        [3, 4],
        ci_runner=fake_runner,
    )

    emitted = []
    worker.finished.connect(emitted.append)

    worker.run()

    assert emitted == [expected]


def test_ci_worker_emits_failed(fitted_result):
    expected_error = RuntimeError("CI failed")

    def fake_runner(result, parameter_indices):
        raise expected_error

    worker = FitTScanCIWorker(
        fitted_result,
        [3],
        ci_runner=fake_runner,
    )

    emitted = []
    worker.failed.connect(emitted.append)

    worker.run()

    assert emitted == [expected_error]