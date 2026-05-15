import numpy as np
import pytest
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.driver._layout import (
    TransientParamLayout,
    DampedOscillationParamLayout,
    BothTransientParamLayout,
)


def test_layout_standard_model():
    layout = TransientParamLayout(num_irf=1, num_t0=1, num_tau=4)

    assert layout.size == 6
    assert layout.irf_slice == slice(0, 1)
    assert layout.t0_slice == slice(1, 2)
    assert layout.tau_slice == slice(2, 6)


def test_layout_unpack():
    layout = TransientParamLayout(num_irf=2, num_t0=3, num_tau=2)
    x = np.array([0.5, 0.1, 1.0, 1.1, 1.2, 5.0, 10.0])

    irf, t0, tau = layout.unpack(x)

    np.testing.assert_array_equal(irf, np.array([0.5, 0.1]))
    np.testing.assert_array_equal(t0, np.array([1.0, 1.1, 1.2]))
    np.testing.assert_array_equal(tau, np.array([5.0, 10.0]))


def test_layout_allows_empty_blocks():
    layout = TransientParamLayout(num_irf=1, num_t0=0, num_tau=2)
    x = np.array([0.5, 5.0, 10.0])

    irf, t0, tau = layout.unpack(x)

    np.testing.assert_array_equal(irf, np.array([0.5]))
    np.testing.assert_array_equal(t0, np.array([]))
    np.testing.assert_array_equal(tau, np.array([5.0, 10.0]))


def test_layout_rejects_short_vector():
    layout = TransientParamLayout(num_irf=1, num_t0=1, num_tau=2)

    with pytest.raises(ValueError, match="expects size 4"):
        layout.unpack(np.array([0.5, 1.0, 5.0]))


def test_layout_rejects_long_vector():
    layout = TransientParamLayout(num_irf=1, num_t0=1, num_tau=2)

    with pytest.raises(ValueError, match="expects size 4"):
        layout.unpack(np.array([0.5, 1.0, 5.0, 10.0, 20.0]))


def test_layout_rejects_non_1d_vector():
    layout = TransientParamLayout(num_irf=1, num_t0=1, num_tau=2)

    with pytest.raises(ValueError, match="1D"):
        layout.unpack(np.zeros((4, 1)))


def test_layout_rejects_negative_block_size():
    with pytest.raises(ValueError, match="non-negative"):
        TransientParamLayout(num_irf=1, num_t0=-1, num_tau=2)


def test_layout_rejects_non_integer_block_size():
    with pytest.raises(TypeError, match="integer"):
        TransientParamLayout(num_irf=1, num_t0=1.5, num_tau=2)


def test_damped_oscillation_layout():
    layout = DampedOscillationParamLayout(num_irf=2, num_t0=3, num_osc=4)

    assert layout.size == 13
    assert layout.irf_slice == slice(0, 2)
    assert layout.t0_slice == slice(2, 5)
    assert layout.tau_osc_slice == slice(5, 9)
    assert layout.period_osc_slice == slice(9, 13)


def test_damped_oscillation_layout_unpack():
    layout = DampedOscillationParamLayout(num_irf=1, num_t0=2, num_osc=2)
    x = list(range(layout.size))

    irf, t0, tau_osc, period_osc = layout.unpack(x)

    assert list(irf) == [0]
    assert list(t0) == [1, 2]
    assert list(tau_osc) == [3, 4]
    assert list(period_osc) == [5, 6]


def test_damped_oscillation_layout_unpack_size_error():
    layout = DampedOscillationParamLayout(num_irf=1, num_t0=1, num_osc=2)

    with pytest.raises(ValueError, match="Expected array of size at least 6"):
        layout.unpack([0, 1, 2])


def test_both_transient_layout():
    layout = BothTransientParamLayout(
        num_irf=2,
        num_t0=3,
        num_decay=4,
        num_osc=2,
    )

    assert layout.size == 13
    assert layout.irf_slice == slice(0, 2)
    assert layout.t0_slice == slice(2, 5)
    assert layout.tau_decay_slice == slice(5, 9)
    assert layout.tau_osc_slice == slice(9, 11)
    assert layout.period_osc_slice == slice(11, 13)


def test_both_transient_layout_unpack():
    layout = BothTransientParamLayout(
        num_irf=1,
        num_t0=2,
        num_decay=2,
        num_osc=2,
    )
    x = list(range(layout.size))

    irf, t0, tau_decay, tau_osc, period_osc = layout.unpack(x)

    assert list(irf) == [0]
    assert list(t0) == [1, 2]
    assert list(tau_decay) == [3, 4]
    assert list(tau_osc) == [5, 6]
    assert list(period_osc) == [7, 8]


def test_both_transient_layout_unpack_size_error():
    layout = BothTransientParamLayout(
        num_irf=1,
        num_t0=1,
        num_decay=1,
        num_osc=1,
    )

    with pytest.raises(ValueError, match="Expected array of size at least 5"):
        layout.unpack([0, 1, 2])