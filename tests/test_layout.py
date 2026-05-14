import numpy as np
import pytest
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.driver._layout import TransientParamLayout


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