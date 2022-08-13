# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.mathfun.deriv_check import check_num_deriv
from TRXASprefitpack import edge_gaussian, edge_lorenzian
from TRXASprefitpack import deriv_edge_gaussian, deriv_edge_lorenzian
from TRXASprefitpack import voigt, voigt_thy
from TRXASprefitpack import deriv_voigt, deriv_voigt_thy


def test_deriv_voigt_1():
    e = np.linspace(-5, 5, 1001)
    fwhm_L = 1
    tst = deriv_voigt(e, 0, fwhm_L)
    ref = check_num_deriv(voigt, e, 0, fwhm_L)
    result = np.allclose(tst[:, [0, 2]], ref[:, [0, 2]])
    assert result is True

def test_deriv_voigt_2():
    e = np.linspace(-5, 5, 1001)
    fwhm_G = 1
    tst = deriv_voigt(e, fwhm_G, 0)
    ref = check_num_deriv(voigt, e, fwhm_G, 0)
    result = np.allclose(tst[:, [0, 1]], ref[:, [0, 1]])
    assert result is True

def test_deriv_voigt_3():
    e = np.linspace(-5, 5, 1001)
    fwhm_G = 1
    fwhm_L = 0.5
    tst = deriv_voigt(e, fwhm_G, fwhm_L)
    ref = check_num_deriv(voigt, e, fwhm_G, fwhm_L)
    result = np.allclose(tst, ref)
    assert result is True

def test_deriv_edge_1():
    e = np.linspace(-5, 5, 1001)
    fwhm_G = 1
    tst = deriv_edge_gaussian(e, fwhm_G)
    ref = check_num_deriv(edge_gaussian,
    e, fwhm_G)
    result = np.allclose(tst, ref)
    assert result is True

def test_deriv_edge_2():
    e = np.linspace(-5, 5, 1001)
    fwhm_L = 1
    tst = deriv_edge_lorenzian(e, fwhm_L)
    ref = check_num_deriv(edge_lorenzian,
    e, fwhm_L)
    result = np.allclose(tst, ref)
    assert result is True

def test_deriv_voigt_thy_1():
    e = np.linspace(-10, 20, 300)
    thy_peak = \
        np.array([[-3, -2, 1, 0, 3, 3.1, 3.5, 4, 10, 15],
        [1e-2, 1e-2, 1e-1, 2e-2, 5e-2, 3e-2, 6e-2, 1e-3, 1e-2, 1.5e-2]]).T
    tst = deriv_voigt_thy(e, thy_peak, 0.3, 0.5, -1.5, 'shift')
    ref = check_num_deriv(lambda e, fwhm_G, fwhm_L, peak_shift: \
        voigt_thy(e, thy_peak, fwhm_G, fwhm_L, peak_shift, 'shift'),
        e, 0.3, 0.5, -1.5)
    result = np.allclose(tst, ref[:, 1:])
    assert result is True

def test_deriv_voigt_thy_2():
    e = np.linspace(-10, 20, 300)
    thy_peak = \
        np.array([[-3, -2, 1, 0, 3, 3.1, 3.5, 4, 10, 15],
        [1e-2, 1e-2, 1e-1, 2e-2, 5e-2, 3e-2, 6e-2, 1e-3, 1e-2, 1.5e-2]]).T
    tst = deriv_voigt_thy(e, thy_peak, 0.3, 0.5, 0.996, 'scale')
    ref = check_num_deriv(lambda e, fwhm_G, fwhm_L, peak_scale: \
        voigt_thy(e, thy_peak, fwhm_G, fwhm_L, peak_scale, 'scale'),
        e, 0.3, 0.5, 0.996)
    result = np.allclose(tst, ref[:, 1:])
    assert result is True

def test_deriv_voigt_thy_3():
    e = np.linspace(-10, 20, 300)
    thy_peak = \
        np.array([[-3, -2, 1, 0, 3, 3.1, 3.5, 4, 10, 15],
        [1e-2, 1e-2, 1e-1, 2e-2, 5e-2, 3e-2, 6e-2, 1e-3, 1e-2, 1.5e-2]]).T
    tst = deriv_voigt_thy(e, thy_peak, 0.3, 0.5,
    np.array([-1.5, 0.996]), 'both')
    ref = check_num_deriv(lambda e, fwhm_G, fwhm_L, peak_shift, peak_scale: \
        voigt_thy(e, thy_peak, fwhm_G, fwhm_L,
        np.array([peak_shift, peak_scale]), 'both'),
        e, 0.3, 0.5, -1.5, 0.996)
    result = np.allclose(tst, ref[:, 1:])
    assert result is True

