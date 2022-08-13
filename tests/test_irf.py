# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.mathfun.deriv_check import check_num_deriv
from TRXASprefitpack import calc_eta, deriv_eta
from TRXASprefitpack import calc_fwhm, deriv_fwhm
from TRXASprefitpack import pvoigt_irf
from TRXASprefitpack import voigt

def test_pvoigt_irf_1():
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.1
    fwhm_L = 0.3
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    max_rel_err = np.max(np.abs(tst-ref))/np.max(ref)
    assert max_rel_err < 2e-2

def test_pvoigt_irf_2():
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.1
    fwhm_L = 0.15
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    max_rel_err = np.max(np.abs(tst-ref))/np.max(ref)
    assert max_rel_err < 2e-2

def test_pvoigt_irf_3():
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.1
    fwhm_L = 0.1
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    max_rel_err = np.max(np.abs(tst-ref))/np.max(ref)
    assert max_rel_err < 2e-2

def test_pvoigt_irf_4():
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.15
    fwhm_L = 0.1
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    max_rel_err = np.max(np.abs(tst-ref))/np.max(ref)
    assert max_rel_err < 2e-2

def test_pvoigt_irf_5():
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.3
    fwhm_L = 0.1
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    max_rel_err = np.max(np.abs(tst-ref))/np.max(ref)
    assert max_rel_err < 2e-2

def test_deriv_eta_1():
    fwhm_G = 0.1
    fwhm_L = 0.3
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_eta_2():
    fwhm_G = 0.1
    fwhm_L = 0.15
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_eta_3():
    fwhm_G = 0.1
    fwhm_L = 0.1
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_eta_4():
    fwhm_G = 0.15
    fwhm_L = 0.1
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_eta_5():
    fwhm_G = 0.3
    fwhm_L = 0.1
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_eta_6():
    fwhm_G = 0.15
    fwhm_L = 0.3
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_fwhm_1():
    fwhm_G = 0.1
    fwhm_L = 0.3
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_fwhm_2():
    fwhm_G = 0.1
    fwhm_L = 0.15
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_fwhm_3():
    fwhm_G = 0.1
    fwhm_L = 0.1
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_fwhm_4():
    fwhm_G = 0.15
    fwhm_L = 0.1
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_fwhm_5():
    fwhm_G = 0.3
    fwhm_L = 0.1
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result is True

def test_deriv_fwhm_6():
    fwhm_G = 0.15
    fwhm_L = 0.3
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)

    result = np.allclose(np.array([d_G, d_L]),
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result is True
