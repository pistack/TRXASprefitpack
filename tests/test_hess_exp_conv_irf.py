# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.mathfun.deriv_check import check_num_hess
from TRXASprefitpack import exp_conv_gau, exp_conv_cauchy
from TRXASprefitpack import hess_exp_conv_gau, hess_exp_conv_cauchy

# loose tolerance is used due to the numerical noise in my poor implementation of
# numerical hessian

def test_hess_exp_conv_gau_1():
    fwhm = 0.15
    t = np.linspace(-1, 100, 2001)
    tst = hess_exp_conv_gau(t, fwhm, 0)
    ref = check_num_hess(exp_conv_gau, t, fwhm, 0)

    assert np.allclose(tst[:, 0], ref[:, 0, 0], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 1], ref[:, 0, 1], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 3], ref[:, 1, 1], rtol=1e-3, atol=1e-5)

def test_hess_exp_conv_gau_2():
    tau = 1
    fwhm = 0.15
    t = np.linspace(-1, 100, 2001)
    tst = hess_exp_conv_gau(t, fwhm, 1/tau)
    ref = check_num_hess(exp_conv_gau, t, fwhm, 1/tau)

    assert np.allclose(tst[:, 0], ref[:, 0, 0], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 1], ref[:, 0, 1], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 2], ref[:, 0, 2], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 3], ref[:, 1, 1], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 4], ref[:, 1, 2], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 5], ref[:, 2, 2], rtol=1e-3, atol=1e-5)

def test_hess_exp_conv_cauchy_1():
    fwhm = 0.15
    t = np.linspace(-1, 100, 2001)
    tst = hess_exp_conv_cauchy(t, fwhm, 0)
    ref = check_num_hess(exp_conv_cauchy, t, fwhm, 0)

    assert np.allclose(tst[:, 0], ref[:, 0, 0], rtol=1e-3, atol=1e-4)
    assert np.allclose(tst[:, 1], ref[:, 0, 1], rtol=1e-3, atol=1e-4)
    assert np.allclose(tst[:, 3], ref[:, 1, 1], rtol=1e-3, atol=1e-4)

def test_hess_exp_conv_cauchy_2():
    tau = 1
    fwhm = 0.15
    t = np.linspace(-1, 100, 2001)
    tst = hess_exp_conv_cauchy(t, fwhm, 1/tau)
    ref = check_num_hess(exp_conv_cauchy, t, fwhm, 1/tau)
    assert np.allclose(tst[:, 0], ref[:, 0, 0], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 1], ref[:, 0, 1], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 2], ref[:, 0, 2], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 3], ref[:, 1, 1], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 4], ref[:, 1, 2], rtol=1e-3, atol=1e-5)
    assert np.allclose(tst[:, 5], ref[:, 2, 2], rtol=1e-3, atol=1e-5)




