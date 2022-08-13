# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np
from scipy.signal import convolve

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import exp_conv_pvoigt, exp_conv_gau, exp_conv_cauchy
from TRXASprefitpack import voigt
from TRXASprefitpack import gau_irf, cauchy_irf
from TRXASprefitpack import calc_fwhm
from TRXASprefitpack import calc_eta

def decay(t, k):
    return np.exp(-k*t)*np.heaviside(t, 1)


def test_exp_conv_gau():
    fwhm_G = 0.15
    tau = 0.3
    t = np.linspace(-2, 2, 2000)
    t_sample = np.hstack((np.arange(-1, -0.5, 0.1),
                          np.arange(-0.5, 0.5, 0.05), np.linspace(0.5, 1, 6)))
    sample_idx = np.searchsorted(t, t_sample)
    gau_ref = gau_irf(t, fwhm_G)
    decay_ref = decay(t, 1/tau)
    ref = convolve(gau_ref, decay_ref, mode='same')*4/2000
    tst = exp_conv_gau(t_sample, fwhm_G, 1/tau)
    max_rel_err = np.max(np.abs(ref[sample_idx]-tst))/np.max(ref)
    assert max_rel_err < 1e-3

def test_exp_conv_cauchy():
    fwhm_L = 0.15
    tau = 0.3
    t = np.linspace(-2, 2, 2000)
    t_sample = np.hstack((np.arange(-1, -0.5, 0.1),
                          np.arange(-0.5, 0.5, 0.05), np.linspace(0.5, 1, 6)))
    sample_idx = np.searchsorted(t, t_sample)
    cauchy_ref = cauchy_irf(t, fwhm_L)
    decay_ref = decay(t, 1/tau)
    ref = convolve(cauchy_ref, decay_ref, mode='same')*4/2000
    tst = exp_conv_cauchy(t_sample, fwhm_L, 1/tau)
    max_rel_err = np.max(np.abs(ref[sample_idx]-tst))/np.max(ref)
    assert max_rel_err < 1e-3


def test_exp_conv_pvoigt():
    fwhm_G = 0.15
    fwhm_L = 0.10
    tau = 0.3
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    t = np.linspace(-2, 2, 2000)
    t_sample = np.hstack((np.arange(-1, -0.5, 0.1),
                          np.arange(-0.5, 0.5, 0.05), np.linspace(0.5, 1, 6)))
    sample_idx = np.searchsorted(t, t_sample)
    voigt_ref = voigt(t, fwhm_G, fwhm_L)
    decay_ref = decay(t, 1/tau)
    ref = convolve(voigt_ref, decay_ref, mode='same')*4/2000
    tst = exp_conv_pvoigt(t_sample, fwhm, eta, 1/tau)
    max_rel_err = np.max(np.abs(ref[sample_idx]-tst))/np.max(ref)
    assert max_rel_err < 1e-2
