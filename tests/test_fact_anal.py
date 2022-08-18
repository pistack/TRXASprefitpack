# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import exp_conv, fact_anal_exp_conv
from TRXASprefitpack import dmp_osc_conv, fact_anal_dmp_osc_conv
from TRXASprefitpack import sum_exp_dmp_osc_conv, fact_anal_sum_exp_dmp_osc_conv



def test_fact_anal_exp_conv_1():
    fwhm = 0.15
    c_ref = np.array([1, -0.5, -0.25])
    tau = np.array([0.3, 10])
    t = np.hstack((np.arange(-1, 1, 0.05), np.linspace(1, 100, 99)))
    model = exp_conv(t, fwhm, tau, c_ref, True, irf='g')
    eps = np.ones_like(model)
    expt = model
    c_tst = fact_anal_exp_conv(t, fwhm, tau, True, irf='g', intensity=expt, eps=eps)
    assert np.allclose(c_ref, c_tst)

def test_fact_anal_exp_conv_2():
    fwhm = 0.15
    c_ref = np.array([1, -0.5, -0.25])
    tau = np.array([0.3, 10])
    t = np.hstack((np.arange(-1, 1, 0.05), np.linspace(1, 100, 99)))
    model = exp_conv(t, fwhm, tau, c_ref, True, irf='c')
    eps = np.ones_like(model)
    expt = model
    c_tst = fact_anal_exp_conv(t, fwhm, tau, True, irf='c', intensity=expt, eps=eps)
    assert np.allclose(c_ref, c_tst)

def test_fact_anal_exp_conv_3():
    fwhm = 0.10
    eta = 0.3
    c_ref = np.array([1, -0.5, -0.25])
    tau = np.array([0.3, 10])
    t = np.hstack((np.arange(-1, 1, 0.05), np.linspace(1, 100, 99)))
    model = exp_conv(t, fwhm, tau, c_ref, True, irf='pv', eta=eta)
    eps = np.ones_like(model)
    expt = model
    c_tst = fact_anal_exp_conv(t, fwhm, tau, True, irf='pv', eta=eta,
    intensity=expt, eps=eps)
    assert np.allclose(c_ref, c_tst)

def test_fact_anal_dmp_osc_conv_1():
    fwhm = 0.15
    c_ref = np.array([1, 0.5])
    tau = np.array([0.3, 1])
    period = np.array([0.5, 1])
    phase = np.array([-np.pi/17, 2*np.pi/4])
    t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
    model = dmp_osc_conv(t, fwhm, tau, period, phase, c_ref, irf='g')
    eps = np.ones_like(model)
    expt = model
    phase_tst, c_tst = fact_anal_dmp_osc_conv(t, fwhm, tau, period, irf='g', intensity=expt, eps=eps)

    assert np.allclose(c_ref, c_tst)
    assert np.allclose(phase, phase_tst)

def test_fact_anal_dmp_osc_conv_2():
    fwhm = 0.15
    c_ref = np.array([1, 0.5])
    tau = np.array([0.3, 1])
    period = np.array([0.5, 1])
    phase = np.array([-np.pi/17, 2*np.pi/4])
    t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
    model = dmp_osc_conv(t, fwhm, tau, period, phase, c_ref, irf='c')
    eps = np.ones_like(model)
    expt = model
    phase_tst, c_tst = fact_anal_dmp_osc_conv(t, fwhm, tau, period, irf='c', intensity=expt, eps=eps)

    assert np.allclose(c_ref, c_tst)
    assert np.allclose(phase, phase_tst)

def test_fact_anal_dmp_osc_conv_3():
    fwhm = 0.10
    eta = 0.3
    c_ref = np.array([1, 0.5])
    tau = np.array([0.3, 1])
    period = np.array([0.5, 1])
    phase = np.array([-np.pi/17, 2*np.pi/4])
    t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
    model = dmp_osc_conv(t, fwhm, tau, period, phase, c_ref, irf='pv', eta=eta)
    eps = np.ones_like(model)
    expt = model
    phase_tst, c_tst = fact_anal_dmp_osc_conv(t, fwhm, tau, period, irf='pv', eta=eta,
    intensity=expt, eps=eps)

    assert np.allclose(c_ref, c_tst)
    assert np.allclose(phase, phase_tst)

def test_fact_anal_sum_exp_dmp_osc_conv_1():
    fwhm = 0.15
    c_ref_decay = np.array([1, -0.5, -0.25])
    c_ref_osc = np.array([0.15, 0.075])
    tau = np.array([0.5, 10])
    tau_osc = np.array([0.3, 1])
    period_osc = np.array([0.5, 1])
    phase_osc = np.array([np.pi/3, -np.pi/4])
    t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
    model = sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, phase_osc,
    c_ref_decay, c_ref_osc, base=True, irf='g')
    eps = np.ones_like(model)
    expt = model
    c_tst_decay, phase_tst_osc, c_tst_osc = \
        fact_anal_sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, base=True,
        irf='g', intensity=expt, eps=eps)
    assert np.allclose(c_ref_decay, c_tst_decay)
    assert np.allclose(c_ref_osc, c_tst_osc)
    assert np.allclose(phase_osc, phase_tst_osc)

def test_fact_anal_sum_exp_dmp_osc_conv_2():
    fwhm = 0.15
    c_ref_decay = np.array([1, -0.5, -0.25])
    c_ref_osc = np.array([0.15, 0.075])
    tau = np.array([0.5, 10])
    tau_osc = np.array([0.3, 1])
    period_osc = np.array([0.5, 1])
    phase_osc = np.array([np.pi/3, -np.pi/4])
    t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
    model = sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, phase_osc,
    c_ref_decay, c_ref_osc, base=True, irf='c')
    eps = np.ones_like(model)
    expt = model
    c_tst_decay, phase_tst_osc, c_tst_osc = \
        fact_anal_sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, base=True,
        irf='c', intensity=expt, eps=eps)
    assert np.allclose(c_ref_decay, c_tst_decay)
    assert np.allclose(c_ref_osc, c_tst_osc)
    assert np.allclose(phase_osc, phase_tst_osc)

def test_fact_anal_sum_exp_dmp_osc_conv_3():
    fwhm = 0.10
    eta = 0.3
    c_ref_decay = np.array([1, -0.5, -0.25])
    c_ref_osc = np.array([0.15, 0.075])
    tau = np.array([0.5, 10])
    tau_osc = np.array([0.3, 1])
    period_osc = np.array([0.5, 1])
    phase_osc = np.array([np.pi/3, -np.pi/4])
    t = np.hstack((np.arange(-1, 1, 0.02), np.linspace(1, 100, 99)))
    model = sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, phase_osc,
    c_ref_decay, c_ref_osc, base=True, irf='pv', eta=eta)
    eps = np.ones_like(model)
    expt = model
    c_tst_decay, phase_tst_osc, c_tst_osc = \
        fact_anal_sum_exp_dmp_osc_conv(t, fwhm, tau, tau_osc, period_osc, base=True,
        irf='pv', eta=eta, intensity=expt, eps=eps)
    assert np.allclose(c_ref_decay, c_tst_decay)
    assert np.allclose(c_ref_osc, c_tst_osc)
    assert np.allclose(phase_osc, phase_tst_osc)

