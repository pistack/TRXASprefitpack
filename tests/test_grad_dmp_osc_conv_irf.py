# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack import deriv_dmp_osc_conv_gau_2, deriv_dmp_osc_conv_cauchy_2
from TRXASprefitpack import deriv_dmp_osc_conv_gau, deriv_dmp_osc_conv_cauchy
from TRXASprefitpack import dmp_osc_conv_gau_2, dmp_osc_conv_cauchy_2
from TRXASprefitpack import dmp_osc_conv_gau, dmp_osc_conv_cauchy
from TRXASprefitpack.mathfun.deriv_check import check_num_deriv
from TRXASprefitpack import calc_eta, calc_fwhm, deriv_eta, deriv_fwhm

def tmp_fun(t, fwhm_G, fwhm_L, k, period, phase):
    fwhm=calc_fwhm(fwhm_G, fwhm_L)
    eta=calc_eta(fwhm_G, fwhm_L)
    gau=dmp_osc_conv_gau(t, fwhm, k, period, phase)
    cauchy=dmp_osc_conv_cauchy(t, fwhm, k, period, phase)
    return gau + eta*(cauchy-gau)

def tmp_fun_2(t, fwhm_G, fwhm_L, k, period, c_pair):
    fwhm=calc_fwhm(fwhm_G, fwhm_L)
    eta=calc_eta(fwhm_G, fwhm_L)
    gau=dmp_osc_conv_gau_2(t, fwhm, k, period, c_pair)
    cauchy=dmp_osc_conv_cauchy_2(t, fwhm, k, period, c_pair)
    return gau + eta*(cauchy-gau)


def test_deriv_dmp_osc_conv_gau_1():
    tau = 1
    fwhm = 0.15
    period = 0.5
    phase = np.pi/3
    t = np.linspace(-1, 100, 2001)
    tst = deriv_dmp_osc_conv_gau(t, fwhm, 1/tau, period, phase)
    ref = check_num_deriv(dmp_osc_conv_gau, t, fwhm, 1/tau, period, phase)
    assert np.allclose(tst, ref)

def test_deriv_dmp_osc_conv_gau_2():
    tau=1
    fwhm=0.15
    period=0.5
    c_pair=(0.3, 0.7)
    t=np.linspace(-1, 100, 2001)
    tst=deriv_dmp_osc_conv_gau_2(t, fwhm, 1/tau, period, c_pair)
    ref=check_num_deriv(lambda t, fwhm, k, period: dmp_osc_conv_gau_2(t, fwhm, k, period, c_pair),
     t, fwhm, 1/tau, period)
    assert np.allclose(tst, ref)

def test_deriv_dmp_osc_conv_cauchy_1():
    tau=1
    fwhm=0.15
    period=0.5
    phase=np.pi/4
    t=np.linspace(-1, 100, 2001)
    tst=deriv_dmp_osc_conv_cauchy(t, fwhm, 1/tau, period, phase)
    ref=check_num_deriv(dmp_osc_conv_cauchy, t, fwhm, 1/tau, period, phase)
    assert np.allclose(tst, ref)

def test_deriv_dmp_osc_conv_cauchy_2():
    tau=1
    fwhm=0.15
    period=0.5
    c_pair=(0.3, 0.7)
    t=np.linspace(-1, 100, 2001)
    tst=deriv_dmp_osc_conv_cauchy_2(t, fwhm, 1/tau, period, c_pair)
    ref=check_num_deriv(lambda t, fwhm, k, period: dmp_osc_conv_cauchy_2(t, fwhm, k, period, c_pair),
     t, fwhm, 1/tau, period)
    assert np.allclose(tst, ref)

def test_deriv_exp_conv_pvoigt():
    tau_1=1
    fwhm_G=0.1
    fwhm_L=0.15
    period=0.5
    phase=np.pi/3
    t=np.linspace(-1, 100, 2001)

    grad=np.empty((t.size, 6))
    fwhm=calc_fwhm(fwhm_G, fwhm_L)
    eta=calc_eta(fwhm_G, fwhm_L)
    dfwhm_G, dfwhm_L=deriv_fwhm(fwhm_G, fwhm_L)
    deta_G, deta_L=deriv_eta(fwhm_G, fwhm_L)
    diff=dmp_osc_conv_cauchy(t, fwhm, 1/tau_1, period, phase) - \
        dmp_osc_conv_gau(t, fwhm, 1/tau_1, period, phase)
    grad_gau=deriv_dmp_osc_conv_gau(t, fwhm, 1/tau_1, period, phase)
    grad_cauchy=deriv_dmp_osc_conv_cauchy(t, fwhm, 1/tau_1, period, phase)
    grad_tot=grad_gau + eta*(grad_cauchy-grad_gau)
    grad[:, 0]=grad_tot[:, 0]
    grad[:, 3]=grad_tot[:, 2]
    grad[:, 4]=grad_tot[:, 3]
    grad[:, 5]=grad_tot[:, 4]
    grad[:, 1]=dfwhm_G*grad_tot[:, 1] + deta_G*diff
    grad[:, 2]=dfwhm_L*grad_tot[:, 1] + deta_L*diff

    ref=check_num_deriv(tmp_fun, t, fwhm_G, fwhm_L, 1/tau_1, period, phase)

    assert np.allclose(grad, ref)

def test_deriv_exp_conv_pvoigt_2():
    c_pair=(0.3, 0.7)

    tau_1=1
    fwhm_G=0.1
    fwhm_L=0.15
    period=0.5
    t=np.linspace(-1, 100, 2001)

    grad=np.empty((t.size, 5))
    fwhm=calc_fwhm(fwhm_G, fwhm_L)
    eta=calc_eta(fwhm_G, fwhm_L)
    dfwhm_G, dfwhm_L=deriv_fwhm(fwhm_G, fwhm_L)
    deta_G, deta_L=deriv_eta(fwhm_G, fwhm_L)
    diff=dmp_osc_conv_cauchy_2(t, fwhm, 1/tau_1, period, c_pair) - \
        dmp_osc_conv_gau_2(t, fwhm, 1/tau_1, period, c_pair)
    grad_gau=deriv_dmp_osc_conv_gau_2(t, fwhm, 1/tau_1, period, c_pair)
    grad_cauchy=deriv_dmp_osc_conv_cauchy_2(t, fwhm, 1/tau_1, period, c_pair)
    grad_tot=grad_gau + eta*(grad_cauchy-grad_gau)
    grad[:, 0]=grad_tot[:, 0]
    grad[:, 3]=grad_tot[:, 2]
    grad[:, 4]=grad_tot[:, 3]
    grad[:, 1]=dfwhm_G*grad_tot[:, 1] + deta_G*diff
    grad[:, 2]=dfwhm_L*grad_tot[:, 1] + deta_L*diff

    ref=check_num_deriv(lambda t, fwhm_G, fwhm_L, k, period: \
        tmp_fun_2(t, fwhm_G, fwhm_L, k, period, c_pair),
    t, fwhm_G, fwhm_L, 1/tau_1, period)

    assert np.allclose(grad, ref)
