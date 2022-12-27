# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import solve_seq_model
from TRXASprefitpack import compute_signal_gau, compute_signal_cauchy, compute_signal_pvoigt
from TRXASprefitpack import sads, dads, sads_svd, dads_svd
from TRXASprefitpack import voigt, edge_gaussian

def test_driver_sads_1():
    fwhm = 0.15
    e = np.linspace(-10, 20, 120)
    gs = voigt(e+5, 0.3, 2.0) + 0.1*edge_gaussian(e, 3)
    ex_1 = 0.7*voigt(e+3, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e-1, 3)
    ex_2 = 0.5*voigt(e+6, 0.3, 2.0) + 0.2*voigt(e+5, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e+1, 3)
    tau = np.array([0.8, 300])
    y0 = np.array([1, 0, 0])
    eigval, V, c = solve_seq_model(tau, y0)
    escan_time = np.array([-1, 0.3, 1, 100, 500])
    eps = np.ones((e.size, escan_time.size))
    y = compute_signal_gau(escan_time, fwhm, eigval, V, c)
    diff_abs = np.vstack((ex_1-gs, ex_2-gs, np.zeros_like(gs))).T
    escan = diff_abs @ y
    sads_seq, _, sads_fit = sads(escan_time, fwhm, eigval, V, c,
    irf='g', intensity=escan, eps=eps)

    sads_seq_svd, sads_fit_svd = sads_svd(escan_time, fwhm, eigval, V, c,
    irf='g', intensity=escan)

    assert np.allclose(sads_seq.T, diff_abs)
    assert np.allclose(sads_fit, escan)
    assert np.allclose(sads_seq_svd, diff_abs)
    assert np.allclose(sads_fit_svd, escan)


def test_driver_sads_2():
    fwhm = 0.15
    e = np.linspace(-10, 20, 120)
    gs = voigt(e+5, 0.3, 2.0) + 0.1*edge_gaussian(e, 3)
    ex_1 = 0.7*voigt(e+3, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e-1, 3)
    ex_2 = 0.5*voigt(e+6, 0.3, 2.0) + 0.2*voigt(e+5, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e+1, 3)
    tau = np.array([0.8, 300])
    y0 = np.array([1, 0, 0])
    eigval, V, c = solve_seq_model(tau, y0)
    escan_time = np.array([-1, 0.3, 1, 100, 500])
    eps = np.ones((e.size, escan_time.size))
    y = compute_signal_cauchy(escan_time, fwhm, eigval, V, c)
    diff_abs = np.vstack((ex_1-gs, ex_2-gs, np.zeros_like(gs))).T
    escan = diff_abs @ y
    sads_seq, _, sads_fit = sads(escan_time, fwhm, eigval, V, c,
    irf='c', intensity=escan, eps=eps)

    sads_seq_svd, sads_fit_svd = sads_svd(escan_time, fwhm, eigval, V, c,
    irf='c', intensity=escan)

    assert np.allclose(sads_seq.T, diff_abs)
    assert np.allclose(sads_fit, escan)
    assert np.allclose(sads_seq_svd, diff_abs)
    assert np.allclose(sads_fit_svd, escan)


def test_driver_sads_3():
    fwhm = 0.15
    eta = 0.3
    e = np.linspace(-10, 20, 120)
    gs = voigt(e+5, 0.3, 2.0) + 0.1*edge_gaussian(e, 3)
    ex_1 = 0.7*voigt(e+3, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e-1, 3)
    ex_2 = 0.5*voigt(e+6, 0.3, 2.0) + 0.2*voigt(e+5, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e+1, 3)
    tau = np.array([0.8, 300])
    y0 = np.array([1, 0, 0])
    eigval, V, c = solve_seq_model(tau, y0)
    escan_time = np.array([-1, 0.3, 1, 100, 500])
    eps = np.ones((e.size, escan_time.size))
    y = compute_signal_pvoigt(escan_time, fwhm, eta, eigval, V, c)
    diff_abs = np.vstack((ex_1-gs, ex_2-gs, np.zeros_like(gs))).T
    escan = diff_abs @ y

    sads_seq, _, sads_fit = sads(escan_time, fwhm, eigval, V, c,
    irf='pv', eta=eta, intensity=escan, eps=eps)

    sads_seq_svd, sads_fit_svd = sads(escan_time, fwhm, eigval, V, c,
    irf='pv', eta=eta, intensity=escan)

    assert np.allclose(sads_seq.T, diff_abs)
    assert np.allclose(sads_fit, escan)
    assert np.allclose(sads_seq_svd, diff_abs)
    assert np.allclose(sads_fit_svd, escan)



def test_driver_dads_1():
    fwhm = 0.15
    e = np.linspace(-10, 20, 120)
    gs = voigt(e+5, 0.3, 2.0) + 0.1*edge_gaussian(e, 3)
    ex_1 = 0.7*voigt(e+3, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e-1, 3)
    ex_2 = 0.5*voigt(e+6, 0.3, 2.0) + 0.2*voigt(e+5, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e+1, 3)
    tau = np.array([0.8, 300])
    y0 = np.array([1, 0, 0])
    eigval, V, c = solve_seq_model(tau, y0)
    escan_time = np.array([-1, 0.3, 1, 100, 500])
    eps = np.ones((e.size, escan_time.size))
    y = compute_signal_gau(escan_time, fwhm, eigval, V, c)
    diff_abs = np.vstack((ex_1-gs, ex_2-gs, np.zeros_like(gs))).T
    escan = diff_abs @ y
    dads_seq, _, dads_fit = dads(escan_time, fwhm, tau, False,
    irf='g', intensity=escan, eps=eps)
    V_scale = np.einsum('j,ij->ij', c, V)
    sads_dads = np.linalg.solve(V_scale[:-1, :-1].T, dads_seq)

    assert np.allclose(sads_dads.T, diff_abs[:, :-1])
    assert np.allclose(dads_fit, escan)


def test_driver_dads_2():
    fwhm = 0.15
    e = np.linspace(-10, 20, 120)
    gs = voigt(e+5, 0.3, 2.0) + 0.1*edge_gaussian(e, 3)
    ex_1 = 0.7*voigt(e+3, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e-1, 3)
    ex_2 = 0.5*voigt(e+6, 0.3, 2.0) + 0.2*voigt(e+5, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e+1, 3)
    tau = np.array([0.8, 300])
    y0 = np.array([1, 0, 0])
    eigval, V, c = solve_seq_model(tau, y0)
    escan_time = np.array([-1, 0.3, 1, 100, 500])
    eps = np.ones((e.size, escan_time.size))
    y = compute_signal_cauchy(escan_time, fwhm, eigval, V, c)
    diff_abs = np.vstack((ex_1-gs, ex_2-gs, np.zeros_like(gs))).T
    escan = diff_abs @ y
    dads_seq, _, dads_fit = dads(escan_time, fwhm, tau, False,
    irf='c', intensity=escan, eps=eps)
    V_scale = np.einsum('j,ij->ij', c, V)
    sads_dads = np.linalg.solve(V_scale[:-1, :-1].T, dads_seq)

    assert np.allclose(sads_dads.T, diff_abs[:, :-1])
    assert np.allclose(dads_fit, escan)


def test_driver_dads_3():
    fwhm = 0.15
    eta = 0.3
    e = np.linspace(-10, 20, 120)
    gs = voigt(e+5, 0.3, 2.0) + 0.1*edge_gaussian(e, 3)
    ex_1 = 0.7*voigt(e+3, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e-1, 3)
    ex_2 = 0.5*voigt(e+6, 0.3, 2.0) + 0.2*voigt(e+5, 0.3, 2.0) + 0.3*voigt(e+4, 0.3, 2.0) + \
        0.1*edge_gaussian(e+1, 3)
    tau = np.array([0.8, 300])
    y0 = np.array([1, 0, 0])
    eigval, V, c = solve_seq_model(tau, y0)
    escan_time = np.array([-1, 0.3, 1, 100, 500])
    eps = np.ones((e.size, escan_time.size))
    y = compute_signal_pvoigt(escan_time, fwhm, eta, eigval, V, c)
    diff_abs = np.vstack((ex_1-gs, ex_2-gs, np.zeros_like(gs))).T
    escan = diff_abs @ y
    dads_seq, _, dads_fit = dads(escan_time, fwhm, tau, False,
    irf='pv', eta=eta, intensity=escan, eps=eps)
    V_scale = np.einsum('j,ij->ij', c, V)
    sads_dads = np.linalg.solve(V_scale[:-1, :-1].T, dads_seq)

    assert np.allclose(sads_dads.T, diff_abs[:, :-1])
    assert np.allclose(dads_fit, escan)





