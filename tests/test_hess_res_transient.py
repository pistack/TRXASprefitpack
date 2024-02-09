# pylint: disable = missing-module-docstring, wrong-import-position, invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import calc_eta, calc_fwhm
from TRXASprefitpack import residual_decay
from TRXASprefitpack import res_hess_decay
from TRXASprefitpack.mathfun.deriv_check import check_num_hess
from TRXASprefitpack import solve_seq_model, rate_eq_conv

rel_tol = 1e-3
abs_tol = 1e-6
epsilon = 5e-8


def test_res_hess_decay_1():
    tau_1 = 0.5
    tau_2 = 10
    tau_3 = 1000
    fwhm = 0.100
    # initial condition
    y0 = np.array([1, 0, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0]
    abs_2 = [0.5, 0.8, 0.2, 0]
    abs_3 = [-0.5, 0.7, 0.9, 0]
    abs_4 = [0.6, 0.3, -1, 0]

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')

    eps_obs_1 = np.ones_like(y_obs_1)
    eps_obs_2 = np.ones_like(y_obs_2)
    eps_obs_3 = np.ones_like(y_obs_3)
    eps_obs_4 = np.ones_like(y_obs_4)

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    x0 = np.array([0.15, 0, 0, 0, 0, 0.4, 9, 990])

    hess_ref = check_num_hess(lambda fwhm, t0_1, t0_2, t0_3, t0_4, \
    tau_1, tau_2, tau_3 : 1/2*np.sum(residual_decay(np.array([fwhm, t0_1, t0_2, t0_3, t0_4, 
    tau_1, tau_2, tau_3]), False, 'g', t=t, intensity=intensity, eps=eps)**2),
    0.15, 0, 0, 0, 0, 0.4, 9, 990)

    hess_tst = res_hess_decay(x0, 3, False, 'g', 
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_2():
    tau_1 = 0.5
    tau_2 = 10
    tau_3 = 1000
    fwhm = 0.100
    # initial condition
    y0 = np.array([1, 0, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0.3]
    abs_2 = [0.5, 0.8, 0.2, 0.1]
    abs_3 = [-0.5, 0.7, 0.9, 0.6]
    abs_4 = [0.6, 0.3, -1, 0.25]

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')

    eps_obs_1 = np.ones_like(y_obs_1)
    eps_obs_2 = np.ones_like(y_obs_2)
    eps_obs_3 = np.ones_like(y_obs_3)
    eps_obs_4 = np.ones_like(y_obs_4)

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    x0 = np.array([0.15, 0, 0, 0, 0, 0.4, 9, 990])

    hess_ref = check_num_hess(lambda fwhm, t0_1, t0_2, t0_3, t0_4, \
    tau_1, tau_2, tau_3 : 1/2*np.sum(residual_decay(np.array([fwhm, t0_1, t0_2, t0_3, t0_4, 
    tau_1, tau_2, tau_3]), True, 'g', t=t, intensity=intensity, eps=eps)**2),
    0.15, 0, 0, 0, 0, 0.4, 9, 990)

    hess_tst = res_hess_decay(x0, 3, True, 'g', 
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_3():
    tau_1 = 0.5
    tau_2 = 10
    tau_3 = 1000
    fwhm = 0.100
    # initial condition
    y0 = np.array([1, 0, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0]
    abs_2 = [0.5, 0.8, 0.2, 0]
    abs_3 = [-0.5, 0.7, 0.9, 0]
    abs_4 = [0.6, 0.3, -1, 0]

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='c')
    y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='c')
    y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='c')
    y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='c')

    eps_obs_1 = np.ones_like(y_obs_1)
    eps_obs_2 = np.ones_like(y_obs_2)
    eps_obs_3 = np.ones_like(y_obs_3)
    eps_obs_4 = np.ones_like(y_obs_4)

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    x0 = np.array([0.15, 0, 0, 0, 0, 0.4, 9, 990])

    hess_ref = check_num_hess(lambda fwhm, t0_1, t0_2, t0_3, t0_4, \
    tau_1, tau_2, tau_3 : 1/2*np.sum(residual_decay(np.array([fwhm, t0_1, t0_2, t0_3, t0_4, 
    tau_1, tau_2, tau_3]), False, 'c', t=t, intensity=intensity, eps=eps)**2),
    0.15, 0, 0, 0, 0, 0.4, 9, 990)

    hess_tst = res_hess_decay(x0, 3, False, 'c', 
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_4():
    tau_1 = 0.5
    tau_2 = 10
    tau_3 = 1000
    fwhm_G = 0.15
    fwhm_L = 0.10
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    # initial condition
    y0 = np.array([1, 0, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0]
    abs_2 = [0.5, 0.8, 0.2, 0]
    abs_3 = [-0.5, 0.7, 0.9, 0]
    abs_4 = [0.6, 0.3, -1, 0]

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)
    y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)
    y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)
    y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)

    eps_obs_1 = np.ones_like(y_obs_1)
    eps_obs_2 = np.ones_like(y_obs_2)
    eps_obs_3 = np.ones_like(y_obs_3)
    eps_obs_4 = np.ones_like(y_obs_4)

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    x0 = np.array([0.15, 0.10, 0, 0, 0, 0, 0.4, 9, 990])

    hess_ref = check_num_hess(lambda fwhm_G, fwhm_L, t0_1, t0_2, t0_3, t0_4, \
    tau_1, tau_2, tau_3 : 1/2*np.sum(residual_decay(np.array([fwhm_G, fwhm_L,
    t0_1, t0_2, t0_3, t0_4, tau_1, tau_2, tau_3]), 
    False, 'pv', t=t, intensity=intensity, eps=eps)**2),
    0.15, 0.10, 0, 0, 0, 0, 0.4, 9, 990)

    hess_tst = res_hess_decay(x0, 3, False, 'pv', 
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_5():
    tau_1 = 0.5
    tau_2 = 10
    tau_3 = 1000
    fwhm_G = 0.15
    fwhm_L = 0.10
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    # initial condition
    y0 = np.array([1, 0, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0.2]
    abs_2 = [0.5, 0.8, 0.2, 0.3]
    abs_3 = [-0.5, 0.7, 0.9, 0.4]
    abs_4 = [0.6, 0.3, -1, 0.12]

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)
    y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)
    y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)
    y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, 
                           irf='pv', eta=eta)

    eps_obs_1 = np.ones_like(y_obs_1)
    eps_obs_2 = np.ones_like(y_obs_2)
    eps_obs_3 = np.ones_like(y_obs_3)
    eps_obs_4 = np.ones_like(y_obs_4)

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    x0 = np.array([0.15, 0.10, 0, 0, 0, 0, 0.4, 9, 990])

    hess_ref = check_num_hess(lambda fwhm_G, fwhm_L, t0_1, t0_2, t0_3, t0_4, \
    tau_1, tau_2, tau_3 : 1/2*np.sum(residual_decay(np.array([fwhm_G, fwhm_L,
    t0_1, t0_2, t0_3, t0_4, tau_1, tau_2, tau_3]), 
    True, 'pv', t=t, intensity=intensity, eps=eps)**2),
    0.15, 0.10, 0, 0, 0, 0, 0.4, 9, 990)

    hess_tst = res_hess_decay(x0, 3, True, 'pv', 
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_6():
    tau_1 = 0.5
    tau_2 = 2
    tau_3 = 10
    tau_4 = 1000
    fwhm = 0.100
    # initial condition
    y0_1 = np.array([1, 0, 0, 0])
    y0_2 = np.array([1, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq_1, V_seq_1, c_seq_1 = \
        solve_seq_model(np.array([tau_1, tau_3, tau_4]), y0_1)
    eigval_seq_2, V_seq_2, c_seq_2 = \
        solve_seq_model(np.array([tau_2, tau_3]), y0_2)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1_1 = [1, 1, 1, 0]
    abs_2_1 = [0.5, 0.8, 0.2, 0]
    abs_3_1 = [-0.5, 0.7, 0.9, 0]
    abs_4_1 = [0.6, 0.3, -1, 0]

    abs_1_2 = [1, 1, 0]
    abs_2_2 = [0.5, 0.8, 0]
    abs_3_2 = [-0.5, 0.7, 0]
    abs_4_2 = [0.6, 0.3, 0]

    t0 = np.random.uniform(-0.2, 0.2, 8) # perturb time zero of each scan

    # generate measured data
    y_obs_1_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_1_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_2_1 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_2_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_3_1 = \
        rate_eq_conv(t_seq-t0[2], fwhm, abs_3_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_4_1 = \
        rate_eq_conv(t_seq-t0[3], fwhm, abs_4_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')

    y_obs_1_2 = \
        rate_eq_conv(t_seq-t0[4], fwhm, abs_1_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_2_2 = \
        rate_eq_conv(t_seq-t0[5], fwhm, abs_2_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_3_2 = \
        rate_eq_conv(t_seq-t0[6], fwhm, abs_3_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_4_2 = \
        rate_eq_conv(t_seq-t0[7], fwhm, abs_4_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')

    # generate measured intensity
    i_obs_1 = np.vstack((y_obs_1_1, y_obs_2_1, y_obs_3_1, y_obs_4_1)).T
    i_obs_2 = np.vstack((y_obs_1_2, y_obs_2_2, y_obs_3_2, y_obs_4_2)).T

    eps_obs = np.ones_like(i_obs_1)

    tau_mask = [np.array([True, False, True, True]),
                np.array([False, True, True, False])]

    t = [t_seq, t_seq]
    intensity = [i_obs_1, i_obs_2]
    eps = [eps_obs, eps_obs]

    x0 = np.array([0.15, 
                   0, 0, 0, 0, 0, 0, 0, 0,
                     0.3, 1, 8, 800])

    hess_ref = check_num_hess(lambda fwhm, t0_1, t0_2, t0_3, t0_4, \
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4 : \
    1/2*np.sum(residual_decay(np.array([fwhm, t0_1, t0_2, t0_3, t0_4,
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4]), 
    False, 'g', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)**2),
    0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 8, 800)

    hess_tst = res_hess_decay(x0, 4, False, 'g', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_7():
    tau_1 = 0.5
    tau_2 = 2
    tau_3 = 10
    tau_4 = 1000
    fwhm = 0.100
    # initial condition
    y0_1 = np.array([1, 0, 0, 0])
    y0_2 = np.array([1, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq_1, V_seq_1, c_seq_1 = \
        solve_seq_model(np.array([tau_1, tau_3, tau_4]), y0_1)
    eigval_seq_2, V_seq_2, c_seq_2 = \
        solve_seq_model(np.array([tau_2, tau_3]), y0_2)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1_1 = [1, 1, 1, 0.1]
    abs_2_1 = [0.5, 0.8, 0.2, 0.2]
    abs_3_1 = [-0.5, 0.7, 0.9, 0.12]
    abs_4_1 = [0.6, 0.3, -1, 0.32]

    abs_1_2 = [1, 1, 0.13]
    abs_2_2 = [0.5, 0.8, 0.23]
    abs_3_2 = [-0.5, 0.7, 0.43]
    abs_4_2 = [0.6, 0.3, 0.29]

    t0 = np.random.uniform(-0.2, 0.2, 8) # perturb time zero of each scan

    # generate measured data
    y_obs_1_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_1_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_2_1 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_2_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_3_1 = \
        rate_eq_conv(t_seq-t0[2], fwhm, abs_3_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_4_1 = \
        rate_eq_conv(t_seq-t0[3], fwhm, abs_4_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')

    y_obs_1_2 = \
        rate_eq_conv(t_seq-t0[4], fwhm, abs_1_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_2_2 = \
        rate_eq_conv(t_seq-t0[5], fwhm, abs_2_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_3_2 = \
        rate_eq_conv(t_seq-t0[6], fwhm, abs_3_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_4_2 = \
        rate_eq_conv(t_seq-t0[7], fwhm, abs_4_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')

    # generate measured intensity
    i_obs_1 = np.vstack((y_obs_1_1, y_obs_2_1, y_obs_3_1, y_obs_4_1)).T
    i_obs_2 = np.vstack((y_obs_1_2, y_obs_2_2, y_obs_3_2, y_obs_4_2)).T

    eps_obs = np.ones_like(i_obs_1)

    tau_mask = [np.array([True, False, True, True, True]),
                np.array([False, True, True, False, True])]

    t = [t_seq, t_seq]
    intensity = [i_obs_1, i_obs_2]
    eps = [eps_obs, eps_obs]

    x0 = np.array([0.15, 
                   0, 0, 0, 0, 0, 0, 0, 0,
                     0.3, 1, 8, 800])

    hess_ref = check_num_hess(lambda fwhm, t0_1, t0_2, t0_3, t0_4, \
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4 : \
    1/2*np.sum(residual_decay(np.array([fwhm, t0_1, t0_2, t0_3, t0_4,
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4]), 
    True, 'g', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)**2),
    0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 8, 800)

    hess_tst = res_hess_decay(x0, 4, True, 'g', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)


def test_res_hess_decay_8():
    tau_1 = 0.5
    tau_2 = 2
    tau_3 = 10
    tau_4 = 1000
    fwhm = 0.100
    # initial condition
    y0_1 = np.array([1, 0, 0, 0])
    y0_2 = np.array([1, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq_1, V_seq_1, c_seq_1 = \
        solve_seq_model(np.array([tau_1, tau_3, tau_4]), y0_1)
    eigval_seq_2, V_seq_2, c_seq_2 = \
        solve_seq_model(np.array([tau_2, tau_3]), y0_2)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1_1 = [1, 1, 1, 0]
    abs_2_1 = [0.5, 0.8, 0.2, 0]
    abs_3_1 = [-0.5, 0.7, 0.9, 0]
    abs_4_1 = [0.6, 0.3, -1, 0]

    abs_1_2 = [1, 1, 0]
    abs_2_2 = [0.5, 0.8, 0]
    abs_3_2 = [-0.5, 0.7, 0]
    abs_4_2 = [0.6, 0.3, 0]

    t0 = np.random.uniform(-0.2, 0.2, 8) # perturb time zero of each scan

    # generate measured data
    y_obs_1_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_1_1, eigval_seq_1, V_seq_1, c_seq_1, irf='c')
    y_obs_2_1 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_2_1, eigval_seq_1, V_seq_1, c_seq_1, irf='c')
    y_obs_3_1 = \
        rate_eq_conv(t_seq-t0[2], fwhm, abs_3_1, eigval_seq_1, V_seq_1, c_seq_1, irf='c')
    y_obs_4_1 = \
        rate_eq_conv(t_seq-t0[3], fwhm, abs_4_1, eigval_seq_1, V_seq_1, c_seq_1, irf='c')

    y_obs_1_2 = \
        rate_eq_conv(t_seq-t0[4], fwhm, abs_1_2, eigval_seq_2, V_seq_2, c_seq_2, irf='c')
    y_obs_2_2 = \
        rate_eq_conv(t_seq-t0[5], fwhm, abs_2_2, eigval_seq_2, V_seq_2, c_seq_2, irf='c')
    y_obs_3_2 = \
        rate_eq_conv(t_seq-t0[6], fwhm, abs_3_2, eigval_seq_2, V_seq_2, c_seq_2, irf='c')
    y_obs_4_2 = \
        rate_eq_conv(t_seq-t0[7], fwhm, abs_4_2, eigval_seq_2, V_seq_2, c_seq_2, irf='c')

    # generate measured intensity
    i_obs_1 = np.vstack((y_obs_1_1, y_obs_2_1, y_obs_3_1, y_obs_4_1)).T
    i_obs_2 = np.vstack((y_obs_1_2, y_obs_2_2, y_obs_3_2, y_obs_4_2)).T

    eps_obs = np.ones_like(i_obs_1)

    tau_mask = [np.array([True, False, True, True]),
                np.array([False, True, True, False])]

    t = [t_seq, t_seq]
    intensity = [i_obs_1, i_obs_2]
    eps = [eps_obs, eps_obs]

    x0 = np.array([0.15, 
                   0, 0, 0, 0, 0, 0, 0, 0,
                     0.3, 1, 8, 800])

    hess_ref = check_num_hess(lambda fwhm, t0_1, t0_2, t0_3, t0_4, \
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4 : \
    1/2*np.sum(residual_decay(np.array([fwhm, t0_1, t0_2, t0_3, t0_4,
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4]), 
    False, 'c', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)**2),
    0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 8, 800)

    hess_tst = res_hess_decay(x0, 4, False, 'c', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_9():
    tau_1 = 0.5
    tau_2 = 2
    tau_3 = 10
    tau_4 = 1000
    fwhm_G = 0.15
    fwhm_L = 0.10
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    # initial condition
    y0_1 = np.array([1, 0, 0, 0])
    y0_2 = np.array([1, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq_1, V_seq_1, c_seq_1 = \
        solve_seq_model(np.array([tau_1, tau_3, tau_4]), y0_1)
    eigval_seq_2, V_seq_2, c_seq_2 = \
        solve_seq_model(np.array([tau_2, tau_3]), y0_2)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1_1 = [1, 1, 1, 0]
    abs_2_1 = [0.5, 0.8, 0.2, 0]
    abs_3_1 = [-0.5, 0.7, 0.9, 0]
    abs_4_1 = [0.6, 0.3, -1, 0]

    abs_1_2 = [1, 1, 0]
    abs_2_2 = [0.5, 0.8, 0]
    abs_3_2 = [-0.5, 0.7, 0]
    abs_4_2 = [0.6, 0.3, 0]

    t0 = np.random.uniform(-0.2, 0.2, 8) # perturb time zero of each scan

    # generate measured data
    y_obs_1_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_1_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)
    y_obs_2_1 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_2_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)
    y_obs_3_1 = \
        rate_eq_conv(t_seq-t0[2], fwhm, abs_3_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)
    y_obs_4_1 = \
        rate_eq_conv(t_seq-t0[3], fwhm, abs_4_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)

    y_obs_1_2 = \
        rate_eq_conv(t_seq-t0[4], fwhm, abs_1_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)
    y_obs_2_2 = \
        rate_eq_conv(t_seq-t0[5], fwhm, abs_2_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)
    y_obs_3_2 = \
        rate_eq_conv(t_seq-t0[6], fwhm, abs_3_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)
    y_obs_4_2 = \
        rate_eq_conv(t_seq-t0[7], fwhm, abs_4_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)

    # generate measured intensity
    i_obs_1 = np.vstack((y_obs_1_1, y_obs_2_1, y_obs_3_1, y_obs_4_1)).T
    i_obs_2 = np.vstack((y_obs_1_2, y_obs_2_2, y_obs_3_2, y_obs_4_2)).T

    eps_obs = np.ones_like(i_obs_1)

    tau_mask = [np.array([True, False, True, True]),
                np.array([False, True, True, False])]

    t = [t_seq, t_seq]
    intensity = [i_obs_1, i_obs_2]
    eps = [eps_obs, eps_obs]

    x0 = np.array([0.15, 0.10, 
                   0, 0, 0, 0, 0, 0, 0, 0,
                     0.3, 1, 8, 800])

    hess_ref = check_num_hess(lambda fwhm_G, fwhm_L, t0_1, t0_2, t0_3, t0_4, \
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4 : \
    1/2*np.sum(residual_decay(np.array([fwhm_G,  fwhm_L, t0_1, t0_2, t0_3, t0_4,
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4]), 
    False, 'pv', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)**2),
    0.15, 0.10, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 8, 800)

    hess_tst = res_hess_decay(x0, 4, False, 'pv', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

def test_res_hess_decay_10():
    tau_1 = 0.5
    tau_2 = 2
    tau_3 = 10
    tau_4 = 1000
    fwhm_G = 0.15
    fwhm_L = 0.10
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    # initial condition
    y0_1 = np.array([1, 0, 0, 0])
    y0_2 = np.array([1, 0, 0])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 2000, 2)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq_1, V_seq_1, c_seq_1 = \
        solve_seq_model(np.array([tau_1, tau_3, tau_4]), y0_1)
    eigval_seq_2, V_seq_2, c_seq_2 = \
        solve_seq_model(np.array([tau_2, tau_3]), y0_2)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1_1 = [1, 1, 1, 0.2]
    abs_2_1 = [0.5, 0.8, 0.2, 0.3]
    abs_3_1 = [-0.5, 0.7, 0.9, 0.12]
    abs_4_1 = [0.6, 0.3, -1, 0.34]

    abs_1_2 = [1, 1, 0.13]
    abs_2_2 = [0.5, 0.8, 0.12]
    abs_3_2 = [-0.5, 0.7, 0.77]
    abs_4_2 = [0.6, 0.3, 0.86]

    t0 = np.random.uniform(-0.2, 0.2, 8) # perturb time zero of each scan

    # generate measured data
    y_obs_1_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_1_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)
    y_obs_2_1 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_2_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)
    y_obs_3_1 = \
        rate_eq_conv(t_seq-t0[2], fwhm, abs_3_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)
    y_obs_4_1 = \
        rate_eq_conv(t_seq-t0[3], fwhm, abs_4_1, eigval_seq_1, V_seq_1, c_seq_1, 
                     irf='pv', eta=eta)

    y_obs_1_2 = \
        rate_eq_conv(t_seq-t0[4], fwhm, abs_1_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)
    y_obs_2_2 = \
        rate_eq_conv(t_seq-t0[5], fwhm, abs_2_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)
    y_obs_3_2 = \
        rate_eq_conv(t_seq-t0[6], fwhm, abs_3_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)
    y_obs_4_2 = \
        rate_eq_conv(t_seq-t0[7], fwhm, abs_4_2, eigval_seq_2, V_seq_2, c_seq_2, 
                     irf='pv', eta=eta)

    # generate measured intensity
    i_obs_1 = np.vstack((y_obs_1_1, y_obs_2_1, y_obs_3_1, y_obs_4_1)).T
    i_obs_2 = np.vstack((y_obs_1_2, y_obs_2_2, y_obs_3_2, y_obs_4_2)).T

    eps_obs = np.ones_like(i_obs_1)

    tau_mask = [np.array([True, False, True, True, True]),
                np.array([False, True, True, False, True])]

    t = [t_seq, t_seq]
    intensity = [i_obs_1, i_obs_2]
    eps = [eps_obs, eps_obs]

    x0 = np.array([0.15, 0.10, 
                   0, 0, 0, 0, 0, 0, 0, 0,
                     0.3, 1, 8, 800])

    hess_ref = check_num_hess(lambda fwhm_G, fwhm_L, t0_1, t0_2, t0_3, t0_4, \
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4 : \
    1/2*np.sum(residual_decay(np.array([fwhm_G,  fwhm_L, t0_1, t0_2, t0_3, t0_4,
    t0_5, t0_6, t0_7, t0_8, tau_1, tau_2, tau_3, tau_4]), 
    True, 'pv', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)**2),
    0.15, 0.10, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 8, 800)

    hess_tst = res_hess_decay(x0, 4, True, 'pv', tau_mask=tau_mask,
    t=t, intensity=intensity, eps=eps)

    assert np.allclose(hess_ref, hess_tst, rtol=1e-2, atol=1e-4)

