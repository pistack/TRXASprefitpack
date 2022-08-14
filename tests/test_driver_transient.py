# pylint: disable = missing-module-docstring,wrong-import-position,invalid-name,multiple-statements
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import solve_seq_model, rate_eq_conv, dmp_osc_conv
from TRXASprefitpack import calc_eta, calc_fwhm
from TRXASprefitpack import fit_transient_exp, fit_transient_dmp_osc
from TRXASprefitpack import save_TransientResult, load_TransientResult

def test_driver_transient_exp_1():
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
    t_seq7 = np.linspace(1000, 5000, 5)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0.2]
    abs_2 = [0.5, 0.8, 0.2, -0.05]
    abs_3 = [-0.5, 0.7, 0.9, 0.03]
    abs_4 = [0.6, 0.3, -1, 0.25]

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')
    y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')

    eps_obs_1 = np.ones_like(y_obs_1)/1000
    eps_obs_2 = np.ones_like(y_obs_2)/1000
    eps_obs_3 = np.ones_like(y_obs_3)/1000
    eps_obs_4 = np.ones_like(y_obs_4)/1000

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    ans = np.array([fwhm, t0[0], t0[1], t0[2], t0[3],
    tau_1, tau_2, tau_3])

    bound_fwhm = [(0.05, 0.2)]
    bound_t0 = [(-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)]
    bound_tau = [(0.1, 1), (1, 100), (100, 10000)]
    fwhm_init = np.random.uniform(0.05, 0.2)
    t0_init = np.random.uniform(-0.4, 0.4, 4)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])


    result_ampgo = fit_transient_exp('g', fwhm_init, t0_init, tau_init, True,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0, bound_tau=bound_tau,
    t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_exp_1')
    load_result_ampgo = load_TransientResult('test_driver_transient_exp_1')
    os.remove('test_driver_transient_exp_1.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert str(result_ampgo) == str(load_result_ampgo)

def test_driver_transient_exp_2():
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
    t_seq7 = np.linspace(1000, 5000, 5)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0.2]
    abs_2 = [0.5, 0.8, 0.2, -0.05]
    abs_3 = [-0.5, 0.7, 0.9, 0.03]
    abs_4 = [0.6, 0.3, -1, 0.25]

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='c')
    y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='c')
    y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='c')
    y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='c')

    eps_obs_1 = np.ones_like(y_obs_1)/1000
    eps_obs_2 = np.ones_like(y_obs_2)/1000
    eps_obs_3 = np.ones_like(y_obs_3)/1000
    eps_obs_4 = np.ones_like(y_obs_4)/1000

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    ans = np.array([fwhm, t0[0], t0[1], t0[2], t0[3],
    tau_1, tau_2, tau_3])

    bound_fwhm = [(0.05, 0.2)]
    bound_t0 = [(-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)]
    bound_tau = [(0.1, 1), (1, 100), (100, 10000)]
    fwhm_init = np.random.uniform(0.05, 0.2)
    t0_init = np.random.uniform(-0.4, 0.4, 4)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])


    result_ampgo = fit_transient_exp('c', fwhm_init, t0_init, tau_init, True,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0, bound_tau=bound_tau,
    t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_exp_2')
    load_result_ampgo = load_TransientResult('test_driver_transient_exp_2')
    os.remove('test_driver_transient_exp_2.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert str(result_ampgo) == str(load_result_ampgo)

def test_driver_transient_exp_3():
    tau_1 = 0.5
    tau_2 = 10
    tau_3 = 1000
    fwhm_G = 0.100
    fwhm_L = 0.05
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
    t_seq7 = np.linspace(1000, 5000, 5)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))
    eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = [1, 1, 1, 0.2]
    abs_2 = [0.5, 0.8, 0.2, -0.05]
    abs_3 = [-0.5, 0.7, 0.9, 0.03]
    abs_4 = [0.6, 0.3, -1, 0.25]

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

    eps_obs_1 = np.ones_like(y_obs_1)/1000
    eps_obs_2 = np.ones_like(y_obs_2)/1000
    eps_obs_3 = np.ones_like(y_obs_3)/1000
    eps_obs_4 = np.ones_like(y_obs_4)/1000

    # generate measured intensity
    i_obs_1 = y_obs_1
    i_obs_2 = y_obs_2
    i_obs_3 = y_obs_3
    i_obs_4 = y_obs_4

    t = [t_seq]
    intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
    eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

    ans = np.array([fwhm_G, fwhm_L, t0[0], t0[1], t0[2], t0[3],
    tau_1, tau_2, tau_3])

    bound_fwhm = [(0.05, 0.2), (0.025, 0.1)]
    bound_t0 = [(-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)]
    bound_tau = [(0.1, 1), (1, 100), (100, 10000)]
    fwhm_init = np.array([np.random.uniform(0.05, 0.2),
    np.random.uniform(0.025, 0.1)])
    t0_init = np.random.uniform(-0.4, 0.4, 4)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])


    result_ampgo = fit_transient_exp('pv', fwhm_init, t0_init, tau_init, True,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0, bound_tau=bound_tau,
    t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_exp_3')
    load_result_ampgo = load_TransientResult('test_driver_transient_exp_3')
    os.remove('test_driver_transient_exp_3.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert str(result_ampgo) == str(load_result_ampgo)

def test_driver_transient_dmp_osc_1():
    fwhm = 0.100
    tau = np.array([0.5, 10, 1000])
    period = np.array([0.2, 3, 200])
    phase_1 = np.array([np.pi/17, -np.pi/6, np.pi/2])
    phase_2 = np.array([np.pi/10, -np.pi/3, np.pi])
    phase_3 = np.array([np.pi/8, -np.pi/2, 2*np.pi/3])
    phase_4 = np.array([np.pi/2, -np.pi, np.pi/18])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 5000, 5)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = np.array([1, 1, 1])
    abs_2 = np.array([0.5, 0.8, 0.2])
    abs_3 = np.array([-0.5, 0.7, 0.9])
    abs_4 = np.array([0.6, 0.3, -1])

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = dmp_osc_conv(t_seq-t0[0], fwhm, tau, period, phase_1, abs_1, irf='g')
    y_obs_2 = dmp_osc_conv(t_seq-t0[1], fwhm, tau, period, phase_2, abs_2, irf='g')
    y_obs_3 = dmp_osc_conv(t_seq-t0[2], fwhm, tau, period, phase_3, abs_3, irf='g')
    y_obs_4 = dmp_osc_conv(t_seq-t0[3], fwhm, tau, period, phase_4, abs_4, irf='g')

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

    ans = np.array([fwhm, t0[0], t0[1], t0[2], t0[3],
    tau[0], tau[1], tau[2], period[0], period[1], period[2]])

    bound_fwhm = [(0.05, 0.2)]
    bound_t0 = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]
    bound_tau = [(0.1, 1), (1, 100), (100, 10000)]
    bound_period = [(0.1, 1), (1, 10), (100, 1000)]
    fwhm_init = np.random.uniform(0.05, 0.2)
    t0_init = np.random.uniform(-0.2, 0.2, 4)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])
    period_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 10), np.random.uniform(100, 1000)])

    result_ampgo = fit_transient_dmp_osc('g', fwhm_init, t0_init, tau_init, period_init,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0,
    bound_tau=bound_tau, bound_period=bound_period, t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_dmp_osc_1')
    load_result_ampgo = load_TransientResult('test_driver_transient_dmp_osc_1')
    os.remove('test_driver_transient_dmp_osc_1.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert str(result_ampgo) == str(load_result_ampgo)

def test_driver_transient_dmp_osc_2():
    fwhm = 0.100
    tau = np.array([0.5, 10, 1000])
    period = np.array([0.2, 3, 200])
    phase_1 = np.array([np.pi/17, -np.pi/6, np.pi/2])
    phase_2 = np.array([np.pi/10, -np.pi/3, np.pi])
    phase_3 = np.array([np.pi/8, -np.pi/2, 2*np.pi/3])
    phase_4 = np.array([np.pi/2, -np.pi, np.pi/18])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 5000, 5)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = np.array([1, 1, 1])
    abs_2 = np.array([0.5, 0.8, 0.2])
    abs_3 = np.array([-0.5, 0.7, 0.9])
    abs_4 = np.array([0.6, 0.3, -1])

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = dmp_osc_conv(t_seq-t0[0], fwhm, tau, period, phase_1, abs_1, irf='c')
    y_obs_2 = dmp_osc_conv(t_seq-t0[1], fwhm, tau, period, phase_2, abs_2, irf='c')
    y_obs_3 = dmp_osc_conv(t_seq-t0[2], fwhm, tau, period, phase_3, abs_3, irf='c')
    y_obs_4 = dmp_osc_conv(t_seq-t0[3], fwhm, tau, period, phase_4, abs_4, irf='c')

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

    ans = np.array([fwhm, t0[0], t0[1], t0[2], t0[3],
    tau[0], tau[1], tau[2], period[0], period[1], period[2]])

    bound_fwhm = [(0.05, 0.2)]
    bound_t0 = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]
    bound_tau = [(0.1, 1), (1, 100), (100, 10000)]
    bound_period = [(0.1, 1), (1, 10), (100, 1000)]
    fwhm_init = np.random.uniform(0.05, 0.2)
    t0_init = np.random.uniform(-0.2, 0.2, 4)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])
    period_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 10), np.random.uniform(100, 1000)])

    result_ampgo = fit_transient_dmp_osc('c', fwhm_init, t0_init, tau_init, period_init,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0,
    bound_tau=bound_tau, bound_period=bound_period, t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_dmp_osc_2')
    load_result_ampgo = load_TransientResult('test_driver_transient_dmp_osc_2')
    os.remove('test_driver_transient_dmp_osc_2.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert str(result_ampgo) == str(load_result_ampgo)

def test_driver_transient_dmp_osc_3():
    fwhm_G = 0.100
    fwhm_L = 0.05
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    tau = np.array([0.5, 10, 1000])
    period = np.array([0.2, 3, 200])
    phase_1 = np.array([np.pi/17, -np.pi/6, np.pi/2])
    phase_2 = np.array([np.pi/10, -np.pi/3, np.pi])
    phase_3 = np.array([np.pi/8, -np.pi/2, 2*np.pi/3])
    phase_4 = np.array([np.pi/2, -np.pi, np.pi/18])

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 2, 0.02)
    t_seq3 = np.arange(2, 5, 0.2)
    t_seq4 = np.arange(5, 10, 1)
    t_seq5 = np.arange(10, 100, 10)
    t_seq6 = np.arange(100, 1000, 100)
    t_seq7 = np.linspace(1000, 5000, 5)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))

    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = np.array([1, 1, 1])
    abs_2 = np.array([0.5, 0.8, 0.2])
    abs_3 = np.array([-0.5, 0.7, 0.9])
    abs_4 = np.array([0.6, 0.3, -1])

    t0 = np.random.uniform(-0.2, 0.2, 4) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = dmp_osc_conv(t_seq-t0[0], fwhm, tau, period, phase_1, abs_1,
    irf='pv', eta=eta)
    y_obs_2 = dmp_osc_conv(t_seq-t0[1], fwhm, tau, period, phase_2, abs_2,
    irf='pv', eta=eta)
    y_obs_3 = dmp_osc_conv(t_seq-t0[2], fwhm, tau, period, phase_3, abs_3,
    irf='pv', eta=eta)
    y_obs_4 = dmp_osc_conv(t_seq-t0[3], fwhm, tau, period, phase_4, abs_4,
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

    ans = np.array([fwhm_G, fwhm_L, t0[0], t0[1], t0[2], t0[3],
    tau[0], tau[1], tau[2], period[0], period[1], period[2]])

    bound_fwhm = [(0.05, 0.2), (0.025, 0.1)]
    bound_t0 = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]
    bound_tau = [(0.1, 1), (1, 100), (100, 10000)]
    bound_period = [(0.1, 1), (1, 10), (100, 1000)]
    fwhm_init = np.array([np.random.uniform(0.05, 0.2),
    np.random.uniform(0.025, 0.1)])
    t0_init = np.random.uniform(-0.2, 0.2, 4)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])
    period_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])

    result_ampgo = fit_transient_dmp_osc('pv', fwhm_init, t0_init, tau_init, period_init,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0,
    bound_tau=bound_tau, bound_period=bound_period, t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_dmp_osc_3')
    load_result_ampgo = load_TransientResult('test_driver_transient_dmp_osc_3')
    os.remove('test_driver_transient_dmp_osc_3.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert str(result_ampgo) == str(load_result_ampgo)

