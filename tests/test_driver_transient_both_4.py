# pylint: disable = missing-module-docstring,wrong-import-position,invalid-name,multiple-statements
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import solve_seq_model, rate_eq_conv, dmp_osc_conv
from TRXASprefitpack import fit_transient_exp, fit_transient_both
from TRXASprefitpack import save_TransientResult, load_TransientResult

def test_driver_transient_both_4():
    tau_1 = 1
    tau_2 = 10
    tau_3 = 1000
    fwhm = 0.100
    tau_osc = np.array([0.5, 0.8])
    period_osc = np.array([0.3, 0.6])
    phase_1 = np.random.uniform(-np.pi, np.pi, 2)
    phase_2 = np.random.uniform(-np.pi, np.pi, 2)
    phase_3 = np.random.uniform(-np.pi, np.pi, 2)
    phase_4 = np.random.uniform(-np.pi, np.pi, 2)
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
    abs_osc_1 = np.array([0.05, 0.03])
    abs_osc_2 = np.array([0.02, 0.01])
    abs_osc_3 = np.array([0.05, 0.1])
    abs_osc_4 = np.array([0.025, 0.125])
    abs_osc = np.vstack((abs_osc_1, abs_osc_2, abs_osc_3, abs_osc_4))

    t0 = np.random.uniform(-0.2, 0.2, 1) # perturb time zero of each scan

    # generate measured data
    y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g') + \
        dmp_osc_conv(t_seq-t0[0], fwhm, tau_osc, period_osc, phase_1, abs_osc_1, 'g')
    y_obs_2 = rate_eq_conv(t_seq-t0[0], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g') + \
        dmp_osc_conv(t_seq-t0[0], fwhm, tau_osc, period_osc, phase_2, abs_osc_2, 'g')
    y_obs_3 = rate_eq_conv(t_seq-t0[0], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g') + \
        dmp_osc_conv(t_seq-t0[0], fwhm, tau_osc, period_osc, phase_3, abs_osc_3, 'g')
    y_obs_4 = rate_eq_conv(t_seq-t0[0], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g') + \
        dmp_osc_conv(t_seq-t0[0], fwhm, tau_osc, period_osc, phase_4, abs_osc_4, 'g')

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

    ans = np.array([fwhm, t0[0],
    tau_1, tau_2, tau_3, tau_osc[0], tau_osc[1], period_osc[0], period_osc[1]])

    bound_fwhm = [(0.05, 0.2)]
    bound_t0 = [(-0.4, 0.4)]
    bound_tau = [(0.1, 1), (1, 100), (100, 10000)]
    bound_tau_osc = [(0.1, 0.5), (0.5, 1)]
    bound_period_osc = [(0.1, 0.5), (0.5, 1)]
    fwhm_init = np.random.uniform(0.05, 0.2)
    t0_init = np.random.uniform(-0.4, 0.4, 1)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])
    tau_osc_init = np.array([np.random.uniform(0.1, 0.5),
    np.random.uniform(0.5, 1)])
    period_osc_init = np.array([np.random.uniform(0.1, 0.5),
    np.random.uniform(0.5, 1)])

    result_ampgo_exp = fit_transient_exp('g', fwhm_init, t0_init, tau_init, True,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0, bound_tau=bound_tau,
    same_t0=True,
    t=t, intensity=intensity, eps=eps)

    fwhm_init_2 = result_ampgo_exp['x'][0]
    t0_init_2 = result_ampgo_exp['x'][1:2]
    tau_init_2 = result_ampgo_exp['x'][2:]

    result_ampgo = fit_transient_both('g', fwhm_init_2, t0_init_2,
    tau_init_2, tau_osc_init, period_osc_init, True, method_glb='ampgo',
    bound_fwhm=bound_fwhm, bound_t0=bound_t0, bound_tau=bound_tau,
    bound_tau_osc=bound_tau_osc, bound_period_osc=bound_period_osc,
    same_t0=True,
    t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_both_1')
    load_result_ampgo = load_TransientResult('test_driver_transient_both_1')
    os.remove('test_driver_transient_both_1.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert np.allclose(result_ampgo['c'][0][4:, :], abs_osc.T)
    assert str(result_ampgo) == str(load_result_ampgo)


