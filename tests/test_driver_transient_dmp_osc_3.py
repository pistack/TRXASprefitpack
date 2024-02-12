# pylint: disable = missing-module-docstring,wrong-import-position,invalid-name,multiple-statements
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import dmp_osc_conv
from TRXASprefitpack import calc_eta, calc_fwhm
from TRXASprefitpack import fit_transient_dmp_osc
from TRXASprefitpack import save_TransientResult, load_TransientResult

def test_driver_transient_dmp_osc_3():
    fwhm_G = 0.05
    fwhm_L = 0.03
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    tau = np.array([0.5, 10, 1000])
    period = np.array([0.3, 3, 200])
    phase_1 = np.random.uniform(-np.pi, np.pi, 3)
    phase_2 = np.random.uniform(-np.pi, np.pi, 3)
    phase_3 = np.random.uniform(-np.pi, np.pi, 3)
    phase_4 = np.random.uniform(-np.pi, np.pi, 3)

    # set time range (mixed step)
    t_seq1 = np.arange(-2, -1, 0.2)
    t_seq2 = np.arange(-1, 1, 0.05)
    t_seq3 = np.arange(1, 20, 0.5)
    t_seq4 = np.arange(20, 100, 10)
    t_seq5 = np.arange(100, 2000, 50)
    t_seq6 = np.linspace(2000, 5000, 4)

    t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6))
    # Now generates measured transient signal
    # Last element is ground state
    abs_1 = np.array([1, 1, 1])
    abs_2 = np.array([0.5, 0.8, 0.2])
    abs_3 = np.array([0.5, 0.7, 0.9])
    abs_4 = np.array([0.6, 0.3, 1])
    abs_osc = np.vstack((abs_1, abs_2, abs_3, abs_4))

    t0 = np.random.uniform(-0.1, 0.1, 4) # perturb time zero of each scan

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

    bound_fwhm = [(0.025, 0.1), (0.025, 0.1)]
    bound_t0 = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
    bound_tau = [(0.1, 1), (1, 100), (100, 5000)]
    bound_period = [(0.1, 0.5), (1, 10), (100, 1000)]
    fwhm_init = np.random.uniform(0.025, 0.1)
    t0_init = np.random.uniform(-0.1, 0.1, 4)
    tau_init = np.array([np.random.uniform(0.1, 1),
    np.random.uniform(1, 100), np.random.uniform(100, 5000)])
    period_init = np.array([np.random.uniform(0.1, 0.5),
    np.random.uniform(1, 10), np.random.uniform(100, 500)])

    result_ampgo = fit_transient_dmp_osc('pv', fwhm_init, t0_init, tau_init, period_init,
    method_glb='ampgo', bound_fwhm=bound_fwhm, bound_t0=bound_t0,
    bound_tau=bound_tau, bound_period=bound_period, t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_dmp_osc_3')
    load_result_ampgo = load_TransientResult('test_driver_transient_dmp_osc_3')
    os.remove('test_driver_transient_dmp_osc_3.h5')

    assert np.allclose(result_ampgo['x'][-6:], ans[-6:], rtol=1e-2)
    assert np.allclose(result_ampgo['c'][0], abs_osc.T, rtol=5e-2)
    assert str(result_ampgo) == str(load_result_ampgo)

