# pylint: disable = missing-module-docstring,wrong-import-position,invalid-name,multiple-statements
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import solve_seq_model, rate_eq_conv
from TRXASprefitpack import fit_transient_exp
from TRXASprefitpack import save_TransientResult, load_TransientResult

def test_driver_transient_exp_same_t0_4():
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
    # Last element is ground state or long lived species
    abs_1_1 = [1, 1, 1, 0.3]
    abs_2_1 = [0.5, 0.8, 0.2, 0.4]
    abs_3_1 = [-0.5, 0.7, 0.9, 0.6]
    abs_4_1 = [0.6, 0.3, -1, 0.7]

    abs_1_2 = [1, 1, 0]
    abs_2_2 = [0.5, 0.8, 0]
    abs_3_2 = [-0.5, 0.7, 0]
    abs_4_2 = [0.6, 0.3, 0]

    t0 = np.random.uniform(-0.2, 0.2, 2) # perturb time zero of each scan

    # generate measured data
    y_obs_1_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_1_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_2_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_2_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_3_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_3_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')
    y_obs_4_1 = \
        rate_eq_conv(t_seq-t0[0], fwhm, abs_4_1, eigval_seq_1, V_seq_1, c_seq_1, irf='g')

    y_obs_1_2 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_1_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_2_2 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_2_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_3_2 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_3_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')
    y_obs_4_2 = \
        rate_eq_conv(t_seq-t0[1], fwhm, abs_4_2, eigval_seq_2, V_seq_2, c_seq_2, irf='g')

    # generate measured intensity
    i_obs_1 = np.vstack((y_obs_1_1, y_obs_2_1, y_obs_3_1, y_obs_4_1)).T
    i_obs_2 = np.vstack((y_obs_1_2, y_obs_2_2, y_obs_3_2, y_obs_4_2)).T

    eps_obs = np.ones_like(i_obs_1)

    tau_mask = [np.array([True, False, True, True, True]),
                np.array([False, True, True, False, False])]

    t = [t_seq, t_seq]
    intensity = [i_obs_1, i_obs_2]
    eps = [eps_obs/1000, eps_obs/1000]


    ans = np.array([fwhm, t0[0], t0[1], tau_1, tau_2, tau_3, tau_4])

    bound_fwhm = [(0.05, 0.2)]
    bound_t0 = [(-0.4, 0.4)]*2
    bound_tau = [(0.1, 1), (1, 10), (1, 100), (100, 10000)]
    fwhm_init = np.random.uniform(0.05, 0.2)
    t0_init = np.random.uniform(-0.4, 0.4, 2)
    tau_init = np.array([np.random.uniform(0.1, 1), np.random.uniform(1, 10),
    np.random.uniform(1, 100), np.random.uniform(100, 10000)])


    result_ampgo = fit_transient_exp('g', fwhm_init, t0_init, tau_init, True,
    method_glb='ampgo', tau_mask=tau_mask, 
    bound_fwhm=bound_fwhm, bound_t0=bound_t0, bound_tau=bound_tau,
    same_t0=True,
    t=t, intensity=intensity, eps=eps)

    save_TransientResult(result_ampgo, 'test_driver_transient_exp_t0_4')
    load_result_ampgo = load_TransientResult('test_driver_transient_exp_t0_4')
    os.remove('test_driver_transient_exp_t0_4.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert str(result_ampgo) == str(load_result_ampgo)



