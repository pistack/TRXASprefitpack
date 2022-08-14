# pylint: disable = missing-module-docstring,wrong-import-position,invalid-name,multiple-statements
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import voigt, voigt_thy, edge_gaussian, edge_lorenzian
from TRXASprefitpack import fit_static_thy, fit_static_voigt
from TRXASprefitpack import save_StaticResult, load_StaticResult
from TRXASprefitpack import static_spectrum

rel = 1e-4
epsilon = 5e-8


def test_driver_static_voigt_1():
    ans = np.array([8987, 9000, 0.8, 0.9, 3, 9, 8992, 7])

    # set scan range
    e = np.linspace(8960, 9020, 160)

    # generate model spectrum
    base_line = 1e-1
    model_static = 0.1*voigt(e-ans[0], ans[2], ans[4]) + \
        0.7*voigt(e-ans[1], ans[3], ans[5]) + \
            0.2*edge_gaussian(e-ans[6], ans[7]) + \
                base_line
    eps_static = np.ones_like(model_static)/1000

    e0_init = np.empty(2)
    fwhm_G_init = np.empty(2)
    fwhm_L_init = np.empty(2)
    e0_edge_init = np.empty(1)
    fwhm_edge_init = np.empty(1)

    e0_init[0] = np.random.uniform(8984, 8990)
    e0_init[1] = np.random.uniform(8991, 9009)
    fwhm_G_init[0] = np.random.uniform(0.4, 1.6)
    fwhm_G_init[1] = np.random.uniform(0.45, 1.8)
    fwhm_L_init[0] = np.random.uniform(1.5, 6)
    fwhm_L_init[1] = np.random.uniform(4.5, 18)
    e0_edge_init[0] = np.random.uniform(8978, 9006)
    fwhm_edge_init[0] = np.random.uniform(3.5, 14)

    bound_e0 = [(8984, 8990), (8991, 9009)]
    bound_fwhm_G = [(0.4, 1.6), (0.45,  1.8)]
    bound_fwhm_L = [(1.5, 6), (4.5, 18)]
    bound_e0_edge = [(8978, 9006)]
    bound_fwhm_edge = [(3.5, 14)]

    result_ampgo = fit_static_voigt(e0_init, fwhm_G_init,
    fwhm_L_init, edge='g', edge_pos_init=e0_edge_init,
    edge_fwhm_init=fwhm_edge_init, base_order=0,
    method_glb='ampgo',
    bound_e0=bound_e0, bound_fwhm_G=bound_fwhm_G,
    bound_fwhm_L=bound_fwhm_L, bound_edge_pos=bound_e0_edge,
    bound_edge_fwhm=bound_fwhm_edge, e=e, intensity=model_static,
    eps=eps_static)

    static_ampgo = static_spectrum(e, result_ampgo)
    deriv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) -
    static_spectrum(e-epsilon*e, result_ampgo))/(2*epsilon*e)
    dderiv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) +
    static_spectrum(e-epsilon*e, result_ampgo)-2*static_ampgo)/(epsilon*e)**2


    cond_sol_ampgo = np.allclose(result_ampgo['x'], ans)
    cond_static_ampgo = np.allclose(static_ampgo, model_static-base_line)
    cond_deriv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=1),
    deriv_static_ampgo)
    cond_dderiv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=2),
    dderiv_static_ampgo)

    save_StaticResult(result_ampgo, 'test_driver_static_voigt_1')
    load_result_ampgo = load_StaticResult('test_driver_static_voigt_1')
    os.remove('test_driver_static_voigt_1.h5')

    assert (cond_sol_ampgo, cond_static_ampgo) == (True, True)
    assert (cond_deriv_ampgo, cond_dderiv_ampgo) == (True, True)
    assert np.allclose(result_ampgo['x'], load_result_ampgo['x']) is True
    assert str(result_ampgo) == str(load_result_ampgo)

def test_driver_static_voigt_2():
    ans = np.array([8987, 9000, 0.8, 0.9, 3, 9, 8992, 7])

    # set scan range
    e = np.linspace(8960, 9020, 160)

    # generate model spectrum
    base_line = 1e-1
    model_static = 0.1*voigt(e-ans[0], ans[2], ans[4]) + \
        0.7*voigt(e-ans[1], ans[3], ans[5]) + \
            0.2*edge_lorenzian(e-ans[6], ans[7]) + \
                base_line
    eps_static = np.ones_like(model_static)/1000

    e0_init = np.empty(2)
    fwhm_G_init = np.empty(2)
    fwhm_L_init = np.empty(2)
    e0_edge_init = np.empty(1)
    fwhm_edge_init = np.empty(1)

    e0_init[0] = np.random.uniform(8984, 8990)
    e0_init[1] = np.random.uniform(8991, 9009)
    fwhm_G_init[0] = np.random.uniform(0.4, 1.6)
    fwhm_G_init[1] = np.random.uniform(0.45, 1.8)
    fwhm_L_init[0] = np.random.uniform(1.5, 6)
    fwhm_L_init[1] = np.random.uniform(4.5, 18)
    e0_edge_init[0] = np.random.uniform(8978, 9006)
    fwhm_edge_init[0] = np.random.uniform(3.5, 14)

    bound_e0 = [(8984, 8990), (8991, 9009)]
    bound_fwhm_G = [(0.4, 1.6), (0.45,  1.8)]
    bound_fwhm_L = [(1.5, 6), (4.5, 18)]
    bound_e0_edge = [(8978, 9006)]
    bound_fwhm_edge = [(3.5, 14)]

    result_ampgo = fit_static_voigt(e0_init, fwhm_G_init,
    fwhm_L_init, edge='l', edge_pos_init=e0_edge_init,
    edge_fwhm_init=fwhm_edge_init, base_order=0,
    method_glb='ampgo',
    bound_e0=bound_e0, bound_fwhm_G=bound_fwhm_G,
    bound_fwhm_L=bound_fwhm_L, bound_edge_pos=bound_e0_edge,
    bound_edge_fwhm=bound_fwhm_edge, e=e, intensity=model_static,
    eps=eps_static)

    static_ampgo = static_spectrum(e, result_ampgo)
    deriv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) -
    static_spectrum(e-epsilon*e, result_ampgo))/(2*epsilon*e)
    dderiv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) +
    static_spectrum(e-epsilon*e, result_ampgo)-2*static_ampgo)/(epsilon*e)**2


    cond_sol_ampgo = np.allclose(result_ampgo['x'], ans)
    cond_static_ampgo = np.allclose(static_ampgo, model_static-base_line)
    cond_deriv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=1),
    deriv_static_ampgo)
    cond_dderiv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=2),
    dderiv_static_ampgo)

    save_StaticResult(result_ampgo, 'test_driver_static_voigt_2')
    load_result_ampgo = load_StaticResult('test_driver_static_voigt_2')
    os.remove('test_driver_static_voigt_2.h5')

    assert (cond_sol_ampgo, cond_static_ampgo) == (True, True)
    assert (cond_deriv_ampgo, cond_dderiv_ampgo) == (True, True)
    assert np.allclose(result_ampgo['x'], load_result_ampgo['x']) is True
    assert str(result_ampgo) == str(load_result_ampgo)

def test_driver_static_thy_1():
    ans = np.array([0.3, 0.5, 862.5, 863, 860.5, 862, 1, 1.5])
    mixing = np.array([0.3, 0.7])
    mixing_edge = np.array([0.3, 0.7])
    thy_peak = np.empty(2, dtype=object)
    thy_peak[0] = np.genfromtxt(path+'/'+'Ni_example_1.stk')
    thy_peak[1] = np.genfromtxt(path+'/'+'Ni_example_2.stk')

    # set scan range
    e = np.linspace(852.5, 865, 51)
    base_line = 5e-1

    # generate model spectrum
    model_static = mixing[0]*voigt_thy(e, thy_peak[0], ans[0], ans[1],
    ans[2], policy='shift')+\
        mixing[1]*voigt_thy(e, thy_peak[1], ans[0], ans[1],
        ans[3], policy='shift')+\
            mixing_edge[0]*edge_gaussian(e-ans[4], ans[6])+\
                mixing_edge[1]*edge_gaussian(e-ans[5], ans[7])+\
                    base_line

    eps_static = np.ones_like(model_static)/1000

    # set boundary
    bound_fwhm_G_thy = (0.15, 0.6)
    bound_fwhm_L_thy = (0.25, 1)
    bound_peak_shift = [(860.5, 863.0), (862, 864)]
    bound_e0_edge = [(859, 861), (861, 863)]
    bound_fwhm_edge = [(0.5, 2), (0.75, 3)]
    fwhm_G_thy_init = np.random.uniform(0.15, 0.6)
    fwhm_L_thy_init = np.random.uniform(0.25, 1)
    peak_shift_init = np.array([np.random.uniform(860.5, 863.0),
    np.random.uniform(862,864)])
    e0_edge_init = np.array([np.random.uniform(859, 861),
    np.random.uniform(861, 863)])
    fwhm_edge_init = np.array([np.random.uniform(0.5, 2),
    np.random.uniform(0.75, 3)])

    result_ampgo = fit_static_thy(thy_peak, fwhm_G_thy_init,
    fwhm_L_thy_init, 'shift', peak_shift=peak_shift_init,
     edge='g', edge_pos_init=e0_edge_init,
    edge_fwhm_init=fwhm_edge_init, base_order=0,
    method_glb='ampgo', bound_fwhm_G=bound_fwhm_G_thy,
    bound_fwhm_L=bound_fwhm_L_thy, bound_peak_shift=bound_peak_shift,
    bound_edge_pos=bound_e0_edge, bound_edge_fwhm=bound_fwhm_edge,
    e=e, intensity=model_static,
    eps=eps_static)

    static_ampgo = static_spectrum(e, result_ampgo)
    deriv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) -
    static_spectrum(e-epsilon*e, result_ampgo))/(2*epsilon*e)
    dderiv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) +
    static_spectrum(e-epsilon*e, result_ampgo)-2*static_ampgo)/(epsilon*e)**2


    cond_sol_ampgo = np.allclose(result_ampgo['x'], ans)
    cond_static_ampgo = np.allclose(static_ampgo, model_static-base_line)
    cond_deriv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=1),
    deriv_static_ampgo)
    cond_dderiv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=2),
    dderiv_static_ampgo, rtol=1e-4, atol=1e-2)

    save_StaticResult(result_ampgo, 'test_driver_static_thy_1')
    load_result_ampgo = load_StaticResult('test_driver_static_thy_1')
    os.remove('test_driver_static_thy_1.h5')

    assert (cond_sol_ampgo, cond_static_ampgo) == (True, True)
    assert (cond_deriv_ampgo, cond_dderiv_ampgo) == (True, True)
    assert np.allclose(result_ampgo['x'], load_result_ampgo['x']) is True
    assert str(result_ampgo) == str(load_result_ampgo)
    assert np.allclose(result_ampgo['thy_peak'][0], load_result_ampgo['thy_peak'][0]) is True
    assert np.allclose(result_ampgo['thy_peak'][1], load_result_ampgo['thy_peak'][1]) is True

def test_driver_static_thy_2():
    ans = np.array([0.3, 0.5, 862.5, 863, 860.5, 862, 1, 1.5])
    mixing = np.array([0.3, 0.7])
    mixing_edge = np.array([0.3, 0.7])
    thy_peak = np.empty(2, dtype=object)
    thy_peak[0] = np.genfromtxt(path+'/'+'Ni_example_1.stk')
    thy_peak[1] = np.genfromtxt(path+'/'+'Ni_example_2.stk')

    # set scan range
    e = np.linspace(852.5, 865, 51)
    base_line = 5e-1

    # generate model spectrum
    model_static = mixing[0]*voigt_thy(e, thy_peak[0], ans[0], ans[1],
    ans[2], policy='shift')+\
        mixing[1]*voigt_thy(e, thy_peak[1], ans[0], ans[1],
        ans[3], policy='shift')+\
            mixing_edge[0]*edge_lorenzian(e-ans[4], ans[6])+\
                mixing_edge[1]*edge_lorenzian(e-ans[5], ans[7])+\
                    base_line

    eps_static = np.ones_like(model_static)/1000

    # set boundary
    bound_fwhm_G_thy = (0.15, 0.6)
    bound_fwhm_L_thy = (0.25, 1)
    bound_peak_shift = [(860.5, 863.0), (862, 864)]
    bound_e0_edge = [(859, 861), (861, 863)]
    bound_fwhm_edge = [(0.5, 2), (0.75, 3)]
    fwhm_G_thy_init = np.random.uniform(0.15, 0.6)
    fwhm_L_thy_init = np.random.uniform(0.25, 1)
    peak_shift_init = np.array([np.random.uniform(860.5, 863.0),
    np.random.uniform(862,864)])
    e0_edge_init = np.array([np.random.uniform(859, 861),
    np.random.uniform(861, 863)])
    fwhm_edge_init = np.array([np.random.uniform(0.5, 2),
    np.random.uniform(0.75, 3)])

    result_ampgo = fit_static_thy(thy_peak, fwhm_G_thy_init,
    fwhm_L_thy_init, 'shift', peak_shift=peak_shift_init,
     edge='l', edge_pos_init=e0_edge_init,
    edge_fwhm_init=fwhm_edge_init, base_order=0,
    method_glb='ampgo', bound_fwhm_G=bound_fwhm_G_thy,
    bound_fwhm_L=bound_fwhm_L_thy, bound_peak_shift=bound_peak_shift,
    bound_edge_pos=bound_e0_edge, bound_edge_fwhm=bound_fwhm_edge,
    e=e, intensity=model_static,
    eps=eps_static)

    static_ampgo = static_spectrum(e, result_ampgo)
    deriv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) -
    static_spectrum(e-epsilon*e, result_ampgo))/(2*epsilon*e)
    dderiv_static_ampgo = (static_spectrum(e+epsilon*e, result_ampgo) +
    static_spectrum(e-epsilon*e, result_ampgo)-2*static_ampgo)/(epsilon*e)**2

    cond_sol_ampgo = np.allclose(result_ampgo['x'], ans)
    cond_static_ampgo = np.allclose(static_ampgo, model_static-base_line)
    cond_deriv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=1),
    deriv_static_ampgo)
    cond_dderiv_ampgo = np.allclose(static_spectrum(e, result_ampgo, deriv_order=2),
    dderiv_static_ampgo, rtol=1e-4, atol=1e-2)

    save_StaticResult(result_ampgo, 'test_driver_static_thy_1')
    load_result_ampgo = load_StaticResult('test_driver_static_thy_1')
    os.remove('test_driver_static_thy_1.h5')

    assert (cond_sol_ampgo, cond_static_ampgo) == (True, True)
    assert (cond_deriv_ampgo, cond_dderiv_ampgo) == (True, True)
    assert np.allclose(result_ampgo['x'], load_result_ampgo['x']) is True
    assert str(result_ampgo) == str(load_result_ampgo)
    assert np.allclose(result_ampgo['thy_peak'][0], load_result_ampgo['thy_peak'][0]) is True
    assert np.allclose(result_ampgo['thy_peak'][1], load_result_ampgo['thy_peak'][1]) is True

