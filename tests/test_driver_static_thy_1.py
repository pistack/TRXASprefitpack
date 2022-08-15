# pylint: disable = missing-module-docstring,wrong-import-position,invalid-name,multiple-statements
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack import voigt_thy, edge_gaussian
from TRXASprefitpack import fit_static_thy
from TRXASprefitpack import save_StaticResult, load_StaticResult
from TRXASprefitpack import static_spectrum

rel = 1e-4
epsilon = 5e-8

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


    save_StaticResult(result_ampgo, 'test_driver_static_thy_1')
    load_result_ampgo = load_StaticResult('test_driver_static_thy_1')
    os.remove('test_driver_static_thy_1.h5')

    assert np.allclose(result_ampgo['x'], ans)
    assert np.allclose(static_ampgo, model_static-base_line)
    assert np.allclose(static_spectrum(e, result_ampgo, deriv_order=1),
    deriv_static_ampgo)
    assert np.allclose(static_spectrum(e, result_ampgo, deriv_order=2),
    dderiv_static_ampgo, rtol=1e-4, atol=1e-2)
    assert np.allclose(result_ampgo['x'], load_result_ampgo['x'])
    assert str(result_ampgo) == str(load_result_ampgo)
    assert np.allclose(result_ampgo['thy_peak'][0], load_result_ampgo['thy_peak'][0])
    assert np.allclose(result_ampgo['thy_peak'][1], load_result_ampgo['thy_peak'][1])

