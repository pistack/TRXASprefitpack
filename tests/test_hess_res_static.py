# pylint: disable = missing-module-docstring,wrong-import-position,invalid-name
import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.mathfun.deriv_check import check_num_hess
from TRXASprefitpack import residual_voigt
from TRXASprefitpack import res_hess_voigt
from TRXASprefitpack import voigt
from TRXASprefitpack import edge_gaussian, edge_lorenzian


def test_res_hess_voigt_1():
    e0_1 = 8987
    e0_2 = 9000
    e0_edge = 8992
    fwhm_G_1 = 0.8
    fwhm_G_2 = 0.9
    fwhm_L_1 = 3
    fwhm_L_2 = 9
    fwhm_edge = 7

    # set scan range
    e = np.linspace(8960, 9020, 160)

    # generate model spectrum
    model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
        0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
            0.2*edge_gaussian(e-e0_edge, fwhm_edge)
    eps_static = np.ones_like(model_static)

    x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

    ref_hess = check_num_hess(lambda e0_1, e0_2, fwhm_G_1, \
    fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge:
    np.sum(residual_voigt(np.array([e0_1, e0_2, 
    fwhm_G_1, fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge]), 
    2, 'g', 1, None, e=e, intensity=model_static, eps=eps_static)**2)/2,
    x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7])

    hess_tst = res_hess_voigt(x0, 2, 'g', 1,
    None, e=e, intensity=model_static, eps=eps_static)

    print('ref')
    print(ref_hess)
    print('tst')
    print(hess_tst)
    assert np.allclose(ref_hess, hess_tst, rtol=1e-4, atol=1e-6)

def test_res_hess_voigt_2():
    e0_1 = 8987
    e0_2 = 9000
    e0_edge = 8992
    fwhm_G_1 = 0.8
    fwhm_G_2 = 0.9
    fwhm_L_1 = 3
    fwhm_L_2 = 9
    fwhm_edge = 7

    # set scan range
    e = np.linspace(8960, 9020, 160)

    # generate model spectrum
    model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
        0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
            0.2*edge_lorenzian(e-e0_edge, fwhm_edge)
    eps_static = np.ones_like(model_static)

    x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

    ref_hess = check_num_hess(lambda e0_1, e0_2, fwhm_G_1, \
    fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge:
    np.sum(residual_voigt(np.array([e0_1, e0_2, 
    fwhm_G_1, fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge]), 
    2, 'l', 1, None, e=e, intensity=model_static, eps=eps_static)**2)/2,
    x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7])

    hess_tst = res_hess_voigt(x0, 2, 'l', 1,
    None, e=e, intensity=model_static, eps=eps_static)

    assert np.allclose(ref_hess, hess_tst, rtol=1e-4, atol=1e-6)

def test_res_hess_voigt_3():
    e0_1 = 8987
    e0_2 = 9000
    e0_edge = 8992
    fwhm_G_1 = 0.8
    fwhm_G_2 = 0.9
    fwhm_L_1 = 3
    fwhm_L_2 = 9
    fwhm_edge = 7

    # set scan range
    e = np.linspace(8960, 9020, 160)

    # generate model spectrum
    model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
        0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
            0.2*edge_gaussian(e-e0_edge, fwhm_edge)+\
            3e-5*(e-e0_edge)**2+1e-3*(e-e0_edge)+1e-1
    eps_static = np.ones_like(model_static)

    x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

    ref_hess = check_num_hess(lambda e0_1, e0_2, fwhm_G_1, \
    fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge:
    np.sum(residual_voigt(np.array([e0_1, e0_2, 
    fwhm_G_1, fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge]), 
    2, 'g', 1, 2, e=e, intensity=model_static, eps=eps_static)**2)/2,
    x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7])

    hess_tst = res_hess_voigt(x0, 2, 'g', 1,
    2, e=e, intensity=model_static, eps=eps_static)

    print('ref')
    print(ref_hess)
    print('tst')
    print(hess_tst)
    print(np.allclose(ref_hess, hess_tst, rtol=1e-4, atol=1e-6))

def test_res_hess_voigt_4():
    e0_1 = 8987
    e0_2 = 9000
    e0_edge = 8992
    fwhm_G_1 = 0.8
    fwhm_G_2 = 0.9
    fwhm_L_1 = 3
    fwhm_L_2 = 9
    fwhm_edge = 7

    # set scan range
    e = np.linspace(8960, 9020, 160)

    # generate model spectrum
    model_static = 0.1*voigt(e-e0_1, fwhm_G_1, fwhm_L_1) + \
        0.7*voigt(e-e0_2, fwhm_G_2, fwhm_L_2) + \
            0.2*edge_lorenzian(e-e0_edge, fwhm_edge)+\
            3e-5*(e-e0_edge)**2+1e-3*(e-e0_edge)+1e-1
    eps_static = np.ones_like(model_static)

    x0 = [8985, 8997, 0.7, 0.8, 2, 7, 8990, 9]

    ref_hess = check_num_hess(lambda e0_1, e0_2, fwhm_G_1, \
    fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge:
    np.sum(residual_voigt(np.array([e0_1, e0_2, 
    fwhm_G_1, fwhm_G_2, fwhm_L_1, fwhm_L_2, e0_edge, fwhm_edge]), 
    2, 'l', 1, 2, e=e, intensity=model_static, eps=eps_static)**2)/2,
    x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7])

    hess_tst = res_hess_voigt(x0, 2, 'l', 1,
    2, e=e, intensity=model_static, eps=eps_static)

    assert np.allclose(ref_hess, hess_tst, rtol=1e-4, atol=1e-6)
