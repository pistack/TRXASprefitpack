# pylint: disable = missing-module-docstring, wrong-import-position
import os
import sys
import numpy as np
from scipy.optimize import approx_fprime

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.driver._ampgo import tunnel, grad_tunnel, fun_grad_tunnel
from TRXASprefitpack.driver._ampgo import check_vaild
from TRXASprefitpack import ampgo

# Global minimum of SIX HUMP CAMELBACK
# f_glopt = -1.0316 at x_opt = (0.0898, -0.7126) and (-0.0898, 0.7126)
# Bound x0: [-5, 5], x1: [-5, 5]

f_opt = -1.0316
bounds = [[-5, 5], [-5, 5]]


def six_hump_camelback(x):
    return (4-2.1*x[0]**2+x[0]**4/3)*x[0]**2 + \
        x[0]*x[1] + (-4+4*x[1]**2)*x[1]**2


def grad_six_hump_camelback(x):
    y0 = x[1] + 2*x[0]*(4-2.1*x[0]**2+x[0]**4/3) + \
        x[0]**3*(4*x[0]**2-12.6)/3
    y1 = x[0] + 8*x[1]*(2*x[1]**2 - 1)
    return np.array([y0, y1])


def fun_grad_six_hump_camelback(x):
    return six_hump_camelback(x), grad_six_hump_camelback(x)


def test_grad_six_hump_camelback():
    grad_ref = np.empty((10, 2))
    grad_tst = np.empty((10, 2))
    for i in range(10):
        x0 = np.random.uniform(-5, 5, 2)
        grad_ref[i, :] = approx_fprime(x0, six_hump_camelback, 5e-8)
        grad_tst[i, :] = grad_six_hump_camelback(x0)

    assert np.allclose(grad_ref, grad_tst, rtol=1e-3, atol=1e-6)


def test_ampgo_1():
    x0 = np.random.uniform(-5, 5, 2)
    res = ampgo(six_hump_camelback, x0)

    assert np.allclose(res['fun'], f_opt, atol=1e-4)


def test_ampgo_2():
    x0 = np.random.uniform(-5, 5, 2)
    res = ampgo(six_hump_camelback, x0,
    minimizer_kwargs={'jac': grad_six_hump_camelback})

    assert  np.allclose(res['fun'], f_opt, atol=1e-4)


def test_ampgo_3():
    x0 = np.random.uniform(-5, 5, 2)
    res = ampgo(fun_grad_six_hump_camelback, x0,
    minimizer_kwargs={'jac': True})

    assert np.allclose(res['fun'], f_opt, atol=1e-4)


def test_grad_tunnel():
    tabusize = 5
    tabulist = 5*[None]
    for i in range(tabusize):
        tabulist[i] = np.random.uniform(-5, 5, 2)

    vaild = False
    while not vaild:
        x0 = np.random.uniform(-5, 5, 2)
        vaild = check_vaild(x0, tabulist)

    ttf_args = (six_hump_camelback, f_opt, tabulist,
                grad_six_hump_camelback)

    grad_ref = approx_fprime(x0, lambda x0: tunnel(x0, *ttf_args), 5e-8)
    grad = grad_tunnel(x0, *ttf_args)
    assert np.allclose(grad, grad_ref)


def test_fun_grad_tunnel():
    tabusize = 5
    tabulist = 5*[None]
    for i in range(tabusize):
        tabulist[i] = np.random.uniform(-5, 5, 2)

    vaild = False
    while not vaild:
        x0 = np.random.uniform(-5, 5, 2)
        vaild = check_vaild(x0, tabulist)

    ttf_args_ref = (six_hump_camelback, f_opt, tabulist,
                    grad_six_hump_camelback)
    ttf_args_tst = (fun_grad_six_hump_camelback, f_opt, tabulist)

    ttf_ref = tunnel(x0, *ttf_args_ref)
    grad_ref = approx_fprime(x0, lambda x0: tunnel(x0, *ttf_args_ref), 5e-8)
    ttf_tst, grad_tst = fun_grad_tunnel(x0, *ttf_args_tst)

    assert np.allclose(ttf_tst, ttf_ref)
    assert np.allclose(grad_tst, grad_ref)
