# pylint: disable = missing-module-docstring, wrong-import-position
import os
import sys
import numpy as np
from scipy.optimize import approx_fprime

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+'/../src/')

from TRXASprefitpack.driver._shgo import _wrapper_shgo 

# Global minimum of SIX HUMP CAMELBACK
# f_glopt = -1.0316 at x_opt = (0.0898, -0.7126) and (-0.0898, 0.7126)
# Bound x0: [-5, 5], x1: [-5, 5]

f_opt = -1.0316


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

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum((1-x)**2) + 100*np.sum((x[1:]-x[:-1]**2)**2)

def griewangk(x):
    prod = 1
    tmp = np.cos(x/np.sqrt(np.array(list(range(1, x.size+1)))))
    for i in range(x.size):
        prod = prod*tmp[i]
    return  1+1/4000*np.sum(x**2) - prod

def grad_sphere(x):
    return 2*x

def grad_rosenbrock(x):
    part_1 = np.empty(x.size)
    part_2 = np.empty(x.size)
    part_1[0] = 0
    part_2[-1] = 0
    part_1[1:] = 200*(x[1:]-x[:-1]**2)
    part_2[:-1] = -400*x[:-1]*(x[1:]-x[:-1]**2)
    return 2*(x-1) + part_1 + part_2

def grad_griewangk(x):
    scale = np.sqrt(np.array(list(range(1, x.size+1))))
    tmp = np.cos(x/scale)
    prod = 1
    for i in range(x.size):
        prod = prod * tmp[i]
    return x/2000 + prod*np.sum(np.tan(x/scale)/scale)

def test_shgo_1():
    bounds = 5*[(-10, 10)]
    x0 = np.random.uniform(-100, 100, 5)
    res = _wrapper_shgo(sphere, x0,
    minimizer_kwargs={'bounds': bounds, 'jac': grad_sphere})

    assert np.allclose(res['fun'], 0)

def test_shgo_2():
    bounds = 5*[(-10, 10)]
    x0 = np.random.uniform(-10, 10, 5)
    res = _wrapper_shgo(rosenbrock, x0,
    minimizer_kwargs={'bounds': bounds, 'jac': grad_rosenbrock})

    assert np.allclose(res['fun'], 0, atol=1e-6)

def test_shgo_5():
    bounds = 2*[(-5, 5)]
    x0 = np.random.uniform(-5, 5, 2)
    res = _wrapper_shgo(six_hump_camelback, x0,
                        minimizer_kwargs={'bounds':bounds})

    assert np.allclose(res['fun'], f_opt, atol=1e-4)


def test_shgo_6():
    bounds = 2*[(-5, 5)]
    x0 = np.random.uniform(-5, 5, 2)
    res = _wrapper_shgo(six_hump_camelback, x0,
    minimizer_kwargs={'bounds': bounds, 'jac': grad_six_hump_camelback})

    assert  np.allclose(res['fun'], f_opt, atol=1e-4)


def test_shgo_7():
    bounds = 2*[(-5, 5)]
    x0 = np.random.uniform(-5, 5, 2)
    res = _wrapper_shgo(fun_grad_six_hump_camelback, x0,
    minimizer_kwargs={'jac': True, 'bounds': bounds})

    assert np.allclose(res['fun'], f_opt, atol=1e-4)
