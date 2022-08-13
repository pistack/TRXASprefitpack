import os
import sys
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path+"/../src/")

from TRXASprefitpack.mathfun.deriv_check import check_num_deriv
from TRXASprefitpack import calc_eta, deriv_eta
from TRXASprefitpack import calc_fwhm, deriv_fwhm
from TRXASprefitpack import pvoigt_irf
from TRXASprefitpack import voigt

def test_pvoigt_irf_1():
    '''
     Test pseudo voigt approximation (fwhm_L = 3 fwhm_G)
    '''
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.1; fwhm_L = 0.3
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    result = np.max(np.abs(tst-ref))/np.max(ref) < 2e-2
    assert result == True

def test_pvoigt_irf_2():
    '''
     Test pseudo voigt approximation (fwhm_L = 1.5 fwhm_G)
    '''
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.1; fwhm_L = 0.15
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    result = np.max(np.abs(tst-ref))/np.max(ref) < 2e-2
    assert result == True

def test_pvoigt_irf_3():
    '''
     Test pseudo voigt approximation (fwhm_L = 1 fwhm_G)
    '''
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.1; fwhm_L = 0.1
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    result = np.max(np.abs(tst-ref))/np.max(ref) < 2e-2
    assert result == True

def test_pvoigt_irf_4():
    '''
     Test pseudo voigt approximation (fwhm_G = 1.5 fwhm_L)
    '''
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.15; fwhm_L = 0.1
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    result = np.max(np.abs(tst-ref))/np.max(ref) < 2e-2
    assert result == True

def test_pvoigt_irf_5():
    '''
     Test pseudo voigt approximation (fwhm_G = 3 fwhm_L)
    '''
    t = np.linspace(-1, 1, 201)
    fwhm_G = 0.3; fwhm_L = 0.1
    fwhm = calc_fwhm(fwhm_G, fwhm_L)
    eta = calc_eta(fwhm_G, fwhm_L)
    ref = voigt(t, fwhm_G, fwhm_L)
    tst = pvoigt_irf(t, fwhm, eta)
    result = np.max(np.abs(tst-ref))/np.max(ref) < 2e-2
    assert result == True
    
def test_deriv_eta_1():
    '''
     Test gradient of mixing parameter eta (fwhm_L = 3 fwhm_G)
    '''
    fwhm_G = 0.1; fwhm_L = 0.3
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_eta_2():
    '''
     Test gradient of mixing parameter eta (fwhm_L = 1.5 fwhm_G)
    '''
    fwhm_G = 0.1; fwhm_L = 0.15
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_eta_3():
    '''
     Test gradient of mixing parameter eta (fwhm_L = fwhm_G)
    '''
    fwhm_G = 0.1; fwhm_L = 0.1
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_eta_4():
    '''
     Test gradient of mixing parameter eta (fwhm_G = 1.5 fwhm_L)
    '''
    fwhm_G = 0.15; fwhm_L = 0.1
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_eta_5():
    '''
     Test gradient of mixing parameter eta (fwhm_G = 3 fwhm_L)
    '''
    fwhm_G = 0.3; fwhm_L = 0.1
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_eta_6():
    '''
     Test gradient of mixing parameter eta (fwhm_L = 2 fwhm_G)
    '''
    fwhm_G = 0.15; fwhm_L = 0.3
    d_G, d_L = deriv_eta(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_eta, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_fwhm_1():
    '''
     Test gradient of unifrom fwhm parameter (fwhm_L = 3 fwhm_G)
    '''
    fwhm_G = 0.1; fwhm_L = 0.3
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_fwhm_2():
    '''
     Test gradient of unifrom fwhm parameter (fwhm_L = 1.5 fwhm_G)
    '''
    fwhm_G = 0.1; fwhm_L = 0.15
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_fwhm_3():
    '''
     Test gradient of unifrom fwhm parameter (fwhm_L = fwhm_G)
    '''
    fwhm_G = 0.1; fwhm_L = 0.1
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_fwhm_4():
    '''
     Test gradient of uniform fwhm parameter (fwhm_G = 1.5 fwhm_L)
    '''
    fwhm_G = 0.15; fwhm_L = 0.1
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_fwhm_5():
    '''
     Test gradient of unifrom fwhm parameter (fwhm_G = 3 fwhm_L)
    '''
    fwhm_G = 0.3; fwhm_L = 0.1
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result == True

def test_deriv_fwhm_6():
    '''
     Test gradient of unifrom fwhm parameter (fwhm_L = 2 fwhm_L)
    '''
    fwhm_G = 0.15; fwhm_L = 0.3
    d_G, d_L = deriv_fwhm(fwhm_G, fwhm_L)
    
    result = np.allclose(np.array([d_G, d_L]), 
    check_num_deriv(calc_fwhm, fwhm_G, fwhm_L))
    assert result == True
