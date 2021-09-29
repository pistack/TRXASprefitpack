# mathfun.py
# Author: Pistack
# Date: 2021. 08. 29.
# Email: pistatex@yonsei.ac.kr
# Part of XAS-pre-fit-pack package
# Useful mathematical function for pre-fitting

import numpy as np
import scipy.linalg as LA
from scipy.special import erf, expi


def exp_conv_gau(t, fwhm, k):

    '''
    Compute exponential function convolved with normalized gaussian 
    distribution

    Note.
    We assume temporal pulse of x-ray is normalized gaussian distribution
    
    .. math::

       \sigma = \\frac{fwhm}{2\sqrt{2\log{2}}}


    .. math::

       IRF(t) = \\frac{1}{\sigma \sqrt{2\pi}}\exp\\left(-\\frac{t^2}{2\sigma^2}\\right)


    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param k: rate constant (inverse of life time)
    :type k: float

    
    :return: convolution of normalized gaussian distribution and exp(-kt)

    .. math::

       \\frac{1}{2}\exp\\left(\\frac{k^2}{2\sigma^2}-kt\\right)\\left(1+{erf}\\left(\\frac{1}{\sqrt{2}}\\left(\\frac{t}{\sigma}-k\sigma\\right)\\right)\\right)


    :rtype: numpy_1d_array
    '''

    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    return 1/2*np.exp((k*sigma)**2/2.0-k*t) * \
        (1+erf((t/sigma-k*sigma)/np.sqrt(2.0)))


def exp_conv_cauchy(t, fwhm, k):

    '''
    Compute exponential function convolved with normalized cauchy 
    distribution

    Note.
    We assume temporal pulse of x-ray is normalized cauchy distribution

    .. math::
       \\gamma = \\frac{fwhm}{2}

    .. math::
       IRF(t) = \\frac{\\gamma}{\\pi}\\frac{1}{(x-t)^2+\\gamma^2}
    

    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param k: rate constant (inverse of life time)
    :type k: float

   
    :return: convolution of normalized cauchy distribution and exp(-kt)

    .. math::
       \\frac{\\exp(-kt)}{\\pi}\\Im\\left(\\exp(-ik\\gamma)\cdot\\left(i\\pi - {Ei}\\left(kt+ik\gamma\\right)\\right)\\right)


    :rtype: numpy_1d_array 
    '''

    gamma = fwhm/2
    if k == 0:
        ans = complex(0, 1)*(0.5+1/np.pi*np.arctan(t/gamma))
    else:
        ans = 1/np.pi*np.exp(-k*t)*np.exp(complex(0, -k*gamma)) * \
            (complex(0, np.pi)-expi(k*t+complex(0, k*gamma)))
    
    return ans.imag


def make_A_matrix(t, k):

    A = np.zeros((k.shape[0], t.shape[0]))
    for i in range(k.shape[0]):
        A[i, :] = np.exp(-k[i]*t)
    A = A*np.heaviside(t, 1)
    A[np.isnan(A)] = 0
    
    return A


def make_A_matrix_gau(t, fwhm, k):

    A = np.zeros((k.shape[0], t.shape[0]))
    for i in range(k.shape[0]):
        A[i, :] = exp_conv_gau(t, fwhm, k[i])
    A[np.isnan(A)] = 0
    
    return A


def make_A_matrix_cauchy(t, fwhm, k):

    A = np.zeros((k.shape[0], t.shape[0]))
    for i in range(k.shape[0]):
        A[i, :] = exp_conv_cauchy(t, fwhm, k[i])
    A[np.isnan(A)] = 0
    A[np.isinf(A)] = 0
    
    return A


def make_A_matrix_pvoigt(t, fwhm_G, fwhm_L, eta, k):
    return (1-eta)*make_A_matrix_gau(t, fwhm_G, k) + \
        eta*make_A_matrix_cauchy(t, fwhm_L, k)


def solve_model(equation, y0):

    '''
    Solve system of first order rate equation

    
    :param equation: matrix corresponding to model
    :type equation: numpy_nd_array
    :param y0: initial condition
    :type y0: numpy_1d_array

    
    :return: eigenvalue of equation
    :rtype: numpy_1d_array
    :return: eigenvectors for equation
    :rtype: numpy_nd_array
    :return: coefficient where y_0 = Vc
    :rtype: numpy_1d_array
    '''

    eigval, V = LA.eig(equation)
    c = LA.solve(V, y0)

    return eigval.real, V, c


def compute_model(t, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    Note: eigval, V, c should be obtained from solve_model

    
    :param t: time
    :type t: numpy_1d_array
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array

    
    :return: solution of rate equation
    :rtype: numpy_nd_array
    '''

    A = make_A_matrix(t, -eigval)
    y = (c * V) @ A
    return y


def compute_signal_gau(t, fwhm, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized gaussian distribution
    Note: eigval, V, c should be obtained from solve_model

    
    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array
  

    :return: 
     solution of rate equation convolved with normalized gaussian distribution
    :rtype: numpy_nd_array 
    '''

    A = make_A_matrix_gau(t, fwhm, -eigval)
    y_signal = (c * V) @ A
    return y_signal


def compute_signal_cauchy(t, fwhm, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized cauchy distribution
    Note: eigval, V, c should be obtained from solve_model

    
    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array
  

    :return: 
     solution of rate equation convolved with normalized cachy distribution
    :rtype: numpy_nd_array 
    '''

    A = make_A_matrix_cauchy(t, fwhm, -eigval)
    y_signal = (c * V) @ A
    return y_signal


def compute_signal_pvoigt(t, fwhm_G, fwhm_L, eta, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized pseudo voigt profile
    (:math:`pvoigt = (1-\\eta) G(t) + \\eta L(t)`,
    G(t): stands for normalized gaussian
    L(t): stands for normalized cauchy(lorenzian) distribution)

    Note: eigval, V, c should be obtained from solve_model

    :param t: time
    :type t: numpy_1d_array
    :param fwhm_G: 
     full width at half maximum of x-ray temporal pulse (gaussian part)
    :type fwhm_G: float
    :param fwhm_L:
     full width at half maximum of x-ray temporal pulse (lorenzian part)
    :type fwhm_L: float
    :param float eta: mixing parameter

    :math:`(0 < \\eta < 1)`

    :type eta: float
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array

    :return: 
     solution of rate equation convolved with normalized pseudo voigt profile
    :rtype: numpy_nd_array
    '''

    A = make_A_matrix_pvoigt(t, fwhm_G, fwhm_L, eta, -eigval)
    y_signal = (c * V) @ A
    return y_signal


def model_n_comp_conv(t, fwhm, tau, c, base=True, irf='g', eta=None):

    '''
    model for n component fitting
    n exponential function convolved with irf
    irf
    'g': normalized gaussian distribution

    'c': normalized cauchy distribution

    'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`


    :param numpy_1d_array t: time
    :param numpy_1d_array fwhm: fwhm of X-ray temporal pulse 
     if irf == 'g' or 'c' then
        fwhm = [fwhm]
     if irf == 'pv' then
        fwhm = [fwhm_G, fwhm_L]
    :param numpy_1d_array tau: life time for each component
    :param numpy_1d_array c: coefficient
     (num_comp+1,) if base=True
     (num_comp,)   if base=False
    :param base: whether or not include baseline [default: True]
    :type base: bool, optional
    :param irf: shape of instrumental response function [default: g]
     'g': normalized gaussian distribution
     'c': normalized cauchy distribution
     'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`

    :type irf: string, optional
    :param eta: mixing parameter for pseudo voigt profile 
     (only needed for pseudo voigt profile, Default value is guessed according to Journal of Applied Crystallography. 33 (6): 1311–1316.)
    :type eta: float, optional
    
    :return: fit
    :rtype: numpy_1d_array
    '''

    k = 1/tau
    if base:
        k = np.concatenate((k, np.array([0])))

    if irf == 'g':
        A = make_A_matrix_gau(t, fwhm[0], k)
    elif irf == 'c':
        A = make_A_matrix_cauchy(t, fwhm[0], k)
    elif irf == 'pv':
        if eta is None:
            f = fwhm[0]**5+2.69269*fwhm[0]**4*fwhm[1] + \
                2.42843*fwhm[0]**3*fwhm[1]**2 + \
                4.47163*fwhm[0]**2*fwhm[1]**3 + \
                0.07842*fwhm[0]*fwhm[1]**4 + \
                fwhm[1]**5
            f = f**(1/5)
            x = fwhm[1]/f
            eta = 1.36603*x-0.47719*x**2+0.11116*x**3
        A = make_A_matrix_pvoigt(t, fwhm[0], fwhm[1],
                                 eta, k)

    y = c@A

    return y


def fact_anal_exp_conv(t, fwhm, tau, irf='g', eta=None,
                       data=None, eps=None, base=True):
    
    '''
    Estimate the best coefficiets when fwhm and tau are given

    Before fitting with Escan data, you want fit your model to tscan data
    to know how many component needed for well description of tscan data
    To do this you have good initial guess for not only life time of
    each component but also coefficients.
    For this, It will solve linear least square problem

    :param numpy_1d_array t: time
    :param numpy_1d_array fwhm: fwhm of X-ray temporal pulse 
     if irf == 'g' or 'c' then
        fwhm = [fwhm]
     if irf == 'pv' then
        fwhm = [fwhm_G, fwhm_L]
    :param numpy_1d_array tau: life time for each component
    :param irf: shape of instrumental response function [default: g]
     'g': normalized gaussian distribution
     'c': normalized cauchy distribution
     'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
    :type irf: string, optional
    :param eta: mixing parameter for pseudo voigt profile 
     (only needed for pseudo voigt profile, Default value is guessed according to Journal of Applied Crystallography. 33 (6): 1311–1316.)
    :type eta: float, optional
    :param numpy_1d_array data: tscan data
    :param numpy_1d_array eps: error for tscan data 
    :param base: whether or not include baseline [default: True]
    :type base: bool, optional
    

    :return: best coefficient for given fwhm, tau
     (num_comp+1,) , if base=True
     (num_comp,)   , otherwise
    :rtype: numpy_1d_array
    '''

    k = 1/tau
    if base:
        k = np.concatenate((k, np.array([0])))

    if irf == 'g':
        A = make_A_matrix_gau(t, fwhm[0], k)
    elif irf == 'c':
        A = make_A_matrix_cauchy(t, fwhm[0], k)
    elif irf == 'pv':
        if eta is None:
            f = fwhm[0]**5+2.69269*fwhm[0]**4*fwhm[1] + \
                2.42843*fwhm[0]**3*fwhm[1]**2 + \
                4.47163*fwhm[0]**2*fwhm[1]**3 + \
                0.07842*fwhm[0]*fwhm[1]**4 + \
                fwhm[1]**5
            f = f**(1/5)
            x = fwhm[1]/f
            eta = 1.36603*x-0.47719*x**2+0.11116*x**3
        A = make_A_matrix_pvoigt(t, fwhm[0], fwhm[1], eta, k)

    if eps is not None:
        y = data/eps
        for i in range(k.shape[0]):
            A[i, :] = A[i, :]/eps
    else:
        y = data
    c, _, _, _ = LA.lstsq(A.T, y)

    return c
