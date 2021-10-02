'''
exp_conv_irf:
submodule for fitting data with sum of exponential decay convolved with irf

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

import numpy as np
import scipy.linalg as LA
from .A_matrix import make_A_matrix_cauchy
from .A_matrix import make_A_matrix_gau, make_A_matrix_pvoigt


def model_n_comp_conv(t, fwhm, tau, c, base=True, irf='g', eta=None):

    '''
    model for n component fitting
    n exponential function convolved with irf; 'g': normalized gaussian distribution, 'c': normalized cauchy distribution, 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`

    :param numpy_1d_array t: time
    :param numpy_1d_array fwhm: fwhm of X-ray temporal pulse, if irf == 'g' or 'c' then fwhm = [fwhm], if irf == 'pv' then fwhm = [fwhm_G, fwhm_L]
    :param numpy_1d_array tau: life time for each component
    :param numpy_1d_array c: coefficient
     (num_comp+1,) if base=True
     (num_comp,)   if base=False
    :param base: whether or not include baseline [default: True]
    :type base: bool, optional
    :param irf: shape of instrumental response function [default: g], 
     'g': normalized gaussian distribution, 
     'c': normalized cauchy distribution, 
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
    :param numpy_1d_array fwhm: fwhm of X-ray temporal pulse, if irf == 'g' or 'c' then fwhm = [fwhm], if irf == 'pv' then fwhm = [fwhm_G, fwhm_L]
    :param numpy_1d_array tau: life time for each component
    :param irf: shape of instrumental response function [default: g], 
     'g': normalized gaussian distribution, 
     'c': normalized cauchy distribution, 
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
