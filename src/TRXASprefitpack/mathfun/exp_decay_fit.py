'''
exp_conv_irf:
submodule for fitting data with sum of exponential decay convolved with irf

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Union, Optional
import numpy as np
import scipy.linalg as LA
from .A_matrix import make_A_matrix_cauchy
from .A_matrix import make_A_matrix_gau, make_A_matrix_pvoigt


def model_n_comp_conv(t: np.ndarray,
                      fwhm: Union[float, np.ndarray],
                      tau: np.ndarray,
                      c: np.ndarray,
                      base: Optional[bool] = True,
                      irf: Optional[str] = 'g',
                      eta: Optional[float] = None
                      ) -> np.ndarray:

    '''
    Constructs the model for the convolution of n exponential and
    instrumental response function
    Supported instrumental response function are

    irf
      * g: gaussian distribution
      * c: cauchy distribution
      * pv: pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       tau: life time for each component
       c: coefficient for each component
       base: whether or not include baseline [default: True]
       irf: shape of instrumental
            response function [default: g]

              * 'g': normalized gaussian distribution,
              * 'c': normalized cauchy distribution,
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile,
            default value is guessed according to
            Journal of Applied Crystallography. 33 (6): 1311–1316.)

    Returns:
      Convolution of the sum of n exponential decays and instrumental
      response function.

    Note:
     1. *fwhm* For gaussian and cauchy distribution,
        only one value of fwhm is needed,
        so fwhm is assumed to be float
        However, for pseudo voigt profile,
        it needs two value of fwhm, one for gaussian part and
        the other for cauchy part.
        So, in this case,
        fwhm is assumed to be numpy.ndarray with size 2.
     2. *c* size of c is assumed to be
        num_comp+1 when base is set to true.
        Otherwise, it is assumed to be num_comp.
    '''
    k = 1/tau
    if base:
        k = np.concatenate((k, np.array([0])))

    if irf == 'g':
        A = make_A_matrix_gau(t, fwhm, k)
    elif irf == 'c':
        A = make_A_matrix_cauchy(t, fwhm, k)
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


def fact_anal_exp_conv(t: np.ndarray,
                       fwhm: Union[float, np.ndarray],
                       tau: np.ndarray,
                       base: Optional[bool] = True,
                       irf: Optional[str] = 'g',
                       eta: Optional[float] = None,
                       data: Optional[np.ndarray] = None,
                       eps: Optional[np.ndarray] = None
                       ) -> np.ndarray:

    '''
    Estimate the best coefficiets when full width at half maximum fwhm
    and life constant tau are given
    
    When you fits your model to tscan data, you need to have
    good initial guess for not only life time of
    each component but also coefficients. To help this it solves
    linear least square problem to find best coefficients when fwhm and
    tau are given.

    Supported instrumental response functions are 

    irf
       1. 'g': gaussian distribution
       2. 'c': cauchy distribution
       3. 'pv': pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       tau: life time for each component
       base: whether or not include baseline [default: True]
       irf: shape of instrumental
            response function [default: g]

              * 'g': normalized gaussian distribution,
              * 'c': normalized cauchy distribution,
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile,
            default value is guessed according to
            Journal of Applied Crystallography. 33 (6): 1311–1316.)
       data: time scan data to fit
       eps: standard error of data

    Returns:
     Best coefficient for given fwhm and tau, if base is set to `True` then
     size of coefficient is `num_comp + 1`, otherwise is `num_comp`.

    Note:
     data should not contain time range and
     the dimension of the data must be one.
    '''

    k = 1/tau
    if base:
        k = np.concatenate((k, np.array([0])))

    if irf == 'g':
        A = make_A_matrix_gau(t, fwhm, k)
    elif irf == 'c':
        A = make_A_matrix_cauchy(t, fwhm, k)
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
