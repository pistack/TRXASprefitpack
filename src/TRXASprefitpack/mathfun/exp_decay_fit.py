'''
exp_conv_irf:
submodule for fitting data with sum of exponential decay or damped oscillation convolved with irf

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Tuple, Optional
import numpy as np
import scipy.linalg as LA

from .rate_eq import compute_signal_irf
from .A_matrix import make_A_matrix_exp, make_A_matrix_dmp_osc


def exp_conv(t: np.ndarray, fwhm: float, tau: np.ndarray,
             c: np.ndarray, base: Optional[bool] = True,
             irf: Optional[str] = 'g', eta: Optional[float] = None) -> np.ndarray:

    '''
    Constructs the model for the convolution of n exponential and
    instrumental response function
    Supported instrumental response function are

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
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile)

    Returns:
      Convolution of the sum of n exponential decays and instrumental
      response function.

    Note:
     Size of weight `c` is `num_comp+1` when base is set to true.
     Otherwise, its size is `num_comp`.
    '''
    A = make_A_matrix_exp(t, fwhm, tau, base, irf, eta)
    y = c@A

    return y


def fact_anal_exp_conv(t: np.ndarray,
                       fwhm: float,
                       tau: np.ndarray,
                       base: Optional[bool] = True,
                       irf: Optional[str] = 'g',
                       eta: Optional[float] = None,
                       intensity: Optional[np.ndarray] = None,
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
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile)
       intensity: intensity of time scan data to fit
       eps: standard error of data

    Returns:
     Best coefficient for given fwhm and tau, if base is set to `True` then
     size of coefficient is `num_comp + 1`, otherwise is `num_comp`.

    Note:
     the dimension of the intensity must be one.
    '''

    A = make_A_matrix_exp(t, fwhm, tau, base, irf, eta)
    if eps is None:
        eps = np.ones_like(intensity)
    
    y = intensity/eps
    A = np.einsum('j,ij->ij', 1/eps, A)
    c, _, _, _ = LA.lstsq(A.T, y, cond=1e-2)

    return c

def rate_eq_conv(t: np.ndarray, fwhm: float,
abs: np.ndarray, eigval: np.ndarray, V: np.ndarray, c: np.ndarray, 
irf: Optional[str] = 'g', eta: Optional[float] = None) -> np.ndarray:

    '''
    Constructs signal model rate equation with
    instrumental response function
    Supported instrumental response function are

      * g: gaussian distribution
      * c: cauchy distribution
      * pv: pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       abs: coefficient for each excited state
       eigval: eigenvalue of rate equation matrix 
       V: eigenvector of rate equation matrix 
       c: coefficient to match initial condition of rate equation
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
      Convolution of the solution of the rate equation and instrumental
      response function.
    '''
    A = compute_signal_irf(t, eigval, V, c, fwhm, irf, eta)
    y = abs@A

    return y

def fact_anal_rate_eq_conv(t: np.ndarray, fwhm: float,
eigval: np.ndarray, V: np.ndarray, c: np.ndarray, 
exclude: Optional[str] = None, irf: Optional[str] = 'g',
eta: Optional[float] = None, intensity: Optional[np.ndarray] = None, 
eps: Optional[np.ndarray] = None) -> np.ndarray:

    '''
    Estimate the best coefficiets when full width at half maximum fwhm
    and eigenvector and eigenvalue of rate equation matrix are given

    Supported instrumental response functions are 

       1. 'g': gaussian distribution
       2. 'c': cauchy distribution
       3. 'pv': pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       eigval: eigenvalue of rate equation matrix 
       V: eigenvector of rate equation matrix 
       c: coefficient to match initial condition of rate equation
       exclude: exclude either 'first' or 'last' element or both 'first' and 'last' element.
                
                * 'first' : exclude first element
                * 'last' : exclude last element
                * 'first_and_last' : exclude both first and last element  
                * None : Do not exclude any element [default]
       irf: shape of instrumental
            response function [default: g]

              * 'g': normalized gaussian distribution,
              * 'c': normalized cauchy distribution,
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile)
       intensity: intensity of time scan data to fit
       eps: standard error of data

    Returns:
     Best coefficient for each component.

    Note:
     1. eigval, V, c should be obtained from solve_model
     2. The dimension of the intensity should be one.
    '''

    A = compute_signal_irf(t, eigval, V, c, fwhm, irf, eta)
    
    abs = np.zeros(A.shape[0])

    if eps is None:
        eps = np.ones_like(intensity)
    
    y = intensity/eps
    
    if exclude == 'first_and_last':
        B = np.einsum('j,ij->ij', 1/eps, A[1:-1,:])
    elif exclude == 'first':
        B = np.einsum('j,ij->ij', 1/eps, A[1:,:])
    elif exclude == 'last':
        B = np.einsum('j,ij->ij', 1/eps, A[:-1,:])
    else:
        B = np.einsum('j,ij->ij', 1/eps, A)
    
    coeff, _, _, _ = LA.lstsq(B.T, y, cond=1e-2)

    if exclude == 'first_and_last':
        abs[1:-1] = coeff
    elif exclude == 'first':
        abs[1:] = coeff
    elif exclude == 'last':
        abs[:-1] = coeff
    else:
        abs = coeff

    return abs


def dmp_osc_conv(t: np.ndarray, fwhm: float,
                      tau: np.ndarray,
                      T: np.ndarray,
                      phase: np.ndarray,
                      c: np.ndarray,
                      irf: Optional[str] = 'g',
                      eta: Optional[float] = None
                      ) -> np.ndarray:

    '''
    Constructs convolution of sum of damped oscillation and
    instrumental response function
    Supported instrumental response function are

      * g: gaussian distribution
      * c: cauchy distribution
      * pv: pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       tau: lifetime of vibration
       T: period of vibration
       phase: phase factor
       c: coefficient for each damping oscillation component
       irf: shape of instrumental
            response function [default: g]

              * 'g': normalized gaussian distribution,
              * 'c': normalized cauchy distribution,
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile)

    Returns:
      Convolution of sum of damped oscillation and instrumental
      response function.
    '''
    A = make_A_matrix_dmp_osc(t, fwhm, tau, T, phase, irf, eta)
    y = c@A

    return y

def fact_anal_dmp_osc_conv(t: np.ndarray,
                       fwhm: float,
                       tau: np.ndarray, T: np.ndarray, phase: np.ndarray,
                       irf: Optional[str] = 'g',
                       eta: Optional[float] = None,
                       intensity: Optional[np.ndarray] = None,
                       eps: Optional[np.ndarray] = None
                       ) -> np.ndarray:

    '''
    Estimate the best coefficiets when full width at half maximum fwhm
    , life constant tau, period of vibration T and phase factor are given

    Supported instrumental response functions are 

       1. 'g': gaussian distribution
       2. 'c': cauchy distribution
       3. 'pv': pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       tau: life time for each component
       T: period of vibration of each component
       phase: phase factor for each component
       irf: shape of instrumental
            response function [default: g]

              * 'g': normalized gaussian distribution,
              * 'c': normalized cauchy distribution,
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile)
       intensity: intensity of time scan data to fit
       eps: standard error of data

    Returns:
     Best coefficient for given damped oscillation component.

    Note:
     the dimension of the intensity should be one.
    '''
    
    A = make_A_matrix_dmp_osc(t, fwhm, tau, T, phase, irf, eta)

    if eps is None:
        eps = np.ones_like(intensity)
    
    y = intensity/eps
    A = np.einsum('j,ij->ij', 1/eps, A)
    c, _, _, _ = LA.lstsq(A.T, y, cond=1e-2)

    return c

def sum_exp_dmp_osc_conv(t: np.ndarray, fwhm: float,
                         tau: np.ndarray,
                         tau_osc: np.ndarray,
                         T: np.ndarray,
                         phase: np.ndarray,
                         c: np.ndarray,
                         c_osc: np.ndarray,
                         base: Optional[bool] = True,
                         irf: Optional[str] = 'g',
                         eta: Optional[float] = None
                         ) -> np.ndarray:
    '''
    Constructs convolution of sum of exponential decay and damped oscillation and
    instrumental response function
    Supported instrumental response function are

      * g: gaussian distribution
      * c: cauchy distribution
      * pv: pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       tau: lifetime of decay
       tau_osc: lifetime of vibration
       T: period of vibration
       phase: phase factor
       c: coefficient for each decay component
       c_osc: coefficient for each vibrational component
       base: Whether or not use baseline feature for exponential decay component [default: True]
       irf: shape of instrumental
            response function [default: g]

              * 'g': normalized gaussian distribution,
              * 'c': normalized cauchy distribution,
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile)

    Returns:
      Convolution of sum of exponential decay and damped oscillation and instrumental
      response function.
    '''
    A = make_A_matrix_exp(t, fwhm, tau, base, irf, eta)
    A_osc = make_A_matrix_dmp_osc(t, fwhm, tau_osc, T, phase, irf, eta)
    y = c@A + c_osc@A_osc

    return y

def fact_anal_sum_exp_dmp_osc_conv(t: np.ndarray, fwhm: float,
                                   tau: np.ndarray, tau_osc: np.ndarray,
                                   T: np.ndarray, phase: np.ndarray,
                                   base: Optional[bool] = True,
                                   irf: Optional[str] = 'g',
                                   eta: Optional[float] = None,
                                   intensity: Optional[np.ndarray] = None,
                                   eps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Estimate the best coefficiets when full width at half maximum fwhm
    , lifetime constant of decay component tau, lifetime constant of 
    oscillation component tau_osc,
    period of vibration T and phase factor are given

    Supported instrumental response functions are 

       1. 'g': gaussian distribution
       2. 'c': cauchy distribution
       3. 'pv': pseudo voigt profile

    Args:
       t: time
       fwhm: full width at half maximum of instrumental response function
       tau: life time for each decay component
       tau_osc: life time for each vibration component
       base: Whether or not use baseline feature for exponential decay component
        [default: True]
       T: period of vibration of each component
       phase: phase factor for each component
       irf: shape of instrumental
            response function [default: g]

              * 'g': normalized gaussian distribution,
              * 'c': normalized cauchy distribution,
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile)
       intensity: intensity of time scan data to fit
       eps: standard error of data

    Returns:
     Tuple of best coefficient for given decay and damped oscillation component.
     (c_(decay), c_(osc))

    Note:
     the dimension of the intensity should be one.
    '''

    A = np.empty((tau.size+1*base+tau_osc.size, t.size))
    A[:tau.size+1*base, :] = make_A_matrix_exp(t, fwhm, tau, base, irf, eta)
    A[tau.size+1*base:, :] = make_A_matrix_dmp_osc(t, fwhm, tau_osc, T, phase, irf, eta)

    if eps is None:
        eps = np.ones_like(intensity)
    
    y = intensity/eps
    A = np.einsum('j,ij->ij', 1/eps, A)
    c, _, _, _ = LA.lstsq(A.T, y, cond=1e-2)

    return c[:tau.size+1*base], c[tau.size+1*base:]

