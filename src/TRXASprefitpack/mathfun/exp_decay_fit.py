'''
exp_conv_irf:
submodule for fitting data with sum of exponential decay convolved with irf

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Tuple, Union, Optional
import numpy as np
import scipy.linalg as LA

from .rate_eq import compute_signal_irf
from .A_matrix import make_A_matrix_exp, make_A_matrix_dmp_osc


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
    A = make_A_matrix_exp(t, fwhm, tau, base, irf, eta)
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

    A = make_A_matrix_exp(t, fwhm, tau, base, irf, eta)
    if eps is None:
        eps = np.ones_like(data)
    
    y = data/eps
    A = np.einsum('j,ij->ij', 1/eps, A)
    c, _, _, _ = LA.lstsq(A.T, y)

    return c

def rate_eq_conv(t: np.ndarray,
                      fwhm: Union[float, np.ndarray],
                      abs: np.ndarray,
                      eigval: np.ndarray, V: np.ndarray, c: np.ndarray, 
                      irf: Optional[str] = 'g',
                      eta: Optional[float] = None
                      ) -> np.ndarray:

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

    Note:
        *fwhm* For gaussian and cauchy distribution,
        only one value of fwhm is needed,
        so fwhm is assumed to be float
        However, for pseudo voigt profile,
        it needs two value of fwhm, one for gaussian part and
        the other for cauchy part.
        So, in this case,
        fwhm is assumed to be numpy.ndarray with size 2.
    '''
    A = compute_signal_irf(t, eigval, V, c, fwhm, irf, eta)
    y = abs@A

    return y

def fact_anal_rate_eq_conv(t: np.ndarray, fwhm: Union[float, np.ndarray],
eigval: np.ndarray, V: np.ndarray, c: np.ndarray, 
exclude: Optional[str] = None, irf: Optional[str] = 'g',
eta: Optional[float] = None, data: Optional[np.ndarray] = None, 
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
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile,
            default value is guessed according to
            Journal of Applied Crystallography. 33 (6): 1311–1316.)
       data: time scan data to fit
       eps: standard error of data

    Returns:
     Best coefficient for each component.

    Note:
     1. eigval, V, c should be obtained from solve_model
     2. data should not contain time range and the dimension of the data must be one.
    '''

    A = compute_signal_irf(t, eigval, V, c, fwhm, irf, eta)
    
    abs = np.zeros(A.shape[0])

    if eps is None:
        eps = np.ones_like(data)
    
    y = data/eps
    
    if exclude == 'first_and_last':
        B = np.einsum('j,ij->ij', 1/eps, A[1:-1,:])
    elif exclude == 'first':
        B = np.einsum('j,ij->ij', 1/eps, A[1:,:])
    elif exclude == 'last':
        B = np.einsum('j,ij->ij', 1/eps, A[:-1,:])
    else:
        B = np.einsum('j,ij->ij', 1/eps, A)
    
    coeff, _, _, _ = LA.lstsq(B.T, y)

    if exclude == 'first_and_last':
        abs[1:-1] = coeff
    elif exclude == 'first':
        abs[1:] = coeff
    elif exclude == 'last':
        abs[:-1] = coeff
    else:
        abs = coeff

    return abs


def dmp_osc_conv(t: np.ndarray, fwhm: Union[float, np.ndarray],
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
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile,
            default value is guessed according to
            Journal of Applied Crystallography. 33 (6): 1311–1316.)

    Returns:
      Convolution of sum of damped oscillation and instrumental
      response function.

    Note:
        *fwhm* For gaussian and cauchy distribution,
        only one value of fwhm is needed,
        so fwhm is assumed to be float
        However, for pseudo voigt profile,
        it needs two value of fwhm, one for gaussian part and
        the other for cauchy part.
        So, in this case,
        fwhm is assumed to be numpy.ndarray with size 2.
    '''
    A = make_A_matrix_dmp_osc(t, fwhm, tau, T, phase, irf, eta)
    y = c@A

    return y

def fact_anal_dmp_osc_conv(t: np.ndarray,
                       fwhm: Union[float, np.ndarray],
                       tau: np.ndarray, T: np.ndarray, phase: np.ndarray,
                       irf: Optional[str] = 'g',
                       eta: Optional[float] = None,
                       data: Optional[np.ndarray] = None,
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
              * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
       eta: mixing parameter for pseudo voigt profile
            (only needed for pseudo voigt profile,
            default value is guessed according to
            Journal of Applied Crystallography. 33 (6): 1311–1316.)
       data: time scan data to fit
       eps: standard error of data

    Returns:
     Best coefficient for given damped oscillation component.

    Note:
     data should not contain time range and
     the dimension of the data must be one.
    '''
    
    A = make_A_matrix_dmp_osc(t, fwhm, tau, T, phase, irf, eta)

    if eps is None:
        eps = np.ones_like(data)
    
    y = data/eps
    A = np.einsum('j,ij->ij', 1/eps, A)
    c, _, _, _ = LA.lstsq(A.T, y)

    return c


def dads(escan_time: np.ndarray, fwhm: Union[float, np.ndarray], tau: np.ndarray, base: Optional[bool] = True,
irf: Optional[str] = 'g', eta: Optional[float] = None,
data: Optional[np.ndarray] = None, eps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate decay associated difference spectrum from experimental energy scan data

    Args:
      escan_time: time delay for each energy scan data
      fwhm: full width at half maximum of instrumental response function
      tau: life time for each component
      base: whether or not include baseline [default: True]
      irf: shape of instrumental response function [default: g]
           * 'g': normalized gaussian distribution,
           * 'c': normalized cauchy distribution,
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
      eta: mixing parameter for pseudo voigt profile
           (only needed for pseudo voigt profile, default value is guessed according to
           Journal of Applied Crystallography. 33 (6): 1311–1316.)
      data: energy scan data
      eps: standard error of data 
    
    Returns:
     Tuple of calculated decay associated difference spectrum of each component and estimated error
    
    Note:
     To calculate decay associated difference spectrum of n component exponential decay, you should measure at least n+1
     energy scan
    '''
    # initialization
    if base:
      c = np.empty((tau.size+1, data.shape[0]))
      dof = escan_time.size - (tau.size+1)
    else:
      c = np.empty((tau.size, data.shape[0]))
      dof = escan_time.size - (tau.size)

    if eps is None:
      eps = np.ones_like(data)
    
    c_eps = np.empty_like(c)
    
    A = make_A_matrix_exp(escan_time, fwhm, tau, base, irf)
    data_scaled = data/eps

    # evaluates dads
    for i in range(data.shape[0]):
      A_scaled = np.einsum('j,ij->ij', 1/eps[i,:], A)
      U, s, Vh = LA.svd(A_scaled.T, full_matrices=False)
      cov = Vh.T @ np.einsum('i,ij->ij', 1/s**2, Vh)
      c[:,i] = np.einsum('j,ij->ij', 1/s, Vh.T) @ (U.T @ data_scaled[i,:])
      res = data_scaled[i,:] - (c[:,i] @ A_scaled)
      red_chi2 = np.sum(res**2)/dof
      c_eps[:,i] = np.sqrt(red_chi2*np.diag(cov))


    return c, c_eps

def sads(escan_time: np.ndarray, fwhm: Union[float, np.ndarray], eigval: np.ndarray, V: np.ndarray, c: np.ndarray,
exclude: Optional[str] = None, irf: Optional[str] = 'g', eta: Optional[float] = None,
data: Optional[np.ndarray] = None, eps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate species associated difference spectrum from experimental energy scan data

    Args:
      escan_time: time delay for each energy scan data
      fwhm: full width at half maximum of instrumental response function
      eigval: eigenvalue of rate equation matrix 
      V: eigenvector of rate equation matrix 
      c: coefficient to match initial condition of rate equation
      exclude: exclude either 'first' or 'last' element or both 'first' and 'last' element.
               * 'first' : exclude first element
               * 'last' : exclude last element
               * 'first_and_last' : exclude both first and last element  
               * None : Do not exclude any element [default]
      irf: shape of instrumental response function [default: g]
           * 'g': normalized gaussian distribution,
           * 'c': normalized cauchy distribution,
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
      eta: mixing parameter for pseudo voigt profile (only needed for pseudo voigt profile,
           default value is guessed according to Journal of Applied Crystallography. 33 (6): 1311–1316.)
      data: energy scan data
      eps: standard error of data 
    
    Returns:
     Tuple of calculated species associated difference spectrum of each component and estimated error
    
    Note:
     1. eigval, V, c should be obtained from solve_model
     2. To calculate species associated difference spectrum of n excited state species, you should measure at least n+1 energy scan
     3. Difference spectrum of ground state is zero, so ground state species should be excluded from rate equation or via exclude option.
    '''
    # initialization
    if exclude is None:
      abs = np.empty((eigval.size, data.shape[0]))
      dof = escan_time.size - eigval.size
    elif exclude in ['first', 'last']:
      abs = np.empty((eigval.size-1, data.shape[0]))
      dof = escan_time.size - (eigval.size-1)
    else:
      abs = np.empty((eigval.size-2, data.shape[0]))
      dof = escan_time.size - (eigval.size-2)

    if eps is None:
      eps = np.ones_like(data)
    
    abs_eps = np.empty_like(abs)
    
    A = compute_signal_irf(escan_time, eigval, V, c, fwhm, irf)
    if exclude == 'first':
      B = A[1:, :]
    elif exclude == 'last':
      B = A[:-1, :]
    elif exclude == 'first_and_last':
      B = A[1:-1, :]
    else:
      B = A

    data_scaled = data/eps

    # evaluates sads
    for i in range(data.shape[0]):
      A_scaled = np.einsum('j,ij->ij', 1/eps[i,:], B)
      U, s, Vh = LA.svd(A_scaled.T, full_matrices= False)
      cov = Vh.T @ np.einsum('i,ij->ij', 1/s**2, Vh)
      abs[:,i] = np.einsum('j,ij->ij', 1/s, Vh.T) @ (U.T @ data_scaled[i,:])
      res = data_scaled[i,:] - (abs[:,i] @ A_scaled)
      red_chi2 = np.sum(res**2)/dof
      abs_eps[:,i] = np.sqrt(red_chi2*np.diag(cov))


    return abs, abs_eps

