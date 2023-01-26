'''
ads:
submodule for driver routine of associated difference spectrum

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Tuple
import numpy as np
from scipy.linalg import svd, lstsq
from ..mathfun.A_matrix import make_A_matrix_dmp_osc, make_A_matrix_exp
from ..mathfun.rate_eq import compute_signal_irf


def dads(escan_time: np.ndarray, fwhm: float, tau: np.ndarray, base: Optional[bool] = True,
         irf: Optional[str] = 'g', eta: Optional[float] = None,
         intensity: Optional[np.ndarray] = None, eps: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`

      eta: mixing parameter for pseudo voigt profile
           (only needed for pseudo voigt profile)
      intensity: intensity of energy scan dataset
      eps: standard error of dataset

    Returns:
     Tuple of calculated decay associated difference spectrum of each component, estimated error,
     and retrieved energy scan intensity from dads and decay components

    Note:
     To calculate decay associated difference spectrum of n component exponential decay, you should measure at least n+1
     energy scan
    '''
    # initialization
    if base:
        c = np.empty((tau.size+1, intensity.shape[0]))
        dof = escan_time.size - (tau.size+1)
    else:
        c = np.empty((tau.size, intensity.shape[0]))
        dof = escan_time.size - (tau.size)

    if eps is None:
        eps = np.ones_like(intensity)

    c_eps = np.empty_like(c)

    A = make_A_matrix_exp(escan_time, fwhm, tau, base, irf, eta)
    data_scaled = intensity/eps

    # evaluates dads
    cond = 1e-2
    for i in range(intensity.shape[0]):
        A_scaled = np.einsum('j,ij->ij', 1/eps[i, :], A)
        U, s, Vh = svd(A_scaled.T, full_matrices=False)
        mask = s > cond*s[0]
        U_turn = U[:, mask]
        s_turn = s[mask]
        Vh_turn = Vh[mask, :]
        c[:, i] = np.einsum('j,ij->ij', 1/s_turn,
                            Vh_turn.T) @ (U_turn.T @ data_scaled[i, :])
        res = data_scaled[i, :] - (c[:, i] @ A_scaled)
        if dof != 0:
            red_chi2 = np.sum(res**2)/dof
            cov_scale = red_chi2 *\
                Vh_turn.T @ np.einsum('i,ij->ij', 1/s_turn**2, Vh_turn)
            c_eps[:, i] = np.sqrt(np.diag(cov_scale))

    return c, c_eps, c.T @ A

def dads_svd(escan_time: np.ndarray, fwhm: float, tau: np.ndarray, base: Optional[bool] = True,
             irf: Optional[str] = 'g', eta: Optional[float] = None,
             intensity: Optional[np.ndarray] = None, cond_num: Optional[float] = 0) \
             -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate decay associated difference spectrum from experimental energy scan data
    (using svd)

    Args:
      escan_time: time delay for each energy scan data
      fwhm: full width at half maximum of instrumental response function
      tau: life time for each component
      base: whether or not include baseline [default: True]
      irf: shape of instrumental response function [default: g]

           * 'g': normalized gaussian distribution,
           * 'c': normalized cauchy distribution,
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`

      eta: mixing parameter for pseudo voigt profile
           (only needed for pseudo voigt profile)
      intensity: intensity of energy scan dataset
      cond_num: conditional number to turncate svd

    Returns:
     Tuple of calculated decay associated difference spectrum of each component,
     and retrieved energy scan intensity from dads and decay components

    Note:
     To calculate decay associated difference spectrum of n component exponential decay, you should measure at least n
     energy scan
    '''
    # svd of data matrix
    U_data, s_data, Vh_data = svd(intensity, full_matrices=False)
    N_survived = np.sum((s_data > cond_num*s_data[0]))
    U_data_trun = U_data[:, :N_survived]
    s_data_trun = s_data[:N_survived]
    Vh_turn = Vh_data[:N_survived, :]
    A = make_A_matrix_exp(escan_time, fwhm, tau, base, irf, eta)
    c, _, _, _ = lstsq(A.T, Vh_turn.T, cond=1e-2)
    coeff = np.einsum('j,ij->ij', s_data_trun, U_data_trun) @ c.T
    fit = coeff @ A

    return coeff, fit

def dads_osc(escan_time: np.ndarray, fwhm: float, tau: np.ndarray, 
             tau_osc: np.ndarray, period_osc: np.ndarray, base: Optional[bool] = True,
             irf: Optional[str] = 'g', eta: Optional[float] = None,
             intensity: Optional[np.ndarray] = None, eps: Optional[np.ndarray] = None) \
             -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate decay associated difference spectrum with damped oscillation 
    from experimental energy scan data

    Args:
      escan_time: time delay for each energy scan data
      fwhm: full width at half maximum of instrumental response function
      tau: life time for each component
      tau_osc: life time for each oscillation component
      period_osc: period of each oscillation
      base: whether or not include baseline [default: True]
      irf: shape of instrumental response function [default: g]

           * 'g': normalized gaussian distribution,
           * 'c': normalized cauchy distribution,
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`

      eta: mixing parameter for pseudo voigt profile
           (only needed for pseudo voigt profile)
      intensity: intensity of energy scan dataset
      eps: standard error of dataset

    Returns:
     1. decay associated differenece spectrum
     2. damped oscillation associated difference spectrum
     3. estimated error of dads
     4. estimated error of doads
     5. retrieved energy scan from dads
     6. retrieved energy scan from doads

    Note:
     1. To calculate decay associated difference spectrum of n component of exponential decay and 
     m component of damped oscillation, you should measure at least :math:`n+2m+1` energy scan
     2. 0 to m-1 raw of doads is doads of cosine component and m to :math:`2m-1` raw of doads is doads of sine component 
    '''
    # initialization
    if base:
        c = np.empty((tau.size+1+2*tau_osc.size, intensity.shape[0]))
        A = np.empty((tau.size+2*tau_osc.size+1, escan_time.size))
        dof = escan_time.size - (tau.size+1+2*tau_osc.size)
        idx_osc = tau.size+1
    else:
        c = np.empty((tau.size+2*tau_osc.size, intensity.shape[0]))
        A = np.empty((tau.size+2*tau_osc.size, escan_time.size))
        dof = escan_time.size - (tau.size+2*tau_osc.size)
        idx_osc = tau.size

    if eps is None:
        eps = np.ones_like(intensity)

    c_eps = np.empty_like(c)

    A[:idx_osc, :] = make_A_matrix_exp(escan_time, fwhm, tau, base, irf, eta)
    A[idx_osc:, :] = make_A_matrix_dmp_osc(escan_time, fwhm, tau_osc, 
    period_osc, irf, eta)
    data_scaled = intensity/eps

    # evaluates dads
    cond = 1e-2
    for i in range(intensity.shape[0]):
        A_scaled = np.einsum('j,ij->ij', 1/eps[i, :], A)
        U, s, Vh = svd(A_scaled.T, full_matrices=False)
        mask = s > cond*s[0]
        U_turn = U[:, mask]
        s_turn = s[mask]
        Vh_turn = Vh[mask, :]
        c[:, i] = np.einsum('j,ij->ij', 1/s_turn,
                            Vh_turn.T) @ (U_turn.T @ data_scaled[i, :])
        res = data_scaled[i, :] - (c[:, i] @ A_scaled)
        if dof != 0:
            red_chi2 = np.sum(res**2)/dof
            cov_scale = red_chi2 *\
                Vh_turn.T @ np.einsum('i,ij->ij', 1/s_turn**2, Vh_turn)
            c_eps[:, i] = np.sqrt(np.diag(cov_scale))

    return c[:idx_osc,:], c[idx_osc:, :], \
           c_eps[:idx_osc, :], c_eps[idx_osc:, :], \
           c[:idx_osc, :].T @ A[:idx_osc, :], \
           c[idx_osc:, :].T @ A[idx_osc:, :] 


def sads(escan_time: np.ndarray, fwhm: float, eigval: np.ndarray, V: np.ndarray, c: np.ndarray,
         exclude: Optional[str] = None, irf: Optional[str] = 'g', eta: Optional[float] = None,
         intensity: Optional[np.ndarray] = None, eps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`

      eta: mixing parameter for pseudo voigt profile (only needed for pseudo voigt profile)
      intensity: intensity of energy scan dataset
      eps: standard error of data

    Returns:
     Tuple of calculated species associated difference spectrum of each component, estimated error and
     retrieved intensity of energy scan from sads and model excited state components

    Note:
     1. eigval, V, c should be obtained from solve_model
     2. To calculate species associated difference spectrum of n excited state species, you should measure at least n+1 energy scan
     3. Difference spectrum of ground state is zero, so ground state species should be excluded from rate equation or via exclude option.
    '''
    # initialization
    if exclude is None:
        diff_abs = np.empty((eigval.size, intensity.shape[0]))
        dof = escan_time.size - eigval.size
    elif exclude in ['first', 'last']:
        diff_abs = np.empty((eigval.size-1, intensity.shape[0]))
        dof = escan_time.size - (eigval.size-1)
    else:
        diff_abs = np.empty((eigval.size-2, intensity.shape[0]))
        dof = escan_time.size - (eigval.size-2)

    if eps is None:
        eps = np.ones_like(intensity)

    diff_abs_eps = np.empty_like(diff_abs)

    A = compute_signal_irf(escan_time, eigval, V, c, fwhm, irf, eta)
    if exclude == 'first':
        B = A[1:, :]
    elif exclude == 'last':
        B = A[:-1, :]
    elif exclude == 'first_and_last':
        B = A[1:-1, :]
    else:
        B = A

    data_scaled = intensity/eps

    # evaluates sads
    cond = 1e-2
    for i in range(intensity.shape[0]):
        A_scaled = np.einsum('j,ij->ij', 1/eps[i, :], B)
        U, s, Vh = svd(A_scaled.T, full_matrices=False)
        mask = s > cond*s[0]
        U_turn = U[:, mask]
        s_turn = s[mask]
        Vh_turn = Vh[mask, :]
        cov = Vh_turn.T @ np.einsum('i,ij->ij', 1/s_turn**2, Vh_turn)
        diff_abs[:, i] = np.einsum('j,ij->ij', 1/s_turn,
                              Vh_turn.T) @ (U_turn.T @ data_scaled[i, :])
        res = data_scaled[i, :] - (diff_abs[:, i] @ A_scaled)
        if dof != 0:
            red_chi2 = np.sum(res**2)/dof
            diff_abs_eps[:, i] = np.sqrt(red_chi2*np.diag(cov))

    return diff_abs, diff_abs_eps, diff_abs.T @ B

def sads_svd(escan_time: np.ndarray, fwhm: float, eigval: np.ndarray, V: np.ndarray, c: np.ndarray,
             exclude: Optional[str] = None, irf: Optional[str] = 'g', eta: Optional[float] = None,
             intensity: Optional[np.ndarray] = None, cond_num: Optional[float] = 0) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate species associated difference spectrum from experimental energy scan data
    (using svd)

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
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`

      eta: mixing parameter for pseudo voigt profile (only needed for pseudo voigt profile)
      intensity: intensity of energy scan dataset
      cond_num: conditional number to turncate svd

    Returns:
     Tuple of calculated species associated difference spectrum of each component, and
     retrieved intensity of energy scan from sads and model excited state components

    Note:
     1. eigval, V, c should be obtained from solve_model
     2. To calculate species associated difference spectrum of n excited state species, you should measure at least n+1 energy scan
     3. Difference spectrum of ground state is zero, so ground state species should be excluded from rate equation or via exclude option.
    '''
    # svd

    U_data, s_data, Vh_data = svd(intensity, full_matrices=False)
    N_survived = np.sum((s_data > cond_num*s_data[0]))
    U_data_trun = U_data[:, :N_survived]
    s_data_trun = s_data[:N_survived]
    Vh_turn = Vh_data[:N_survived, :]


    A = compute_signal_irf(escan_time, eigval, V, c, fwhm, irf, eta)
    if exclude == 'first':
        B = A[1:, :]
    elif exclude == 'last':
        B = A[:-1, :]
    elif exclude == 'first_and_last':
        B = A[1:-1, :]
    else:
        B = A

    c, _, _, _ = lstsq(B.T, Vh_turn.T, cond=1e-2)
    coeff = np.einsum('j,ij->ij', s_data_trun, U_data_trun) @ c.T
    fit = coeff @ B

    return coeff, fit

def sads_osc(escan_time: np.ndarray, fwhm: float, 
             eigval: np.ndarray, V: np.ndarray, c: np.ndarray,
             tau_osc: np.ndarray, period_osc: np.ndarray,
             exclude: Optional[str] = None, 
             irf: Optional[str] = 'g', eta: Optional[float] = None,
             intensity: Optional[np.ndarray] = None, eps: Optional[np.ndarray] = None) \
             -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate species associated difference spectrum with damped oscillation
    from experimental energy scan data

    Args:
      escan_time: time delay for each energy scan data
      fwhm: full width at half maximum of instrumental response function
      eigval: eigenvalue of rate equation matrix
      V: eigenvector of rate equation matrix
      c: coefficient to match initial condition of rate equation
      tau_osc: life time for each oscillation component
      period_osc: period of each oscillation
      exclude: exclude either 'first' or 'last' element or both 'first' and 'last' element.

               * 'first' : exclude first element
               * 'last' : exclude last element
               * 'first_and_last' : exclude both first and last element
               * None : Do not exclude any element [default]

      irf: shape of instrumental response function [default: g]

           * 'g': normalized gaussian distribution,
           * 'c': normalized cauchy distribution,
           * 'pv': pseudo voigt profile :math:`(1-\\eta)g(t, {fwhm}) + \\eta c(t, {fwhm})`

      eta: mixing parameter for pseudo voigt profile (only needed for pseudo voigt profile)
      intensity: intensity of energy scan dataset
      eps: standard error of data

    Returns:
     1. species associated difference spectrum
     2. damped oscillation asscoiated difference spectrum
     3. estimated error of sads
     4. estimated error of doads
     5. retrieved energy scan from sads
     6. retrieved energy scan from doads

    Note:
     1. eigval, V, c should be obtained from solve_model
     2. To calculate species associated difference spectrum of n excited state species with m damped oscillation component, 
     you should measure at least :math:`n+2m+1` energy scan
     3. Difference spectrum of ground state is zero, 
     so ground state species should be excluded from rate equation or via exclude option.
     4. 0 to m-1 raw of doads is doads of cosine component and m to :math:`2m-1` raw of doads is doads of sine component
    '''
    # initialization
    if exclude is None:
        B = np.empty((eigval.size+2*tau_osc.size, escan_time.size))
        diff_abs = np.empty((eigval.size+2*tau_osc.size, intensity.shape[0]))
        dof = escan_time.size - (eigval.size+2*tau_osc.size)
        idx_osc = eigval.size
    elif exclude in ['first', 'last']:
        B = np.empty((eigval.size-1+2*tau_osc.size, escan_time.size))
        diff_abs = np.empty((eigval.size-1+2*tau_osc.size, intensity.shape[0]))
        dof = escan_time.size - (eigval.size-1+2*tau_osc.size)
        idx_osc = eigval.size-1
    else:
        B = np.empty((eigval.size-2+2*tau_osc.size, escan_time.size))
        diff_abs = np.empty((eigval.size-2, intensity.shape[0]))
        dof = escan_time.size - (eigval.size-2+2*tau_osc.size)
        idx_osc = eigval.size-2

    if eps is None:
        eps = np.ones_like(intensity)

    diff_abs_eps = np.empty_like(diff_abs)

    A = compute_signal_irf(escan_time, eigval, V, c, fwhm, irf, eta)
    if exclude == 'first':
        B[:idx_osc, :] = A[1:, :]
    elif exclude == 'last':
        B[:idx_osc, :] = A[:-1, :]
    elif exclude == 'first_and_last':
        B[:idx_osc, :] = A[1:-1, :]
    else:
        B[:idx_osc, :] = A
    
    B[idx_osc:, :] = make_A_matrix_dmp_osc(escan_time, fwhm, 
    tau_osc, period_osc, irf, eta)

    data_scaled = intensity/eps

    # evaluates sads
    cond = 1e-2
    for i in range(intensity.shape[0]):
        A_scaled = np.einsum('j,ij->ij', 1/eps[i, :], B)
        U, s, Vh = svd(A_scaled.T, full_matrices=False)
        mask = s > cond*s[0]
        U_turn = U[:, mask]
        s_turn = s[mask]
        Vh_turn = Vh[mask, :]
        cov = Vh_turn.T @ np.einsum('i,ij->ij', 1/s_turn**2, Vh_turn)
        diff_abs[:, i] = np.einsum('j,ij->ij', 1/s_turn,
                              Vh_turn.T) @ (U_turn.T @ data_scaled[i, :])
        res = data_scaled[i, :] - (diff_abs[:, i] @ A_scaled)
        if dof != 0:
            red_chi2 = np.sum(res**2)/dof
            diff_abs_eps[:, i] = np.sqrt(red_chi2*np.diag(cov))

    return diff_abs[:idx_osc, :], diff_abs[idx_osc:, :], \
           diff_abs_eps[:idx_osc, :], diff_abs_eps[idx_osc:, :], \
           diff_abs[:idx_osc, :].T @ B[:idx_osc, :], \
           diff_abs[idx_osc:, :].T @ B[idx_osc:, :]
        
