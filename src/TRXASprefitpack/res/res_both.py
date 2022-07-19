'''
res_both:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of (sum of exponential decay and damped oscillation) and instrumental response function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence
import numpy as np
from ..mathfun.irf import calc_eta
from ..mathfun.A_matrix import make_A_matrix_exp, make_A_matrix_dmp_osc, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_exp_sum_conv_gau, deriv_exp_sum_conv_cauchy
from ..mathfun.exp_conv_irf import deriv_dmp_osc_sum_conv_gau, deriv_dmp_osc_sum_conv_cauchy

# residual and gradient function for exponential decay model + damped oscillation model

def residual_both(params: np.ndarray, num_comp: int, num_comp_osc:int, base: bool, irf: str, 
                   fix_param_idx: Optional[np.ndarray] = None,
                   t: Optional[Sequence[np.ndarray]] = None, 
                   data: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_both
    scipy.optimize.least_squares compatible vector residual function for fitting multiple set of time delay scan with the
    sum of convolution of (sum of exponential decay damped oscillation) and instrumental response function  

    Args:
     params: parameter used for fitting
             if irf == 'g','c':
                param[0]: fwhm_(G/L)
                param[1:1+number of total time delay scan]: time zero of each scan
                param[1+num_tot_scan:1+num_tot_scan+num_tau]: time constant (inverse of rate constant) of each decay component
                param[1+num_tot_scan+num_tau:1+num_tot_scan+num_tau+num_tau_osc]: time constant of each damped oscillation component
                param[1+num_tot_scan+num_tau+num_tau_osc:1+num_tot_scan+num_tau+2*num_tau_osc]: period of each damped oscillation component
                param[1+num_tot_scan+num_tau+2*num_tau_osc:]: phase of each damped oscillation component 
             if irf == 'pv'
                param[0]: fwhm_G
                param[1]: fwhm_L
                param[2:2+number of total time delay scan]: time zero of each scan
                param[2+num_tot_scan:2+num_tot_scan+num_tau]: time constant (inverse of rate constant) of each decay component
                param[2+num_tot_scan+num_tau:2+num_tot_scan+num_tau+num_tau_osc]: time constant of each damped oscillation component
                param[2+num_tot_scan+num_tau+num_tau_osc:2+num_tot_scan+num_tau+2*num_tau_osc]: period of each damped oscillation component
                param[2+num_tot_scan+num_tau+2*num_tau_osc:]: phase of each damped oscillation component 

     fwhm: full width of half maximum of instrumental response function
           if irf in ['g','c']: fwhm is single float
           if irf == 'pv': fwhm = [fwhm_G, fwhm_L]
     num_comp: number of exponential decay component (except base)
     num_comp_osc: number of damped oscillation component
     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
          For pseudo voigt profile, the mixing parameter eta is calculated by calc_eta routine
     fix_param_idx: idx for fixed parameter (masked array for `params`)
     t: time points for each data set
     data: sequence of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Residual vector
    
    Note:
     each dataset does not contain time range
    '''
    params = np.atleast_1d(params)
    
    if irf in ['g', 'c']:
        num_irf = 1
        fwhm = params[0]
        eta = None
    else:
            num_irf = 2
            fwhm = np.array([params[0], params[1]])
            eta = calc_eta(params[0], params[1])

    num_t0 = 0; sum = 0
    for d in data:
        num_t0 = d.shape[1] + num_t0
        sum = sum + d.size
    
    chi = np.empty(sum)
    tau = np.empty(num_comp, dtype=float)
    tau_osc = np.empty(num_comp_osc, dtype=float)
    period_osc = np.empty(num_comp_osc, dtype=float)
    phase_osc = np.empty(num_comp_osc, dtype=float)
    
    for i in range(num_comp):
        tau[i] = params[num_irf+num_t0+i]
    for i in range(num_comp_osc):
        tau_osc[i] = params[num_irf+num_t0+num_comp+i]
        period_osc[i] = params[num_irf+num_t0+num_comp+num_comp_osc+i]
        phase_osc[i] = params[num_irf+num_t0+num_comp+2*num_comp_osc+i]
    

    end = 0; t0_idx = num_irf
    for ti,d,e in zip(t,data,eps):
        A = np.empty((num_comp+1*base+num_comp_osc, d.shape[0]))
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            A[:num_comp+1*base, :] = make_A_matrix_exp(ti-t0, fwhm, tau, base, irf, eta)
            A[num_comp+1*base:, :] = make_A_matrix_dmp_osc(ti-t0, fwhm, tau_osc, period_osc, phase_osc, irf, eta)
            c = fact_anal_A(A, d[:,j], e[:,j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1

    return chi
    
def jac_res_both(params: np.ndarray, num_comp: int, num_comp_osc:int, base: bool, irf: str, 
                   fix_param_idx: Optional[np.ndarray] = None,
                   t: Optional[Sequence[np.ndarray]] = None, 
                   data: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    jac_res_both
    scipy.optimize.least_squares compatible gradient of vector residual function for fitting multiple set of time delay scan with the
    sum of convolution of (sum of exponential decay damped oscillation) and instrumental response function  

    Args:
     params: parameter used for fitting
             if irf == 'g','c':
                param[0]: fwhm_(G/L)
                param[1:1+number of total time delay scan]: time zero of each scan
                param[1+num_tot_scan:1+num_tot_scan+num_tau]: time constant (inverse of rate constant) of each decay component
                param[1+num_tot_scan+num_tau:1+num_tot_scan+num_tau+num_tau_osc]: time constant of each damped oscillation component
                param[1+num_tot_scan+num_tau+num_tau_osc:1+num_tot_scan+num_tau+2*num_tau_osc]: period of each damped oscillation component
                param[1+num_tot_scan+num_tau+2*num_tau_osc:]: phase of each damped oscillation component 
             if irf == 'pv'
                param[0]: fwhm_G
                param[1]: fwhm_L
                param[2:2+number of total time delay scan]: time zero of each scan
                param[2+num_tot_scan:2+num_tot_scan+num_tau]: time constant (inverse of rate constant) of each decay component
                param[2+num_tot_scan+num_tau:2+num_tot_scan+num_tau+num_tau_osc]: time constant of each damped oscillation component
                param[2+num_tot_scan+num_tau+num_tau_osc:2+num_tot_scan+num_tau+2*num_tau_osc]: period of each damped oscillation component
                param[2+num_tot_scan+num_tau+2*num_tau_osc:]: phase of each damped oscillation component 

     fwhm: full width of half maximum of instrumental response function
           if irf in ['g','c']: fwhm is single float
           if irf == 'pv': fwhm = [fwhm_G, fwhm_L]
     num_comp: number of exponential decay component (except base)
     num_comp_osc: number of damped oscillation component
     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
          For pseudo voigt profile, the mixing parameter eta is calculated by calc_eta routine
     fix_param_idx: idx for fixed parameter (masked array for `params`)
     t: time points for each data set
     data: sequence of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Gradient of residual vector
    
    Note:
     Gradient is implemented for gaussian and cauchy irf.
     For pseudo voigt shape irf, gradient is implemented only when both fwhm_G and fwhm_L are fixed.
    '''

    params = np.atleast_1d(params)
    
    if irf in ['g', 'c']:
        num_irf = 1
        fwhm = params[0]
        eta = None
    else:
            num_irf = 2
            fwhm = np.array([params[0], params[1]])
            eta = calc_eta(params[0], params[1])

    num_t0 = 0; sum = 0
    for d in data:
        num_t0 = d.shape[1] + num_t0
        sum = sum + d.size

    tau = np.empty(num_comp, dtype=float)
    tau_osc = np.empty(num_comp_osc, dtype=float)
    period_osc = np.empty(num_comp_osc, dtype=float)
    phase_osc = np.empty(num_comp_osc, dtype=float)
    
    for i in range(num_comp):
        tau[i] = params[num_irf+num_t0+i]
    for i in range(num_comp_osc):
        tau_osc[i] = params[num_irf+num_t0+num_comp+i]
        period_osc[i] = params[num_irf+num_t0+num_comp+num_comp_osc+i]
        phase_osc[i] = params[num_irf+num_t0+num_comp+2*num_comp_osc+i]
    
    num_param = num_irf+num_t0+num_comp+3*num_comp_osc

    df = np.zeros((sum, num_param))
    
    end = 0; t0_idx = num_irf; tau_start = num_t0 + t0_idx; tau_osc_start = tau_start + num_comp
    for ti,d,e in zip(t,data,eps):
        step = d.shape[0]
        A = np.empty((num_comp+1*base+num_comp_osc, step))
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            A[:num_comp+1*base, :] = make_A_matrix_exp(ti-t0, fwhm, tau, base, irf, eta)
            A[num_comp+1*base:, :] = make_A_matrix_dmp_osc(ti-t0, fwhm, tau_osc, period_osc, phase_osc, irf, eta)
            c = fact_anal_A(A, d[:,j], e[:,j])

            if irf == 'g':
                grad_decay = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_osc = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
            elif irf == 'c':
                grad_decay = deriv_exp_sum_conv_cauchy(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_osc = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
            else:
                grad_gau_decay = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_gau_osc = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
                grad_cauchy_decay = deriv_exp_sum_conv_cauchy(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_cauchy_osc = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
                grad_decay = grad_gau_decay + eta*(grad_cauchy_decay-grad_gau_decay)
                grad_osc = grad_gau_osc + eta*(grad_cauchy_osc-grad_gau_osc)
            
            grad_decay = np.einsum('j,ij->ij', 1/e[:, j], grad_decay)
            grad_osc = np.einsum('j,ij->ij', 1/e[:, j], grad_osc)

            df[end:end+step, tau_start:tau_start+num_comp] = \
                np.einsum('i,ij->ij', -1/tau**2, grad_decay[2:,:]).T
            df[end:end+step, tau_osc_start:tau_osc_start+num_comp_osc] = \
                np.einsum('i,ij->ij', -1/tau_osc**2, grad_osc[2:2+num_comp_osc,:]).T
            df[end:end+step, tau_osc_start+num_comp_osc:] = grad_osc[2+num_comp_osc:,:].T
            df[end:end+step, t0_idx] = -grad_decay[0, :]-grad_osc[0, :]
            df[end:end+step, 0] = grad_decay[1, :]+grad_osc[1, :]

            end = end + step
            t0_idx = t0_idx + 1

    if fix_param_idx is not None:
        df[:, fix_param_idx] = 0

    return df