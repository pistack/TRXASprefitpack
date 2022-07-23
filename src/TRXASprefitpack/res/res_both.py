'''
res_both:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of (sum of exponential decay and damped oscillation) and instrumental response function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence
import numpy as np
from ..mathfun.irf import calc_eta, deriv_calc_eta
from ..mathfun.A_matrix import make_A_matrix_gau, make_A_matrix_cauchy
from ..mathfun.A_matrix import make_A_matrix_gau_osc, make_A_matrix_cauchy_osc, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_exp_sum_conv_gau, deriv_exp_sum_conv_cauchy
from ..mathfun.exp_conv_irf import deriv_dmp_osc_sum_conv_gau, deriv_dmp_osc_sum_conv_cauchy

# residual and gradient function for exponential decay model + damped oscillation model

def residual_both(params: np.ndarray, num_comp: int, num_comp_osc:int, base: bool, irf: str, 
                   fix_param_idx: Optional[np.ndarray] = None,
                   t: Optional[Sequence[np.ndarray]] = None, 
                   intensity: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
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
     intensity: sequence of intensity of datasets
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
            fwhm_G = params[0], fwhm_L = params[1]
            eta = calc_eta(fwhm_G, fwhm_L)

    num_t0 = 0; sum = 0
    for d in intensity:
        num_t0 = d.shape[1] + num_t0
        sum = sum + d.size
    
    chi = np.empty(sum)

    tau = params[num_irf+num_t0:num_irf+num_t0+num_comp]
    tau_osc = params[num_irf+num_t0+num_comp:num_irf+num_t0+num_comp+num_comp_osc]
    period_osc = params[num_irf+num_t0+num_comp+num_comp_osc:num_irf+num_t0+num_comp+2*num_comp_osc]
    phase_osc = params[num_irf+num_t0+num_comp+2*num_comp_osc:]

    if base:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau; k[-1] = 0
    else:
        k = 1/tau
    
    end = 0; t0_idx = num_irf
    for ti,d,e in zip(t,intensity,eps):
        A = np.empty((num_comp+1*base+num_comp_osc, d.shape[0]))
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            if irf == 'g':
                A[:num_comp+1*base, :] = make_A_matrix_gau(ti-t0, fwhm, k)
                A[num_comp+1*base:, :] = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc)
            elif irf == 'c':
                A[:num_comp+1*base, :] = make_A_matrix_cauchy(ti-t0, fwhm, k)
                A[num_comp+1*base:, :] = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc)
            else:
                tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k); 
                tmp_gau_osc = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc)
                tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
                tmp_cauchy_osc = make_A_matrix_cauchy_osc(ti-t0, fwhm, k)
                A[:num_comp+1*base, :] = tmp_gau + eta*(tmp_cauchy-tmp_gau)
                A[num_comp+1*base:, :] = tmp_gau_osc + eta*(tmp_cauchy_osc-tmp_gau_osc)

            c = fact_anal_A(A, d[:,j], e[:,j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1

    return chi
    
def jac_res_both(params: np.ndarray, num_comp: int, num_comp_osc:int, base: bool, irf: str, 
                   fix_param_idx: Optional[np.ndarray] = None,
                   t: Optional[Sequence[np.ndarray]] = None, 
                   intensity: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
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
     data: sequence of intensity of datasets
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
            fwhm_G = params[0], fwhm_L = params[1]
            eta = calc_eta(fwhm_G, fwhm_L)
            etap_fwhm_G, etap_fwhm_L = deriv_calc_eta(fwhm_G, fwhm_L)

    num_t0 = 0; sum = 0
    for d in intensity:
        num_t0 = d.shape[1] + num_t0
        sum = sum + d.size
    
    tau = params[num_irf+num_t0:num_irf+num_t0+num_comp]
    tau_osc = params[num_irf+num_t0+num_comp:num_irf+num_t0+num_comp+num_comp_osc]
    period_osc = params[num_irf+num_t0+num_comp+num_comp_osc:num_irf+num_t0+num_comp+2*num_comp_osc]
    phase_osc = params[num_irf+num_t0+num_comp+2*num_comp_osc:]

    if base:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau; k[-1] = 0
    else:
        k = 1/tau
    
    num_param = num_irf+num_t0+num_comp+3*num_comp_osc

    df = np.zeros((sum, num_param))
    
    end = 0; t0_idx = num_irf; tau_start = num_t0 + t0_idx; tau_osc_start = tau_start + num_comp
    for ti,d,e in zip(t,intensity,eps):
        step = d.shape[0]
        A = np.empty((num_comp+1*base+num_comp_osc, step))
        if irf == 'pv':
            grad_decay = np.empty((3+num_comp, ti.size))
            grad_osc = np.empty((3+3*num_comp_osc, ti.size))
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            if irf == 'g':
                A[:num_comp+1*base, :] = make_A_matrix_gau(ti-t0, fwhm, k)
                A[num_comp+1*base:, :] = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc)
            elif irf == 'c':
                A[:num_comp+1*base, :] = make_A_matrix_cauchy(ti-t0, fwhm, k)
                A[num_comp+1*base:, :] = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc)
            else:
                tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k); 
                tmp_gau_osc = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc)
                tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
                tmp_cauchy_osc = make_A_matrix_cauchy_osc(ti-t0, fwhm, k)
                diff = tmp_cauchy-tmp_gau; diff_osc = tmp_cauchy_osc-tmp_gau_osc
                A[:num_comp+1*base, :] = tmp_gau + eta*diff
                A[num_comp+1*base:, :] = tmp_gau_osc + eta*diff_osc

            c = fact_anal_A(A, d[:,j], e[:,j])

            if irf == 'g':
                grad_decay = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_osc = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
            elif irf == 'c':
                grad_decay = deriv_exp_sum_conv_cauchy(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_osc = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
            else:
                cdiff = c[:num_comp+1*base]@diff; cdiff_osc = c[num_comp+1*base:]@diff_osc
                grad_gau_decay = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_gau_osc = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
                grad_cauchy_decay = deriv_exp_sum_conv_cauchy(ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_cauchy_osc = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm, 1/tau_osc, period_osc, phase_osc, c[num_comp+1*base:])
                grad_decay[0, :] = grad_gau_decay[0, :] + eta*(grad_cauchy_decay[0, :]-grad_gau_decay[0, :])
                grad_osc[0, :] = grad_gau_osc[0, :] + eta*(grad_cauchy_osc[0, :]-grad_gau_osc[0, :])
                grad_decay[1, :] = (1-eta)*grad_gau_decay[1, :] + etap_fwhm_G*cdiff
                grad_osc[1, :] = (1-eta)*grad_gau_decay[1, :] + etap_fwhm_G*cdiff_osc
                grad_decay[2, :] = eta*grad_cauchy_decay[2, :] + etap_fwhm_L*cdiff
                grad_osc[2, :] = eta*grad_cauchy_osc[2, :] + etap_fwhm_L*cdiff_osc
                grad_osc[3:, :] = grad_gau_osc[3:, :] + eta*(grad_cauchy_osc[3:, :]-grad_gau_osc[3:, :])
                grad_decay[3:, :] = grad_gau_decay[3:, :] + eta*(grad_cauchy_decay[3:, :]-grad_gau_decay[3:, :])
                grad_osc[3:, :] = grad_gau_osc[3:, :] + eta*(grad_cauchy_osc[3:, :]-grad_gau_osc[3:, :])
            
            grad_decay = np.einsum('j,ij->ij', 1/e[:, j], grad_decay)
            grad_osc = np.einsum('j,ij->ij', 1/e[:, j], grad_osc)

            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_decay[1, :]+grad_osc[1, :]
                df[end:end+step, tau_start:tau_start+num_comp] = \
                    np.einsum('i,ij->ij', -1/tau**2, grad_decay[2:,:]).T
                df[end:end+step, tau_osc_start:tau_osc_start+num_comp_osc] = \
                    np.einsum('i,ij->ij', -1/tau_osc**2, grad_osc[2:2+num_comp_osc,:]).T
                df[end:end+step, tau_osc_start+num_comp_osc:] = grad_osc[2+num_comp_osc:,:].T
            else:
                df[end:end+step, 0] = grad_decay[1, :]+grad_osc[1, :]
                df[end:end+step, 1] = grad_decay[2, :]+grad_osc[2, :]
                df[end:end+step, tau_start:tau_start+num_comp] = \
                    np.einsum('i,ij->ij', -1/tau**2, grad_decay[3:,:]).T
                df[end:end+step, tau_osc_start:tau_osc_start+num_comp_osc] = \
                    np.einsum('i,ij->ij', -1/tau_osc**2, grad_osc[3:3+num_comp_osc,:]).T
                df[end:end+step, tau_osc_start+num_comp_osc:] = grad_osc[3+num_comp_osc:,:].T
            df[end:end+step, t0_idx] = -grad_decay[0, :]-grad_osc[0, :]


            end = end + step
            t0_idx = t0_idx + 1

    if fix_param_idx is not None:
        df[:, fix_param_idx] = 0

    return df