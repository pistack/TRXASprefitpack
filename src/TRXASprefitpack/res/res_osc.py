'''
res_osc:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of damped oscillation and instrumental response function 

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence
import numpy as np
from ..mathfun.irf import calc_eta, deriv_calc_eta
from ..mathfun.A_matrix import make_A_matrix_gau_osc, make_A_matrix_cauchy_osc, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_dmp_osc_sum_conv_gau, deriv_dmp_osc_sum_conv_cauchy

# residual and gradient function for damped oscillation model 

def residual_dmp_osc(params: np.ndarray, num_comp: int, irf: str, 
                    fix_param_idx: Optional[np.ndarray] = None,
                    t: Optional[Sequence[np.ndarray]] = None, 
                    intensity: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_dmp_osc
    scipy.optimize.least_squares compatible gradient of vector residual function for fitting multiple set of time delay scan with the
    sum of convolution of damped oscillation and instrumental response function  

    Args:
     params: parameter used for fitting
             if irf == 'g','c':
                param[0]: fwhm_(G/L)
                param[1:1+number of total time delay scan]: time zero of each scan
                param[1+num_tot_scan:1+num_tot_scan+num_comp]: time constant (inverse of rate constant) of each damped oscillation component
                param[1+num_tot_scan+num_comp:1+num_tot_scan+2*num_comp]: period of each damped oscillation component
                param[1+num_tot_scan+2*num_comp:]: phase factor of each damped oscillation component
             if irf == 'pv'
                param[0]: fwhm_G
                param[1]: fwhm_L
                param[2:2+number of total time delay scan]: time zero of each scan
                param[2:2+number of total time delay scan]: time zero of each scan
                param[2+num_tot_scan:2+num_tot_scan+num_comp]: time constant (inverse of rate constant) of each damped oscillation component
                param[2+num_tot_scan+num_comp:2+num_tot_scan+2*num_comp]: period of each damped oscillation component
                param[2+num_tot_scan+2*num_comp:]: phase factor of each damped oscillation component
           
     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
          For pseudo voigt profile, the mixing parameter eta is calculated by calc_eta routine
     fix_param_idx: index for fixed parameter (masked array for `params`)
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
    else:
            num_irf = 2
            fwhm_G = params[0]; fwhm_L = params[1]
            eta = calc_eta(fwhm_G, fwhm_L)

    num_t0 = 0; sum = 0
    for d in intensity:
        num_t0 = d.shape[1] + num_t0
        sum = sum + d.size
    
    chi = np.empty(sum)
    
    tau = params[num_irf+num_t0:num_irf+num_t0+num_comp]
    period = params[num_irf+num_t0+num_comp:num_irf+num_t0+2*num_comp]
    phase = params[num_irf+num_t0+2*num_comp:num_irf+num_t0+3*num_comp]

    end = 0; t0_idx = num_irf
    for ti,d,e in zip(t,intensity,eps):
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period, phase)
            elif irf == 'c':
                A = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period, phase)
            else:
                A_gau = make_A_matrix_gau_osc(ti-t0, fwhm_G, 1/tau, period, phase)
                A_cauchy = make_A_matrix_cauchy_osc(ti-t0, fwhm_L, 1/tau, period, phase)
                A = A_gau + eta*(A_cauchy-A_gau)
            c = fact_anal_A(A, d[:,j], e[:,j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1

    return chi
    
def jac_res_dmp_osc(params: np.ndarray, num_comp: int, irf: str, 
                    fix_param_idx: Optional[np.ndarray] = None,
                    t: Optional[Sequence[np.ndarray]] = None, 
                    intensity: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    jac_res_dmp_osc
    scipy.optimize.least_squares compatible gradient of vector residual function for fitting multiple set of time delay scan with the
    sum of convolution of damped oscillation and instrumental response function  

    Args:
     params: parameter used for fitting
             if irf == 'g','c':
                param[0]: fwhm_(G/L)
                param[1:1+number of total time delay scan]: time zero of each scan
                param[1+num_tot_scan:1+num_tot_scan+num_comp]: time constant (inverse of rate constant) of each damped oscillation component
                param[1+num_tot_scan+num_comp:1+num_tot_scan+2*num_comp]: period of each damped oscillation component
                param[1+num_tot_scan+2*num_comp:]: phase factor of each damped oscillation component
             if irf == 'pv'
                param[0]: fwhm_G
                param[1]: fwhm_L
                param[2:2+number of total time delay scan]: time zero of each scan
                param[2:2+number of total time delay scan]: time zero of each scan
                param[2+num_tot_scan:2+num_tot_scan+num_comp]: time constant (inverse of rate constant) of each damped oscillation component
                param[2+num_tot_scan+num_comp:2+num_tot_scan+2*num_comp]: period of each damped oscillation component
                param[2+num_tot_scan+2*num_comp:]: phase factor of each damped oscillation component
           
     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
          For pseudo voigt profile, the mixing parameter eta is calculated by calc_eta routine
     fix_param_idx: index for fixed parameter (masked array for `params`)
     t: time points for each data set
     intensity: sequence of intensity of datasets
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
    else:
        num_irf = 2
        fwhm_G = params[0], fwhm_L = params[1]
        eta = calc_eta(fwhm_G, fwhm_L)
        etap_fwhm_G, etap_fwhm_L = deriv_calc_eta(fwhm_G, fwhm_L)

    num_t0 = 0; sum = 0
    for d in intensity:
        num_t0 = num_t0 + d.shape[1]
        sum = sum + d.size

    tau = params[num_irf+num_t0:num_irf+num_t0+num_comp]
    period = params[num_irf+num_t0+num_comp:num_irf+num_t0+2*num_comp]
    phase = params[num_irf+num_t0+2*num_comp:num_irf+num_t0+3*num_comp]
    
    num_param = num_irf+num_t0+3*num_comp

    df = np.zeros((sum, num_param))

    end = 0; t0_idx = num_irf; tau_start = num_t0 + t0_idx

    for ti,d,e in zip(t, intensity, eps):
        step = d.shape[0]
        if irf == 'pv':
            grad = np.empty((3*tau.size+3, ti.size))
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period, phase)
            elif irf == 'c':
                A = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period, phase)
            else:
                A_gau = make_A_matrix_gau_osc(ti-t0, fwhm_G, 1/tau, period, phase)
                A_cauchy = make_A_matrix_cauchy_osc(ti-t0, fwhm_L, 1/tau, period, phase)
                diff = A_cauchy-A_gau
                A = A_gau + eta*diff
            c = fact_anal_A(A, d[:,j], e[:,j])
                
            if irf == 'g':
                grad = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm, 1/tau, period, phase, c)
            elif irf == 'c':
                grad = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm, 1/tau, period, phase, c)
            else:
                cdiff = c@diff
                grad_gau = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm[0], 1/tau, period, phase, c)
                grad_cauchy = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm[1], 1/tau, period, phase, c)
                grad[1, :] = (1-eta)*grad_gau[0, :] + etap_fwhm_G*cdiff
                grad[2, :] = eta*grad_cauchy[0, :] + etap_fwhm_L*cdiff
                grad[0, :] = grad_gau[0, :] + eta*(grad_cauchy[0, :]-grad_gau[0, :])
                grad[3:, :] = grad_gau[3:, :] + eta*(grad_cauchy[3:, :]-grad_gau[3:, :])
   
            grad = np.einsum('j,ij->ij', 1/e[:, j], grad)
            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad[1, :]
                df[end:end+step, tau_start:tau_start+num_comp] = np.einsum('i,ij->ij', -1/tau**2, grad[2:2+num_comp,:]).T
                df[end:end+step, tau_start+num_comp:] = grad[2+num_comp:,:].T
            else:
                df[end:end+step, :2] = grad[1:3, :].T
                df[end:end+step, tau_start:tau_start+num_comp] = np.einsum('i,ij->ij', -1/tau**2, grad[3:3+num_comp,:]).T
                df[end:end+step, tau_start+num_comp:] = grad[3+num_comp:,:].T

            df[end:end+step, t0_idx] = -grad[0, :]

                
            end = end + step
            t0_idx = t0_idx + 1

    if fix_param_idx is not None:
        df[:, fix_param_idx] = 0

    return df