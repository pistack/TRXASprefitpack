'''
res_osc:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of damped oscillation and instrumental response function 

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence
import numpy as np
from ..mathfun.irf import calc_eta
from ..mathfun.A_matrix import make_A_matrix_dmp_osc, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_dmp_osc_sum_conv_gau, deriv_dmp_osc_sum_conv_cauchy

# residual and gradient function for damped oscillation model 

def residual_dmp_osc(params: np.ndarray, num_comp: int, irf: str, 
                    fix_param_idx: Optional[np.ndarray] = None,
                    t: Optional[Sequence[np.ndarray]] = None, 
                    data: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
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

     fwhm: full width of half maximum of instrumental response function
           if irf in ['g','c']: fwhm is single float
           if irf == 'pv': fwhm = [fwhm_G, fwhm_L]
           
     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
          For pseudo voigt profile, the mixing parameter eta is calculated by calc_eta routine
     fix_param_idx: index for fixed parameter (masked array for `params`)
     t: time points for each data set
     data: sequence of datasets
     eps: sequence of estimated error of datasets
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
    period = np.empty(num_comp, dtype=float)
    phase = np.empty(num_comp, dtype=float)
    
    for i in range(num_comp):
        tau[i] = params[num_irf+num_t0+i]
        period[i] = params[num_irf+num_t0+num_comp+i]
        phase[i] = params[num_irf+num_t0+2*num_comp+i]

    end = 0; t0_idx = num_irf
    for ti,d,e in zip(t,data,eps):
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            A = make_A_matrix_dmp_osc(ti-t0, fwhm, 1/tau, period, phase, irf, eta)
            c = fact_anal_A(A, d[:,j], e[:,j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1

    return chi
    
def jac_res_dmp_osc(params: np.ndarray, num_comp: int, irf: str, 
                    fix_param_idx: Optional[np.ndarray] = None,
                    t: Optional[Sequence[np.ndarray]] = None, 
                    data: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
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

     fwhm: full width of half maximum of instrumental response function
           if irf in ['g','c']: fwhm is single float
           if irf == 'pv': fwhm = [fwhm_G, fwhm_L]
           
     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g + \\eta c`
          For pseudo voigt profile, the mixing parameter eta is calculated by calc_eta routine
     fix_param_idx: index for fixed parameter (masked array for `params`)
     t: time points for each data set
     data: sequence of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Gradient of residucal vector

    Note:
     Gradient is implemented for gaussian and cauchy irf.
     For pseudo voigt shape irf, gradient is implemented only when both fwhm_G and fwhm_L are fixed.
    '''
    params = np.atleast_1d(params)
    
    if irf in ['g', 'c']:
            num_irf = 1 
            eta = None
            fwhm = params[0]
    else:
        num_irf = 2
        fwhm = np.array([fwhm[0], fwhm[1]])
        eta = calc_eta(fwhm[0], fwhm[1])

    num_t0 = 0; sum = 0
    for d in data:
        num_t0 = num_t0 + d.shape[1]
        sum = sum + d.size
        
    tau = np.empty(num_comp, dtype=float)
    period = np.empty(num_comp, dtype=float)
    phase = np.empty(num_comp, dtype=float)

    for i in range(num_comp):
        tau[i] = params[num_irf+num_t0+i]
        period[i] = params[num_irf+num_t0+num_comp+i]
        phase[i] = params[num_irf+num_t0+2*num_comp+i]
    
    num_param = num_irf+num_t0+3*num_comp

    df = np.zeros((sum, num_param))

    end = 0; t0_idx = num_irf; tau_start = num_t0 + t0_idx

    for ti,d,e in zip(t, data, eps):
        step = d.shape[0]
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            A = make_A_matrix_dmp_osc(ti-t0, fwhm, 1/tau, period, phase, irf, eta)
            c = fact_anal_A(A, d[:,j], e[:,j])
                
            if irf == 'g':
                grad = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm, 1/tau, period, phase, c)
            elif irf == 'c':
                grad = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm, 1/tau, period, phase, c)
            else:
                grad_gau = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm[0], 1/tau, period, phase, c)
                grad_cauchy = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm[1], 1/tau, period, phase, c)
                grad = grad_gau + eta*(grad_cauchy-grad_gau)
   
            grad = np.einsum('j,ij->ij', 1/e[:, j], grad)
            df[end:end+step, tau_start:tau_start+num_comp] = np.einsum('i,ij->ij', -1/tau**2, grad[2:2+num_comp,:])
            df[end:end+step, tau_start+num_comp: tau_start+2*num_comp] = grad[2+num_comp:2+2*num_comp,:]
            df[end:end+step, tau_start+2*num_comp:] = grad[2+2*num_comp:,:]
            df[end:end+step, t0_idx] = -grad[0, :]
            df[end:end+step, 0] = grad[1, :]
                
            end = end + step
            t0_idx = t0_idx + 1

    if fix_param_idx is not None:
        df[:, fix_param_idx] = 0

    return df