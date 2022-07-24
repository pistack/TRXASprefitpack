'''
res_osc:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of damped oscillation and instrumental response function 

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Tuple, Optional, Sequence
import numpy as np
from ..mathfun.irf import calc_eta, deriv_eta
from ..mathfun.irf import calc_fwhm, deriv_fwhm
from ..mathfun.A_matrix import make_A_matrix_gau_osc, make_A_matrix_cauchy_osc, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_dmp_osc_sum_conv_gau, deriv_dmp_osc_sum_conv_cauchy

# residual and gradient function for damped oscillation model 

def residual_dmp_osc(params: np.ndarray, num_comp: int, irf: str, 
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
     t: time points for each data set
     intensity: sequence of intensity of datasets
     eps: sequence of estimated error of datasets
    
    Returns:
     Residual vector
    '''

    params = np.atleast_1d(params)
    
    if irf in ['g', 'c']:
        num_irf = 1
        fwhm = params[0]
    else:
            num_irf = 2
            fwhm = calc_fwhm(params[0], params[1])
            eta = calc_eta(params[0], params[1])

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
                A_gau = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period, phase)
                A_cauchy = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period, phase)
                A = A_gau + eta*(A_cauchy-A_gau)
            c = fact_anal_A(A, d[:,j], e[:,j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1

    return chi
    
def res_grad_dmp_osc(params: np.ndarray, num_comp: int, irf: str, 
                     fix_param_idx: Optional[np.ndarray] = None,
                     t: Optional[Sequence[np.ndarray]] = None, 
                     intensity: Optional[Sequence[np.ndarray]] = None, 
                     eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_dmp_osc
    scipy.optimize.minimize compatible pair of scalar residual function and its gradient for fitting multiple set of time delay scan with the
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
     Tuple of scalar residual function :math:`(\\frac{1}{2}\\sum_i {res}^2_i)` and its gradient
    '''
    params = np.atleast_1d(params)
    
    if irf in ['g', 'c']:
            num_irf = 1 
            fwhm = params[0]
    else:
        num_irf = 2
        eta = calc_eta(params[0], params[1])
        fwhm = calc_eta(params[0], params[1])
        dfwhm_G, dfwhm_L = deriv_fwhm(params[0], params[1])
        deta_G, deta_L = deriv_eta(params[0], params[1])

    num_t0 = 0; sum = 0
    for d in intensity:
        num_t0 = num_t0 + d.shape[1]
        sum = sum + d.size

    tau = params[num_irf+num_t0:num_irf+num_t0+num_comp]
    period = params[num_irf+num_t0+num_comp:num_irf+num_t0+2*num_comp]
    phase = params[num_irf+num_t0+2*num_comp:num_irf+num_t0+3*num_comp]
    
    num_param = num_irf+num_t0+3*num_comp
    chi = np.empty(sum)
    df = np.zeros((sum, num_param))

    end = 0; t0_idx = num_irf; tau_start = num_t0 + t0_idx

    for ti,d,e in zip(t, intensity, eps):
        step = d.shape[0]
        for j in range(d.shape[1]):
            t0 = params[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period, phase)
            elif irf == 'c':
                A = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period, phase)
            else:
                A_gau = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period, phase)
                A_cauchy = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period, phase)
                diff = A_cauchy-A_gau
                A = A_gau + eta*diff
            c = fact_anal_A(A, d[:,j], e[:,j])
            chi[end:end+step] = (c@A-d[:,j])/e[:, j]
                
            if irf == 'g':
                grad = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm, 1/tau, period, phase, c)
            elif irf == 'c':
                grad = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm, 1/tau, period, phase, c)
            else:
                grad_gau = deriv_dmp_osc_sum_conv_gau(ti-t0, fwhm[0], 1/tau, period, phase, c)
                grad_cauchy = deriv_dmp_osc_sum_conv_cauchy(ti-t0, fwhm[1], 1/tau, period, phase, c)
                grad = grad_gau + eta*(grad_cauchy-grad_gau)
   
            grad = np.einsum('i,ij->ij', 1/e[:, j], grad)
            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad[:, 1]
            else:
                cdiff = (c@diff)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad[:, 1]+deta_L*cdiff

            df[end:end+step, t0_idx] = -grad[:, 0]
            df[end:end+step, tau_start:tau_start+num_comp] = \
                np.einsum('j,ij->ij', -1/tau**2, grad[:, 2:2+num_comp])
            df[end:end+step, tau_start+num_comp:] = grad[:, 2+num_comp:]

                
            end = end + step
            t0_idx = t0_idx + 1

    if fix_param_idx is not None:
        df[:, fix_param_idx] = 0

    return np.sum(chi**2)/2, chi@df