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
from ..mathfun.exp_conv_irf import deriv_dmp_osc_sum_conv_gau_2, deriv_dmp_osc_sum_conv_cauchy_2

# residual and gradient function for damped oscillation model


def residual_dmp_osc(x0: np.ndarray, num_comp: int, irf: str,
                     t: Optional[Sequence[np.ndarray]] = None,
                     intensity: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_dmp_osc
    `scipy.optimize.least_squares` compatible gradient of vector residual function 
    for fitting multiple set of time delay scan with the
    sum of convolution of damped oscillation and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{scan}`: time zero of each scan
        * :math:`2+N_{scan}+i`: time constant of each damped oscillation
        * :math:`2+N_{scan}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{scan}`: time zero of each scan
        * :math:`3+N_{scan}+i`: time constant of each damped oscillation
        * :math:`3+N_{scan}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     t: time points for each data set
     intensity: sequence of intensity of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Residual vector
    '''

    x0 = np.atleast_1d(x0)

    if irf in ['g', 'c']:
        num_irf = 1
        fwhm = x0[0]
    else:
        num_irf = 2
        fwhm = calc_fwhm(x0[0], x0[1])
        eta = calc_eta(x0[0], x0[1])

    num_t0 = 0
    count = 0
    for d in intensity:
        num_t0 = d.shape[1] + num_t0
        count = count + d.size

    chi = np.empty(count)

    tau = x0[num_irf+num_t0:num_irf+num_t0+num_comp]
    period = x0[num_irf+num_t0+num_comp:num_irf+num_t0+2*num_comp]

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period)
            elif irf == 'c':
                A = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period)
            else:
                A_gau = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period)
                A_cauchy = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period)
                A = A_gau + eta*(A_cauchy-A_gau)
            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1

    return chi


def res_grad_dmp_osc(x0: np.ndarray, num_comp: int, irf: str,
                     fix_param_idx: Optional[np.ndarray] = None,
                     t: Optional[Sequence[np.ndarray]] = None,
                     intensity: Optional[Sequence[np.ndarray]] = None,
                     eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_dmp_osc
    scipy.optimize.minimize compatible pair of scalar residual function 
    and its gradient for fitting multiple set of time delay scan with the
    sum of convolution of damped oscillation and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{scan}`: time zero of each scan
        * :math:`2+N_{scan}+i`: time constant of each damped oscillation
        * :math:`2+N_{scan}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{scan}`: time zero of each scan
        * :math:`3+N_{scan}+i`: time constant of each damped oscillation
        * :math:`3+N_{scan}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     fix_param_idx: index for fixed parameter (masked array for `x0`)
     t: time points for each data set
     intensity: sequence of intensity of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Tuple of scalar residual function :math:`(\\frac{1}{2}\\sum_i {res}^2_i)` and its gradient
    '''
    x0 = np.atleast_1d(x0)

    if irf in ['g', 'c']:
        num_irf = 1
        fwhm = x0[0]
    else:
        num_irf = 2
        eta = calc_eta(x0[0], x0[1])
        fwhm = calc_fwhm(x0[0], x0[1])
        dfwhm_G, dfwhm_L = deriv_fwhm(x0[0], x0[1])
        deta_G, deta_L = deriv_eta(x0[0], x0[1])

    num_t0 = 0
    count = 0
    for d in intensity:
        num_t0 = num_t0 + d.shape[1]
        count = count + d.size

    tau = x0[num_irf+num_t0:num_irf+num_t0+num_comp]
    period = x0[num_irf+num_t0+num_comp:num_irf+num_t0+2*num_comp]

    num_param = num_irf+num_t0+2*num_comp
    chi = np.empty(count)
    df = np.zeros((count, num_irf+2*num_comp))
    grad = np.empty(num_param)

    end = 0
    t0_idx = num_irf

    for ti, d, e in zip(t, intensity, eps):
        step = d.shape[0]
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period)
            elif irf == 'c':
                A = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period)
            else:
                A_gau = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau, period)
                A_cauchy = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau, period)
                diff = A_cauchy-A_gau
                A = A_gau + eta*diff
            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+step] = (c@A-d[:, j])/e[:, j]

            if irf == 'g':
                grad_tmp = deriv_dmp_osc_sum_conv_gau_2(
                    ti-t0, fwhm, 1/tau, period, c)
            elif irf == 'c':
                grad_tmp = deriv_dmp_osc_sum_conv_cauchy_2(
                    ti-t0, fwhm, 1/tau, period, c)
            else:
                grad_gau = deriv_dmp_osc_sum_conv_gau_2(
                    ti-t0, fwhm, 1/tau, period, c)
                grad_cauchy = deriv_dmp_osc_sum_conv_cauchy_2(
                    ti-t0, fwhm, 1/tau, period, c)
                grad_tmp = grad_gau + eta*(grad_cauchy-grad_gau)

            grad_tmp = np.einsum('i,ij->ij', 1/e[:, j], grad_tmp)
            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_tmp[:, 1]
            else:
                cdiff = (c@diff)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad_tmp[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad_tmp[:, 1]+deta_L*cdiff

            grad[t0_idx] = -chi[end:end+step]@grad_tmp[:, 0]
            df[end:end+step, num_irf:num_irf+num_comp] = \
                np.einsum('j,ij->ij', -1/tau**2, grad_tmp[:, 2:2+num_comp])
            df[end:end+step, num_irf+num_comp:] = grad_tmp[:, 2+num_comp:2+2*num_comp]

            end = end + step
            t0_idx = t0_idx + 1

    mask = np.ones(num_param, dtype=bool)
    mask[num_irf:num_irf+num_t0] = False
    grad[mask] = chi@df

    if fix_param_idx is not None:
        grad[fix_param_idx] = 0

    return np.sum(chi**2)/2, grad

def residual_dmp_osc_same_t0(x0: np.ndarray, num_comp: int, irf: str,
                             t: Optional[Sequence[np.ndarray]] = None,
                             intensity: Optional[Sequence[np.ndarray]] = None, 
                             eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_dmp_osc_same_t0
    `scipy.optimize.least_squares` compatible gradient of vector residual function 
    for fitting multiple set of time delay scan with the
    sum of convolution of damped oscillation and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{dset}`: time zero of each data set
        * :math:`2+N_{dset}+i`: time constant of each damped oscillation
        * :math:`2+N_{dset}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{dset}`: time zero of each data set
        * :math:`3+N_{dset}+i`: time constant of each damped oscillation
        * :math:`3+N_{dset}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     t: time points for each data set
     intensity: sequence of intensity of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Residual vector
    '''

    num_irf = 1 + 1*(irf == 'pv')
    num_dset = len(t)
    num_dset_each = np.empty(num_dset, dtype=int)
    for i in range(num_dset):
        num_dset_each[i] = intensity[i].shape[1]
    num_dset_tot = np.sum(num_dset_each)
    x0 = np.atleast_1d(x0)
    x1 = np.empty(x0.size-num_dset+num_dset_tot)
    x1[:num_irf] = x0[:num_irf]
    x1[num_irf+num_dset_tot:] = x0[num_irf+num_dset:]
    start_t0_idx = num_irf
    for i in range(num_dset):
        for j in range(num_dset_each[i]):
            x1[start_t0_idx+j] = x0[i]
        start_t0_idx = start_t0_idx + num_dset_each[i]

    return residual_dmp_osc(x1, num_comp, irf, 
    t=t, intensity=intensity, eps=eps)

def res_grad_dmp_osc_same_t0(x0: np.ndarray, num_comp: int, irf: str,
                             fix_param_idx: Optional[np.ndarray] = None,
                             t: Optional[Sequence[np.ndarray]] = None,
                             intensity: Optional[Sequence[np.ndarray]] = None,
                             eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_dmp_osc_same_t0
    scipy.optimize.minimize compatible pair of scalar residual function 
    and its gradient for fitting multiple set of time delay scan with the
    sum of convolution of damped oscillation and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{dset}`: time zero of each data set
        * :math:`2+N_{dset}+i`: time constant of each damped oscillation
        * :math:`2+N_{dset}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{dset}`: time zero of each data set
        * :math:`3+N_{dset}+i`: time constant of each damped oscillation
        * :math:`3+N_{dset}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of damped oscillation component
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     fix_param_idx: index for fixed parameter (masked array for `x0`)
     t: time points for each data set
     intensity: sequence of intensity of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Tuple of scalar residual function :math:`(\\frac{1}{2}\\sum_i {res}^2_i)` and its gradient
    '''

    num_irf = 1 + 1*(irf == 'pv')
    num_dset = len(t)
    num_dset_each = np.empty(num_dset, dtype=int)
    for i in range(num_dset):
        num_dset_each[i] = intensity[i].shape[1]
    num_dset_tot = np.sum(num_dset_each)
    x0 = np.atleast_1d(x0)
    x1 = np.empty(x0.size-num_dset+num_dset_tot)
    fix_param_idx_1 = np.empty(x0.size-num_dset+num_dset_tot, dtype=bool)
    x1[:num_irf] = x0[:num_irf]
    x1[num_irf+num_dset_tot:] = x0[num_irf+num_dset:]
    fix_param_idx_1[:num_irf] = fix_param_idx[:num_irf]
    fix_param_idx_1[num_irf+num_dset_tot:] = fix_param_idx[num_irf+num_dset:]
    start_t0_idx = num_irf

    for i in range(num_dset):
        for j in range(num_dset_each[i]):
            x1[start_t0_idx+j] = x0[i]
            fix_param_idx_1[start_t0_idx+j] = fix_param_idx[i]
        start_t0_idx = start_t0_idx + num_dset_each[i]
    
    res, grad_0 = res_grad_dmp_osc(x1, num_comp, irf, fix_param_idx_1,
    t=t, intensity=intensity, eps=eps)
    grad = np.empty(x0.size)
    grad[:num_irf] = grad_0[:num_irf]
    grad[num_irf+num_dset:] = grad_0[num_irf+num_dset_tot:]
    start_t0_idx = num_irf
    for i in range(num_dset):
        grad[num_irf+i] = 0
        for j in range(num_dset_each[i]):
            grad[num_irf+i] = grad[num_irf+i] + grad_0[start_t0_idx+j]
        start_t0_idx = start_t0_idx + num_dset_each[i]
        
    return res, grad
