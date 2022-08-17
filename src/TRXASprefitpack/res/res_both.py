'''
res_both:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of (sum of exponential decay and damped oscillation) and instrumental response function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence, Tuple
import numpy as np
from ..mathfun.irf import calc_eta, deriv_eta
from ..mathfun.irf import calc_fwhm, deriv_fwhm
from ..mathfun.A_matrix import make_A_matrix_gau, make_A_matrix_cauchy
from ..mathfun.A_matrix import make_A_matrix_gau_osc, make_A_matrix_cauchy_osc, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_exp_conv_gau, deriv_exp_conv_cauchy
from ..mathfun.exp_conv_irf import deriv_dmp_osc_conv_gau_2, deriv_dmp_osc_conv_cauchy_2
from ..mathfun.exp_conv_irf import deriv_exp_sum_conv_gau, deriv_exp_sum_conv_cauchy
from ..mathfun.exp_conv_irf import deriv_dmp_osc_sum_conv_gau_2, deriv_dmp_osc_sum_conv_cauchy_2

# residual and gradient function for exponential decay model + damped oscillation model


def residual_both(x0: np.ndarray, num_comp: int, num_comp_osc: int, base: bool, irf: str,
                  t: Optional[Sequence[np.ndarray]] = None,
                  intensity: Optional[Sequence[np.ndarray]] = None,
                  eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_both
    `scipy.optimize.least_squares` compatible vector residual function
    for fitting multiple set of time delay scan with the
    sum of convolution of (sum of exponential decay damped oscillation) and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{scan}`: time zero of each scan
        * :math:`2+N_{scan}` to :math:`2+N_{scan}+N_{\\tau}`: time constant of each decay component
        * :math:`2+N_{scan}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`2+N_{scan}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{scan}`: time zero of each scan
        * :math:`3+N_{scan}` to :math:`3+N_{scan}+N_{\\tau}`: time constant of each decay component
        * :math:`3+N_{scan}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`3+N_{scan}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of exponential decay component (except base)
     num_comp_osc: number of damped oscillation component
     base: whether or not include baseline (i.e. very long lifetime component)
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
        eta = None
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
    tau_osc = x0[num_irf+num_t0+num_comp:num_irf+num_t0+num_comp+num_comp_osc]
    period_osc = x0[num_irf+num_t0+num_comp +
                    num_comp_osc:num_irf+num_t0+num_comp+2*num_comp_osc]

    if base:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0
    else:
        k = 1/tau

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        A = np.empty((num_comp+1*base+2*num_comp_osc, d.shape[0]))
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A[:num_comp+1*base, :] = make_A_matrix_gau(ti-t0, fwhm, k)
                A[num_comp+1*base:,
                    :] = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            elif irf == 'c':
                A[:num_comp+1*base, :] = make_A_matrix_cauchy(ti-t0, fwhm, k)
                A[num_comp+1*base:,
                    :] = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            else:
                tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k)
                tmp_gau_osc = make_A_matrix_gau_osc(
                    ti-t0, fwhm, 1/tau_osc, period_osc)
                tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
                tmp_cauchy_osc = make_A_matrix_cauchy_osc(
                    ti-t0, fwhm, 1/tau_osc, period_osc)
                A[:num_comp+1*base, :] = tmp_gau + eta*(tmp_cauchy-tmp_gau)
                A[num_comp+1*base:, :] = tmp_gau_osc + \
                    eta*(tmp_cauchy_osc-tmp_gau_osc)

            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1
    return chi


def res_grad_both(x0: np.ndarray, num_comp: int, num_comp_osc: int, base: bool, irf: str,
                  fix_param_idx: Optional[np.ndarray] = None,
                  t: Optional[Sequence[np.ndarray]] = None,
                  intensity: Optional[Sequence[np.ndarray]] = None,
                  eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_both
    `scipy.optimize.minimize` compatible scalar residual and
    its gradient function for fitting multiple set of time delay scan with the
    sum of convolution of (sum of exponential decay damped oscillation) and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{scan}`: time zero of each scan
        * :math:`2+N_{scan}` to :math:`2+N_{scan}+N_{\\tau}`: time constant of each decay component
        * :math:`2+N_{scan}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`2+N_{scan}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{scan}`: time zero of each scan
        * :math:`3+N_{scan}` to :math:`3+N_{scan}+N_{\\tau}`: time constant of each decay component
        * :math:`3+N_{scan}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`3+N_{scan}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of exponential decay component (except base)
     num_comp_osc: number of damped oscillation component
     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     fix_param_idx: idx for fixed parameter (masked array for `x0`)
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
        eta = None
    else:
        num_irf = 2
        eta = calc_eta(x0[0], x0[1])
        fwhm = calc_fwhm(x0[0], x0[1])
        deta_G, deta_L = deriv_eta(x0[0], x0[1])
        dfwhm_G, dfwhm_L = deriv_fwhm(x0[0], x0[1])

    num_t0 = 0
    count = 0
    for d in intensity:
        num_t0 = d.shape[1] + num_t0
        count = count + d.size

    tau = x0[num_irf+num_t0:num_irf+num_t0+num_comp]
    tau_osc = x0[num_irf+num_t0+num_comp:num_irf+num_t0+num_comp+num_comp_osc]
    period_osc = x0[num_irf+num_t0+num_comp +
                    num_comp_osc:num_irf+num_t0+num_comp+2*num_comp_osc]

    if base:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0
    else:
        k = 1/tau

    num_param = num_irf+num_t0+num_comp+2*num_comp_osc

    chi = np.empty(count)
    df = np.zeros((count, num_irf+num_comp+2*num_comp_osc))
    grad = np.empty(num_param)

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        step = d.shape[0]
        A = np.empty((num_comp+1*base+2*num_comp_osc, step))
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A[:num_comp+1*base, :] = make_A_matrix_gau(ti-t0, fwhm, k)
                A[num_comp+1*base:,
                    :] = make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            elif irf == 'c':
                A[:num_comp+1*base, :] = make_A_matrix_cauchy(ti-t0, fwhm, k)
                A[num_comp+1*base:,
                    :] = make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            else:
                tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k)
                tmp_gau_osc = make_A_matrix_gau_osc(
                    ti-t0, fwhm, 1/tau_osc, period_osc)
                tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
                tmp_cauchy_osc = make_A_matrix_cauchy_osc(
                    ti-t0, fwhm, 1/tau_osc, period_osc)
                diff = tmp_cauchy-tmp_gau
                diff_osc = tmp_cauchy_osc-tmp_gau_osc
                A[:num_comp+1*base, :] = tmp_gau + eta*diff
                A[num_comp+1*base:, :] = tmp_gau_osc + eta*diff_osc

            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+step] = (c@A-d[:, j])/e[:, j]

            if irf == 'g':
                grad_decay = deriv_exp_sum_conv_gau(
                    ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_osc = deriv_dmp_osc_sum_conv_gau_2(
                    ti-t0, fwhm, 1/tau_osc, period_osc, c[num_comp+1*base:])
            elif irf == 'c':
                grad_decay = deriv_exp_sum_conv_cauchy(
                    ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_osc = deriv_dmp_osc_sum_conv_cauchy_2(
                    ti-t0, fwhm, 1/tau_osc, period_osc, c[num_comp+1*base:])
            else:
                grad_gau_decay = deriv_exp_sum_conv_gau(
                    ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_gau_osc = deriv_dmp_osc_sum_conv_gau_2(
                    ti-t0, fwhm, 1/tau_osc, period_osc, c[num_comp+1*base:])
                grad_cauchy_decay = deriv_exp_sum_conv_cauchy(
                    ti-t0, fwhm, 1/tau, c[:num_comp+1*base], base)
                grad_cauchy_osc = deriv_dmp_osc_sum_conv_cauchy_2(
                    ti-t0, fwhm, 1/tau_osc, period_osc, c[num_comp+1*base:])
                grad_decay = grad_gau_decay + eta * \
                    (grad_cauchy_decay-grad_gau_decay)
                grad_osc = grad_gau_osc + eta*(grad_cauchy_osc-grad_gau_osc)

            grad_decay = np.einsum('i,ij->ij', 1/e[:, j], grad_decay)
            grad_osc = np.einsum('i,ij->ij', 1/e[:, j], grad_osc)
            grad_sum = grad_decay[:, :2] + grad_osc[:, :2]

            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_sum[:, 1]

            else:
                cdiff = (c[:num_comp+1*base]@diff +
                         c[num_comp+1*base:]@diff_osc)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad_sum[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad_sum[:, 1]+deta_L*cdiff

            grad[t0_idx] = -chi[end:end+step]@grad_sum[:, 0]
            df[end:end+step, num_irf:num_irf+num_comp] = \
                np.einsum('j,ij->ij', -1/tau**2, grad_decay[:, 2:])
            df[end:end+step, num_irf+num_comp:num_irf+num_comp+num_comp_osc] = \
                np.einsum('j,ij->ij', -1/tau_osc**2,
                          grad_osc[:, 2:2+num_comp_osc])
            df[end:end+step, num_irf+num_comp+num_comp_osc:] = \
                grad_osc[:, 2+num_comp_osc:2+2*num_comp_osc]

            end = end + step
            t0_idx = t0_idx + 1

    mask = np.ones(num_param, dtype=bool)
    mask[num_irf:num_irf+num_t0] = False
    grad[mask] = chi@df

    if fix_param_idx is not None:
        grad[fix_param_idx] = 0

    return np.sum(chi**2)/2, grad

def residual_both_same_t0(x0: np.ndarray, num_comp: int, num_comp_osc: int, base: bool, irf: str,
                          t: Optional[Sequence[np.ndarray]] = None,
                          intensity: Optional[Sequence[np.ndarray]] = None,
                          eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_both_same_t0
    `scipy.optimize.least_squares` compatible vector residual function for fitting multiple set of time delay scan with the
    sum of convolution of (sum of exponential decay damped oscillation) and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{dset}`: time zero of each dataset
        * :math:`2+N_{dset}` to :math:`2+N_{dset}+N_{\\tau}`: time constant of each decay component
        * :math:`2+N_{dset}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`2+N_{dset}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{dset}`: time zero of each dataset
        * :math:`3+N_{dset}` to :math:`3+N_{dset}+N_{\\tau}`: time constant of each decay component
        * :math:`3+N_{dset}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`3+N_{dset}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of exponential decay component (except base)
     num_comp_osc: number of damped oscillation component
     base: whether or not include baseline (i.e. very long lifetime component)
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
        eta = None
    else:
        num_irf = 2
        fwhm = calc_fwhm(x0[0], x0[1])
        eta = calc_eta(x0[0], x0[1])

    num_dataset = len(t)
    count = 0
    for i in range(num_dataset):
        count = count + intensity[i].size

    chi = np.empty(count)

    tau = x0[num_irf+num_dataset:num_irf+num_dataset+num_comp]
    tau_osc = x0[num_irf+num_dataset+num_comp:
    num_irf+num_dataset+num_comp+num_comp_osc]
    period_osc = x0[num_irf+num_dataset+num_comp +
                    num_comp_osc:num_irf+num_dataset+num_comp+2*num_comp_osc]

    if base:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0
    else:
        k = 1/tau

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        A = np.empty((num_comp+1*base+2*num_comp_osc, d.shape[0]))
        t0 = x0[t0_idx]
        if irf == 'g':
            A[:num_comp+1*base, :] = make_A_matrix_gau(ti-t0, fwhm, k)
            A[num_comp+1*base:, :] = \
                make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
        elif irf == 'c':
            A[:num_comp+1*base, :] = make_A_matrix_cauchy(ti-t0, fwhm, k)
            A[num_comp+1*base:, :] = \
                make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
        else:
            tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k)
            tmp_gau_osc = \
                make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
            tmp_cauchy_osc = \
                make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            A[:num_comp+1*base, :] = tmp_gau + eta*(tmp_cauchy-tmp_gau)
            A[num_comp+1*base:, :] = tmp_gau_osc + \
                eta*(tmp_cauchy_osc-tmp_gau_osc)

        for j in range(d.shape[1]):
            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]
            end = end + d.shape[0]
        t0_idx = t0_idx + 1
    return chi

def res_grad_both_same_t0(x0: np.ndarray, num_comp: int, num_comp_osc: int, base: bool, irf: str,
                          fix_param_idx: Optional[np.ndarray] = None,
                          t: Optional[Sequence[np.ndarray]] = None,
                          intensity: Optional[Sequence[np.ndarray]] = None,
                          eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_both_same_t0
    `scipy.optimize.minimize` compatible scalar residual and
    its gradient function for fitting multiple set of time delay scan with the
    sum of convolution of (sum of exponential decay damped oscillation) and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{dset}`: time zero of each dataset
        * :math:`2+N_{dset}` to :math:`2+N_{dset}+N_{\\tau}`: time constant of each decay component
        * :math:`2+N_{dset}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`2+N_{dset}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{dset}`: time zero of each dataset
        * :math:`3+N_{dset}` to :math:`3+N_{dset}+N_{\\tau}`: time constant of each decay component
        * :math:`3+N_{dset}+N_{\\tau}+i`: time constant of each damped oscillation
        * :math:`3+N_{dset}+N_{\\tau}+N_{osc}+i`: period of each damped oscillation

     num_comp: number of exponential decay component (except base)
     num_comp_osc: number of damped oscillation component
     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     fix_param_idx: idx for fixed parameter (masked array for `x0`)
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
        eta = None
    else:
        num_irf = 2
        eta = calc_eta(x0[0], x0[1])
        fwhm = calc_fwhm(x0[0], x0[1])
        deta_G, deta_L = deriv_eta(x0[0], x0[1])
        dfwhm_G, dfwhm_L = deriv_fwhm(x0[0], x0[1])

    num_dataset = len(t)
    count = 0
    for i in range(num_dataset):
        count = count + intensity[i].size

    tau = x0[num_irf+num_dataset:num_irf+num_dataset+num_comp]
    tau_osc = x0[num_irf+num_dataset+num_comp:
    num_irf+num_dataset+num_comp+num_comp_osc]
    period_osc = x0[num_irf+num_dataset+num_comp+num_comp_osc:
                    num_irf+num_dataset+num_comp+2*num_comp_osc]

    if base:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0
    else:
        k = 1/tau

    num_param = num_irf+num_dataset+num_comp+2*num_comp_osc

    chi = np.empty(count)
    df = np.zeros((count, num_irf+num_comp+2*num_comp_osc))
    grad = np.zeros(num_param)

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        step = d.shape[0]
        A = np.empty((num_comp+1*base+2*num_comp_osc, step))
        A_grad_decay = np.empty((num_comp+1*base, step, 3))
        A_grad_osc = np.empty((2*num_comp_osc, step, 4))
        grad_decay = np.empty((step, num_comp+2))
        grad_osc = np.empty((step, 2*num_comp_osc+2))

        t0 = x0[t0_idx]
        if irf == 'g':
            A[:num_comp+1*base, :] = make_A_matrix_gau(ti-t0, fwhm, k)
            A[num_comp+1*base:, :] = \
                make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            for i in range(num_comp):
                A_grad_decay[i, :, :] = deriv_exp_conv_gau(ti-t0, fwhm, k[i])
            if base:
                A_grad_decay[-1, :, :] = deriv_exp_conv_gau(ti-t0, fwhm, 0)
            for i in range(num_comp_osc):
                A_grad_osc[i, :, :] = \
                    deriv_dmp_osc_conv_gau_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (1, 0))
                A_grad_osc[num_comp_osc+i, :, :] = \
                    deriv_dmp_osc_conv_gau_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (0, 1))
        elif irf == 'c':
            A[:num_comp+1*base, :] = make_A_matrix_cauchy(ti-t0, fwhm, k)
            A[num_comp+1*base:, :] = \
                make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            for i in range(num_comp):
                A_grad_decay[i, :, :] = deriv_exp_conv_cauchy(ti-t0, fwhm, k[i])
            if base:
                A_grad_decay[-1, :, :] = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
            for i in range(num_comp_osc):
                A_grad_osc[i, :, :] = \
                    deriv_dmp_osc_conv_cauchy_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (1, 0))
                A_grad_osc[num_comp_osc+i, :, :] = \
                    deriv_dmp_osc_conv_cauchy_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (0, 1))
        else:
            tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k)
            tmp_gau_osc = \
                make_A_matrix_gau_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
            tmp_cauchy_osc = \
                make_A_matrix_cauchy_osc(ti-t0, fwhm, 1/tau_osc, period_osc)
            diff = tmp_cauchy-tmp_gau
            diff_osc = tmp_cauchy_osc-tmp_gau_osc
            A[:num_comp+1*base, :] = tmp_gau + eta*diff
            A[num_comp+1*base:, :] = tmp_gau_osc + eta*diff_osc
            for i in range(num_comp):
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, k[i])
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, k[i])
                A_grad_decay[i, :, :] = tmp_grad_gau + eta*(tmp_grad_cauchy-tmp_grad_gau)
            if base:
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
                A_grad_decay[-1, :, :] = tmp_grad_gau + eta*(tmp_grad_cauchy-tmp_grad_gau)
            for i in range(num_comp_osc):
                tmp_grad_gau_osc_1 = \
                    deriv_dmp_osc_conv_gau_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (1, 0))
                tmp_grad_gau_osc_2 = \
                    deriv_dmp_osc_conv_gau_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (0, 1))
                tmp_grad_cauchy_osc_1 = \
                    deriv_dmp_osc_conv_cauchy_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (1, 0))
                tmp_grad_cauchy_osc_2 = \
                    deriv_dmp_osc_conv_cauchy_2(ti-t0, fwhm, 1/tau_osc[i], period_osc[i], (0, 1))
                A_grad_osc[i, :, :] = \
                    tmp_grad_gau_osc_1 + eta*(tmp_grad_cauchy_osc_1-tmp_grad_gau_osc_1)
                A_grad_osc[num_comp_osc+i, :, :] = \
                    tmp_grad_gau_osc_2 + eta*(tmp_grad_cauchy_osc_2-tmp_grad_gau_osc_2)

        for j in range(d.shape[1]):
            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+step] = (c@A-d[:, j])/e[:, j]

            grad_decay[:, :2] = \
                np.tensordot(c[:num_comp+1*base], A_grad_decay[:, :, :2], axes=1)
            grad_osc[:, :2] = \
                np.tensordot(c[num_comp+1*base:], A_grad_osc[:, :, :2], axes=1)
            for i in range(num_comp):
                grad_decay[:, 2+i] = c[i]*A_grad_decay[i, :, 2]
            for i in range(num_comp_osc):
                grad_osc[:, 2+i] = \
                    c[num_comp+1*base+i]*A_grad_osc[i, :, 2] + \
                        c[num_comp+1*base+num_comp_osc+i] * \
                            A_grad_osc[num_comp_osc+i, :, 2]
                grad_osc[:, 2+num_comp_osc+i] = \
                    c[num_comp+1*base+i]*A_grad_osc[i, :, 3] + \
                        c[num_comp+1*base+num_comp_osc+i] * \
                            A_grad_osc[num_comp_osc+i, :, 3]

            grad_decay = np.einsum('i,ij->ij', 1/e[:, j], grad_decay)
            grad_osc = np.einsum('i,ij->ij', 1/e[:, j], grad_osc)
            grad_sum = grad_decay[:, :2] + grad_osc[:, :2]

            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_sum[:, 1]

            else:
                cdiff = (c[:num_comp+1*base]@diff +
                         c[num_comp+1*base:]@diff_osc)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad_sum[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad_sum[:, 1]+deta_L*cdiff

            grad[t0_idx] = grad[t0_idx] -chi[end:end+step]@grad_sum[:, 0]
            df[end:end+step, num_irf:num_irf+num_comp] = \
                np.einsum('j,ij->ij', -1/tau**2, grad_decay[:, 2:])
            df[end:end+step, num_irf+num_comp:num_irf+num_comp+num_comp_osc] = \
                np.einsum('j,ij->ij', -1/tau_osc**2,
                          grad_osc[:, 2:2+num_comp_osc])
            df[end:end+step, num_irf+num_comp+num_comp_osc:] = \
                grad_osc[:, 2+num_comp_osc:2+2*num_comp_osc]

            end = end + step

        t0_idx = t0_idx + 1

    mask = np.ones(num_param, dtype=bool)
    mask[num_irf:num_irf+num_dataset] = False
    grad[mask] = chi@df

    if fix_param_idx is not None:
        grad[fix_param_idx] = 0

    return np.sum(chi**2)/2, grad
