'''
res_raise:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of raise_model and instrumental response function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence, Tuple
import numpy as np
from ..mathfun.irf import calc_eta, deriv_eta
from ..mathfun.irf import calc_fwhm, deriv_fwhm
from ..mathfun.A_matrix import make_A_matrix_gau, make_A_matrix_cauchy, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_exp_conv_gau, deriv_exp_conv_cauchy
from ..mathfun.exp_conv_irf import deriv_exp_sum_conv_gau, deriv_exp_sum_conv_cauchy

# residual and gradient function for exponential decay model


def residual_raise(x0: np.ndarray, base: bool, irf: str,
                   t: Optional[Sequence[np.ndarray]] = None,
                   intensity: Optional[Sequence[np.ndarray]] = None, eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_raise
    scipy.optimize.least_squares compatible vector residual function for fitting multiple set of time delay scan with the
    convolution of raise_model :math:`(\\exp(-t/\\tau_{i+1})-\\exp(-t/\\tau_1))` and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{scan}`: time zero of each scan
        * :math:`2+N_{scan}` to :math:`2+N_{scan}+N_{\\tau}`: time constant of each decay component

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{scan}`: time zero of each scan
        * :math:`3+N_{scan}` to :math:`3+N_{scan}+N_{\\tau}`: time constant of each decay component

     num_comp: number of exponential decay component (except base)
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
    tau = x0[num_irf+num_t0:]
    if not base:
        k = 1/tau
    else:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau(ti-t0, fwhm, k)
            elif irf == 'c':
                A = make_A_matrix_cauchy(ti-t0, fwhm, k)
            else:
                A_gau = make_A_matrix_gau(ti-t0, fwhm, k)
                A_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
                A = A_gau + eta*(A_cauchy-A_gau)
            A[1:, :] = A[1:, :] - A[0, :]
            c = fact_anal_A(A[1:, :], d[:, j], e[:, j])
            chi[end:end+d.shape[0]] = ((c@A[1:, :]) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1

    return chi


def res_grad_raise(x0: np.ndarray, num_comp: int, base: bool, irf: str,
                   fix_param_idx: Optional[np.ndarray] = None,
                   t: Optional[Sequence[np.ndarray]] = None,
                   intensity: Optional[Sequence[np.ndarray]] = None,
                   eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_raise
    scipy.optimize.minimize compatible scalar residual and its gradient function for fitting multiple set of time delay scan with the
    sum of convolution of raise_model :math:`(\\exp(-t/\\tau_{i+1})-\\exp(-t/\\tau_1))` and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{scan}`: time zero of each scan
        * :math:`2+N_{scan}` to :math:`2+N_{scan}+N_{\\tau}`: time constant of each decay component

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{scan}`: time zero of each scan
        * :math:`3+N_{scan}` to :math:`3+N_{scan}+N_{\\tau}`: time constant of each decay component

     num_comp: number of exponential decay component (except base)
     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     t: time points for each data set
     fix_param_idx: index for fixed parameter (masked array for `x0`)
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

    tau = x0[num_irf+num_t0:]

    if not base:
        k = 1/tau
    else:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0

    num_param = num_irf+num_t0+num_comp
    chi = np.empty(count)
    df = np.empty((count, tau.size+num_irf))
    grad = np.empty(num_param)

    end = 0
    t0_idx = num_irf

    for ti, d, e in zip(t, intensity, eps):
        step = d.shape[0]
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau(ti-t0, fwhm, k)
            elif irf == 'c':
                A = make_A_matrix_cauchy(ti-t0, fwhm, k)
            else:
                A_gau = make_A_matrix_gau(ti-t0, fwhm, k)
                A_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
                diff = A_cauchy-A_gau
                A = A_gau + eta*diff
            A[1:, :] = A[1:, :] - A[0, :]
            c = fact_anal_A(A[1:, :], d[:, j], e[:, j])
            chi[end:end+step] = (c@A[1:, :]-d[:, j])/e[:, j]

            c_grad = np.hstack((np.array([-np.sum(c)]), c))

            if irf == 'g':
                grad_tmp = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c_grad, base)
            elif irf == 'c':
                grad_tmp = deriv_exp_sum_conv_cauchy(
                    ti-t0, fwhm, 1/tau, c, base)
            else:
                grad_gau = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c_grad, base)
                grad_cauchy = deriv_exp_sum_conv_cauchy(
                    ti-t0, fwhm, 1/tau, c_grad, base)
                grad_tmp = grad_gau + eta*(grad_cauchy-grad_gau)

            grad_tmp = np.einsum('i,ij->ij', 1/e[:, j], grad_tmp)
            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_tmp[:, 1]
            else:
                cdiff = (c_grad@diff)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad_tmp[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad_tmp[:, 1]+deta_L*cdiff
            grad[t0_idx] = -chi[end:end+step]@grad_tmp[:, 0]
            df[end:end+step,
                num_irf:] = np.einsum('j,ij->ij', -1/tau**2, grad_tmp[:, 2:])

            end = end + step
            t0_idx = t0_idx + 1

    mask = np.ones(num_param, dtype=bool)
    mask[num_irf:num_irf+num_t0] = False
    grad[mask] = chi@df

    if fix_param_idx is not None:
        grad[fix_param_idx] = 0

    return np.sum(chi**2)/2, grad

def residual_raise_same_t0(x0: np.ndarray, base: bool, irf: str,
                           t: Optional[Sequence[np.ndarray]] = None,
                           intensity: Optional[Sequence[np.ndarray]] = None,
                           eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_raise_same_t0
    scipy.optimize.least_squares compatible vector residual function
    for fitting multiple set of time delay scan with the
    sum of convolution of raise_model :math:`(\\exp(-t/\\tau_{i+1})-\\exp(-t/\\tau_1))`
    and instrumental response function
    Set Time Zero of every time dset in same dataset same

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{dset}`: time zero of each data set
        * :math:`2+N_{dset}` to :math:`2+N_{dset}+N_{\\tau}`: time constant of each decay component

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{dset}`: time zero of each data set
        * :math:`3+N_{dset}` to :math:`3+N_{dset}+N_{\\tau}`: time constant of each decay component

     num_comp: number of exponential decay component (except base)
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
    else:
        num_irf = 2
        fwhm = calc_fwhm(x0[0], x0[1])
        eta = calc_eta(x0[0], x0[1])

    num_dataset = len(t)
    count = 0
    for i in range(num_dataset):
        count = count + intensity[i].size

    chi = np.empty(count)
    tau = x0[num_irf+num_dataset:]
    if not base:
        k = 1/tau
    else:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        t0 = x0[t0_idx]
        if irf == 'g':
            A = make_A_matrix_gau(ti-t0, fwhm, k)
        elif irf == 'c':
            A = make_A_matrix_cauchy(ti-t0, fwhm, k)
        else:
            A_gau = make_A_matrix_gau(ti-t0, fwhm, k)
            A_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
            A = A_gau + eta*(A_cauchy-A_gau)
        A[1:, :] = A[1:, :] - A[0, :]
        for j in range(d.shape[1]):
            c = fact_anal_A(A[1:, :], d[:, j], e[:, j])
            chi[end:end+d.shape[0]] = ((c@A[1:, :]) - d[:, j])/e[:, j]

            end = end + d.shape[0]
        t0_idx = t0_idx + 1

    return chi

def res_grad_raise_same_t0(x0: np.ndarray, num_comp: int, base: bool, irf: str,
                           fix_param_idx: Optional[np.ndarray] = None,
                           t: Optional[Sequence[np.ndarray]] = None,
                           intensity: Optional[Sequence[np.ndarray]] = None,
                           eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_raise_same_t0
    scipy.optimize.minimize compatible scalar residual
    and its gradient function for fitting multiple set of time delay scan with the
    sum of convolution of raise_model :math:`(\\exp(-t/\\tau_{i+1})-\\exp(-t/\\tau_1))`
    and instrumental response function

    Args:
     x0: initial parameter,
      if irf == 'g','c':

        * 1st: fwhm_(G/L)
        * 2nd to :math:`2+N_{dset}`: time zero of each dataset
        * :math:`2+N_{dset}` to :math:`2+N_{dset}+N_{\\tau}`: time constant of each decay component

      if irf == 'pv':

        * 1st and 2nd: fwhm_G, fwhm_L
        * 3rd to :math:`3+N_{dset}`: time zero of each dataset
        * :math:`3+N_{dset}` to :math:`3+N_{dset}+N_{\\tau}`: time constant of each decay component

     num_comp: number of exponential decay component (except base)
     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     t: time points for each data set
     fix_param_idx: index for fixed parameter (masked array for `x0`)
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

    if base:
        k = np.empty(tau.size+1)
        k[:-1] = 1/tau
        k[-1] = 0
    else:
        k = 1/tau

    num_param = num_irf+num_dataset+num_comp

    chi = np.empty(count)
    df = np.zeros((count, num_irf+num_comp))
    grad = np.zeros(num_param)

    end = 0
    t0_idx = num_irf
    for ti, d, e in zip(t, intensity, eps):
        step = d.shape[0]
        A = np.empty((num_comp+1*base, step))
        A_grad_decay = np.empty((num_comp+1*base, step, 3))
        grad_decay = np.empty((step, num_comp+2))

        t0 = x0[t0_idx]
        if irf == 'g':
            A[:num_comp+1*base, :] = make_A_matrix_gau(ti-t0, fwhm, k)
            for i in range(num_comp):
                A_grad_decay[i, :, :] = deriv_exp_conv_gau(ti-t0, fwhm, k[i])
            if base:
                A_grad_decay[-1, :, :] = deriv_exp_conv_gau(ti-t0, fwhm, 0)
        elif irf == 'c':
            A[:num_comp+1*base, :] = make_A_matrix_cauchy(ti-t0, fwhm, k)
            for i in range(num_comp):
                A_grad_decay[i, :, :] = deriv_exp_conv_cauchy(ti-t0, fwhm, k[i])
            if base:
                A_grad_decay[-1, :, :] = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
        else:
            tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k)
            tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
            diff = tmp_cauchy-tmp_gau
            A[:num_comp+1*base, :] = tmp_gau + eta*diff
            for i in range(num_comp):
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, k[i])
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, k[i])
                A_grad_decay[i, :, :] = tmp_grad_gau + eta*(tmp_grad_cauchy-tmp_grad_gau)
            if base:
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
                A_grad_decay[-1, :, :] = tmp_grad_gau + eta*(tmp_grad_cauchy-tmp_grad_gau)
 
        A[1:, :] = A[1:, :] - A[0, :]

        for j in range(d.shape[1]):
            c = fact_anal_A(A[1:, :], d[:, j], e[:, j])
            chi[end:end+step] = (c@A[1:, :]-d[:, j])/e[:, j]

            c_grad = np.hstack((np.array([-np.sum(c)]), c))

            grad_decay[:, :2] = \
                np.tensordot(c_grad[:num_comp+1*base], A_grad_decay[:, :, :2], axes=1)
            for i in range(num_comp):
                grad_decay[:, 2+i] = c_grad[i]*A_grad_decay[i, :, 2]

            grad_decay = np.einsum('i,ij->ij', 1/e[:, j], grad_decay)

            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_decay[:, 1]

            else:
                cdiff = (c_grad[:num_comp+1*base]@diff)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad_decay[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad_decay[:, 1]+deta_L*cdiff

            grad[t0_idx] = grad[t0_idx] -chi[end:end+step]@grad_decay[:, 0]
            df[end:end+step, num_irf:num_irf+num_comp] = \
                np.einsum('j,ij->ij', -1/tau**2, grad_decay[:, 2:])

            end = end + step

        t0_idx = t0_idx + 1

    mask = np.ones(num_param, dtype=bool)
    mask[num_irf:num_irf+num_dataset] = False
    grad[mask] = chi@df

    if fix_param_idx is not None:
        grad[fix_param_idx] = 0

    return np.sum(chi**2)/2, grad
