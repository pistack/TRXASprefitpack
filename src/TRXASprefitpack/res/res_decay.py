'''
res_decay:
submodule for residual function and gradient for fitting time delay scan with the
convolution of sum of exponential decay and instrumental response function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence, Tuple
import numpy as np
from ..mathfun.irf import calc_eta, deriv_eta
from ..mathfun.irf import calc_fwhm, deriv_fwhm
from ..mathfun.A_matrix import make_A_matrix_gau, make_A_matrix_cauchy, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_exp_conv_gau, deriv_exp_conv_cauchy
from ..mathfun.exp_conv_irf import hess_exp_conv_gau, hess_exp_conv_cauchy
from ..mathfun.exp_conv_irf import deriv_exp_sum_conv_gau, deriv_exp_sum_conv_cauchy

# residual and gradient function for exponential decay model


def residual_decay(x0: np.ndarray, base: bool, irf: str, 
                   tau_mask: Optional[Sequence[np.ndarray]] = None,
                   t: Optional[Sequence[np.ndarray]] = None,
                   intensity: Optional[Sequence[np.ndarray]] = None, 
                   eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_decay
    scipy.optimize.least_squares compatible vector residual function for fitting multiple set of time delay scan with the
    sum of convolution of exponential decay and instrumental response function

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

     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine
     tau_mask (sequence of boolean np.ndarray): whether or not include jth time constant in ith dataset fitting (tau_mask[i][j])
      If base is True, size of tau_mask[i] should be `num_tau+1`.

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
    dset_idx = 0
    for ti, d, e in zip(t, intensity, eps):
        if tau_mask is None:
            tm = np.ones_like(k, dtype=bool)
        else:
            tm = tau_mask[dset_idx]
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau(ti-t0, fwhm, k[tm])
            elif irf == 'c':
                A = make_A_matrix_cauchy(ti-t0, fwhm, k[tm])
            else:
                A_gau = make_A_matrix_gau(ti-t0, fwhm, k[tm])
                A_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k[tm])
                A = A_gau + eta*(A_cauchy-A_gau)
            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
            t0_idx = t0_idx + 1
        dset_idx = dset_idx + 1

    return chi


def res_grad_decay(x0: np.ndarray, num_comp: int, base: bool, irf: str,
                   fix_param_idx: Optional[np.ndarray] = None,
                   tau_mask: Optional[np.ndarray] = None,
                   t: Optional[Sequence[np.ndarray]] = None,
                   intensity: Optional[Sequence[np.ndarray]] = None,
                   eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_decay
    scipy.optimize.minimize compatible scalar residual and its gradient function for fitting multiple set of time delay scan with the
    sum of convolution of exponential decay and instrumental response function

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

     fix_param_idx: index for fixed parameter (masked array for `x0`)
     tau_mask (sequence of boolean np.ndarray): whether or not include jth time constant in ith dataset fitting (tau_mask[i][j])
      If base is True, size of tau_mask[i] should be `num_tau+1`.

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
    dset_idx = 0

    for ti, d, e in zip(t, intensity, eps):
        step = d.shape[0]
        if tau_mask is None:
            tm = np.ones_like(k, dtype=bool)
        else:
            tm = tau_mask[dset_idx]
        for j in range(d.shape[1]):
            t0 = x0[t0_idx]
            if irf == 'g':
                A = make_A_matrix_gau(ti-t0, fwhm, k[tm])
            elif irf == 'c':
                A = make_A_matrix_cauchy(ti-t0, fwhm, k[tm])
            else:
                A_gau = make_A_matrix_gau(ti-t0, fwhm, k[tm])
                A_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k[tm])
                diff = A_cauchy-A_gau
                A = A_gau + eta*diff
            cm = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+step] = (cm@A-d[:, j])/e[:, j]

            c = np.zeros_like(k)
            c[tm] = cm

            if irf == 'g':
                grad_tmp = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c, base)
            elif irf == 'c':
                grad_tmp = deriv_exp_sum_conv_cauchy(
                    ti-t0, fwhm, 1/tau, c, base)
            else:
                grad_gau = deriv_exp_sum_conv_gau(ti-t0, fwhm, 1/tau, c, base)
                grad_cauchy = deriv_exp_sum_conv_cauchy(
                    ti-t0, fwhm, 1/tau, c, base)
                grad_tmp = grad_gau + eta*(grad_cauchy-grad_gau)

            grad_tmp = np.einsum('i,ij->ij', 1/e[:, j], grad_tmp)
            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_tmp[:, 1]
            else:
                cdiff = (c@diff)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad_tmp[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad_tmp[:, 1]+deta_L*cdiff
            grad[t0_idx] = -chi[end:end+step]@grad_tmp[:, 0]
            df[end:end+step,
                num_irf:] = np.einsum('j,ij->ij', -1/tau**2, grad_tmp[:, 2:])

            end = end + step
            t0_idx = t0_idx + 1
        dset_idx = dset_idx+1

    mask = np.ones(num_param, dtype=bool)
    mask[num_irf:num_irf+num_t0] = False
    grad[mask] = chi@df

    if fix_param_idx is not None:
        grad[fix_param_idx] = 0

    return np.sum(chi**2)/2, grad

def res_hess_decay(x0: np.ndarray, num_comp: int, base: bool, irf: str,
                   tau_mask: Optional[np.ndarray] = None,
                   t: Optional[Sequence[np.ndarray]] = None,
                   intensity: Optional[Sequence[np.ndarray]] = None,
                   eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_hess_decay
     Hessian for fitting multiple set of time delay scan with the
    sum of convolution of exponential decay and instrumental response function

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

     tau_mask (sequence of boolean np.ndarray): whether or not include jth time constant in ith dataset fitting (tau_mask[i][j])
      If base is True, size of tau_mask[i] should be `num_tau+1`.

     t: time points for each data set
     intensity: sequence of intensity of datasets
     eps: sequence of estimated error of datasets

    Returns:
     Hessian of scalar residual function :math:`(\\frac{1}{2}\\sum_i {res}^2_i)` based on the seperation scheme
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
    Hcx = np.zeros((tau.size+1*base, tau.size+num_irf+1))
    Hcorr = np.zeros((num_param, num_param))
    Hx_1st = np.zeros((num_param, num_param))
    Hx_2nd = np.zeros((num_param, num_param))

    t0_idx = num_irf
    dset_idx = 0

    for ti, d, e in zip(t, intensity, eps):
        if tau_mask is None:
            tm = np.ones_like(k, dtype=bool)
        else:
            tm = tau_mask[dset_idx]
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
            cm = fact_anal_A(A[tm, :], d[:, j], e[:, j])
            chi = (cm@A[tm, :]-d[:, j])/e[:, j]

            c = np.zeros_like(k)
            c[tm] = cm

            dc = np.einsum('ij,j->ij', A[tm, :], 1/e[:, j])
            Hc = dc @ dc.T
            
            grad_sum = np.zeros((chi.size, 1+num_irf+tau.size))

            if irf == 'g':
                for i in range(tau.size):
                    tmp_grad = deriv_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                    tmp_hess = c[i]*hess_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                    tmp_hess[:, 1] = -tmp_hess[:, 1] # d^2 f / d(-t)d(fwhm)
                    tmp_hess[:, 2] = tmp_hess[:, 2]/tau[i]**2 # d^2 f / d(-t)d(1/k)
                    tmp_hess[:, 4] = -tmp_hess[:, 4]/tau[i]**2 # d^2 f / d(fwhm)d(1/k)
                    tmp_hess[:, 5] = (tmp_hess[:, 5]/tau[i]+2*tmp_grad[:, 2])/tau[i]**3 # d^2 f / d(1/k)^2
                    tmp_grad[:, 0] = -tmp_grad[:, 0]
                    tmp_grad[:, 2] = -tmp_grad[:, 2]/tau[i]**2 # d f / d(1/k)

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[i, 0] = tmp_chi_grad[1] #fwhm
                    Hcx[i, 1] = tmp_chi_grad[0] #t
                    Hcx[i, i+2] = tmp_chi_grad[2] #tau_i

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[3] #d(fwhm)^2
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[1] # dtd(fwhm)
                    Hx_2nd[0, 1+num_t0+i] = Hx_2nd[0, 1+num_t0+i] + tmp_chi_hess[4] #d(fwhm)d(tau)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[0] # dt^2
                    Hx_2nd[t0_idx, 1+num_t0+i] = tmp_chi_hess[2] # dt dtau_i
                    Hx_2nd[1+num_t0+i, 1+num_t0+i] = \
                        Hx_2nd[1+num_t0+i, 1+num_t0+i] + tmp_chi_hess[5] # d(tau_i)^2 

                    #Jf
                    grad_sum[:, :2] = grad_sum[:, :2]+c[i]*tmp_grad[:, :2]
                    grad_sum[:, 2+i] = c[i]*tmp_grad[:, 2]
                
                if base:
                    tmp_grad = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                    tmp_hess = c[-1]*hess_exp_conv_gau(ti-t0, fwhm, 0)
                    tmp_hess[:, 1] = -tmp_hess[:, 1] # d^2 f / d(-t)d(fwhm)
                    tmp_grad[:, 0] = -tmp_grad[:, 0]

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[-1, 0] = tmp_chi_grad[1] #fwhm
                    Hcx[-1, 1] = tmp_chi_grad[0] #t

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[3] #d(fwhm)^2
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[1] # dtd(fwhm)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[0] # dt^2

                    #Jf
                    grad_sum[:, :2] = grad_sum[:, :2]+c[-1]*tmp_grad[:, :2]
                
                Hx_1st_tmp = grad_sum.T @ grad_sum

                b = np.linalg.solve(Hc, Hcx)
                b[~tm, :] = 0
                Hcorr_tmp = b.T @ Hcx

                # fwhm
                Hx_1st[0, 0] = Hx_1st[0, 0] + Hx_1st_tmp[0, 0]
                Hcorr[0, 0] = Hcorr[0, 0] + Hcorr_tmp[0, 0]

                Hx_1st[0, t0_idx] = Hx_1st_tmp[0, 1]
                Hcorr[0, t0_idx] = Hcorr[0, 1]

                Hx_1st[0, 1+num_t0:] = Hx_1st[0, 1+num_t0:] + Hx_1st_tmp[0, 2:]
                Hcorr[0, 1+num_t0:] = Hcorr[0, 1+num_t0:] + Hcorr_tmp[0, 2:]

                # t0
                Hx_1st[t0_idx, t0_idx] = Hx_1st_tmp[1, 1]
                Hcorr[t0_idx, t0_idx] = Hcorr_tmp[1, 1]

                Hx_1st[t0_idx, 1+num_t0:] = Hx_1st_tmp[1, 2:]
                Hcorr[t0_idx, 1+num_t0:] = Hcorr_tmp[1, 2:]

                # tau
                Hx_1st[1+num_t0:, 1+num_t0:] = \
                    Hx_1st[1+num_t0:, 1+num_t0:] + Hx_1st_tmp[2:, 2:]
                Hcorr[1+num_t0:, 1+num_t0:] = \
                    Hcorr[1+num_t0:, 1+num_t0:] + Hcorr_tmp[2:, 2:]


            t0_idx = t0_idx + 1
        dset_idx = dset_idx+1

        for i in range(num_param):
            for j in range(i+1, num_param):
                Hx_1st[j, i] = Hx_1st[i, j]
                Hx_2nd[j, i] = Hx_2nd[i, j]
                Hcorr[j, i] = Hcorr[i, j]
    
    H = Hx_1st + Hx_2nd - Hcorr

    return H

def residual_decay_same_t0(x0: np.ndarray, base: bool, irf: str,
                           tau_mask: Optional[Sequence[np.ndarray]] = None,
                           t: Optional[Sequence[np.ndarray]] = None,
                           intensity: Optional[Sequence[np.ndarray]] = None,
                           eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    residual_decay_same_t0
    scipy.optimize.least_squares compatible vector residual function
    for fitting multiple set of time delay scan with the
    sum of convolution of exponential decay and instrumental response function
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

     base: whether or not include baseline (i.e. very long lifetime component)
     irf: shape of instrumental response function

          * 'g': normalized gaussian distribution,
          * 'c': normalized cauchy distribution,
          * 'pv': pseudo voigt profile :math:`(1-\\eta)g(f) + \\eta c(f)`

        For pseudo voigt profile, the mixing parameter :math:`\\eta(f_G, f_L)` and
        uniform fwhm paramter :math:`f(f_G, f_L)` are calculated by `calc_eta` and `calc_fwhm` routine

     tau_mask (sequence of boolean np.ndarray): whether or not include jth time constant in ith dataset fitting (tau_mask[i][j])
      If base is True, size of tau_mask[i] should be `num_tau+1`.

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
    dset_idx = 0
    for ti, d, e in zip(t, intensity, eps):
        t0 = x0[t0_idx]
        if tau_mask is None:
            tm = np.ones_like(k, dtype=bool)
        else:
            tm = tau_mask[dset_idx]
        if irf == 'g':
            A = make_A_matrix_gau(ti-t0, fwhm, k[tm])
        elif irf == 'c':
            A = make_A_matrix_cauchy(ti-t0, fwhm, k[tm])
        else:
            A_gau = make_A_matrix_gau(ti-t0, fwhm, k[tm])
            A_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k[tm])
            A = A_gau + eta*(A_cauchy-A_gau)
        for j in range(d.shape[1]):
            c = fact_anal_A(A, d[:, j], e[:, j])
            chi[end:end+d.shape[0]] = ((c@A) - d[:, j])/e[:, j]

            end = end + d.shape[0]
        t0_idx = t0_idx + 1
        dset_idx = dset_idx + 1

    return chi

def res_grad_decay_same_t0(x0: np.ndarray, num_comp: int, base: bool, irf: str,
                           fix_param_idx: Optional[np.ndarray] = None,
                           tau_mask: Optional[np.ndarray] = None,
                           t: Optional[Sequence[np.ndarray]] = None,
                           intensity: Optional[Sequence[np.ndarray]] = None,
                           eps: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    res_grad_decay_same_t0
    scipy.optimize.minimize compatible scalar residual
    and its gradient function for fitting multiple set of time delay scan with the
    sum of convolution of exponential decay and instrumental response function

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
     fix_param_idx: index for fixed parameter (masked array for `x0`)
     tau_mask (sequence of boolean np.ndarray): whether or not include jth time constant in ith dataset fitting (tau_mask[i][j])
      If base is True, size of tau_mask[i] should be `num_tau+1`.
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
    dset_idx = 0
    for ti, d, e in zip(t, intensity, eps):

        # initialize
        step = d.shape[0]
        t0 = x0[t0_idx]
        A = np.empty((num_comp+1*base, step))
        A_grad_decay = np.empty((num_comp+1*base, step, 3))
        grad_decay = np.empty((step, num_comp+2))

        if tau_mask is None:
            tm = np.ones_like(k, dtype=bool)
        else:
            tm = tau_mask[dset_idx]

        # caching
        if irf == 'g':
            A = make_A_matrix_gau(ti-t0, fwhm, k)
            for i in range(num_comp):
                A_grad_decay[i, :, :] = deriv_exp_conv_gau(ti-t0, fwhm, k[i])
            if base:
                A_grad_decay[-1, :, :] = deriv_exp_conv_gau(ti-t0, fwhm, 0)
        elif irf == 'c':
            A = make_A_matrix_cauchy(ti-t0, fwhm, k)
            for i in range(num_comp):
                A_grad_decay[i, :, :] = deriv_exp_conv_cauchy(ti-t0, fwhm, k[i])
            if base:
                A_grad_decay[-1, :, :] = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
        else:
            tmp_gau = make_A_matrix_gau(ti-t0, fwhm, k)
            tmp_cauchy = make_A_matrix_cauchy(ti-t0, fwhm, k)
            diff = tmp_cauchy-tmp_gau
            A = tmp_gau + eta*diff
            for i in range(num_comp):
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, k[i])
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, k[i])
                A_grad_decay[i, :, :] = tmp_grad_gau + eta*(tmp_grad_cauchy-tmp_grad_gau)
            if base:
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
                A_grad_decay[-1, :, :] = tmp_grad_gau + eta*(tmp_grad_cauchy-tmp_grad_gau)

        for j in range(d.shape[1]):
            cm = fact_anal_A(A[tm, :], d[:, j], e[:, j])
            chi[end:end+step] = (cm@A[tm, :]-d[:, j])/e[:, j]

            c = np.zeros_like(k)
            c[tm] = cm

            grad_decay[:, :2] = \
                np.tensordot(c, A_grad_decay[:, :, :2], axes=1)
            for i in range(num_comp):
                grad_decay[:, 2+i] = c[i]*A_grad_decay[i, :, 2]

            grad_decay = np.einsum('i,ij->ij', 1/e[:, j], grad_decay)

            if irf in ['g', 'c']:
                df[end:end+step, 0] = grad_decay[:, 1]

            else:
                cdiff = (c@diff)/e[:, j]
                df[end:end+step, 0] = dfwhm_G*grad_decay[:, 1]+deta_G*cdiff
                df[end:end+step, 1] = dfwhm_L*grad_decay[:, 1]+deta_L*cdiff

            grad[t0_idx] = grad[t0_idx] -chi[end:end+step]@grad_decay[:, 0]
            df[end:end+step, num_irf:num_irf+num_comp] = \
                np.einsum('j,ij->ij', -1/tau**2, grad_decay[:, 2:])

            end = end + step

        t0_idx = t0_idx + 1
        dset_idx = dset_idx + 1

    mask = np.ones(num_param, dtype=bool)
    mask[num_irf:num_irf+num_dataset] = False
    grad[mask] = chi@df

    if fix_param_idx is not None:
        grad[fix_param_idx] = 0

    return np.sum(chi**2)/2, grad
