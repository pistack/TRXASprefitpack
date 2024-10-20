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
from ..mathfun.irf import hess_fwhm_eta
from ..mathfun.A_matrix import make_A_matrix_gau, make_A_matrix_cauchy, fact_anal_A
from ..mathfun.exp_conv_irf import deriv_exp_conv_gau, deriv_exp_conv_cauchy
from ..mathfun.exp_conv_irf import deriv_exp_sum_conv_gau, deriv_exp_sum_conv_cauchy
from ..mathfun.exp_conv_irf import hess_exp_conv_gau, hess_exp_conv_cauchy

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

def res_hess_raise(x0: np.ndarray, num_comp: int, base: bool, irf: str,
                   t: Optional[Sequence[np.ndarray]] = None,
                   intensity: Optional[Sequence[np.ndarray]] = None,
                   eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    res_hess_raise
     Hessian for fitting multiple set of time delay scan with the
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
        hess_fwhm, hess_eta = hess_fwhm_eta(x0[0], x0[1])

    num_t0 = 0
    for d in intensity:
        num_t0 = num_t0 + d.shape[1]

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

        grad_sum = np.zeros((d.shape[0], 1+num_irf+tau.size))

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
            cm = fact_anal_A(A[1:, :], d[:, j], e[:, j])
            chi = (cm@A[1:, :]-d[:, j])/e[:, j]

            c = np.zeros_like(k)
            c[1:] = cm
            c[0] = -np.sum(cm)

            dc = np.einsum('ij,j->ij', A, 1/e[:, j])
            Hc = dc @ dc.T
            
            grad_sum[:, :] = 0
            Hcx[:, :] = 0

            if irf in ['g', 'c']:
                for i in range(tau.size):
                    if irf == 'g':
                        tmp_grad = deriv_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                        tmp_hess = c[i]*hess_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                    else:
                        tmp_grad = deriv_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                        tmp_hess = c[i]*hess_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                    
                    tmp_hess[:, 1] = -tmp_hess[:, 1] # d^2 f / d(-t)d(fwhm)
                    tmp_hess[:, 2] = tmp_hess[:, 2]/tau[i]**2 # d^2 f / d(-t)d(1/k)
                    tmp_hess[:, 4] = -tmp_hess[:, 4]/tau[i]**2 # d^2 f / d(fwhm)d(1/k)
                    tmp_hess[:, 5] = (tmp_hess[:, 5]/tau[i]+2*c[i]*tmp_grad[:, 2])/tau[i]**3 # d^2 f / d(1/k)^2
                    tmp_grad[:, 0] = -tmp_grad[:, 0] # df/d(-t)
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
                    grad_sum[:, 0] = grad_sum[:, 0]+c[i]*tmp_grad[:, 1]
                    grad_sum[:, 1] = grad_sum[:, 1]+c[i]*tmp_grad[:, 0]
                    grad_sum[:, 2+i] = c[i]*tmp_grad[:, 2]
                
                if base:
                    if irf == 'g':
                        tmp_grad = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                        tmp_hess = c[-1]*hess_exp_conv_gau(ti-t0, fwhm, 0)
                    else:
                        tmp_grad = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
                        tmp_hess = c[-1]*hess_exp_conv_cauchy(ti-t0, fwhm, 0)
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
                    grad_sum[:, 0] = grad_sum[:, 0]+c[-1]*tmp_grad[:, 1]
                    grad_sum[:, 1] = grad_sum[:, 1]+c[-1]*tmp_grad[:, 0]
            else:
                for i in range(tau.size):
                    tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                    tmp_hess_gau = c[i]*hess_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                    tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                    tmp_hess_cauchy = c[i]*hess_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                    
                    tmp_hess_gau[:, 1] = -tmp_hess_gau[:, 1] # d^2 f / d(-t)d(fwhm)
                    tmp_hess_gau[:, 2] = tmp_hess_gau[:, 2]/tau[i]**2 # d^2 f / d(-t)d(1/k)
                    tmp_hess_gau[:, 4] = -tmp_hess_gau[:, 4]/tau[i]**2 # d^2 f / d(fwhm)d(1/k)
                    tmp_hess_gau[:, 5] = \
                        (tmp_hess_gau[:, 5]/tau[i]+2*c[i]*tmp_grad_gau[:, 2])/tau[i]**3 # d^2 f / d(1/k)^2
                    tmp_grad_gau[:, 0] = -tmp_grad_gau[:, 0] # df/d(-t)
                    tmp_grad_gau[:, 2] = -tmp_grad_gau[:, 2]/tau[i]**2 # d f / d(1/k)

                    tmp_hess_cauchy[:, 1] = -tmp_hess_cauchy[:, 1] # d^2 f / d(-t)d(fwhm)
                    tmp_hess_cauchy[:, 2] = tmp_hess_cauchy[:, 2]/tau[i]**2 # d^2 f / d(-t)d(1/k)
                    tmp_hess_cauchy[:, 4] = -tmp_hess_cauchy[:, 4]/tau[i]**2 # d^2 f / d(fwhm)d(1/k)
                    tmp_hess_cauchy[:, 5] = \
                        (tmp_hess_cauchy[:, 5]/tau[i]+2*c[i]*tmp_grad_cauchy[:, 2])/tau[i]**3 # d^2 f / d(1/k)^2
                    tmp_grad_cauchy[:, 0] = -tmp_grad_cauchy[:, 0] # df/d(-t)
                    tmp_grad_cauchy[:, 2] = -tmp_grad_cauchy[:, 2]/tau[i]**2 # d f / d(1/k)

                    tmp_grad = np.zeros((chi.size, 4)) # fwhm_G fwhm_L t tau
                    tmp_hess = np.zeros((chi.size, 10))

                    ## Construct tmp_grad and tmp_hess for pseudo voigt

                    # gradient
                    tmp_grad[:, 0] = dfwhm_G*\
                        (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                        deta_G*diff[i, :]
                    tmp_grad[:, 1] = dfwhm_L*\
                        (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                        deta_L*diff[i, :]
                    tmp_grad[:, 2] = tmp_grad_gau[:, 0]+\
                        eta*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])
                    tmp_grad[:, 3] = tmp_grad_gau[:, 2]+\
                        eta*(tmp_grad_cauchy[:, 2]-tmp_grad_gau[:, 2])
                    
                    #hessian

                    # fwhm_G, fwhm_G
                    tmp_hess[:, 0] = c[i]*hess_fwhm[0]*\
                    (tmp_grad_gau[:, 1]+
                     eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    dfwhm_G*(dfwhm_G*(tmp_hess_gau[:, 3]+
                    eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                    2*c[i]*deta_G*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    c[i]*hess_eta[0]*diff[i, :]

                    # fwhm_L, fwhm_L
                    tmp_hess[:, 4] = c[i]*hess_fwhm[2]*\
                    (tmp_grad_gau[:, 1]+
                     eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    dfwhm_L*(dfwhm_L*(tmp_hess_gau[:, 3]+
                    eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                    2*c[i]*deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    c[i]*hess_eta[2]*diff[i, :]

                    # fwhm_G, fwhm_L
                    tmp_hess[:, 1] = c[i]*hess_fwhm[1]*\
                    (tmp_grad_gau[:, 1]+
                    eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    dfwhm_G*(dfwhm_L*(tmp_hess_gau[:, 3]+
                    eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                    c[i]*deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    c[i]*hess_eta[1]*diff[i, :] + \
                    c[i]*deta_G*dfwhm_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])

                    # fwhm_G, other
                    tmp_hess[:, 2] = c[i]*deta_G*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                    dfwhm_G*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))
                    tmp_hess[:, 3] = c[i]*deta_G*(tmp_grad_cauchy[:, 2]-tmp_grad_gau[:, 2])+\
                    dfwhm_G*(tmp_hess_gau[:, 4]+eta*(tmp_hess_cauchy[:, 4]-tmp_hess_gau[:, 4]))

                    # fwhm_L, other
                    tmp_hess[:, 5] = c[i]*deta_L*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                    dfwhm_L*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))
                    tmp_hess[:, 6] = c[i]*deta_L*(tmp_grad_cauchy[:, 2]-tmp_grad_gau[:, 2])+\
                    dfwhm_L*(tmp_hess_gau[:, 4]+eta*(tmp_hess_cauchy[:, 4]-tmp_hess_gau[:, 4]))

                    # other, other
                    tmp_hess[:, 7] = tmp_hess_gau[:, 0]+\
                    eta*(tmp_hess_cauchy[:, 0]-tmp_hess_gau[:, 0])
                    tmp_hess[:, 8] = tmp_hess_gau[:, 2]+\
                    eta*(tmp_hess_cauchy[:, 2]-tmp_hess_gau[:, 2])
                    tmp_hess[:, 9] = tmp_hess_gau[:, 5]+\
                    eta*(tmp_hess_cauchy[:, 5]-tmp_hess_gau[:, 5])

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[i, :num_irf+1] = tmp_chi_grad[:3] #fwhm_(G,L) t
                    Hcx[i, i+1+num_irf] = tmp_chi_grad[3] #tau_i

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[0] #d(fwhm_G)^2
                    Hx_2nd[0, 1] = Hx_2nd[0, 1] + tmp_chi_hess[1] # d(fwhm_G)d(fwhm_L)
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[2] # dtd(fwhm_G)
                    Hx_2nd[0, num_irf+num_t0+i] = \
                        Hx_2nd[0, num_irf+num_t0+i] + tmp_chi_hess[3] #d(fwhm_G)d(tau)
                    Hx_2nd[1, 1] = Hx_2nd[1, 1] + tmp_chi_hess[4] #d(fwhm_L)^2
                    Hx_2nd[1, t0_idx] = Hx_2nd[1, t0_idx] + tmp_chi_hess[5] # dtd(fwhm_L)
                    Hx_2nd[1, num_irf+num_t0+i] = \
                        Hx_2nd[1, num_irf+num_t0+i] + tmp_chi_hess[6] #d(fwhm_L)d(tau)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[7] # dt^2
                    Hx_2nd[t0_idx, num_irf+num_t0+i] = tmp_chi_hess[8] # dt dtau_i
                    Hx_2nd[num_irf+num_t0+i, num_irf+num_t0+i] = \
                        Hx_2nd[num_irf+num_t0+i, num_irf+num_t0+i] + tmp_chi_hess[9] # d(tau_i)^2 

                    #Jf
                    grad_sum[:, :num_irf+1] = \
                        grad_sum[:, :num_irf+1]+c[i]*tmp_grad[:, :num_irf+1]
                    grad_sum[:, 1+i+num_irf] = c[i]*tmp_grad[:, 3]
                
                if base:
                    tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                    tmp_hess_gau = c[-1]*hess_exp_conv_gau(ti-t0, fwhm, 0)
                    tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
                    tmp_hess_cauchy = c[-1]*hess_exp_conv_cauchy(ti-t0, fwhm, 0)
                    
                    tmp_hess_gau[:, 1] = -tmp_hess_gau[:, 1] # d^2 f / d(-t)d(fwhm)
                    tmp_grad_gau[:, 0] = -tmp_grad_gau[:, 0] # df/d(-t)

                    tmp_hess_cauchy[:, 1] = -tmp_hess_cauchy[:, 1] # d^2 f / d(-t)d(fwhm)
                    tmp_grad_cauchy[:, 0] = -tmp_grad_cauchy[:, 0] # df/d(-t)

                    tmp_grad = np.zeros((chi.size, 3)) # fwhm_G fwhm_L t tau
                    tmp_hess = np.zeros((chi.size, 6))

                    ## Construct tmp_grad and tmp_hess for pseudo voigt

                    # gradient
                    tmp_grad[:, 0] = dfwhm_G*\
                        (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                        deta_G*diff[-1, :]
                    tmp_grad[:, 1] = dfwhm_L*\
                        (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                        deta_L*diff[-1, :]
                    tmp_grad[:, 2] = tmp_grad_gau[:, 0]+\
                        eta*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])
                    #hessian

                    # fwhm_G, fwhm_G
                    tmp_hess[:, 0] = c[-1]*hess_fwhm[0]*\
                    (tmp_grad_gau[:, 1]+
                    eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    dfwhm_G*(dfwhm_G*(tmp_hess_gau[:, 3]+
                    eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                    2*c[-1]*deta_G*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    c[-1]*hess_eta[0]*diff[-1, :]

                    # fwhm_L, fwhm_L
                    tmp_hess[:, 3] = c[-1]*hess_fwhm[2]*\
                    (tmp_grad_gau[:, 1]+
                    eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    dfwhm_L*(dfwhm_L*(tmp_hess_gau[:, 3]+
                    eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                    2*c[-1]*deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    c[-1]*hess_eta[2]*diff[-1, :]

                    # fwhm_G, fwhm_L
                    tmp_hess[:, 1] = c[-1]*hess_fwhm[1]*\
                    (tmp_grad_gau[:, 1]+
                    eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    dfwhm_G*(dfwhm_L*(tmp_hess_gau[:, 3]+
                    eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                    c[-1]*deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                    c[-1]*hess_eta[1]*diff[-1, :] + \
                    c[-1]*deta_G*dfwhm_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])

                    # fwhm_G, other
                    tmp_hess[:, 2] = c[-1]*deta_G*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                    dfwhm_G*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))

                    # fwhm_L, other
                    tmp_hess[:, 4] = c[-1]*deta_L*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                    dfwhm_L*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))

                    # other, other
                    tmp_hess[:, 5] = tmp_hess_gau[:, 0]+\
                    eta*(tmp_hess_cauchy[:, 0]-tmp_hess_gau[:, 0])

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[-1, :num_irf+1] = tmp_chi_grad[:3] #fwhm_(G,L) t

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[0] #d(fwhm_G)^2
                    Hx_2nd[0, 1] = Hx_2nd[0, 1] + tmp_chi_hess[1] # d(fwhm_G)d(fwhm_L)
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[2] # dtd(fwhm_G)
                    Hx_2nd[1, 1] = Hx_2nd[1, 1] + tmp_chi_hess[3] #d(fwhm_L)^2
                    Hx_2nd[1, t0_idx] = Hx_2nd[1, t0_idx] + tmp_chi_hess[4] # dtd(fwhm_L)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[5] # dt^2

                    #Jf
                    grad_sum[:, :num_irf+1] = \
                        grad_sum[:, :num_irf+1]+c[-1]*tmp_grad[:, :num_irf+1]

            # IRF independent part    
            Hx_1st_tmp = grad_sum.T @ grad_sum

            Hcx = Hcx + dc@grad_sum

            Hc_tau = Hc[:k.size, :k.size]
            Hcx_tau = np.zeros((k.size, 1+num_irf+num_comp))
            Hcx_tau[:, :] = Hcx[:k.size, :num_irf+1+num_comp]
            b = np.linalg.solve(Hc_tau, Hcx_tau)
            Hcorr_tmp = b.T @ (Hcx_tau)

            # fwhm
            Hx_1st[:num_irf, :num_irf] = \
                Hx_1st[:num_irf, :num_irf] + Hx_1st_tmp[:num_irf, :num_irf]
            Hcorr[:num_irf, :num_irf] = \
                Hcorr[:num_irf, :num_irf] + Hcorr_tmp[:num_irf, :num_irf]

            Hx_1st[:num_irf, t0_idx] = Hx_1st_tmp[:num_irf, num_irf]
            Hcorr[:num_irf, t0_idx] = Hcorr_tmp[:num_irf, num_irf]

            Hx_1st[:num_irf, num_irf+num_t0:] = \
                Hx_1st[:num_irf, num_irf+num_t0:] + \
                    Hx_1st_tmp[:num_irf, num_irf+1:]
            Hcorr[:num_irf, num_irf+num_t0:num_irf+num_t0+num_comp] = \
                Hcorr[:num_irf, num_irf+num_t0:num_irf+num_t0+num_comp] + \
                    Hcorr_tmp[:num_irf, num_irf+1:]

            # t0
            Hx_1st[t0_idx, t0_idx] = Hx_1st_tmp[num_irf, num_irf]
            Hcorr[t0_idx, t0_idx] = Hcorr_tmp[num_irf, num_irf]

            Hx_1st[t0_idx, num_irf+num_t0:] = \
                Hx_1st_tmp[num_irf, 1+num_irf:]
            Hcorr[t0_idx, num_irf+num_t0:num_irf+num_t0+num_comp] = \
                Hcorr_tmp[num_irf, 1+num_irf:]

            # tau
            Hx_1st[num_irf+num_t0:, num_irf+num_t0:] = \
                Hx_1st[num_irf+num_t0:, num_irf+num_t0:] + \
                    Hx_1st_tmp[num_irf+1:, num_irf+1:]
            Hcorr[num_irf+num_t0:num_irf+num_t0+num_comp, 
                  num_irf+num_t0:num_irf+num_t0+num_comp] = \
                Hcorr[num_irf+num_t0:num_irf+num_t0+num_comp, 
                      num_irf+num_t0:num_irf+num_t0+num_comp] + \
                    Hcorr_tmp[num_irf+1:, num_irf+1:]

            t0_idx = t0_idx + 1
        dset_idx = dset_idx+1

    H = Hx_1st + Hx_2nd - Hcorr
 
    for i in range(num_param):
        for j in range(i+1, num_param):
            H[j, i] = H[i, j]
 
    return H

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


def res_hess_raise_same_t0(x0: np.ndarray, num_comp: int, base: bool, irf: str,
                   t: Optional[Sequence[np.ndarray]] = None,
                   intensity: Optional[Sequence[np.ndarray]] = None,
                   eps: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    '''
    res_hess_raise_same_t0
     Hessian for fitting multiple set of time delay scan with the
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
     fix_param_idx: index for fixed parameter (masked array for `x0`)
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
        hess_fwhm, hess_eta = hess_fwhm_eta(x0[0], x0[1])

    num_t0 = len(intensity) 
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

        cache_grad = np.zeros((d.shape[0], num_irf+2, k.size))
        cache_hess = np.zeros((d.shape[0], 
                               ((num_irf+3)*(num_irf+2))//2, k.size))
        grad_sum = np.zeros((d.shape[0], 1+num_irf+tau.size))
        
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
        
        # caching
        if irf in ['g', 'c']:
            for i in range(tau.size):
                if irf == 'g':
                    tmp_grad = deriv_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                    tmp_hess = hess_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                else:
                    tmp_grad = deriv_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                    tmp_hess = hess_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                    
                tmp_hess[:, 1] = -tmp_hess[:, 1] # d^2 f / d(-t)d(fwhm)
                tmp_hess[:, 2] = tmp_hess[:, 2]/tau[i]**2 # d^2 f / d(-t)d(1/k)
                tmp_hess[:, 4] = -tmp_hess[:, 4]/tau[i]**2 # d^2 f / d(fwhm)d(1/k)
                tmp_hess[:, 5] = (tmp_hess[:, 5]/tau[i]+2*tmp_grad[:, 2])/tau[i]**3 # d^2 f / d(1/k)^2
                tmp_grad[:, 0] = -tmp_grad[:, 0] # df/d(-t)
                tmp_grad[:, 2] = -tmp_grad[:, 2]/tau[i]**2 # d f / d(1/k)

                cache_grad[:, 0, i] = tmp_grad[:, 1]
                cache_grad[:, 1, i] = tmp_grad[:, 0]
                cache_grad[:, 2, i] = tmp_grad[:, 2]

                cache_hess[:, 0, i] = tmp_hess[:, 3]
                cache_hess[:, 1, i] = tmp_hess[:, 1]
                cache_hess[:, 2, i] = tmp_hess[:, 4]
                cache_hess[:, 3, i] = tmp_hess[:, 0]
                cache_hess[:, 4, i] = tmp_hess[:, 2]
                cache_hess[:, 5, i] = tmp_hess[:, 5]
                
            if base:
                if irf == 'g':
                    tmp_grad = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                    tmp_hess = hess_exp_conv_gau(ti-t0, fwhm, 0)
                else:
                    tmp_grad = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
                    tmp_hess = hess_exp_conv_cauchy(ti-t0, fwhm, 0)
                tmp_hess[:, 1] = -tmp_hess[:, 1] # d^2 f / d(-t)d(fwhm)
                tmp_grad[:, 0] = -tmp_grad[:, 0]

                cache_grad[:, 0, -1] = tmp_grad[:, 1]
                cache_grad[:, 1, -1] = tmp_grad[:, 0]

                cache_hess[:, 0, -1] = tmp_hess[:, 3]
                cache_hess[:, 1, -1] = tmp_hess[:, 1]
                cache_hess[:, 2, -1] = tmp_hess[:, 0]
        else:
            for i in range(tau.size):
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                tmp_hess_gau = hess_exp_conv_gau(ti-t0, fwhm, 1/tau[i])
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                tmp_hess_cauchy = hess_exp_conv_cauchy(ti-t0, fwhm, 1/tau[i])
                    
                tmp_hess_gau[:, 1] = -tmp_hess_gau[:, 1] # d^2 f / d(-t)d(fwhm)
                tmp_hess_gau[:, 2] = tmp_hess_gau[:, 2]/tau[i]**2 # d^2 f / d(-t)d(1/k)
                tmp_hess_gau[:, 4] = -tmp_hess_gau[:, 4]/tau[i]**2 # d^2 f / d(fwhm)d(1/k)
                tmp_hess_gau[:, 5] = \
                    (tmp_hess_gau[:, 5]/tau[i]+2*tmp_grad_gau[:, 2])/tau[i]**3 # d^2 f / d(1/k)^2
                tmp_grad_gau[:, 0] = -tmp_grad_gau[:, 0] # df/d(-t)
                tmp_grad_gau[:, 2] = -tmp_grad_gau[:, 2]/tau[i]**2 # d f / d(1/k)

                tmp_hess_cauchy[:, 1] = -tmp_hess_cauchy[:, 1] # d^2 f / d(-t)d(fwhm)
                tmp_hess_cauchy[:, 2] = tmp_hess_cauchy[:, 2]/tau[i]**2 # d^2 f / d(-t)d(1/k)
                tmp_hess_cauchy[:, 4] = -tmp_hess_cauchy[:, 4]/tau[i]**2 # d^2 f / d(fwhm)d(1/k)
                tmp_hess_cauchy[:, 5] = \
                    (tmp_hess_cauchy[:, 5]/tau[i]+2*tmp_grad_cauchy[:, 2])/tau[i]**3 # d^2 f / d(1/k)^2
                tmp_grad_cauchy[:, 0] = -tmp_grad_cauchy[:, 0] # df/d(-t)
                tmp_grad_cauchy[:, 2] = -tmp_grad_cauchy[:, 2]/tau[i]**2 # d f / d(1/k)

                ## Construct tmp_grad and tmp_hess for pseudo voigt

                # gradient
                cache_grad[:, 0, i] = dfwhm_G*\
                    (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                        deta_G*diff[i, :]
                cache_grad[:, 1, i] = dfwhm_L*\
                    (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                        deta_L*diff[i, :]
                cache_grad[:, 2, i] = tmp_grad_gau[:, 0]+\
                    eta*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])
                cache_grad[:, 3, i] = tmp_grad_gau[:, 2]+\
                    eta*(tmp_grad_cauchy[:, 2]-tmp_grad_gau[:, 2])
                    
                #hessian

                # fwhm_G, fwhm_G
                cache_hess[:, 0, i] = hess_fwhm[0]*\
                (tmp_grad_gau[:, 1]+
                 eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                dfwhm_G*(dfwhm_G*(tmp_hess_gau[:, 3]+
                eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                2*deta_G*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                hess_eta[0]*diff[i, :]

                # fwhm_L, fwhm_L
                cache_hess[:, 4, i] = hess_fwhm[2]*\
                (tmp_grad_gau[:, 1]+
                 eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                dfwhm_L*(dfwhm_L*(tmp_hess_gau[:, 3]+
                eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                2*deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                hess_eta[2]*diff[i, :]

                # fwhm_G, fwhm_L
                cache_hess[:, 1, i] = hess_fwhm[1]*\
                (tmp_grad_gau[:, 1]+
                 eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                dfwhm_G*(dfwhm_L*(tmp_hess_gau[:, 3]+
                eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                hess_eta[1]*diff[i, :] + \
                deta_G*dfwhm_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])

                # fwhm_G, other
                cache_hess[:, 2, i] = deta_G*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                dfwhm_G*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))
                cache_hess[:, 3, i] = deta_G*(tmp_grad_cauchy[:, 2]-tmp_grad_gau[:, 2])+\
                dfwhm_G*(tmp_hess_gau[:, 4]+eta*(tmp_hess_cauchy[:, 4]-tmp_hess_gau[:, 4]))

                # fwhm_L, other
                cache_hess[:, 5, i] = deta_L*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                dfwhm_L*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))
                cache_hess[:, 6, i] = deta_L*(tmp_grad_cauchy[:, 2]-tmp_grad_gau[:, 2])+\
                dfwhm_L*(tmp_hess_gau[:, 4]+eta*(tmp_hess_cauchy[:, 4]-tmp_hess_gau[:, 4]))

                # other, other
                cache_hess[:, 7, i] = tmp_hess_gau[:, 0]+\
                eta*(tmp_hess_cauchy[:, 0]-tmp_hess_gau[:, 0])
                cache_hess[:, 8, i] = tmp_hess_gau[:, 2]+\
                    eta*(tmp_hess_cauchy[:, 2]-tmp_hess_gau[:, 2])
                cache_hess[:, 9, i] = tmp_hess_gau[:, 5]+\
                    eta*(tmp_hess_cauchy[:, 5]-tmp_hess_gau[:, 5])
            if base:
                tmp_grad_gau = deriv_exp_conv_gau(ti-t0, fwhm, 0)
                tmp_hess_gau = hess_exp_conv_gau(ti-t0, fwhm, 0)
                tmp_grad_cauchy = deriv_exp_conv_cauchy(ti-t0, fwhm, 0)
                tmp_hess_cauchy = hess_exp_conv_cauchy(ti-t0, fwhm, 0)
                    
                tmp_hess_gau[:, 1] = -tmp_hess_gau[:, 1] # d^2 f / d(-t)d(fwhm)
                tmp_grad_gau[:, 0] = -tmp_grad_gau[:, 0] # df/d(-t)

                tmp_hess_cauchy[:, 1] = -tmp_hess_cauchy[:, 1] # d^2 f / d(-t)d(fwhm)
                tmp_grad_cauchy[:, 0] = -tmp_grad_cauchy[:, 0] # df/d(-t)


                ## Construct tmp_grad and tmp_hess for pseudo voigt

                # gradient
                cache_grad[:, 0, -1] = dfwhm_G*\
                (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                deta_G*diff[-1, :]
                cache_grad[:, 1, -1] = dfwhm_L*\
                (tmp_grad_gau[:, 1]+eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                deta_L*diff[-1, :]
                cache_grad[:, 2, -1] = tmp_grad_gau[:, 0]+\
                eta*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])
                    
                #hessian

                # fwhm_G, fwhm_G
                cache_hess[:, 0, -1] = hess_fwhm[0]*\
                (tmp_grad_gau[:, 1]+
                 eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                dfwhm_G*(dfwhm_G*(tmp_hess_gau[:, 3]+
                eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                2*deta_G*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                hess_eta[0]*diff[-1, :]

                # fwhm_L, fwhm_L
                cache_hess[:, 3, -1] = hess_fwhm[2]*\
                (tmp_grad_gau[:, 1]+
                eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                dfwhm_L*(dfwhm_L*(tmp_hess_gau[:, 3]+
                eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                2*deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                hess_eta[2]*diff[-1, :]

                # fwhm_G, fwhm_L
                cache_hess[:, 1, -1] = hess_fwhm[1]*\
                (tmp_grad_gau[:, 1]+
                eta*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                dfwhm_G*(dfwhm_L*(tmp_hess_gau[:, 3]+
                eta*(tmp_hess_cauchy[:, 3]-tmp_hess_gau[:, 3]))+
                deta_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])) + \
                hess_eta[1]*diff[-1, :] + \
                deta_G*dfwhm_L*(tmp_grad_cauchy[:, 1]-tmp_grad_gau[:, 1])

                # fwhm_G, other
                cache_hess[:, 2, -1] = deta_G*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                dfwhm_G*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))

                # fwhm_L, other
                cache_hess[:, 4, -1] = deta_L*(tmp_grad_cauchy[:, 0]-tmp_grad_gau[:, 0])+\
                dfwhm_L*(tmp_hess_gau[:, 1]+eta*(tmp_hess_cauchy[:, 1]-tmp_hess_gau[:, 1]))

                # other, other
                cache_hess[:, 5, -1] = tmp_hess_gau[:, 0]+\
                eta*(tmp_hess_cauchy[:, 0]-tmp_hess_gau[:, 0])
                
        
        for j in range(d.shape[1]):
            cm = fact_anal_A(A[1:, :], d[:, j], e[:, j])
            chi = (cm@A[1:, :]-d[:, j])/e[:, j]

            c = np.zeros_like(k)
            c[1:] = cm
            c[0] = -np.sum(cm)

            dc = np.einsum('ij,j->ij', A, 1/e[:, j])
            Hc = dc @ dc.T
            
            grad_sum[:, :] = 0
            Hcx[:, :] = 0

            if irf in ['g', 'c']:
                for i in range(tau.size):

                    tmp_grad = cache_grad[:, :, i]
                    tmp_hess = c[i]*cache_hess[:, :, i]

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[i, 0] = tmp_chi_grad[0] #fwhm
                    Hcx[i, 1] = tmp_chi_grad[1] #t
                    Hcx[i, i+2] = tmp_chi_grad[2] #tau_i

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[0] #d(fwhm)^2
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[1] # dtd(fwhm)
                    Hx_2nd[0, 1+num_t0+i] = Hx_2nd[0, 1+num_t0+i] + tmp_chi_hess[2] #d(fwhm)d(tau)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[3] # dt^2
                    Hx_2nd[t0_idx, 1+num_t0+i] = Hx_2nd[t0_idx, 1+num_t0+i] + tmp_chi_hess[4] # dt dtau_i
                    Hx_2nd[1+num_t0+i, 1+num_t0+i] = \
                        Hx_2nd[1+num_t0+i, 1+num_t0+i] + tmp_chi_hess[5] # d(tau_i)^2 

                    #Jf
                    grad_sum[:, 0] = grad_sum[:, 0]+c[i]*tmp_grad[:, 0]
                    grad_sum[:, 1] = grad_sum[:, 1]+c[i]*tmp_grad[:, 1]
                    grad_sum[:, 2+i] = c[i]*tmp_grad[:, 2]
                
                if base:

                    tmp_grad = cache_grad[:, :, -1]
                    tmp_hess = c[-1]*cache_hess[:, :, -1]

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[-1, 0] = tmp_chi_grad[0] #fwhm
                    Hcx[-1, 1] = tmp_chi_grad[1] #t

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[0] #d(fwhm)^2
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[1] # dtd(fwhm)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[2] # dt^2

                    #Jf
                    grad_sum[:, 0] = grad_sum[:, 0]+c[-1]*tmp_grad[:, 0]
                    grad_sum[:, 1] = grad_sum[:, 1]+c[-1]*tmp_grad[:, 1]
            else:
                for i in range(tau.size):

                    tmp_grad = cache_grad[:, :, i]
                    tmp_hess = c[i]*cache_hess[:, :, i]

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[i, :num_irf+1] = tmp_chi_grad[:3] #fwhm_(G,L) t
                    Hcx[i, i+1+num_irf] = tmp_chi_grad[3] #tau_i

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[0] #d(fwhm_G)^2
                    Hx_2nd[0, 1] = Hx_2nd[0, 1] + tmp_chi_hess[1] # d(fwhm_G)d(fwhm_L)
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[2] # dtd(fwhm_G)
                    Hx_2nd[0, num_irf+num_t0+i] = \
                        Hx_2nd[0, num_irf+num_t0+i] + tmp_chi_hess[3] #d(fwhm_G)d(tau)
                    Hx_2nd[1, 1] = Hx_2nd[1, 1] + tmp_chi_hess[4] #d(fwhm_L)^2
                    Hx_2nd[1, t0_idx] = Hx_2nd[1, t0_idx] + tmp_chi_hess[5] # dtd(fwhm_L)
                    Hx_2nd[1, num_irf+num_t0+i] = \
                        Hx_2nd[1, num_irf+num_t0+i] + tmp_chi_hess[6] #d(fwhm_L)d(tau)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[7] # dt^2
                    Hx_2nd[t0_idx, num_irf+num_t0+i] = Hx_2nd[t0_idx, num_irf+num_t0+i] + \
                        tmp_chi_hess[8] # dt dtau_i
                    Hx_2nd[num_irf+num_t0+i, num_irf+num_t0+i] = \
                        Hx_2nd[num_irf+num_t0+i, num_irf+num_t0+i] + tmp_chi_hess[9] # d(tau_i)^2 

                    #Jf
                    grad_sum[:, :num_irf+1] = \
                        grad_sum[:, :num_irf+1]+c[i]*tmp_grad[:, :num_irf+1]
                    grad_sum[:, 1+i+num_irf] = c[i]*tmp_grad[:, 3]
                
                if base:
                    tmp_grad = cache_grad[:, :, -1]
                    tmp_hess = c[-1]*cache_hess[:, :, -1]

                    tmp_grad = np.einsum('ij,i->ij', tmp_grad, 1/e[:, j])
                    tmp_hess = np.einsum('ij,i->ij', tmp_hess, 1/e[:, j])

                    tmp_chi_grad = chi@tmp_grad; tmp_chi_hess = chi@tmp_hess

                    # Hcx
                    Hcx[-1, :num_irf+1] = tmp_chi_grad[:3] #fwhm_(G,L) t

                    # Hx
                    Hx_2nd[0, 0] = Hx_2nd[0, 0] + tmp_chi_hess[0] #d(fwhm_G)^2
                    Hx_2nd[0, 1] = Hx_2nd[0, 1] + tmp_chi_hess[1] # d(fwhm_G)d(fwhm_L)
                    Hx_2nd[0, t0_idx] = Hx_2nd[0, t0_idx] + tmp_chi_hess[2] # dtd(fwhm_G)
                    Hx_2nd[1, 1] = Hx_2nd[1, 1] + tmp_chi_hess[3] #d(fwhm_L)^2
                    Hx_2nd[1, t0_idx] = Hx_2nd[1, t0_idx] + tmp_chi_hess[4] # dtd(fwhm_L)
                    Hx_2nd[t0_idx, t0_idx] = Hx_2nd[t0_idx, t0_idx] + tmp_chi_hess[5] # dt^2

                    #Jf
                    grad_sum[:, :num_irf+1] = \
                        grad_sum[:, :num_irf+1]+c[-1]*tmp_grad[:, :num_irf+1]

            # IRF independent part    
            Hx_1st_tmp = grad_sum.T @ grad_sum

            Hcx = Hcx + dc@grad_sum

            Hc_tau = Hc[:k.size, :k.size]
            Hcx_tau = np.zeros((k.size, 1+num_irf+num_comp))
            Hcx_tau[:, :] = Hcx[:k.size, :num_irf+1+num_comp]
            b = np.linalg.solve(Hc_tau, Hcx_tau)
            Hcorr_tmp = b.T @ (Hcx_tau)

            # fwhm
            Hx_1st[:num_irf, :num_irf] = \
                Hx_1st[:num_irf, :num_irf] + Hx_1st_tmp[:num_irf, :num_irf]
            Hcorr[:num_irf, :num_irf] = \
                Hcorr[:num_irf, :num_irf] + Hcorr_tmp[:num_irf, :num_irf]

            Hx_1st[:num_irf, t0_idx] = Hx_1st[:num_irf, t0_idx] + \
            Hx_1st_tmp[:num_irf, num_irf]
            Hcorr[:num_irf, t0_idx] = Hcorr[:num_irf, t0_idx] + \
            Hcorr_tmp[:num_irf, num_irf]

            Hx_1st[:num_irf, num_irf+num_t0:] = \
                Hx_1st[:num_irf, num_irf+num_t0:] + \
                    Hx_1st_tmp[:num_irf, num_irf+1:]
            Hcorr[:num_irf, num_irf+num_t0:num_irf+num_t0+num_comp] = \
                Hcorr[:num_irf, num_irf+num_t0:num_irf+num_t0+num_comp] + \
                    Hcorr_tmp[:num_irf, num_irf+1:]

            # t0
            Hx_1st[t0_idx, t0_idx] = Hx_1st[t0_idx, t0_idx] + \
                Hx_1st_tmp[num_irf, num_irf]
            Hcorr[t0_idx, t0_idx] = Hcorr[t0_idx, t0_idx] + \
            Hcorr_tmp[num_irf, num_irf]

            Hx_1st[t0_idx, num_irf+num_t0:] = \
            Hx_1st[t0_idx, num_irf+num_t0:] + \
            Hx_1st_tmp[num_irf, 1+num_irf:]
            Hcorr[t0_idx, num_irf+num_t0:num_irf+num_t0+num_comp] = \
            Hcorr[t0_idx, num_irf+num_t0:num_irf+num_t0+num_comp] + \
            Hcorr_tmp[num_irf, 1+num_irf:]

            # tau
            Hx_1st[num_irf+num_t0:, num_irf+num_t0:] = \
                Hx_1st[num_irf+num_t0:, num_irf+num_t0:] + \
                    Hx_1st_tmp[num_irf+1:, num_irf+1:]
            Hcorr[num_irf+num_t0:num_irf+num_t0+num_comp, 
                  num_irf+num_t0:num_irf+num_t0+num_comp] = \
                Hcorr[num_irf+num_t0:num_irf+num_t0+num_comp, 
                      num_irf+num_t0:num_irf+num_t0+num_comp] + \
                    Hcorr_tmp[num_irf+1:, num_irf+1:]

        t0_idx = t0_idx + 1
        dset_idx = dset_idx+1

    H = Hx_1st + Hx_2nd - Hcorr
 
    for i in range(num_param):
        for j in range(i+1, num_param):
            H[j, i] = H[i, j]
 
    return H