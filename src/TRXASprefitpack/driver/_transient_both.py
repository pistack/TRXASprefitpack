'''
_transient_both:
submodule for fitting time delay scan with the
convolution of sum of (exponential decay and damped oscillation)
and instrumental response function

:copyright: 2021-2026 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Optional, Union, Sequence, Tuple
import numpy as np
from ..mathfun.irf import calc_eta, calc_fwhm
from .transient_result import TransientResult
from scipy.optimize import least_squares
from ..mathfun.A_matrix import make_A_matrix_exp, make_A_matrix_dmp_osc, fact_anal_A
from ..res.parm_bound import set_bound_t0, set_bound_tau
from ..res.res_both import residual_both, res_grad_both
from ..res.res_both import residual_both_same_t0, res_grad_both_same_t0
from ._input import normalize_tscan_inputs, validate_t0_count
from ._transient_common import (
    calc_covariance_from_hessian,
    calc_individual_chi2,
    count_total_scans,
    default_dataset_names,
    get_num_irf,
    make_fixed_mask,
    make_lsq_bounds,
    prepare_lsq_kwargs,
    run_global_optimization,
    validate_transient_driver_options,
)


def fit_transient_both(irf: str, fwhm_init: Union[float, np.ndarray],
                       t0_init: np.ndarray, tau_init: np.ndarray,
                       tau_osc_init: np.ndarray, period_osc_init: np.ndarray,
                       base: bool,
                       method_glb: Optional[str] = None,
                       method_lsq: Optional[str] = 'trf',
                       kwargs_glb: Optional[dict] = None,
                       kwargs_lsq: Optional[dict] = None,
                       bound_fwhm: Optional[Sequence[Tuple[float, float]]] = None,
                       bound_t0: Optional[Sequence[Tuple[float, float]]] = None,
                       bound_tau: Optional[Sequence[Tuple[float, float]]] = None,
                       bound_tau_osc: Optional[Sequence[Tuple[float, float]]] = None,
                       bound_period_osc: Optional[Sequence[Tuple[float, float]]] = None,
                       same_t0: Optional[bool] = False,
                       name_of_dset: Optional[Sequence[str]] = None,
                       t: Optional[Sequence[np.ndarray]] = None,
                       intensity: Optional[Sequence[np.ndarray]] = None,
                       eps: Optional[Sequence[np.ndarray]] = None) -> TransientResult:
    '''
    driver routine for fitting multiple data set of time delay scan data with
    sum of the convolution of exponential decay and instrumental response function and
    sum of the convolution of damped oscillation and insrumental response function.

    It separates linear and non-linear part of the optimization problem to solve non linear least sequare
    optimization problem efficiently.

    Moreover, this driver uses two step method to search best parameter, its covariance and
    estimated parameter error.

    Step 1. (method_glb)
    Use global optimization to find rough global minimum of our objective function.
    In this stage, it use analytic gradient for scalar residual function.

    Step 2. (method_lsq)
    Use least squares optimization algorithm to refine global minimum of objective function and approximate covariance matrix.
    Because of linear and non-linear seperation scheme, the analytic jacobian for vector residual function is hard to optain.
    Thus, in this stage, it uses numerical jacobian.

    Args:
     irf ({'g', 'c', 'pv'}): shape of instrumental response function

      'g': gaussian shape

      'c': cauchy shape

      'pv': pseudo voigt shape (kind 2)
     fwhm_init (float or np.ndarray): initial full width at half maximum for instrumental response function

      * if irf in ['g', 'c'] then fwhm_init is float
      * if irf == 'pv' then fwhm_init is the `numpy.ndarray` with [fwhm_G, fwhm_L]

     t0_init (np.ndarray): time zeros for each scan
     tau_init (np.ndarray): lifetime constant for each decay component
     tau_osc (np.ndarray): lifetime constant for each damped oscillation component
     period_init (np.ndarray): period of each oscillation component
     base (bool): Whether or not include baseline feature (i.e. very long lifetime constant)
     method_glb ({None, 'basinhopping', 'ampgo'}): Method for global optimization
     method_lsq ({'trf', 'dogbox', 'lm'}): method of local optimization for least_squares
                                           minimization (refinement of global optimization solution)
     kwargs_glb: keyward arguments for global optimization solver
     kwargs_lsq: keyward arguments for least square optimization solver
     bound_fwhm (sequence of tuple): boundary for irf parameter. If upper and lower bound are same,
      driver assumes that the parameter is fixed during optimization. If `bound_fwhm` is `None`,
      the upper and lower bound are given as `(fwhm_init/2, 2*fwhm_init)`.
     bound_t0 (sequence of tuple): boundary for time zero parameter.
      If `bound_t0` is `None`, the upper and lower bound are given by `set_bound_t0`.
     bound_tau (sequence of tuple): boundary for lifetime constant for decay component,
      if `bound_tau` is `None`, the upper and lower bound are given by ``set_bound_tau``.
     bound_tau_osc (sequence of tuple): boundary for lifetime constant for damped oscillation component,
      if `bound_tau_osc` is `None`, the upper and lower bound are given by ``set_bound_tau``.
     bound_period_osc (sequence of tuple): boundary for period of damped oscillation component,
      if `bound_period_osc` is `None`, the upper and lower bound are given by ``set_bound_tau``.
     same_t0 (bool): Whether or not time zero of every time delay scan in the same dataset should be same
     name_of_dset (sequence of str): name of each dataset
     t (sequence of np.narray): time scan range for each datasets
     intensity (sequence of np.ndarray): sequence of intensity of datasets
      for time delay scan
     eps (sequence of np.ndarray): sequence of estimated errors of each dataset

     Returns:
      TransientResult class object
    '''
    validate_transient_driver_options(method_glb, method_lsq, irf)
    t, intensity, eps = normalize_tscan_inputs(t, intensity, eps)
    validate_t0_count(t0_init, intensity, same_t0)

    num_irf = get_num_irf(irf)
    num_param = num_irf+t0_init.size+tau_init.size+2*tau_osc_init.size
    param = np.empty(num_param, dtype=float)

    param[:num_irf] = fwhm_init
    param[num_irf:num_irf+t0_init.size] = t0_init
    param[num_irf+t0_init.size:num_irf+t0_init.size+tau_init.size] = tau_init
    osc_start = num_irf+t0_init.size+tau_init.size
    param[osc_start:osc_start+tau_osc_init.size] = tau_osc_init
    param[osc_start+tau_osc_init.size:osc_start +
          2*tau_osc_init.size] = period_osc_init

    bound = num_param*[None]

    if bound_fwhm is None:
        for i in range(num_irf):
            bound[i] = (param[i]/2, 2*param[i])
    else:
        bound[:num_irf] = bound_fwhm

    if bound_t0 is None:
        for i in range(t0_init.size):
            bound[i+num_irf] = set_bound_t0(t0_init[i], fwhm_init)
    else:
        bound[num_irf:num_irf+t0_init.size] = bound_t0

    if bound_tau is None:
        for i in range(tau_init.size):
            bound[i+num_irf +
                  t0_init.size] = set_bound_tau(tau_init[i], fwhm_init)
    else:
        bound[num_irf+t0_init.size:osc_start] = bound_tau

    if bound_tau_osc is None:
        for i in range(tau_osc_init.size):
            bound[osc_start+i] = set_bound_tau(tau_osc_init[i], fwhm_init)
    else:
        bound[osc_start:osc_start+tau_osc_init.size] = bound_tau_osc

    if bound_period_osc is None:
        for i in range(tau_osc_init.size):
            bound[osc_start+tau_osc_init.size +
                  i] = set_bound_tau(period_osc_init[i], fwhm_init)
    else:
        bound[osc_start+tau_osc_init.size:osc_start +
              2*tau_osc_init.size] = bound_period_osc
    
    fix_param_idx = make_fixed_mask(bound)

    go_args = (tau_init.size, tau_osc_init.size,
                   base, irf, fix_param_idx, t, intensity, eps)
    
    res_go = run_global_optimization(
        method_glb,
        param,
        args=go_args,
        bounds=bound,
        grad_func=res_grad_both,
        grad_func_same_t0=res_grad_both_same_t0,
        same_t0=same_t0,
        kwargs_glb=kwargs_glb,
        )

    param_gopt = res_go['x']
    args_lsq = (tau_init.size, tau_osc_init.size, base, irf, t, intensity, eps)
    kwargs_lsq = prepare_lsq_kwargs(kwargs_lsq, args=args_lsq)
    bound_tuple = make_lsq_bounds(bound)

    if same_t0:
        res_lsq = least_squares(residual_both_same_t0, param_gopt,
        method=method_lsq, bounds=bound_tuple, **kwargs_lsq)
    else:
        res_lsq = least_squares(residual_both, param_gopt,
        method=method_lsq, bounds=bound_tuple, **kwargs_lsq)

    param_opt = res_lsq['x']

    fwhm_opt = param_opt[:num_irf]
    tau_opt = param_opt[num_irf+t0_init.size:osc_start]
    tau_osc_opt = param_opt[osc_start:osc_start+tau_osc_init.size]
    period_osc_opt = param_opt[osc_start +
                               tau_osc_init.size:osc_start+2*tau_osc_init.size]

    fit = np.empty(len(t), dtype=object)
    fit_osc = np.empty(len(t), dtype=object)
    fit_decay = np.empty(len(t), dtype=object)
    res = np.empty(len(t), dtype=object)

    num_tot_scan = count_total_scans(intensity)
    for i in range(len(t)):
        fit[i] = np.empty(intensity[i].shape)
        fit_osc[i] = np.empty(intensity[i].shape)
        fit_decay[i] = np.empty(intensity[i].shape)
        res[i] = np.empty(intensity[i].shape)

# Calc individual chi2
    chi = res_lsq['fun']
    num_param_tot = num_tot_scan * \
        (tau_opt.size+1*base+2*tau_osc_opt.size)+num_param-np.sum(fix_param_idx)
    chi2 = 2*res_lsq['cost']
    red_chi2 = chi2/(chi.size-num_param_tot)
    num_param_ind = 2*tau_opt.size+4*tau_osc_opt.size+1*base+2+1*(irf == 'pv')
    chi2_ind, red_chi2_ind = calc_individual_chi2(
        chi,
        intensity,
        num_param_ind=num_param_ind,
        )

    param_name = np.empty(param_opt.size, dtype=object)
    c = np.empty(len(t), dtype=object)
    phase = np.empty(len(t), dtype=object)
    t0_idx = num_irf

    if irf == 'g':
        eta = 0
        fwhm_pv = fwhm_opt[0]
        param_name[0] = 'fwhm_G'
    elif irf == 'c':
        eta = 1
        fwhm_pv = fwhm_opt[0]
        param_name[0] = 'fwhm_L'
    else:
        eta = calc_eta(fwhm_opt[0], fwhm_opt[1])
        fwhm_pv = calc_fwhm(fwhm_opt[0], fwhm_opt[1])
        param_name[0] = 'fwhm_G'
        param_name[1] = 'fwhm_L'

    for i in range(len(t)):
        A = np.empty((tau_init.size+1*base+2*tau_osc_init.size, t[i].size))
        c[i] = np.empty(
            (tau_init.size+1*base+tau_osc_init.size, intensity[i].shape[1]))
        phase[i] = np.empty((tau_osc_init.size, intensity[i].shape[1]))

        if same_t0:
            A[:tau_init.size+1*base, :] = \
                make_A_matrix_exp(t[i]-param_opt[t0_idx],
                fwhm_pv, tau_opt, base, irf, eta)
            A[tau_init.size+1*base:, :] = \
                make_A_matrix_dmp_osc(t[i]-param_opt[t0_idx], fwhm_pv,
                tau_osc_opt, period_osc_opt, irf, eta)

        for j in range(intensity[i].shape[1]):
            if not same_t0:
                A[:tau_init.size+1*base, :] = \
                    make_A_matrix_exp(t[i]-param_opt[t0_idx],
                    fwhm_pv, tau_opt, base, irf, eta)
                A[tau_init.size+1*base:, :] = \
                    make_A_matrix_dmp_osc(t[i]-param_opt[t0_idx], fwhm_pv,
                    tau_osc_opt, period_osc_opt, irf, eta)

            tmp = fact_anal_A(A, intensity[i][:, j], eps[i][:, j])
            fit[i][:, j] = tmp @ A
            fit_decay[i][:, j] = tmp[:tau_init.size +
                                     1*base] @ A[:tau_init.size+1*base, :]
            fit_osc[i][:, j] = tmp[tau_init.size+1 *
                                   base:] @ A[tau_init.size+1*base:, :]
            c[i][:tau_init.size+1*base, j] = tmp[:tau_init.size+1*base]
            c[i][tau_init.size+1*base:, j] = \
                np.sqrt(tmp[tau_init.size+1*base:tau_init.size+1*base+tau_osc_init.size]**2 +
                        tmp[tau_init.size+1*base+tau_osc_init.size:]**2)
            phase[i][:, j] = -np.arctan2(tmp[tau_init.size+1*base+tau_osc_init.size:],
                                         tmp[tau_init.size+1*base:tau_init.size+1*base+tau_osc_init.size])

            if not same_t0:
                param_name[t0_idx] = f't_0_{i+1}_{j+1}'
                t0_idx = t0_idx + 1

        if same_t0:
            param_name[t0_idx] = f't_0_{i}'
            t0_idx = t0_idx + 1

        res[i] = intensity[i] - fit[i]

    for i in range(tau_init.size):
        param_name[num_irf+t0_init.size+i] = f'tau_{i+1}'

    for i in range(tau_osc_init.size):
        param_name[osc_start+i] = f'tau_osc_{i+1}'
        param_name[osc_start+tau_osc_init.size+i] = f'period_osc_{i+1}'

    jac = res_lsq['jac']
    hes = jac.T @ jac
    cov, cov_scaled, corr, param_eps = calc_covariance_from_hessian(
         hes, fix_param_idx, red_chi2)

    result = TransientResult()
    result['same_t0'] = same_t0

    # save experimental fitting data
    if name_of_dset is None:
        name_of_dset = default_dataset_names(len(t))

    result['name_of_dset'] = name_of_dset
    result['t'] = t
    result['intensity'] = intensity
    result['eps'] = eps

    result['model'] = 'both'
    result['fit'] = fit
    result['fit_osc'] = fit_osc
    result['fit_decay'] = fit_decay
    result['res'] = res
    result['irf'] = irf
    result['eta'] = eta
    result['fwhm'] = fwhm_pv

    result['param_name'] = param_name
    result['x'] = param_opt
    result['bounds'] = bound
    result['base'] = base
    result['c'] = c
    result['phase'] = phase
    result['chi2'] = chi2
    result['chi2_ind'] = chi2_ind
    result['aic'] = chi.size*np.log(chi2/chi.size)+2*num_param_tot
    result['bic'] = chi.size * \
        np.log(chi2/chi.size)+num_param_tot*np.log(chi.size)
    result['red_chi2'] = red_chi2
    result['red_chi2_ind'] = red_chi2_ind
    result['nfev'] = res_go['nfev'] + res_lsq['nfev']
    result['n_param'] = num_param_tot
    result['n_param_ind'] = num_param_ind
    result['num_pts'] = chi.size
    result['jac'] = jac
    result['cov'] = cov
    result['corr'] = corr
    result['cov_scaled'] = cov_scaled
    result['x_eps'] = param_eps
    result['method_lsq'] = method_lsq
    result['message_lsq'] = res_lsq['message']
    result['success_lsq'] = res_lsq['success']

    if result['success_lsq']:
        result['status'] = 0
    else:
        result['status'] = -1

    if method_glb is not None:
        result['method_glb'] = method_glb
        result['message_glb'] = res_go['message'][0]
    else:
        result['method_glb'] = None
        result['message_glb'] = None

    result['n_osc'] = tau_osc_init.size
    if tau_init is None:
        result['n_decay'] = 0
    else:
        result['n_decay'] = tau_init.size

    return result
