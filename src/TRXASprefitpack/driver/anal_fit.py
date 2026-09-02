'''
anal_fit:
submodule for
1. comparing two different fitting model
2. calculating confidence interval of parameter
based on f_test

:copyright: 2021-2026 by pistack (Junho Lee).
'''

import warnings
import numpy as np
from scipy.stats import f, norm
from scipy.optimize import brenth, minimize
from ..res import res_grad_decay, res_grad_raise, res_grad_dmp_osc, res_grad_both
from ..res import res_grad_decay_same_t0
from ..res import res_grad_raise_same_t0
from ..res import res_grad_dmp_osc_same_t0
from ..res import res_grad_both_same_t0
from ..res import res_grad_voigt, res_grad_thy


class CIResult(dict):
    '''
    Class for represent confidence interval of each parameter

    Attributes:
     method ({'f'}): method to calculate confidence interval of each parameter
     alpha (float): significant level
     param_name (sequence of str): name of parameter
     x (np.ndarray): best parameter
     ci (sequence of tuple): confidence interval of each parameter at significant level alpha
    '''

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())

    def __str__(self):
        doc_lst = []
        doc_lst.append('[Report for Confidence Interval based on F-test]')
        doc_lst.append(f"    Method: {self['method']}")
        doc_lst.append(f"    Significance level: {self['alpha']: 4e}")
        doc_lst.append(' ')
        doc_lst.append('[Confidence interval]')
        for pn, pv, ci in zip(self['param_name'], self['x'], self['ci']):
            if ci[0] == 0 and ci[1] == 0:
                continue

            if np.isnan(ci[0]) or np.isnan(ci[1]):
                doc_lst.append(f'    {pn}: confidence interval not found')
                continue

            tmp_doc_lst = []
            tmp_doc_lst.append(f'    {pv:.8f}'.rstrip('0').rstrip('.'))
            tmp_doc_lst.append(f'- {-ci[0]: .8f}'.rstrip('0').rstrip('.'))
            tmp_doc_lst.append(f'<= {pn} <=')
            tmp_doc_lst.append(f'{pv: .8f}'.rstrip('0').rstrip('.'))
            tmp_doc_lst.append(f'+ {ci[1]: .8f}'.rstrip('0').rstrip('.'))
            doc_lst.append(' '.join(tmp_doc_lst))
        return '\n'.join(doc_lst)

def _safe_parameter_step(value, eps, fallback_rel=1e-3, fallback_abs=1e-8):
    '''Return a finite positive scan step for CI bracketing'''
    if np.isfinite(eps) and eps > 0:
        return float(eps)
    
    scale = max(abs(float(value)), 1.0)
    return fallback_abs + fallback_rel * scale

def _clip_to_bound(value, lower, upper):
    if lower is not None and value < lower:
        return lower
    if upper is not None and value > upper:
        return upper
    return value

def _find_ci_bracket(
    center,
    step,
    direction,
    scan_func,
    fargs,
    lower_bound,
    upper_bound,
    max_expand,
):
    """Find one side of a confidence interval bracket.

    Returns
    -------
    float or None
        Bracket endpoint where scan_func(endpoint) >= 0.
        Returns None if the endpoint cannot be found.
    """

    if step <= 0 or ~np.isfinite(step):
        raise ValueError("step must be a finite positive number.")
    
    if direction not in (-1, 1):
        raise ValueError("direction must be either -1 or 1.")
    
    current = center

    for _ in range(max_expand):
        candidate = current + direction * step
        candidate = _clip_to_bound(candidate, lower_bound, upper_bound)

        value = scan_func(candidate, *fargs)

        if value >= 0:
            return candidate
        
        if direction < 0 and lower_bound is not None and candidate <= lower_bound:
            return None
        
        if direction > 0 and upper_bound is not None and candidate >= upper_bound:
            return None
        
        current = candidate

    return None

def is_better_fit(result1, result2) -> float:
    '''
    Compare fit based on f-test

    Args:
     result1 ({'StaticResult', 'TransientResult'}): fitting result class
      which has more parameter than result2
     result2 ({'StaticResult', 'TransientResult'}): fitting result class
      which has less parameter than result1

    Returns:
     p value of test, If p is smaller than your significant level,
     result1 is may better fit than result2.
     Otherwise, you cannot say result1 is better fit than result2.

    Note:

     * The number of parameters in result1 should be greater than the number of parameters in result2.
     * The result1 and result2 should be different model for same data.

    '''
    chi2_1 = result1['chi2']
    chi2_2 = result2['chi2']
    num_param_1 = result1['n_param']
    num_param_2 = result2['n_param']
    num_pts_1 = result1['num_pts']
    num_pts_2 = result2['num_pts']

    if num_param_1 <= num_param_2:
        raise ValueError(f'The number of parameter in model 1: {num_param_1}' +
        ' should be strictly greater than' +
        f' the number of parameter in model 2: {num_param_2}')

    if num_pts_1 != num_pts_2:
        raise ValueError(f'The number of data points in result1 is {num_pts_1}, ' +
                         f'which is not the same as result2: {num_pts_2}')

    dfn = num_param_1 - num_param_2
    dfd = num_pts_1 - num_param_1

    F_test = (chi2_2-chi2_1)/dfn/(chi2_1/dfd)
    p = 1- f.cdf(F_test, dfn, dfd)
    return p

def confidence_interval(result, alpha: float, parameter_indices=None) -> CIResult:
    '''
    Calculate 1d confidence interval for selected parameters at significance level alpha
    Based on F-test method

    Args:
     result ({'StaticResult', 'TransientResult'}): fitting result class
     alpha (float): significance level
     parameter_indices:
     Optional sequence of parameter indices to scan. If omitted, all non-fixed parameters are scanned.

    Returns:
     CIResult class instance
    '''

    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a finite number between 0 and 1.")
    
    params = np.atleast_1d(result['x'])
    num_params_total = params.size

    fix_param_idx = np.zeros(len(result['x']), dtype=bool)
    for i in range(num_params_total):
        fix_param_idx[i] = (result['bounds'][i][0] == result['bounds'][i][1])

    if parameter_indices is None:
        requested_idx = np.ones(num_params_total, dtype=bool)
    else:
        parameter_indices = np.asarray(parameter_indices)

        if parameter_indices.ndim != 1:
            raise ValueError(
                "parameter_indices must be a one-dimensional sequence.")

        if parameter_indices.size == 0:
            raise ValueError(
                "At least one parameter must be selected.")

        if not np.issubdtype(parameter_indices.dtype, np.integer):
            raise TypeError(
                "parameter_indices must contain integer indices.")

        parameter_indices = parameter_indices.astype(int)

        if np.any(parameter_indices < 0) or np.any(
            parameter_indices >= num_params_total):
            raise IndexError(
                "parameter_indices contains an out-of-range index.")

        if np.unique(parameter_indices).size != parameter_indices.size:
            raise ValueError(
                "parameter_indices must not contain duplicates.")

        if np.any(fix_param_idx[parameter_indices]):
            raise ValueError(
                "A fixed parameter cannot be selected for CI scanning."
            )

        requested_idx = np.zeros(num_params_total, dtype=bool)
        requested_idx[parameter_indices] = True

    select_idx = fix_param_idx | ~requested_idx
    scan_idx = np.arange(num_params_total)
    
    ci_lst = [(0, 0) for _ in range(len(result['x']))]
    num_param = result['n_param']
    num_pts = result['num_pts']

    chi2_opt = result['chi2']
    dfn = 1
    dfd = num_pts - num_param
    F_alpha = f.ppf(1-alpha, dfn, dfd)
    norm_alpha = np.ceil(norm.ppf(1-alpha/2))

    if result['model'] == 'decay':
        if result['same_t0']:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_decay_same_t0, result['n_decay'],
            result['base'], result['irf'], result['tau_mask'], fix_param_idx,
            result['t'], result['intensity'], result['eps']]
        else:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_decay, result['n_decay'],
            result['base'], result['irf'], result['tau_mask'], fix_param_idx,
            result['t'], result['intensity'], result['eps']]
    elif result['model'] == 'raise':
        if result['same_t0']:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_raise_same_t0, result['n_decay'],
            result['base'], result['irf'], fix_param_idx,
            result['t'], result['intensity'], result['eps']]
        else:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_raise, result['n_decay'],
            result['base'], result['irf'], fix_param_idx,
            result['t'], result['intensity'], result['eps']]
    elif result['model'] == 'dmp_osc':
        if result['same_t0']:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_dmp_osc_same_t0, result['n_osc'],
            result['irf'], fix_param_idx,
            result['t'], result['intensity'], result['eps']]
        else:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_dmp_osc, result['n_osc'],
            result['irf'], fix_param_idx,
            result['t'], result['intensity'], result['eps']]
    elif result['model'] == 'both':
        if result['same_t0']:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_both_same_t0,
            result['n_decay'], result['n_osc'], result['base'], result['irf'],
            fix_param_idx,
            result['t'], result['intensity'], result['eps']]
        else:
            args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
            res_grad_both,
            result['n_decay'], result['n_osc'], result['base'], result['irf'],
            fix_param_idx,
            result['t'], result['intensity'], result['eps']]
    elif result['model'] == 'voigt':
        args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
        res_grad_voigt,
        result['n_voigt'], result['edge'], result['n_edge'],
        result['base_order'], fix_param_idx,
        result['e'], result['intensity'], result['eps']]
    elif result['model'] == 'thy':
        args = [F_alpha, dfn, dfd, chi2_opt, 0, params, result['bounds'],
        res_grad_thy,
        result['policy'], result['thy_peak'],
        result['edge'], result['n_edge'],
        result['base_order'], fix_param_idx,
        result['e'], result['intensity'], result['eps']]
    else:
        raise ValueError(
            f"Unsupported model for confidence_interval: {result['model']}"
        )

    sub_scan_idx = scan_idx[~select_idx]
    for idx in sub_scan_idx:
        p0 = params[idx]
        args[4] = idx
        fargs = tuple(args)
        p_eps = _safe_parameter_step(p0, result['x_eps'][idx])

        p_lb = _find_ci_bracket(p0, norm_alpha*p_eps, -1,
                                ci_scan_opt_f, fargs,
                                lower_bound=result['bounds'][idx][0],
                                upper_bound=result['bounds'][idx][1],
                                max_expand=100)
    
        p_ub = _find_ci_bracket(p0, norm_alpha*p_eps, 1,
                                ci_scan_opt_f, fargs,
                                lower_bound=result['bounds'][idx][0],
                                upper_bound=result['bounds'][idx][1],
                                max_expand=100)
        
        if p_lb is None:
            warnings.warn(
                f"Lower confidence bound was not found for parameter {idx}.",
                RuntimeWarning,
                stacklevel=2
            )
            z1 = np.nan
        else:
            try:
                z1 = brenth(ci_scan_opt_f, p_lb, p0, args=fargs)
            except ValueError:
                warnings.warn(
                    f"Lower confidence bound was not found for parameter {idx}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                z1 = np.nan

        if p_ub is None:
            warnings.warn(
                f"Upper confidence bound was not found for parameter {idx}.",
                RuntimeWarning,
                stacklevel=2
            )
            z2 = np.nan
        else:
            try:
                z2 = brenth(ci_scan_opt_f, p0, p_ub, args=fargs)
            except ValueError:
                warnings.warn(
                    f"Upper confidence bound was not found for parameter {idx}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                z2 = np.nan

        ci_lst[idx] = (z1-p0, z2-p0)

    ci_res = CIResult()
    ci_res['method'] = 'f'
    ci_res['alpha'] = alpha
    ci_res['param_name'] = result['param_name']
    ci_res['x'] = result['x']
    ci_res['ci'] = ci_lst
    return ci_res

def res_scan_opt(p, *args) -> float:
    '''
    res_scan_opt
    Scans minimal value of residual function with fixing
    ith value of parameter to p.

    Args:
     p: value of ith parameter
     args: arguments

           * 1st: i, index of parameter to scan
           * 2nd: parameter
           * 3rd: bounds
           * 4th: objective function which also gives its gradient
           * 5th to last: arguments for objective function
           * :math:`last-3`: fixed_param_idx

    Returns:
     residual value at params[i] = p
    '''
    param = np.atleast_1d(args[1]).copy()
    param[args[0]] = p
    bounds = len(args[2])*[None]
    for i in range(len(args[2])):
        bounds[i] = args[2][i]
    bounds[args[0]] = (p, p)
    func = args[3]
    fixed_param_idx = np.atleast_1d(args[-4]).copy()
    fixed_param_idx[args[0]] = True
    fargs_lst = (len(args)-4)*[None]
    for i in range(4, len(args)):
        fargs_lst[i-4] = args[i]
    fargs_lst[-4] = fixed_param_idx
    fargs = tuple(fargs_lst)
    res = minimize(func, param, args=fargs, bounds=bounds, method='L-BFGS-B', jac=True)
    if not res['success']:
        warnings.warn(f"CI scan optimization did not converge: {res.message}",
                      RuntimeWarning,
                      stacklevel=2,)

    return res['fun']

def ci_scan_opt_f(p, *args):
    '''
    Confidence interval scan with ith parameter is fixed to p. (for f-test based method)
    res_scan_opt returns the scalar least-squares objective (chi2/2)
    Therefore, chi2_opt is divided by 2 here to compare
    objective values on the same scale.
    '''
    F_alpha, dfn, dfd, chi2_opt = args[:4]
    fargs = tuple(args[4:])
    return (res_scan_opt(p, *fargs)-chi2_opt/2)/dfn/(chi2_opt/(2*dfd))-F_alpha




