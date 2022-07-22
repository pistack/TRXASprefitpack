'''
f_test:
submodule for 
1. comparing two different fitting model
2. calculating confidence interval of parameter
based on f_test

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
import numpy as np
from scipy.stats import f
from scipy.optimize import brenth
from ..res import res_scan, jac_scan
from ..res import residual_decay, residual_dmp_osc, residual_both
from ..res import residual_voigt, residual_thy
from ..res import jac_res_decay, jac_res_dmp_osc, jac_res_both
from ..res import jac_res_voigt, jac_res_thy

class CIResult(dict):
    '''
    Class for represent confidence interval of each parameter

    Attributes:
     method ({'f'}): method to calculate confidance interval of each parameter
      currently only supports F-test based method.
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
        doc_lst.append("[Report for Confidence Interval]")
        doc_lst.append(f"    Method: {self['method']})")
        doc_lst.append(f"    Significance level: {self['alpha']: 4e}")
        doc_lst.append(' ')
        doc_lst.append("[Confidence level]")
        for pn, pv, ci in zip(self['param_name'], self['x'], self['ci']):
            doc_lst.append(f"    {pv-ci[0]: .8f}".rstrip('0').rstrip('.')+
            f" <= {pn} <= {pv+ci[1]: .8f}".rstrip('0').rstrip('.'))
        return '\n'.join(doc_lst)



def is_better_model_f(result1, result2) -> float:
    '''
    Compare model based on f_test

    result1 ({'StaticResult', 'TransientResult'}): fitting result class
     which has more parameter than result2
    result2 ({'StaticResult', 'TransientResult'}): fitting result class
     which has less parameter than result1

    Returns:
     p value of test, If p is smaller than your significant level,
     result1 is may better fit than result2.
     Otherwise, you cannot say resul1 is better fit than result2.
    
    Note:
     The number of parameters in result1 should be greather than
     the number of parameters in result2.

     The result1 and result2 should be different model for same data. 
    '''
    chi2_1 = result1['chi2']; chi2_2 = result2['chi2']
    num_param_1 = result1['n_param']; num_param_2 = result2['n_param']
    num_pts_1 = result1['num_pts']; num_pts_2 = result2['num_pts']

    if num_param_1 <= num_param_2:
        raise Exception(f'Number of parameter in model 1: {num_param_1}' + 
        ' should be strictly greather than' + 
        f' the number of parameter in model 2: {num_param_2}')
    
    if num_pts_1 != num_pts_2:
        raise Exception('The result1 and result2 should be different model for same data')
    
    dfn = num_param_1 - num_param_2
    dfd = num_pts_1 - num_param_1

    F_test = (chi2_2-chi2_1)/dfn/(chi2_1/dfd)
    p = 1- f.cdf(F_test, dfn, dfd)
    return p

def ci_scan(p, *args):
    F_alpha, dfn, dfd, chi2_opt = args[:4]
    fargs = tuple(args[4:])
    ans = (res_scan(p, *fargs)-chi2_opt)/dfn/(chi2_opt/dfd)-F_alpha
    print(ans)
    return ans

def jac_ci_scan(p, *args):
    _, dfn, dfd, chi2_opt = args[:4]
    fargs = tuple(args[4:])
    return 2*jac_scan(p, *fargs)/dfn/(chi2_opt/dfd)


def confidence_interval_f(result, alpha: float) -> CIResult:
    '''
    Calculate 1d confidence interval of each parameter at significance level alpha
    via f-test based method

    Args:
     result ({'StaticResult', 'TransientResult'}): fitting result class
     alpha (float): significance level
    
    Returns:
     CIResult class instance
    '''
    params = np.atleast_1d(result['x'])
    fix_param_idx = np.zeros(len(result['x']), dtype=bool)
    for i in range(params.size):
        fix_param_idx[i] = (result['bounds'][i][0] == result['bounds'][i][1])
    scan_idx = np.array(range(len(result['x'])))
    sub_scan_idx = scan_idx[~fix_param_idx]
    ci_lst = len(result['x'])*[(0, 0)]
    num_param = result['n_param']
    num_pts = result['num_pts']

    chi2_opt = result['chi2']
    dfn = len(params) - 1; dfd = num_pts - num_param
    F_alpha = f.ppf(1-alpha, dfn, dfd)

    if result['model'] == 'decay':
        args = [F_alpha, dfn, dfd, chi2_opt, 0, params, residual_decay, jac_res_decay, result['n_decay'],
        result['base'], result['irf'], fix_param_idx, 
        result['t'], result['intensity'], result['eps']]
    elif result['model'] == 'dmp_osc':
        args = [F_alpha, dfn, dfd, chi2_opt, 0, params, residual_dmp_osc, jac_res_dmp_osc,
        result['n_osc'], result['irf'], fix_param_idx,
        result['t'], result['intensity'], result['eps']]
    elif result['model'] == 'both':
        args = [F_alpha, dfn, dfd, chi2_opt, 0, params, residual_both, jac_res_both,
        result['n_decay'], result['n_osc'], result['base'], result['irf'], fix_param_idx,
        result['t'], result['intensity'], result['eps']]
    elif result['model'] == 'voigt':
        args = [F_alpha, dfn, dfd, chi2_opt, 0, params, residual_voigt, jac_res_voigt,
        result['n_voigt'], result['edge'], result['base_order'], fix_param_idx,
        result['e'], result['intensity'], result['eps']]
    elif result['model'] == 'thy':
        args = [F_alpha, dfn, dfd, chi2_opt, 0, params, residual_thy, jac_res_thy,
        result['policy'], result['thy_peak'], 
        result['edge'], result['base_order'], fix_param_idx,
        result['e'], result['intensity'], result['eps']]
    
    for idx in [0]:
        p0 = params[idx]
        args[4] = idx
        fargs = tuple(args)
        z1 = brenth(ci_scan, p0, result['bounds'][idx][1], args=fargs)
        z2 = brenth(ci_scan, result['bounds'][idx][0], p0, args=fargs)
        ci_lst[idx] = (z2-p0, z1-p0)
    
    ci_res = CIResult()
    ci_res['method'] = 'f'
    ci_res['alpha'] = alpha
    ci_res['x'] = result['x']
    ci_res['ci'] = ci_lst
    return ci_res




    