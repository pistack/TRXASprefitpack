'''
res_gen:
submodule which provides compatible layer of residual and radient function
:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

import numpy as np

def residual_scalar(params: np.ndarray, *args) -> float:
    '''
    residual_scaler
    scipy.optimize.minimize compatible scaler residual function
    Args:
     params: parameter used for fitting
     args: arguments
             args[0]: objective function
             args[1]: jacobian of objective function
             args[2:]: arguments for both objective and jocobian  

    Returns:
     Residucal scalar(i.e. square of 2-norm of Residual Vector)
    '''
    func = args[0]
    fargs = ()
    if len(args) > 2:
        fargs = tuple(args[2:])
    return np.sum(func(params, *fargs)**2)

def grad_res_scalar(params: np.ndarray, *args) -> np.ndarray:
    '''
    grad_res_scalar
    scipy.optimize.minimizer compatible gradient of scalar residual function

    Args:
     params: parameter used for fitting
     args: arguments
             args[0]: objective function
             args[1]: jacobian of objective function
             args[2:]: arguments for both objective and jocobian  

    Returns:
     Gradient of residucal scalar
    '''
    func, jac = args[:2]
    fargs = ()
    if len(args) > 2:
        fargs = tuple(args[2:])
    return func(params, *fargs) @ jac(params, *fargs)

def res_scan(p, *args) -> float:
    '''
    res_scan
    Scans residual function though ith parameter

    Args:
     p: value of ith parameter
     args: arguments
           args[0]: i, index of parameter to scan
           args[1]: params
           args[2]: objective function
           args[3]: jacobian of objective function
           args[3:]: arguments for both objective and jocobian
    
    Returns:
     residual value at params[i] = p
    '''
    idx = args[0]
    params = np.atleast_1d(args[1]).copy()
    fargs = ()
    if len(args)>2:
        fargs = tuple(args[2:])
    params[idx] = p
    return residual_scalar(params, *fargs)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

def jac_scan(p, *args) -> float:
    '''
    jac_scan
    partial derivative of residual .w.r.t. ith parameter

    Args:
     p: value of ith parameter
     args: arguments
           args[0]: i, index of parameter to scan
           args[1]: params
           args[2]: objective function
           args[3]: jacobian of objective function
           args[3:]: arguments for both objective and jocobian
    
    Returns:
     residual value at params[i] = p
    '''
    idx = args[0]
    params = np.atleast_1d(args[1]).copy()
    fargs = ()
    if len(args)>2:
        fargs = tuple(args[2:])
    params[idx] = p
    return grad_res_scalar(params, *fargs)[idx]

def res_lmfit(params, *args) -> np.ndarray:
    '''
    res_lmfit
    lmfit compatible layer for vector residual function

    Args:
     params: parameter used for fitting (lmfit.parameter class)
     args: arguments
             args[0]: objective function
             args[1]: jacobian of objective function
             args[2:]: arguments for both objective and jocobian 
    Returns:
     residucal vector
    '''
    func = args[0]
    fargs = ()
    if len(args) > 2:
        fargs = tuple(args[2:])
    return func(params, *fargs)

def jac_lmfit(params, *args) -> np.ndarray:
    '''
    jac_lmfit
    lmfit compatible layer for jacobian of vector residual function

    Args:
     params: parameter used for fitting (lmfit.parameter class)
     args: arguments
             args[0]: objective function
             args[1]: jacobian of objective function
             args[2:]: arguments for both objective and jocobian  

    Returns:
     lmfit compatible Gradient of residucal vector
    '''
    mask = np.empty(len(params), dtype=bool)
    for i in range(len(params)):
        mask[i] = params[i].vary
    jac = args[1]
    fargs = ()
    if len(args) > 2:
        fargs = tuple(args[2:])
    return jac(params, *fargs)[:, mask]