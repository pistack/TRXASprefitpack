'''
res_gen:
submodule which provides compatible layer of residual and radient function
:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
import numpy as np

def residual_scalar(x0: np.ndarray, *args) -> float:
    '''
    residual_scaler
    scipy.optimize.minimize compatible scaler residual function
    Args:
     x0: initial parameter
     args: arguments
             args[0]: objective function
             args[1:]: arguments for both objective function

    Returns:
     Residucal scalar(i.e. square of 2-norm of Residual Vector)
     chi2/2
    '''
    func = args[0]
    fargs = ()
    if len(args) > 1:
        fargs = tuple(args[1:])
    return np.sum(func(x0, *fargs)**2)/2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

def res_lmfit(x0, *args) -> np.ndarray:
    '''
    res_lmfit
    lmfit compatible layer for vector residual function

    Args:
     x0 (lmfit.Parameters): initial parameter
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
    return func(x0, *fargs)