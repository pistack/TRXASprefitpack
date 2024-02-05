'''
deriv_tst:
submodule to test derivative routine of mathfun subpackage

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Callable
import numpy as np


def check_num_deriv(fun: Callable, *args, eps_rel=1e-8, eps_abs=5e-6):
    '''
    Test implementation of derivative via finite difference
    i th column: derivative of f w.r.t ith argument
    '''
    n = len(args)
    if isinstance(args[0], np.ndarray):
        num_grad = np.empty((args[0].size, n))
        for i in range(n):
            f_args = list(args)
            b_args = list(args)
            f_args[i] = f_args[i]*(1+eps_rel)+eps_abs
            b_args[i] = b_args[i]*(1-eps_rel)-eps_abs
            f_args = tuple(f_args)
            b_args = tuple(b_args)
            dx = f_args[i] - b_args[i]
            num_grad[:, i] = (fun(*f_args)-fun(*b_args))/dx
    else:
        num_grad = np.empty(n)
        for i in range(n):
            f_args = list(args)
            b_args = list(args)
            f_args[i] = f_args[i]*(1+eps_rel)+eps_abs
            b_args[i] = b_args[i]*(1-eps_rel)-eps_abs
            f_args = tuple(f_args)
            b_args = tuple(b_args)
            dx = f_args[i] - b_args[i]
            num_grad[i] = (fun(*f_args)-fun(*b_args))/dx

    return num_grad

def check_num_hess(fun: Callable, *args, eps_rel=1e-8, eps_abs=5e-6):
    '''
    Test implementation of hessian via finite difference
    (i, j) element of hessian d^2f/dx_i dx_j
    '''
    n = len(args)
    f0 = fun(*args)
    if isinstance(args[0], np.ndarray):
        num_hess = np.empty((args[0].size, n, n))
        for i in range(n):
            for j in range(i):
                xpp_args = list(args)
                xmp_args = list(args)
                xpm_args = list(args)
                xmm_args = list(args)
                xpp_args[i] = xpp_args[i]*(1+eps_rel)+eps_abs
                xpp_args[j] = xpp_args[j]*(1+eps_rel)+eps_abs
                xmp_args[i] = xmp_args[i]*(1-eps_rel)-eps_abs
                xmp_args[j] = xmp_args[j]*(1+eps_rel)+eps_abs
                xpm_args[i] = xpm_args[i]*(1+eps_rel)+eps_abs
                xpm_args[j] = xpm_args[j]*(1-eps_rel)-eps_abs
                xmm_args[i] = xmm_args[i]*(1-eps_rel)-eps_abs
                xmm_args[j] = xmm_args[j]*(1-eps_rel)-eps_abs
                xpp_args = tuple(xpp_args)
                xmp_args = tuple(xmp_args)
                xpm_args = tuple(xpm_args)
                xmm_args = tuple(xmm_args)
                dxpp_xmp = xpp_args[i]-xmp_args[i]
                dxpp_xpm = xpp_args[j]-xpm_args[j]
                dfpp_fmp = (fun(*xpp_args)-fun(*xmp_args))/(dxpp_xmp)
                dfpm_fmm = (fun(*xpm_args)-fun(*xmm_args))/(dxpp_xmp)
                num_hess[:, i, j] = (dfpp_fmp-dfpm_fmm)/dxpp_xpm
                num_hess[:, j, i] = num_hess[:, i, j]
        for i in range(n):
            xp_args = list(args)
            xm_args = list(args)
            xp_args[i] = xp_args[i]*(1+eps_rel)+eps_abs
            xm_args[i] = xm_args[i]*(1-eps_rel)-eps_abs
            dpx = xp_args[i] - args[i]
            dxm = args[i] - xm_args[i]
            dfp_f = (fun(*xp_args)-f0)/dpx
            df_fm = (f0-fun(*xm_args))/dxm
            num_hess[:, i, i] = (dfp_f-df_fm)/dpx

    else:
        num_hess = np.empty((n, n))
        for i in range(n):
            for j in range(i):
                xpp_args = list(args)
                xmp_args = list(args)
                xpm_args = list(args)
                xmm_args = list(args)
                xpp_args[i] = xpp_args[i]*(1+eps_rel)+eps_abs
                xpp_args[j] = xpp_args[j]*(1+eps_rel)+eps_abs
                xmp_args[i] = xmp_args[i]*(1-eps_rel)-eps_abs
                xmp_args[j] = xmp_args[j]*(1+eps_rel)+eps_abs
                xpm_args[i] = xpm_args[i]*(1+eps_rel)+eps_abs
                xpm_args[j] = xpm_args[j]*(1-eps_rel)-eps_abs
                xmm_args[i] = xmm_args[i]*(1-eps_rel)-eps_abs
                xmm_args[j] = xmm_args[j]*(1-eps_rel)-eps_abs
                xpp_args = tuple(xpp_args)
                xmp_args = tuple(xmp_args)
                xpm_args = tuple(xpm_args)
                xmm_args = tuple(xmm_args)
                dxpp_xmp = xpp_args[i]-xmp_args[i]
                dxpp_xpm = xpp_args[j]-xpm_args[j]
                dfpp_fmp = (fun(*xpp_args)-fun(*xmp_args))/(dxpp_xmp)
                dfpm_fmm = (fun(*xpm_args)-fun(*xmm_args))/(dxpp_xmp)
                num_hess[i, j] = (dfpp_fmp-dfpm_fmm)/dxpp_xpm
                num_hess[j, i] = num_hess[i, j]
        for i in range(n):
            xp_args = list(args)
            xm_args = list(args)
            xp_args[i] = xp_args[i]*(1+eps_rel)+eps_abs
            xm_args[i] = xm_args[i]*(1-eps_rel)-eps_abs
            dpx = xp_args[i] - args[i]
            dxm = args[i] - xm_args[i]
            dfp_f = (fun(*xp_args)-f0)/dpx
            df_fm = (f0-fun(*xm_args))/dxm
            num_hess[i, i] = (dfp_f-df_fm)/dpx

    return num_hess
