'''
res_voigt:
submodule for residual function and dfient for fitting static spectrum with the
sum of voigt function, edge function and base function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Optional
import numpy as np
from ..mathfun.A_matrix import fact_anal_A
from ..mathfun.peak_shape import voigt, edge_gaussian, edge_lorenzian
from ..mathfun.peak_shape import deriv_voigt, deriv_edge_gaussian, deriv_edge_lorenzian

def residual_voigt(params: np.ndarray, num_voigt: int, edge: Optional[str] = None,
                    base_order: Optional[int] = None, 
                    fix_param_idx: Optional[np.ndarray] = None,
                    e: np.ndarray = None, 
                    data: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    residual_voigt
    scipy.optimize.least_squares compatible vector residual function for fitting static spectrum with the 
    sum of voigt function, edge function base function

    Args:
     params: parameter used for fitting

              param[i]: peak position e0_i for each voigt component
              param[num_voigt+i]: fwhm_G of each voigt component
              param[2*num_voigt+i]: fwhm_L of each voigt component

              if edge is not None:

                 param[-2]: edge position

                 param[-1]: fwhm of edge function
     num_voigt: number of voigt component
     edge ({'g', 'l'}): type of edge shape function
                        if edge is not set, it does not include edge function.
     base_order ({0, 1, 2}): polynomial order of baseline function
                             if base_order is not set, it does not include baseline function.
     fix_param_idx: idx for fixed parameter (masked array for `params`)
     e: 1d array of energy points of data (n,)
     data: static data (n,)
     eps: estimated error of data (n,)

    Returns:
     Residucal vector
    
    Note:
     data should not contain energy range.
     If fwhm_G of ith voigt component is zero then it is treated as lorenzian function with fwhm_L
     If fwhm_L of ith voigt component is zero then it is treated as gaussian function with fwhm_G
    '''
    params = np.atleast_1d(params)
    tot_comp = num_voigt

    e0 = params[:num_voigt]
    fwhm_G = params[num_voigt:2*num_voigt]
    fwhm_L = params[2*num_voigt:3*num_voigt]
    
    if edge is not None:
        tot_comp = tot_comp+1
    if base_order is not None:
        tot_comp = tot_comp+base_order+1
    
    A = np.empty((tot_comp, e.size))

    for i in range(num_voigt):
        A[i, :] = voigt(e-e0[i], fwhm_G[i], fwhm_L[i])
    
    base_start = num_voigt
    if edge is not None:
        base_start = base_start+1
        if edge == 'g':
            A[num_voigt, :] = edge_gaussian(e-params[-2], params[-1])
        elif edge == 'l':
            A[num_voigt,:] = edge_lorenzian(e-params[-2], params[-1])
    
    if base_order is not None:
        A[base_start, :] = np.ones(e.size)
        for i in range(base_order):
            A[base_start+i] = e*A[base_start+i-1]
    
    c = fact_anal_A(A, data, eps)

    chi = (data - c@A)/eps

    return chi

def jac_res_voigt(params: np.ndarray, num_voigt: int, edge: Optional[str] = None,
                    base_order: Optional[int] = None, 
                    fix_param_idx: Optional[np.ndarray] = None,
                    e: np.ndarray = None, 
                    data: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    jac_res_voigt
    scipy.optimize.least_squares compatible dfient of vector residual function for fitting static spectrum with the 
    sum of voigt function, edge function base function

    Args:
     params: parameter used for fitting

              param[i]: peak position e0_i for each voigt component
              param[num_voigt+i]: fwhm_G of each voigt component
              param[2*num_voigt+i]: fwhm_L of each voigt component

              if edge is not None:

                 param[-2]: edge position

                 param[-1]: fwhm of edge function
     num_voigt: number of voigt component
     edge ({'g', 'l'}): type of edge shape function
                        if edge is not set, it does not include edge function.
     base_order ({0, 1, 2}): polynomial order of baseline function
                             if base_order is not set, it does not include baseline function.
     fix_param_idx: idx for fixed parameter (masked array for `params`)
     e: 1d array of energy points of data (n,)
     data: data (n,)
     eps: estimated error of data (n,)

    Returns:
     Grandient of residucal vector
    
    Note:
     data should not contain energy range.
     If fwhm_G of ith voigt component is zero then it is treated as lorenzian function with fwhm_L
     If fwhm_L of ith voigt component is zero then it is treated as gaussian function with fwhm_G
    '''
    params = np.atleast_1d(params)
    tot_comp = num_voigt

    e0 = params[:num_voigt]
    fwhm_G = params[num_voigt:2*num_voigt]
    fwhm_L = params[2*num_voigt:3*num_voigt]
    
    if edge is not None:
        tot_comp = tot_comp+1
    if base_order is not None:
        tot_comp = tot_comp+base_order+1
    
    A = np.empty((tot_comp, e.size))

    for i in range(num_voigt):
        A[i, :] = voigt(e-e0[i], fwhm_G[i], fwhm_L[i])
    
    base_start = num_voigt
    if edge is not None:
        base_start = num_voigt+1
        if edge == 'g':
            A[num_voigt, :] = edge_gaussian(e-params[-2], params[-1])
        elif edge == 'l':
            A[num_voigt,:] = edge_lorenzian(e-params[-2], params[-1])
    
    if base_order is not None:
        A[base_start, :] = np.ones(e.size)
        for i in range(base_order):
            A[base_start+i+1] = e*A[base_start+i]
    
    c = fact_anal_A(A, data, eps)

    df = np.empty((data.size, params.size))

    for i in range(num_voigt):
        df_tmp = c[i]*deriv_voigt(e-e0[i], fwhm_G[i], fwhm_L[i])
        df[:, i] = -df_tmp[0]
        df[:, num_voigt+i] = df_tmp[1]
        df[:, 2*num_voigt+i] = df_tmp[2]

    if edge is not None:
        if edge == 'g':
            df_edge = c[num_voigt]*deriv_edge_gaussian(e-params[-2], params[-1])
        elif edge == 'l':
            df_edge = c[num_voigt]*deriv_edge_lorenzian(e-params[-2], params[-1]) 
        
        df[:, -2] = -df_edge[0]
        df[:, -1] = df_edge[1]
    
    df = np.einsum('i,ij->ij', 1/eps, df)

    df[:, fix_param_idx] = 0

    return df