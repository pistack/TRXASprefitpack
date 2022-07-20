'''
res_thy:
submodule for residual function and dfient for fitting static spectrum with the
sum of voigt broadened theoretical spectrum, edge function and base function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Optional
import numpy as np
from ..mathfun.A_matrix import fact_anal_A
from ..mathfun.peak_shape import voigt_thy, edge_gaussian, edge_lorenzian
from ..mathfun.peak_shape import deriv_voigt_thy, deriv_edge_gaussian, deriv_edge_lorenzian

def residual_thy(params: np.ndarray, policy: str, thy_peak: np.ndarray, edge: Optional[str] = None,
                 base_order: Optional[int] = None, 
                 fix_param_idx: Optional[np.ndarray] = None,
                 e: np.ndarray = None, 
                 data: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    residual_thy
    scipy.optimize.least_squares compatible vector residual function for fitting static spectrum with the 
    sum of voigt broadend theoretical spectrum, edge function base function

    Args:
     params: parameter used for fitting
              param[0]: fwhm_G 
              param[1]: fwhm_L
              if policy == 'scale':
                param[2]: peak_scale
              if policy == 'shift':
                param[2]: peak_shift
              if policy == 'both':
                param[2]: peak_shift
                param[3]: peak_scale

              if edge is not None:

                 param[-2]: edge position

                 param[-1]: fwhm of edge function
     policy {'shift', 'scale', 'both'}: Policy to match discrepency 
      between experimental data and theoretical spectrum.

      'shift' : Default option, shift peak position by peak_factor
      'scale' : scale peak position by peak_factor
      'both' : both shift and scale peak postition
        peak_factor = [shift_factor, scale_factor]
     thy_peak: theoretically calculated peak position and intensity
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

    if policy in ['scale', 'shift']:
        peak_factor = params[2]
    elif policy == 'both':
        peak_factor = np.ndarray([params[2], params[3]])
    
    tot_comp = 1
    
    if edge is not None:
        tot_comp = tot_comp+1
    if base_order is not None:
        tot_comp = tot_comp+base_order+1
    
    A = np.empty((tot_comp, e.size))

    A[0, :] = voigt_thy(e, thy_peak, params[0], params[1], peak_factor, policy)
    
    base_start = 1
    if edge is not None:
        base_start = base_start+1
        if edge == 'g':
            A[1, :] = edge_gaussian(e-params[-2], params[-1])
        elif edge == 'l':
            A[1, :] = edge_lorenzian(e-params[-2], params[-1])
    
    if base_order is not None:
        A[base_start, :] = np.ones(e.size)
        for i in range(base_order):
            A[base_start+i+1] = e*A[base_start+i]
    
    c = fact_anal_A(A, data, eps)

    chi = (data - c@A)/eps

    return chi

def jac_res_thy(params: np.ndarray, policy: str, thy_peak: np.ndarray, edge: Optional[str] = None,
                base_order: Optional[int] = None, 
                fix_param_idx: Optional[np.ndarray] = None,
                e: np.ndarray = None, 
                data: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    jac_res_thy
    scipy.optimize.least_squares compatible vector residual function for fitting static spectrum with the 
    sum of voigt broadend theoretical spectrum, edge function base function

    Args:
     params: parameter used for fitting
              param[0]: fwhm_G 
              param[1]: fwhm_L
              if policy == 'scale':
                param[2]: peak_scale
              if policy == 'shift':
                param[2]: peak_shift
              if policy == 'both':
                param[2]: peak_shift
                param[3]: peak_scale

              if edge is not None:

                 param[-2]: edge position

                 param[-1]: fwhm of edge function
     policy {'shift', 'scale', 'both'}: Policy to match discrepency 
      between experimental data and theoretical spectrum.

      'shift' : Default option, shift peak position by peak_factor
      'scale' : scale peak position by peak_factor
      'both' : both shift and scale peak postition
        peak_factor = [shift_factor, scale_factor]
     thy_peak: theoretically calculated peak position and intensity
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

    if policy in ['scale', 'shift']:
        peak_factor = params[2]
    elif policy == 'both':
        peak_factor = np.ndarray([params[2], params[3]])
    
    tot_comp = 1
    
    if edge is not None:
        tot_comp = tot_comp+1
    if base_order is not None:
        tot_comp = tot_comp+base_order+1
    
    A = np.empty((tot_comp, e.size))

    A[0, :] = voigt_thy(e, thy_peak, params[0], params[1], peak_factor, policy)
    
    base_start = 1
    if edge is not None:
        base_start = base_start+1
        if edge == 'g':
            A[1, :] = edge_gaussian(e-params[-2], params[-1])
        elif edge == 'l':
            A[1, :] = edge_lorenzian(e-params[-2], params[-1])
    
    if base_order is not None:
        A[base_start, :] = np.ones(e.size)
        for i in range(base_order):
            A[base_start+i+1] = e*A[base_start+i]
    
    c = fact_anal_A(A, data, eps)

    df = np.empty((data.size, params.size))

    deriv_thy = c[0]*deriv_voigt_thy(e, thy_peak, params[0], params[1], peak_factor, policy)
    df[:, :2] = deriv_thy[:2, :].T
    if policy in ['scale', 'shift']:
        df[:, 2] = deriv_thy[2, :].T
    elif policy == 'both':
        df[:, 2:4] = deriv_thy[2:, :].T 

    if edge is not None:
        if edge == 'g':
            df_edge = c[1]*deriv_edge_gaussian(e-params[-2], params[-1])
        elif edge == 'l':
            df_edge = c[1]*deriv_edge_lorenzian(e-params[-2], params[-1]) 
        
        df[:, -2] = -df_edge[0]
        df[:, -1] = df_edge[1]
    
    df = np.einsum('i,ij->ij', 1/eps, df)

    df[:, fix_param_idx] = 0

    return df