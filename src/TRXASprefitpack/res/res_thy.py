'''
res_thy:
submodule for residual function and dfient for fitting static spectrum with the
sum of voigt broadened theoretical spectrum, edge function and base function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Optional, Sequence
import numpy as np
from numpy.polynomial.legendre import legval
from ..mathfun.A_matrix import fact_anal_A
from ..mathfun.peak_shape import voigt_thy, edge_gaussian, edge_lorenzian
from ..mathfun.peak_shape import deriv_voigt_thy, deriv_edge_gaussian, deriv_edge_lorenzian

def residual_thy(x0: np.ndarray, policy: str, thy_peak: Sequence[np.ndarray], 
                 edge: Optional[str] = None, num_edge: Optional[int] = 0,
                 base_order: Optional[int] = None, 
                 e: np.ndarray = None, 
                 intensity: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    residaul_thy
    `scipy.optimize.least_squares` compatible vector residual function for fitting static spectrum with the 
    sum of voigt broadend theoretical spectrum, edge function base function

    Args:
     x0: initial parameter

      * 1st and 2nd: fwhm_G and fwhm_L

      if policy == 'scale':

      * 3rd to :math:`2+{num}_{thy}` : peak_scale

      if policy == 'shift':

      * 3rd to :math:`2+{num}_{thy}` : peak_shift

      if policy == 'both':

      * 3rd to :math:`2+{num}_{thy}` : peak_shift
      * :math:`2+{num}_{thy}` to :math:`2+2 {num}_{thy}`: peak_scale

      if edge is not None:
      
      * :math:`2+{num}_{thy}+{num}_{edge}+i` or :math:`2+2{num_thy}+{num}_{edge}+i`: i th edge position
      * :math:`2+{num}_{thy}+2{num}_{edge}+i` or :math:`2+2{num_thy}+2{num}_{edge}+i`: fwhm of i th edge

     policy ({'shift', 'scale', 'both'}): Policy to match discrepency 
      between experimental data and theoretical spectrum.

      * 'shift' : Default option, shift peak position by peak_factor
      * 'scale' : scale peak position by peak_factor
      * 'both' : both shift and scale peak postition, peak_factor should be a tuple of `shift_factor` and `scale_factor`.

     thy_peak: theoretically calculated peak position and intensity
     edge ({'g', 'l'}): type of edge shape function
      if edge is not set, it does not include edge function.
     num_edge: the number of edge feature
     base_order (int): polynomial order of baseline function
      if base_order is not set, it does not include baseline function.
     fix_param_idx: idx for fixed parameter (masked array for `x0`)
     e: 1d array of energy points of data (n,)
     intensity: intensity of static data (n,)
     eps: estimated error of data (n,)

    Returns:
     Tuple of scalar residual function :math:`(\\frac{1}{2}\\sum_i {res}^2_i)` and its gradient
    
    Note:
     * If fwhm_G of ith voigt component is zero then it is treated as lorenzian function with fwhm_L
     * If fwhm_L of ith voigt component is zero then it is treated as gaussian function with fwhm_G

    '''
    x0 = np.atleast_1d(x0)

    thy_comp = len(thy_peak)
    tot_comp = thy_comp

    if policy in ['scale', 'shift']:
        peak_factor = x0[2:2+thy_comp]
    elif policy == 'both':
        peak_factor = np.array(thy_comp, dtype=object)
        for i in range(thy_comp):
            peak_factor[i] = np.array([x0[2+i], x0[2+thy_comp+i]])
    
    if edge is not None:
        tot_comp = tot_comp+num_edge
    if base_order is not None:
        tot_comp = tot_comp+base_order+1
    
    A = np.empty((tot_comp, e.size))
    for i in range(thy_comp):
        A[i, :] = \
            voigt_thy(e, thy_peak[i], x0[0], x0[1], peak_factor[i], policy)/np.sum(thy_peak[i][:, 1])
    
    base_start = thy_comp
    if edge is not None:
        base_start = base_start+num_edge
        if policy in ['shift', 'scale']:
            param_edge_start = 2+thy_comp
        else:
            param_edge_start = 2+2*thy_comp
        if edge == 'g':
            for i in range(num_edge):
                A[thy_comp+i, :] = edge_gaussian(e-x0[param_edge_start+i], 
                x0[param_edge_start+num_edge+i])
        elif edge == 'l':
            for i in range(num_edge):
                A[thy_comp+i, :] = edge_lorenzian(e-x0[param_edge_start+i], 
                    x0[param_edge_start+num_edge+i])
    
    if base_order is not None:
        e_max = np.max(e); e_min = np.min(e)
        e_norm = 2*(e-(e_max+e_min)/2)/(e_max-e_min)
        tmp = np.eye(base_order+1)
        A[base_start:, :] = legval(e_norm, tmp, tensor=True)
    
    c = fact_anal_A(A, intensity, eps)
    chi = (c@A - intensity)/eps

    return chi

def res_grad_thy(x0: np.ndarray, policy: str, thy_peak: Sequence[np.ndarray], 
                 edge: Optional[str] = None, num_edge: Optional[int] = 0,
                 base_order: Optional[int] = None, 
                 fix_param_idx: Optional[np.ndarray] = None,
                 e: np.ndarray = None, 
                 intensity: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    res_grad_thy
    `scipy.optimize.minimize` compatible scalar residual function and its gradient for fitting static spectrum with the 
    sum of voigt broadend theoretical spectrum, edge function base function

    Args:
     x0: initial parameter

      * 1st and 2nd: fwhm_G and fwhm_L

      if policy == 'scale':

      * 3rd to :math:`2+{num}_{thy}` : peak_scale

      if policy == 'shift':

      * 3rd to :math:`2+{num}_{thy}` : peak_shift

      if policy == 'both':

      * 3rd to :math:`2+{num}_{thy}` : peak_shift
      * :math:`2+{num}_{thy}` to :math:`2+2 {num}_{thy}`: peak_scale

      if edge is not None:
      
      * :math:`2+{num}_{thy}+{num}_{edge}+i` or :math:`2+2{num_thy}+{num}_{edge}+i`: i th edge position
      * :math:`2+{num}_{thy}+2{num}_{edge}+i` or :math:`2+2{num_thy}+2{num}_{edge}+i`: fwhm of i th edge

     policy ({'shift', 'scale', 'both'}): Policy to match discrepency 
      between experimental data and theoretical spectrum.

      * 'shift' : Default option, shift peak position by peak_factor
      * 'scale' : scale peak position by peak_factor
      * 'both' : both shift and scale peak postition, peak_factor should be a tuple of `shift_factor` and `scale_factor`.

     thy_peak: theoretically calculated peak position and intensity
     edge ({'g', 'l'}): type of edge shape function
      if edge is not set, it does not include edge function.
     num_edge: the number of edge feature
     base_order (int): polynomial order of baseline function
      if base_order is not set, it does not include baseline function.
     fix_param_idx: idx for fixed parameter (masked array for `x0`)
     e: 1d array of energy points of data (n,)
     intensity: intensity of static data (n,)
     eps: estimated error of data (n,)

    Returns:
     Tuple of scalar residual function :math:`(\\frac{1}{2}\\sum_i {res}^2_i)` and its gradient
    
    Note:
     * If fwhm_G of ith voigt component is zero then it is treated as lorenzian function with fwhm_L
     * If fwhm_L of ith voigt component is zero then it is treated as gaussian function with fwhm_G

    '''
    x0 = np.atleast_1d(x0)

    thy_comp = len(thy_peak)
    tot_comp = thy_comp
    area = np.empty(thy_comp)
    for i in range(thy_comp):
        area[i] = np.sum(thy_peak[i][:, 1])

    if policy in ['scale', 'shift']:
        peak_factor = x0[2:2+thy_comp]
    elif policy == 'both':
        peak_factor = np.array(thy_comp, dtype=object)
        for i in range(thy_comp):
            peak_factor[i] = np.array([x0[2+i], x0[2+thy_comp+i]])
    
    if edge is not None:
        tot_comp = tot_comp+num_edge
    if base_order is not None:
        tot_comp = tot_comp+base_order+1
    
    A = np.empty((tot_comp, e.size))
    for i in range(thy_comp):
        A[i, :] = \
            voigt_thy(e, thy_peak[i], x0[0], x0[1], peak_factor[i], policy)/area[i]
    
    base_start = thy_comp
    if edge is not None:
        base_start = base_start+num_edge
        if policy in ['shift', 'scale']:
            param_edge_start = 2+thy_comp
        else:
            param_edge_start = 2+2*thy_comp
        if edge == 'g':
            for i in range(num_edge):
                A[thy_comp+i, :] = edge_gaussian(e-x0[param_edge_start+i], 
                x0[param_edge_start+num_edge+i])
        elif edge == 'l':
            for i in range(num_edge):
                A[thy_comp+i, :] = edge_lorenzian(e-x0[param_edge_start+i], 
                    x0[param_edge_start+num_edge+i])
    
    if base_order is not None:
        e_max = np.max(e); e_min = np.min(e)
        e_norm = 2*(e-(e_max+e_min)/2)/(e_max-e_min)
        tmp = np.eye(base_order+1)
        A[base_start:, :] = legval(e_norm, tmp, tensor=True)
    
    c = fact_anal_A(A, intensity, eps)
    chi = (c@A - intensity)/eps
    df = np.zeros((intensity.size, x0.size))

    for i in range(thy_comp):
        deriv_thy = (c[i]/area[i])*\
            deriv_voigt_thy(e, thy_peak[i], x0[0], x0[1], peak_factor[i], policy)
        df[:, :2] = df[:, :2] + deriv_thy[:, :2]
        if policy in ['scale', 'shift']:
            df[:, 2+i] = deriv_thy[:, 2]
        else:
            df[:, 2+i] = deriv_thy[:, 2]
            df[:, 2+thy_comp+i] = deriv_thy[:, 3] 

    if edge is not None:
        if edge == 'g':
            for i in range(num_edge):
                df_edge = c[thy_comp+i]*deriv_edge_gaussian(e-x0[param_edge_start+i], 
                x0[param_edge_start+num_edge+i])
                df[:, param_edge_start+i] = -df_edge[:, 0]
                df[:, param_edge_start+num_edge+i] = df_edge[:, 1]
        elif edge == 'l':
            for i in range(num_edge):
                df_edge = c[thy_comp+i]*deriv_edge_lorenzian(e-x0[param_edge_start+i], 
                x0[param_edge_start+num_edge+i]) 
                df[:, param_edge_start+i] = -df_edge[:, 0]
                df[:, param_edge_start+num_edge+i] = df_edge[:, 1]
    
    df = np.einsum('i,ij->ij', 1/eps, df)

    df[:, fix_param_idx] = 0

    return np.sum(chi**2)/2, chi@df