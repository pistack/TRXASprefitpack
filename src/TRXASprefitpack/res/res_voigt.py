'''
res_voigt:
submodule for residual function and dfient for fitting static spectrum with the
sum of voigt function, edge function and base function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Optional
import numpy as np
from numpy.polynomial.legendre import legval
from ..mathfun.A_matrix import fact_anal_A
from ..mathfun.peak_shape import voigt, edge_gaussian, edge_lorenzian
from ..mathfun.peak_shape import deriv_voigt, deriv_edge_gaussian, deriv_edge_lorenzian
from ..mathfun.peak_shape import hess_voigt, hess_edge_gaussian, hess_edge_lorenzian


def residual_voigt(x0: np.ndarray, num_voigt: int, edge: Optional[str] = None,
                   num_edge: Optional[int] = 0,
                   base_order: Optional[int] = None,
                   e: np.ndarray = None,
                   intensity: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    residual_voigt
    scipy.optimize.least_squares compatible vector residual function for fitting static spectrum with the
    sum of voigt function, edge function base function

    Args:
     x0: initial parameter

         * i th: peak position e0_i for i th voigt component
         * :math:`{num}_{voigt}+i` th: fwhm_G of i th voigt component
         * :math:`2{num}_{voigt}+i` th: fwhm_L of i th voigt component

         if edge is not None:

         * :math:`3{num}_{voigt}+i` th: ith edge position
         * :math:`3{num}_{voigt}+{num}_{edge}+i` th: fwhm of ith edge function

     num_voigt: number of voigt component
     edge ({'g', 'l'}): type of edge shape function
      if edge is not set, it does not include edge function.
     num_edge: number of edge component
     base_order (int): polynomial order of baseline function
      if base_order is not set, it does not include baseline function.
     e: 1d array of energy points of data (n,)
     intensity: intensity of static data (n,)
     eps: estimated error of data (n,)

    Returns:
     Residucal vector

    Note:

     * If fwhm_G of ith voigt component is zero then it is treated as lorenzian function with fwhm_L
     * If fwhm_L of ith voigt component is zero then it is treated as gaussian function with fwhm_G
    '''
    x0 = np.atleast_1d(x0)
    tot_comp = num_voigt

    e0 = x0[:num_voigt]
    fwhm_G = x0[num_voigt:2*num_voigt]
    fwhm_L = x0[2*num_voigt:3*num_voigt]

    if edge is not None:
        tot_comp = tot_comp+num_edge
    if base_order is not None:
        tot_comp = tot_comp+base_order+1

    A = np.empty((tot_comp, e.size))

    for i in range(num_voigt):
        A[i, :] = voigt(e-e0[i], fwhm_G[i], fwhm_L[i])

    base_start = num_voigt
    if edge is not None:
        base_start = base_start+num_edge
        if edge == 'g':
            for i in range(num_edge):
                A[num_voigt+i, :] = edge_gaussian(e-x0[3*num_voigt+i],
                                                  x0[3*num_voigt+num_edge+i])
        elif edge == 'l':
            for i in range(num_edge):
                A[num_voigt+i, :] = edge_lorenzian(e-x0[3*num_voigt+i],
                                                   x0[3*num_voigt+num_edge+i])

    if base_order is not None:
        e_max = np.max(e)
        e_min = np.min(e)
        e_norm = 2*(e-(e_max+e_min)/2)/(e_max-e_min)
        tmp = np.eye(base_order+1)
        A[base_start:, :] = legval(e_norm, tmp, tensor=True)

    c = fact_anal_A(A, intensity, eps)

    chi = (c@A-intensity)/eps

    return chi


def res_grad_voigt(x0: np.ndarray, num_voigt: int, edge: Optional[str] = None,
                   num_edge: Optional[int] = 0,
                   base_order: Optional[int] = None,
                   fix_param_idx: Optional[np.ndarray] = None,
                   e: np.ndarray = None,
                   intensity: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    res_grad_voigt
    scipy.optimize.minimizer compatible scalar residual function and its gradient for fitting static spectrum with the
    sum of voigt function, edge function base function

    Args:
     x0: initial parameter

         * i th: peak position e0_i for i th voigt component
         * :math:`{num}_{voigt}+i` th: fwhm_G of i th voigt component
         * :math:`2{num}_{voigt}+i` th: fwhm_L of i th voigt component

         if edge is not None:

         * :math:`3{num}_{voigt}+i` th: ith edge position
         * :math:`3{num}_{voigt}+{num}_{edge}+i` th: fwhm of ith edge function

     num_voigt: number of voigt component
     edge ({'g', 'l'}): type of edge shape function
      if edge is not set, it does not include edge function.
     num_edge: number of edge component
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
    tot_comp = num_voigt

    e0 = x0[:num_voigt]
    fwhm_G = x0[num_voigt:2*num_voigt]
    fwhm_L = x0[2*num_voigt:3*num_voigt]

    if edge is not None:
        tot_comp = tot_comp+num_edge
    if base_order is not None:
        tot_comp = tot_comp+base_order+1

    A = np.empty((tot_comp, e.size))

    for i in range(num_voigt):
        A[i, :] = voigt(e-e0[i], fwhm_G[i], fwhm_L[i])

    base_start = num_voigt
    if edge is not None:
        base_start = num_voigt+num_edge
        if edge == 'g':
            for i in range(num_edge):
                A[num_voigt+i, :] = edge_gaussian(e-x0[3*num_voigt+i],
                                                  x0[3*num_voigt+num_edge+i])
        elif edge == 'l':
            for i in range(num_edge):
                A[num_voigt, :] = edge_lorenzian(e-x0[3*num_voigt+i],
                                                 x0[3*num_voigt+num_edge+i])

    if base_order is not None:
        e_max = np.max(e)
        e_min = np.min(e)
        e_norm = 2*(e-(e_max+e_min)/2)/(e_max-e_min)
        tmp = np.eye(base_order+1)
        A[base_start:, :] = legval(e_norm, tmp, tensor=True)

    c = fact_anal_A(A, intensity, eps)

    chi = (c@A-intensity)/eps
    df = np.empty((intensity.size, x0.size))

    for i in range(num_voigt):
        df_tmp = c[i]*deriv_voigt(e-e0[i], fwhm_G[i], fwhm_L[i])
        df[:, i] = -df_tmp[:, 0]
        df[:, num_voigt+i] = df_tmp[:, 1]
        df[:, 2*num_voigt+i] = df_tmp[:, 2]

    if edge is not None:
        if edge == 'g':
            for i in range(num_edge):
                df_edge = c[num_voigt+i]*deriv_edge_gaussian(e-x0[3*num_voigt+i],
                                                             x0[3*num_voigt+num_edge+i])
                df[:, 3*num_voigt+i] = -df_edge[:, 0]
                df[:, 3*num_voigt+num_edge+i] = df_edge[:, 1]
        elif edge == 'l':
            for i in range(num_edge):
                df_edge = c[num_voigt+i]*deriv_edge_lorenzian(e-x0[3*num_voigt+i],
                                                              x0[3*num_voigt+num_edge+i])
                df[:, 3*num_voigt+i] = -df_edge[:, 0]
                df[:, 3*num_voigt+num_edge+i] = df_edge[:, 1]

    df = np.einsum('i,ij->ij', 1/eps, df)

    df[:, fix_param_idx] = 0

    return np.sum(chi**2)/2, chi@df

def res_hess_voigt(x0: np.ndarray, num_voigt: int, edge: Optional[str] = None,
                   num_edge: Optional[int] = 0,
                   base_order: Optional[int] = None,
                   e: np.ndarray = None,
                   intensity: np.ndarray = None, eps: np.ndarray = None) -> np.ndarray:
    '''
    res_hess_voigt

    Compute hessian of 1/2*chi(C(theta), theta) defined in seperation scheme section

    Args:
     x0: initial parameter

         * i th: peak position e0_i for i th voigt component
         * :math:`{num}_{voigt}+i` th: fwhm_G of i th voigt component
         * :math:`2{num}_{voigt}+i` th: fwhm_L of i th voigt component

         if edge is not None:

         * :math:`3{num}_{voigt}+i` th: ith edge position
         * :math:`3{num}_{voigt}+{num}_{edge}+i` th: fwhm of ith edge function

     num_voigt: number of voigt component
     edge ({'g', 'l'}): type of edge shape function
      if edge is not set, it does not include edge function.
     num_edge: number of edge component
     base_order (int): polynomial order of baseline function
      if base_order is not set, it does not include baseline function.
     e: 1d array of energy points of data (n,)
     intensity: intensity of static data (n,)
     eps: estimated error of data (n,)

    Returns:
     Hessian of scalar residual function :math:`(\\frac{1}{2}\\sum_i {res}^2_i)` based on the seperation scheme

    Note:

     * If fwhm_G of ith voigt component is zero then it is treated as lorenzian function with fwhm_L
     * If fwhm_L of ith voigt component is zero then it is treated as gaussian function with fwhm_G
    '''

    x0 = np.atleast_1d(x0)
    tot_comp = num_voigt

    e0 = x0[:num_voigt]
    fwhm_G = x0[num_voigt:2*num_voigt]
    fwhm_L = x0[2*num_voigt:3*num_voigt]

    if edge is not None:
        tot_comp = tot_comp+num_edge
    if base_order is not None:
        tot_comp = tot_comp+base_order+1

    A = np.empty((tot_comp, e.size))

    for i in range(num_voigt):
        A[i, :] = voigt(e-e0[i], fwhm_G[i], fwhm_L[i])

    base_start = num_voigt
    if edge is not None:
        base_start = num_voigt+num_edge
        if edge == 'g':
            for i in range(num_edge):
                A[num_voigt+i, :] = edge_gaussian(e-x0[3*num_voigt+i],
                                                  x0[3*num_voigt+num_edge+i])
        elif edge == 'l':
            for i in range(num_edge):
                A[num_voigt, :] = edge_lorenzian(e-x0[3*num_voigt+i],
                                                 x0[3*num_voigt+num_edge+i])

    if base_order is not None:
        e_max = np.max(e)
        e_min = np.min(e)
        e_norm = 2*(e-(e_max+e_min)/2)/(e_max-e_min)
        tmp = np.eye(base_order+1)
        A[base_start:, :] = legval(e_norm, tmp, tensor=True)

    c = fact_anal_A(A, intensity, eps)

    dc = np.einsum('ij,j->ij', A, 1/eps)
    chi = (c@A-intensity)/eps
    df = np.empty((intensity.size, x0.size))
    
    Hc = dc @ dc.T
    Hcx = np.zeros((c.size, x0.size))
    Hx = np.zeros((x0.size, x0.size))

    # df and Hcx
    for i in range(num_voigt):
        df_tmp = deriv_voigt(e-e0[i], fwhm_G[i], fwhm_L[i])
        df_tmp = np.einsum('ij,i->ij', df_tmp, 1/eps)
        df_tmp[:, 0] = -df_tmp[:, 0]
        hf_tmp = c[i]*hess_voigt(e-e0[i], fwhm_G[i], fwhm_L[i])
        hf_tmp = np.einsum('ij,i->ij', hf_tmp, 1/eps)
        df[:, i] = c[i]*df_tmp[:, 0]
        df[:, num_voigt+i] = c[i]*df_tmp[:, 1]
        df[:, 2*num_voigt+i] = c[i]*df_tmp[:, 2]
        chidf_tmp = chi@df_tmp
        Hcx[i, i] = chidf_tmp[0]
        Hcx[i, num_voigt+i] = chidf_tmp[1]
        Hcx[i, 2*num_voigt+i] = chidf_tmp[2]
        chihf_tmp = chi@hf_tmp
        Hx[i, i] = chihf_tmp[0]
        Hx[i, num_voigt+i] = -chihf_tmp[1]
        Hx[i, 2*num_voigt+i] = -chihf_tmp[2]
        Hx[num_voigt+i, num_voigt+i] = chihf_tmp[3]
        Hx[num_voigt+i, 2*num_voigt+i] = chihf_tmp[4]
        Hx[2*num_voigt+i, 2*num_voigt+i] = chihf_tmp[5]
        Hx[num_voigt+i, i] = Hx[i, num_voigt+i]
        Hx[2*num_voigt+i, i] = Hx[i, 2*num_voigt+i]
        Hx[2*num_voigt+i, num_voigt+i] = Hx[num_voigt+i, 2*num_voigt+i]



    if edge is not None:
        if edge == 'g':
            for i in range(num_edge):
                df_edge = deriv_edge_gaussian(e-x0[3*num_voigt+i],
                                              x0[3*num_voigt+num_edge+i])
                df_edge = np.einsum('ij,i->ij', df_edge, 1/eps)
                dh_edge = c[num_voigt+i]*hess_edge_gaussian(e-x0[3*num_voigt+i],
                                                            x0[3*num_voigt+num_edge+i])
                dh_edge = np.einsum('ij,i->ij', dh_edge, 1/eps)
                df[:, 3*num_voigt+i] = -c[num_voigt+i]*df_edge[:, 0]
                df[:, 3*num_voigt+num_edge+i] = c[num_voigt+i]*df_edge[:, 1]
                chidf_edge = chi@df_edge
                Hcx[num_voigt+i, 3*num_voigt+i] = -chidf_edge[0]
                Hcx[num_voigt+i, 3*num_voigt+num_edge+i] = chidf_edge[1]
                chidh_edge = chi@dh_edge
                Hx[3*num_voigt+i, 3*num_voigt+i] = chidh_edge[0]
                Hx[3*num_voigt+i, 3*num_voigt+num_edge+i] = -chidh_edge[1]
                Hx[3*num_voigt+num_edge+i, 3*num_voigt+num_edge+i] = chidh_edge[2]
                Hx[3*num_voigt+num_edge+i, 3*num_voigt+i] = Hx[3*num_voigt+i, 3*num_voigt+num_edge+i]
        elif edge == 'l':
            for i in range(num_edge):
                df_edge = deriv_edge_lorenzian(e-x0[3*num_voigt+i],
                                              x0[3*num_voigt+num_edge+i])
                df_edge = np.einsum('ij,i->ij', df_edge, 1/eps)
                dh_edge = c[num_voigt+i]*hess_edge_lorenzian(e-x0[3*num_voigt+i],
                                                            x0[3*num_voigt+num_edge+i])
                dh_edge = np.einsum('ij,i->ij', dh_edge, 1/eps)
                df[:, 3*num_voigt+i] = -c[num_voigt+i]*df_edge[:, 0]
                df[:, 3*num_voigt+num_edge+i] = c[num_voigt+i]*df_edge[:, 1]
                chidf_edge = chi@df_edge
                Hcx[num_voigt+i, 3*num_voigt+i] = -chidf_edge[0]
                Hcx[num_voigt+i, 3*num_voigt+num_edge+i] = chidf_edge[1]
                chidh_edge = chi@dh_edge
                Hx[3*num_voigt+i, 3*num_voigt+i] = chidh_edge[0]
                Hx[3*num_voigt+i, 3*num_voigt+num_edge+i] = -chidh_edge[1]
                Hx[3*num_voigt+num_edge+i, 3*num_voigt+num_edge+i] = chidh_edge[2]
                Hx[3*num_voigt+num_edge+i, 3*num_voigt+i] = Hx[3*num_voigt+i, 3*num_voigt+num_edge+i]
    
    Hcx = Hcx + (dc @ df)
    Hx = Hx + (df.T @ df)

    # solve Hcx = Hc B
    B = np.linalg.solve(Hc, Hcx)

    H = Hx - B.T@(Hcx)

    return H
