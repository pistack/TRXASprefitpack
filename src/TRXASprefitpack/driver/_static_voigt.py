'''
_static_voigt:
submodule for static spectrum with the
sum of voigt function, edge function and baseline function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Sequence, Tuple
import numpy as np
from .static_result import StaticResult
from ._ampgo import ampgo
from scipy.optimize import basinhopping
from scipy.optimize import least_squares
from ..mathfun.peak_shape import voigt, edge_gaussian, edge_lorenzian
from ..mathfun.A_matrix import fact_anal_A
from ..res.parm_bound import set_bound_t0
from ..res.res_gen import residual_scalar, grad_res_scalar
from ..res.res_voigt import residual_voigt, jac_res_voigt

GLOBAL_OPTS = {'ampgo': ampgo, 'basinhopping': basinhopping}

def fit_static_voigt(e0_init: np.ndarray, fwhm_G_init: np.ndarray, fwhm_L_init: np.ndarray,
                     edge: Optional[str] = None, 
                     edge_pos_init: Optional[float] = None,
                     edge_fwhm_init: Optional[float] = None,
                     base_order: Optional[int] = None,
                     method_glb: Optional[str] = 'basinhopping', 
                     method_lsq: Optional[str] = 'trf',
                     kwargs_glb: Optional[dict] = None, 
                     kwargs_lsq: Optional[dict] = None,
                     bound_e0: Optional[Sequence[Tuple[float, float]]] = None, 
                     bound_fwhm_G: Optional[Sequence[Tuple[float, float]]] = None, 
                     bound_fwhm_L: Optional[Sequence[Tuple[float, float]]] = None,
                     bound_edge_pos: Optional[Tuple[float, float]] = None,
                     bound_edge_fwhm: Optional[Tuple[float, float]] = None,
                      e: Optional[np.ndarray] = None, 
                      data: Optional[np.ndarray] = None,
                      eps: Optional[np.ndarray] = None) -> StaticResult:
                      
      '''
      driver routine for fitting multiple data set of time delay scan data with
      sum of the convolution of exponential decay and instrumental response function.

      This driver using two step algorithm to search best parameter, its covariance and
      estimated parameter error.

      Model: sum of voigt function, edge function and base function
      :math:`\\sum_{i=1}^n c_i y_i(t-e_0_i, {fwhm}_{(G, i)}, {fwhm}_{(L, i)}) + c_{n+1}{edge} + {base}`
      
      Objective function: chi squared
      :math:`\\chi^2 = \sum_i \\left(\\frac{model-data_i}{eps_i}\\right)^2`
                    

      Step 1. (method_glb)
      Use global optimization to find rough global minimum of our objective function

      Step 2. (method_lsq)
      Use least squares optimization algorithm to refine global minimum of objective function and approximate covariance matrix.
                      
      Moreover two solve non linear least square optimization problem efficiently, it separates linear and non-linear part of the problem.

      Finding weight :math:`c_i` which minimizes :math:`\\chi^2` when any other parameters are given is linear problem, 
      this problem is solved before the each iteration of non linear problem.
      Since :math:`\\frac{\\partial \\chi^2}{\\partial c_i} = 0` is satisfied, the jacobian or gradient for
      our objective function is ,simply, :math:`\\frac{\\partial \\chi^2}{\\partial {param}_i}`. 
      
      Args:
       e0_init (np.ndarray): initial peak position of each voigt component
       fwhm_G_init (np.ndarray): initial gaussian part of fwhm parameter of each voigt component
       fwhm_L_init (np.ndarray): initial lorenzian part of fwhm parameter of each voigt component
       edge ({'g', 'l'}): type of edge function. If edge is not set, edge feature is not included.
       edge_pos_init: initial edge position
       edge_fwhm_init: initial fwhm parameter of edge
       method_glb ({'ampgo', 'basinhopping'}): 
        method of global optimization used in fitting process
       method_lsq ({'trf', 'dogbox', 'lm'}): method of local optimization for least_squares
                                             minimization (refinement of global optimization solution)
       kwargs_glb: keyward arguments for global optimization solver
       kwargs_lsq: keyward arguments for least square optimization solver
       bound_e0 (sequence of tuple): boundary for each voigt componet. If upper and lower bound are same, 
        driver assumes that the parameter is fixed during optimization. If `bound_fwhm` is `None`, 
        the upper and lower bound are given as `(e0-2*fwhm_(init,i), 2*fwhm_(init,i))`.
       bound_fwhm_G (sequence of tuple): boundary for fwhm_G parameter. 
        If `bound_fwhm_G` is `None`, the upper and lower bound are given as `(fwhm_G/2, 2*fwhm_G)`.
       bound_fwhm_L (sequence of tuple): boundary for fwhm_L parameter. 
        If `bound_fwhm_L` is `None`, the upper and lower bound are given as `(fwhm_L/2, 2*fwhm_L)`.
       bound_edge_pos (tuple): boundary for edge position, 
        if `bound_tau` is `None` and `edge` is set, the upper and lower bound are given as `(0.99*edge_pos, 1.01*edge_pos)`.
       e (np.narray): energy range for data
       data (np.ndarray): static spectrum data (it should not contain energy scan range)
       eps (np.ndarray): estimated errors of static spectrum data

       Returns:
        StaticResult class object
       Note:
        if initial fwhm_G is zero then such voigt component is treated as lorenzian component
        if initial fwhm_L is zero then such voigt component is treated as gaussian component
      '''

      if e0_init is None:
            num_voigt = 0
            num_param = 0
      else:
            num_voigt = e0_init.size
            num_param = 3*num_voigt

      num_comp = num_voigt
      if edge is not None:
            num_comp = num_comp+1
            num_param = num_param+2
      
      if base_order is not None:
            num_comp = num_comp + base_order + 1
      
      param = np.empty(num_param, dtype=float)
      fix_param_idx = np.empty(num_param, dtype=bool)

      param[:num_voigt] = e0_init
      param[num_voigt:2*num_voigt] = fwhm_G_init
      param[2*num_voigt:3*num_voigt] = fwhm_L_init
      if edge is not None:
            param[-2] = edge_pos_init
            param[-1] = edge_fwhm_init
      
      bound = num_param*[None]

      if bound_e0 is None:
            tmp = np.empty(2)
            for i in range(num_voigt):
                  tmp[0] = fwhm_G_init[i]
                  tmp[1] = fwhm_L_init[i]
                  bound[i] = set_bound_t0(e0_init[i], tmp)
      else:
            bound[:num_voigt] = bound_e0
                      
      if bound_fwhm_G is None:
            for i in range(num_voigt):
                  bound[i+num_voigt] = (fwhm_G_init[i]/2, 2*fwhm_G_init[i])
      else:
            bound[num_voigt: 2*num_voigt] = bound_fwhm_G
      
      if bound_fwhm_L is None:
            for i in range(num_voigt):
                  bound[i+2*num_voigt] = (fwhm_L_init[i]/2, 2*fwhm_L_init[i])
      else:
            bound[2*num_voigt:3*num_voigt] = bound_fwhm_L
      
      if edge is not None:
            if bound_edge_pos is None:
                  bound[-2] = (0.999*edge_pos_init, 1.001*edge_pos_init)
            else:
                  bound[-2] = bound_edge_pos
            if bound_edge_fwhm is None:
                  bound[-1] = (edge_fwhm_init/2, 2*edge_fwhm_init)
            else:
                  bound[-1] = bound_edge_fwhm

      for i in range(num_param):
            fix_param_idx[i] = (bound[i][0] == bound[i][1])
      
      go_args = (residual_voigt, jac_res_voigt, num_voigt, edge, base_order, fix_param_idx, e, data, eps)
      min_go_kwargs = {'args': go_args, 'jac': grad_res_scalar, 'bounds': bound}
      if kwargs_glb is not None:
            minimizer_kwargs = kwargs_glb.pop('minimizer_kwargs', None)
            if minimizer_kwargs is None:
                  kwargs_glb['minimizer_kwargs'] = min_go_kwargs
            else:
                  kwargs_glb['minimizer_kwargs'] = minimizer_kwargs.update(min_go_kwargs)
      else:
            kwargs_glb = {'minimizer_kwargs' : min_go_kwargs}
      res_go = GLOBAL_OPTS[method_glb](residual_scalar, param, **kwargs_glb)
      param_gopt = res_go['x']

      if kwargs_lsq is not None:
            _ = kwargs_lsq.pop('args', None)
            _ = kwargs_lsq.pop('kwargs', None)
            kwargs_lsq['args'] = tuple(go_args[2:])
      else:
            kwargs_lsq = {'args' : tuple(go_args[2:])}

      bound_tuple = (num_param*[None], num_param*[None])
      for i in range(num_param):
            bound_tuple[0][i] = bound[i][0]
            bound_tuple[1][i] = bound[i][1]
            if bound[i][0] == bound[i][1]:
                  if bound[i][0] > 0:
                        bound_tuple[1][i] = bound[i][1]*(1+1e-4)+1e-8
                  else:
                        bound_tuple[1][i] = bound[i][1]*(1-1e-4)+1e-8
      
      res_lsq = least_squares(residual_voigt, param_gopt, method=method_lsq, jac=jac_res_voigt, bounds=bound_tuple, **kwargs_lsq)
      param_opt = res_lsq['x']

      e0_opt = param_opt[:num_voigt]
      fwhm_G_opt = param_opt[num_voigt:2*num_voigt]
      fwhm_L_opt = param_opt[2*num_voigt:3*num_voigt]


    # Calc individual chi2
      chi = res_lsq['fun']
      num_param_tot = num_comp+num_param-np.sum(fix_param_idx)
      chi2 = 2*res_lsq['cost']
      red_chi2 = chi2/(chi.size-num_param_tot)

      param_name = np.empty(param_opt.size, dtype=object)
      for i in range(num_voigt):
            param_name[i] = f'e0_{i+1}'
            param_name[num_voigt+i] = f'fwhm_(G, {i+1})'
            param_name[2*num_voigt+i] = f'fwhm_(L, {i+1})'
      
      if edge is not None:
            param_name[-2] = f'E0_{edge}'
            if edge == 'g':
                  param_name[-1] = 'fwhm_(G, edge)'
            elif edge == 'l':
                  param_name[-1] = 'fwhm_(L, edge)'
      
      A = np.empty((num_comp, e.size))
      
      for i in range(num_voigt):
            A[i, :] = voigt(e-e0_opt[i], fwhm_G_opt[i], fwhm_L_opt[i])
      
      base_start = num_voigt
      
      if edge is not None:
            base_start = base_start+1
            if edge == 'g':
                  A[num_voigt, :] = edge_gaussian(e-param_opt[-2], param_opt[-1])
            elif edge == 'l':
                  A[num_voigt,:] = edge_lorenzian(e-param_opt[-2], param_opt[-1])
    
      if base_order is not None:
            A[base_start, :] = np.ones(e.size)
            for i in range(base_order):
                  A[base_start+i] = e*A[base_start+i-1]
      
      c = fact_anal_A(A, data, eps)

      fit = c@A

      if edge is not None:
            fit_comp = np.einsum('i,ij->ij', c[:num_voigt+1], A[:num_voigt+1,:])
      else:
            fit_comp = np.einsum('i,ij->ij', c[:num_voigt], A[:num_voigt, :])
      
      base = None

      if base_order is not None:
            base = c[base_start:]@A[base_start:,:]
            
      res = data - fit
      
      jac = res_lsq['jac']
      hes = jac.T @ jac
      cov = np.zeros_like(hes)
      n_free_param = np.sum(~fix_param_idx)
      mask_2d = np.einsum('i,j->ij', ~fix_param_idx, ~fix_param_idx)
      cov[mask_2d] = np.linalg.inv(hes[mask_2d].reshape((n_free_param, n_free_param))).flatten()
      cov_scaled = red_chi2*cov
      param_eps = np.sqrt(np.diag(cov_scaled))
      corr = cov_scaled.copy()
      weight = np.einsum('i,j->ij', param_eps, param_eps)
      corr[mask_2d] = corr[mask_2d]/weight[mask_2d]

      result = StaticResult()
      result['model'] = 'voigt'
      result['fit'] = fit; result['fit_comp'] = fit_comp 
      result['res'] = res; result['base_order'] = base_order
      result['edge'] = edge; result['n_voigt'] = num_voigt
      result['param_name'] = param_name; result['x'] = param_opt
      result['bounds'] = bound; result['base'] = base; result['c'] = c
      result['chi2'] = chi2
      result['aic'] = chi.size*np.log(chi2/chi.size)+2*num_param_tot
      result['bic'] = chi.size*np.log(chi2/chi.size)+num_param_tot*np.log(chi.size)
      result['red_chi2'] = red_chi2
      result['nfev'] = res_go['nfev'] + res_lsq['nfev']
      result['n_param'] = num_param_tot
      result['num_pts'] = chi.size; result['jac'] = jac
      result['cov'] = cov; result['corr'] = corr; result['cov_scaled'] = cov_scaled
      result['x_eps'] = param_eps
      result['success_lsq'] = res_lsq['success']

      if result['success_lsq']:
            result['status'] = 0
      else:
            result['status'] = -1
      
      result['method_glb'] = method_glb
      result['method_lsq'] = method_lsq
      result['message_glb'] = res_go['message']
      result['message_lsq'] = res_lsq['message']

      return result