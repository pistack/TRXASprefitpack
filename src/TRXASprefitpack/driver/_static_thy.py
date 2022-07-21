'''
_static_thy:
submodule for static spectrum with the
sum of voigt broadend theoretical spectrum, edge function and baseline function

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Tuple
import numpy as np
from numpy.polynomial.legendre import legval
from .static_result import StaticResult
from ._ampgo import ampgo
from scipy.optimize import basinhopping
from scipy.optimize import least_squares
from ..mathfun.peak_shape import edge_gaussian, edge_lorenzian, voigt_thy
from ..mathfun.A_matrix import fact_anal_A
from ..res.res_gen import residual_scalar, grad_res_scalar
from ..res.res_thy import residual_thy, jac_res_thy

GLOBAL_OPTS = {'ampgo': ampgo, 'basinhopping': basinhopping}

def fit_static_thy(thy_peak: np.ndarray, fwhm_G_init: np.ndarray, fwhm_L_init: np.ndarray,
                   policy: str, peak_shift: Optional[float] = None,
                   peak_scale: Optional[float] = None,
                   edge: Optional[str] = None, 
                   edge_pos_init: Optional[float] = None,
                   edge_fwhm_init: Optional[float] = None,
                   base_order: Optional[int] = None,
                   method_glb: Optional[str] = 'basinhopping', 
                   method_lsq: Optional[str] = 'trf',
                   kwargs_glb: Optional[dict] = None, 
                   kwargs_lsq: Optional[dict] = None,
                   bound_fwhm_G: Optional[Tuple[float, float]] = None, 
                   bound_fwhm_L: Optional[Tuple[float, float]] = None,
                   bound_peak_shift: Optional[Tuple[float, float]] = None,
                   bound_peak_scale: Optional[Tuple[float, float]] = None,
                   bound_edge_pos: Optional[Tuple[float, float]] = None,
                   bound_edge_fwhm: Optional[Tuple[float, float]] = None,
                   e: Optional[np.ndarray] = None, 
                   data: Optional[np.ndarray] = None,
                   eps: Optional[np.ndarray] = None) -> StaticResult:
                      
      '''
      driver routine for fitting static spectrum with sum of voigt broadend thoretical spectrum,
      edge and polynomial base line.

      This driver using two step algorithm to search best parameter, its covariance and
      estimated parameter error.

      Model: sum of voigt broadened theoretical spectrum, edge function and base function
      :math:`c_{0} y(e, {fwhm}_{(G, i)}, {fwhm}_{(L, i), {peak_factor}}) + c_{1}{edge} + {base}`
      
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
       thy_peak (np.ndarray): peak position and intensity for theoretically calculated spectrum
       fwhm_G_init (float): initial gaussian part of fwhm parameter
       fwhm_L_init (float): initial lorenzian part of fwhm parameter
       policy ({'shift', 'scale', 'both'}): policy to match discrepancy between thoretical spectrum and experimental one 
       peak_shift (float): peak shift parameter
       peak_scale (float): peak scale parameter
       edge ({'g', 'l'}): type of edge function. If edge is not set, edge feature is not included.
       edge_pos_init: initial edge position
       edge_fwhm_init: initial fwhm parameter of edge
       method_glb ({'ampgo', 'basinhopping'}): 
        method of global optimization used in fitting process
       method_lsq ({'trf', 'dogbox', 'lm'}): method of local optimization for least_squares
                                             minimization (refinement of global optimization solution)
       kwargs_glb: keyward arguments for global optimization solver
       kwargs_lsq: keyward arguments for least square optimization solver
       bound_fwhm_G (sequence of tuple): boundary for fwhm_G parameter. 
        If `bound_fwhm_G` is `None`, the upper and lower bound are given as `(fwhm_G/2, 2*fwhm_G)`.
       bound_fwhm_L (sequence of tuple): boundary for fwhm_L parameter. 
        If `bound_fwhm_L` is `None`, the upper and lower bound are given as `(fwhm_L/2, 2*fwhm_L)`.
       bound_peak_shift (tuple): boundary for peak shift parameter. If `bound_peak_shift` is `None`, the upper and lower bound are
        given as `(-2*|peak_shift|,2*|peak_shift|)`.
       bound_peak_scale (tuple): boundary for peak scale parameter. If `bound_peak_scale` is `None`, the upper and lower bound are
        given as `(0.9*peak_scale, 1.1*peak_scale)`.
       bound_edge_pos (tuple): boundary for edge position, 
        if `bound_edge_pos` is `None` and `edge` is set, the upper and lower bound are given as `(0.99*edge_pos, 1.01*edge_pos)`.
       bound_edge_fwhm (tuple): boundary for fwhm parameter of edge feature. 
        If `bound_edge_fwhm` is `None`, the upper and lower bound are given as `(edge_fwhm/2, 2*edge_fwhm)`.
       e (np.narray): energy range for data
       data (np.ndarray): static spectrum data (it should not contain energy scan range)
       eps (np.ndarray): estimated errors of static spectrum data

       Returns:
        StaticResult class object
       Note:
        if initial fwhm_G is zero then such voigt component is treated as lorenzian component
        if initial fwhm_L is zero then such voigt component is treated as gaussian component
      '''
      num_voigt = 1
      num_param = 3 + 1*(policy == 'both')

      num_comp = num_voigt
      if edge is not None:
            num_comp = num_comp+1
            num_param = num_param+2
      
      if base_order is not None:
            num_comp = num_comp + base_order + 1
      
      param = np.empty(num_param, dtype=float)
      fix_param_idx = np.empty(num_param, dtype=bool)

      param[0] = fwhm_G_init
      param[1] = fwhm_L_init
      if policy == 'shift':
            param[2] = peak_shift
      elif policy == 'scale':
            param[2] = peak_scale
      elif policy == 'both':
            param[2] = peak_shift
            param[3] = peak_scale
      if edge is not None:
            param[-2] = edge_pos_init
            param[-1] = edge_fwhm_init
      
      bound = num_param*[None]

      if bound_fwhm_G is None:
            bound[0] = (fwhm_G_init/2, 2*fwhm_G_init)
      else:
            bound[0] = bound_fwhm_G

      if bound_fwhm_L is None:
            bound[1] = (fwhm_L_init/2, 2*fwhm_L_init)
      else:
            bound[1] = bound_fwhm_G
      
      if policy in ['shift', 'both']:
            if bound_peak_shift is None:
                  bound[2] = (-2*np.abs(peak_shift), 2*np.abs(peak_shift))
            else:
                  bound[2] = bound_peak_shift
      elif policy == 'scale':
            if bound_peak_scale is None:
                  bound[2] = (0.9*peak_scale, 1.1*peak_scale)
            else:
                  bound[2] = bound_peak_scale
      
      if policy == 'both':
            if bound_peak_scale is None:
                  bound[3] = (0.9*peak_scale, 1.1*peak_scale)
            else:
                  bound[3] = bound_peak_scale
      
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
      
      go_args = (residual_thy, jac_res_thy, policy, thy_peak, edge, base_order, fix_param_idx, e, data, eps)
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
      
      res_lsq = least_squares(residual_thy, param_gopt, method=method_lsq, jac=jac_res_thy, bounds=bound_tuple, **kwargs_lsq)
      param_opt = res_lsq['x']

      param_name = np.empty(param_opt.size, dtype=object)
      param_name[0] = 'fwhm_G'
      param_name[1] = 'fwhm_L'
      fwhm_G_opt = param_opt[0]
      fwhm_L_opt = param_opt[1]
      if policy in ['scale', 'shift']:
            peak_factor_opt = param_opt[2]
            if policy == 'scale':
                  param_name[2] = 'peak_scale'
            else:
                  param_name[2] = 'peak_shift'
      elif policy == 'both':
            peak_factor_opt = np.ndarray([param_opt[2], param_opt[3]])
            param_name[2] = 'peak_shift'
            param_name[3] = 'peak_scale'


    # Calc individual chi2
      chi = res_lsq['fun']
      num_param_tot = num_comp+num_param-np.sum(fix_param_idx)
      chi2 = 2*res_lsq['cost']
      red_chi2 = chi2/(chi.size-num_param_tot)
      
      if edge is not None:
            param_name[-2] = f'E0_{edge}'
            if edge == 'g':
                  param_name[-1] = 'fwhm_(G, edge)'
            elif edge == 'l':
                  param_name[-1] = 'fwhm_(L, edge)'
      
      A = np.empty((num_comp, e.size))
      A[0, :] = voigt_thy(e, thy_peak, fwhm_G_opt, fwhm_L_opt, peak_factor_opt, policy)      
      base_start = 1
      if edge is not None:
            base_start = base_start+1
            if edge == 'g':
                  A[1, :] = edge_gaussian(e-param_opt[-2], param_opt[-1])
            elif edge == 'l':
                  A[1, :] = edge_lorenzian(e-param_opt[-2], param_opt[-1])
    
      if base_order is not None:
            e_max = np.max(e); e_min = np.min(e)
            e_norm = 2*(e-(e_max+e_min)/2)/(e_max-e_min)
            tmp = np.eye(base_order+1)
            A[base_start:, :] = legval(e_norm, tmp, tensor=True)
      
      c = fact_anal_A(A, data, eps)

      fit = c@A

      fit_comp = np.einsum('i,ij->ij', c[:base_start], A[:base_start,:])
      
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
      result['model'] = 'thy'
      result['e'] = e; result['data'] = data; result['eps'] = eps
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
      if method_glb == 'ampgo':
            result['message_glb'] = res_go['message']
      elif method_glb == 'basinhopping':
            result['message_glb'] = res_go['message'][0]           
      result['message_lsq'] = res_lsq['message']

      return result