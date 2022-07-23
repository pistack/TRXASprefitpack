'''
_transient_exp:
submodule for fitting time delay scan with the
convolution of sum of exponential decay and instrumental response function 

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Optional, Union, Sequence, Tuple
import numpy as np
from ..mathfun.irf import calc_eta
from .transient_result import TransientResult
from scipy.optimize import basinhopping
from scipy.optimize import least_squares
from ..mathfun.A_matrix import make_A_matrix_exp, fact_anal_A
from ..res.parm_bound import set_bound_t0, set_bound_tau
from ..res.res_gen import residual_scalar, res_grad_scalar
from ..res.res_decay import residual_decay, jac_res_decay

def fit_transient_exp(irf: str, fwhm_init: Union[float, np.ndarray], 
                      t0_init: np.ndarray, tau_init: np.ndarray, base: bool, 
                      do_glb: Optional[bool] = False, 
                      method_lsq: Optional[str] = 'trf',
                      kwargs_glb: Optional[dict] = None, 
                      kwargs_lsq: Optional[dict] = None,
                      bound_fwhm: Optional[Sequence[Tuple[float, float]]] = None, 
                      bound_t0: Optional[Sequence[Tuple[float, float]]] = None, 
                      bound_tau: Optional[Sequence[Tuple[float, float]]] = None,
                      name_of_dset: Optional[Sequence[str]] = None,
                      t: Optional[Sequence[np.ndarray]] = None, 
                      intensity: Optional[Sequence[np.ndarray]] = None,
                      eps: Optional[Sequence[np.ndarray]] = None) -> TransientResult:
                      
      '''
      driver routine for fitting multiple data set of time delay scan data with
      sum of the convolution of exponential decay and instrumental response function.

      It separates linear and non-linear part of the optimization problem to solve non linear least sequare
      optimization problem efficiently.

      Finding weight :math:`c_i` which minimizes :math:`\\chi^2` when any other parameters are given is linear problem, 
      this problem is solved before the each iteration of non linear problem.
      Since :math:`\\frac{\\partial \\chi^2}{\\partial c_i} = 0` is satisfied, the gradient for
      our scalar residual function is ,simply, :math:`\\frac{\\partial \\chi^2}{\\partial {param}_i}`. 

      Model: sum of convolution of exponential decay and instrumental response function
      :math:`\\sum_{i=1}^n c_i y_i(t-t_0, {fwhm}, 1/\\tau_i) + {base}*c_{n+1} y_{n+1}(t-t0, {fwhm}, 0)`
      
      Objective function: chi squared
      :math:`\\chi^2 = \sum_i \\left(\\frac{model-data_i}{eps_i}\\right)^2`

      Moreover this driver uses two step algorithm to search best parameter, its covariance and
      estimated parameter error.

      Step 1. (method_glb)
      Use global optimization to find rough global minimum of our objective function.
      In this stage, it use analytic gradient for scalar residual function.

      Step 2. (method_lsq)
      Use least squares optimization algorithm to refine global minimum of objective function and approximate covariance matrix.
      Because of linear and non-linear seperation scheme, the analytic jacobian for vector residual function is hard to obtain.
      Thus, in this stage, it uses numerical jacobian. 
                            
      Args:
       irf ({'g', 'c', 'pv'}): shape of instrumental response function
       
        'g': gaussian shape
        
        'c': cauchy shape
        
        'pv': pseudo voigt shape
       fwhm_init (float or np.ndarray): initial full width at half maximum for instrumental response function
       
        if irf in ['g', 'c'] then fwhm_init is float
        
        if irf == 'pv' then fwhm_init is the numpy.ndarray with [fwhm_G, fwhm_L]
       t0_init (np.ndarray): time zeros for each scan
       tau_init (np.ndarray): lifetime constant for each decay component
       base (bool): Whether or not include baseline feature (i.e. very long lifetime constant)
       do_glb (bool): Whether or not use global optimization algorithm. If True then basinhopping algorithm is used.
       method_lsq ({'trf', 'dogbox', 'lm'}): method of local optimization for least_squares
                                             minimization (refinement of global optimization solution)
       kwargs_glb: keyward arguments for global optimization solver
       kwargs_lsq: keyward arguments for least square optimization solver
       bound_fwhm (sequence of tuple): boundary for irf parameter. If upper and lower bound are same, 
        driver assumes that the parameter is fixed during optimization. If `bound_fwhm` is `None`, 
        the upper and lower bound are given as `(fwhm_init/2, 2*fwhm_init)`.
       bound_t0 (sequence of tuple): boundary for time zero parameter. 
        If `bound_t0` is `None`, the upper and lower bound are given as `(t0-2*fwhm_init, t0+2*fwhm_init)`.
       bound_tau (sequence of tuple): boundary for lifetime constant for decay component, 
        if `bound_tau` is `None`, the upper and lower bound are given by ``set_bound_tau``.
       name_of_dset (sequence of str): name of each dataset
       t (sequence of np.narray): time scan range for each datasets
       intensity (sequence of np.ndarray): sequence of intensity of datasets for time delay scan
       eps (sequence of np.ndarray): sequence of estimated errors of each dataset

       Returns:
        TransientResult class object
      '''
      if tau_init is None:
            num_comp = 0
      else:
            num_comp = tau_init.size

      num_irf = 1*(irf in ['g', 'c'])+2*(irf == 'pv')
      num_param = num_irf+t0_init.size+num_comp
      param = np.empty(num_param, dtype=float)
      fix_param_idx = np.empty(num_param, dtype=bool)

      param[:num_irf] = fwhm_init
      param[num_irf:num_irf+t0_init.size] = t0_init
      param[num_irf+t0_init.size:] = tau_init
      bound = num_param*[None]

      if bound_fwhm is None:
            for i in range(num_irf):
                  bound[i] = (param[i]/2, 2*param[i])
      else:
            bound[:num_irf] = bound_fwhm
                      
      if bound_t0 is None:
            for i in range(t0_init.size):
                  bound[i+num_irf] = set_bound_t0(t0_init[i], fwhm_init)
      else:
            bound[num_irf:num_irf+t0_init.size] = bound_t0
      
      if bound_tau is None:
            for i in range(num_comp):
                  bound[i+num_irf+t0_init.size] = set_bound_tau(tau_init[i], fwhm_init)
      else:
            bound[num_irf+t0_init.size:] = bound_tau

      for i in range(num_param):
            fix_param_idx[i] = (bound[i][0] == bound[i][1])
      
      if do_glb:
            go_args = (residual_decay, jac_res_decay, num_comp, base, irf, fix_param_idx, 
            t, intensity, eps)
            min_go_kwargs = {'args': go_args, 'jac': True, 'bounds': bound}
            if kwargs_glb is not None:
                  minimizer_kwargs = kwargs_glb.pop('minimizer_kwargs', None)
                  if minimizer_kwargs is None:
                        kwargs_glb['minimizer_kwargs'] = min_go_kwargs
                  else:
                        minimizer_kwargs['args'] = min_go_kwargs['args']
                        minimizer_kwargs['jac'] = min_go_kwargs['jac']
                        minimizer_kwargs['bounds'] = min_go_kwargs['bounds']
                        kwargs_glb['minimizer_kwargs'] = minimizer_kwargs
            else:
                  kwargs_glb = {'minimizer_kwargs' : min_go_kwargs}
            res_go = basinhopping(res_grad_scalar, param, **kwargs_glb)
      else:
            res_go = dict()
            res_go['x'] = param
            res_go['message'] = None
            res_go['nfev'] = 0

      param_gopt = res_go['x']
      args_lsq = (num_comp, base, irf, fix_param_idx, t, intensity, eps)

      if kwargs_lsq is not None:
            _ = kwargs_lsq.pop('args', None)
            _ = kwargs_lsq.pop('kwargs', None)
            kwargs_lsq['args'] = args_lsq
      else:
            kwargs_lsq = {'args' : args_lsq}

      bound_tuple = (num_param*[None], num_param*[None])
      for i in range(num_param):
            bound_tuple[0][i] = bound[i][0]
            bound_tuple[1][i] = bound[i][1]
            if bound[i][0] == bound[i][1]:
                  if bound[i][0] > 0:
                        bound_tuple[1][i] = bound[i][1]*(1+1e-8)+1e-16
                  else:
                        bound_tuple[1][i] = bound[i][1]*(1-1e-8)+1e-16
      
      # Since jacobian of vector residual function is inaccurate 
      res_lsq = least_squares(residual_decay, param_gopt, method=method_lsq, bounds=bound_tuple, **kwargs_lsq)
      param_opt = res_lsq['x']

      fwhm_opt = param_opt[:num_irf]
      tau_opt = param_opt[num_irf+t0_init.size:]
      
      fit = np.empty(len(t), dtype=object); res = np.empty(len(t), dtype=object)
      
      num_tot_scan = 0
      for i in range(len(t)):
            num_tot_scan = num_tot_scan + intensity[i].shape[1]
            fit[i] = np.empty(intensity[i].shape)
            res[i] = np.empty(intensity[i].shape)


    # Calc individual chi2
      chi = res_lsq['fun']
      num_param_tot = num_tot_scan*(num_comp+1*base)+num_param-np.sum(fix_param_idx)
      chi2 = 2*res_lsq['cost']
      red_chi2 = chi2/(chi.size-num_param_tot)
      
      start = 0; end = 0; 
      chi2_ind = np.empty(len(t), dtype=object); red_chi2_ind = np.empty(len(t), dtype=object)
      num_param_ind = 2*tau_opt.size+1*base+2+1*(irf == 'pv')

      for i in range(len(t)):
            end = start + intensity[i].size
            chi_aux = chi[start:end].reshape(intensity[i].shape)
            chi2_ind[i] = np.sum(chi_aux**2, axis=0)
            red_chi2_ind[i] = chi2_ind[i]/(intensity[i].shape[0]-num_param_ind)
            start = end

      param_name = np.empty(param_opt.size, dtype=object)
      c = np.empty(len(t), dtype=object)
      t0_idx = num_irf

      if irf == 'g':
            param_name[0] = 'fwhm_G'
      elif irf == 'c':
            param_name[0] = 'fwhm_L'
      else:
            param_name[0] = 'fwhm_G'
            param_name[1] = 'fwhm_L'

      for i in range(len(t)):
            c[i] = np.empty((num_comp+1*base, intensity[i].shape[1]))
            
            for j in range(intensity[i].shape[1]):
                  A = make_A_matrix_exp(t[i]-param_opt[t0_idx], fwhm_opt, tau_opt, base, irf)
                  c[i][:, j] = fact_anal_A(A, intensity[i][:, j], eps[i][:, j])
                  fit[i][:, j] = c[i][:, j] @ A
                  param_name[t0_idx] = f't_0_{i+1}_{j+1}'
                  t0_idx = t0_idx + 1
            
            res[i] = intensity[i] - fit[i]
      
      for i in range(num_comp):
            param_name[num_irf+t0_init.size+i] = f'tau_{i+1}'
      
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

      result = TransientResult()
      result['model'] = 'decay'
      result['fit'] = fit; result['res'] = res; result['irf'] = irf

      if irf == 'g':
            result['eta'] = 0
      elif irf == 'c':
            result['eta'] = 1
      else:
            result['eta'] = calc_eta(fwhm_opt[0], fwhm_opt[1])
      
      # save experimental fitting data
      if name_of_dset is None:
            name_of_dset = np.empty(len(t), dtype=object)
            for i in range(len(t)):
                  name_of_dset[i] = f'dataset_{i+1}'
      result['name_of_dset'] = name_of_dset; result['t'] = t
      result['intensity'] = intensity; result['eps'] = eps

      result['param_name'] = param_name; result['x'] = param_opt
      result['bounds'] = bound; result['base'] = base; result['c'] = c
      result['chi2'] = chi2; result['chi2_ind'] = chi2_ind
      result['aic'] = chi.size*np.log(chi2/chi.size)+2*num_param_tot
      result['bic'] = chi.size*np.log(chi2/chi.size)+num_param_tot*np.log(chi.size)
      result['red_chi2'] = red_chi2; result['red_chi2_ind'] = red_chi2_ind
      result['nfev'] = res_go['nfev'] + res_lsq['nfev']
      result['n_param'] = num_param_tot; result['n_param_ind'] = num_param_ind
      result['num_pts'] = chi.size; result['jac'] = jac
      result['cov'] = cov; result['corr'] = corr; result['cov_scaled'] = cov_scaled
      result['x_eps'] = param_eps
      result['method_lsq'] = method_lsq
      result['message_lsq'] = res_lsq['message']
      result['success_lsq'] = res_lsq['success']

      if result['success_lsq']:
            result['status'] = 0
      else:
            result['status'] = -1
      
      if do_glb:
            result['method_glb'] = 'basinhopping'
            result['message_glb'] = res_go['message'][0]  
      else:
            result['method_glb'] = None
            result['message_glb'] = None

      result['n_osc'] = 0
      if tau_init is None:
            result['n_decay'] = 0
      result['n_decay'] = tau_init.size

      return result