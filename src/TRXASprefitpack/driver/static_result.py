import os
from pathlib import Path
import numpy as np

class StaticResult(dict):
      '''
      Represent results for fitting static driver routine

      Attributes:
       model ({'thy', 'voigt'}): model used for fitting

              `voigt`: sum of voigt function, edge function and base funtion

              `thy`: sum of voigt broadened theoretical lineshape spectrum, edge function and base function
       e (np.ndarray): energy range
       data (np.ndarray): static spectrum
       eps (np.ndarray): estimated error of static spectrum
       fit (np.ndarray): fitting curve for data (n,)
       fit_comp (np.ndarray): curve for each voigt component and edge
       base (np.ndaray): fitting curve for baseline
       res (np.ndarray): residual curve (data-fit) for each data (n,)
       edge ({'gaussian', 'lorenzian'}): type of edge function, if edge is None then edge function is not
       included in the fitting model

            'gaussian': gaussian type edge function

            'lorenzian': lorenzian type edge function
       base_order (int): order of baseline function
                         if base_order is None then baseline is not included in the fitting model
       param_name (np.ndarray): name of parameter
       n_voigt (int): number of voigt component
       x (np.ndarray): best parameter
       bounds (sequence of tuple): boundary of each parameter
       c (np.ndarray): best weight of each voigt component and edge of data
       chi2 (float): chi squared value of fitting
       aic (float): Akaike Information Criterion statistic: :math:`N\\log(\\chi^2/N)+2N_{parm}`
       bic (float): Bayesian Information Criterion statistic: :math:`N\\log(\\chi^2/N)+N_{parm}\log(N)`
       red_chi2 (float): total reduced chi squared value of fitting
       nfev (int): total number of function evaluation
       n_param (int): total number of effective parameter
       num_pts (int): total number of data points 
       jac (np.ndarray): jacobian of objective function at optimal point
       cov (np.ndarray): covariance matrix (i.e. inverse of (jac.T @ jac))
       cov_scaled (np.ndarray): scaled covariance matrix (i.e. `red_chi2` * `cov`)
       corr (np.ndarray): parameter correlation matrix
       x_eps (np.ndarray): estimated error of parameter
        (i.e. square root of diagonal element of `conv_scaled`)
       method_glb ({'ampgo', 'basinhopping'}): 
        method of global optimization used in fitting process
       message_glb (str): messages from global optimization process
       method_lsq ({'trf', 'dogbox', 'lm'}): method of local optimization for least_squares
                                             minimization (refinement of global optimization solution)
       success_lsq (bool): whether or not local least square optimization is successed
       message_lsq (str): messages from local least square optimization process
       status ({0, -1}): status of optimization process

                     `0` : normal termination

                     `-1` : least square optimization process is failed
      '''

      def __getattr__(self, name):
            try:
                  return self[name]
            except KeyError as e:
                  raise AttributeError(name) from e
      
      __setattr__ = dict.__setitem__
      __delattr__ = dict.__delitem__
      
      def __dir__(self):
            return list(self.keys())
      
      def __str__(self, corr_tol: float = 1e-1):
            '''
            Report StaticResult
            Args:
            corr_tol: parameter correlation greather than `corr_tol` would be reported
            '''
            doc_lst = []
            doc_lst.append("[Model information]")
            doc_lst.append(f"    model : {self['model']}")
            if self['edge'] is not None:
                  doc_lst.append(f"    edge: {self['edge']}")
            if self['base_order'] is not None:
                  doc_lst.append(f"    base_order: {self['base_order']}")
            doc_lst.append(' ')
            doc_lst.append("[Optimization Method]")
            doc_lst.append(f"    global: {self['method_glb']}")
            doc_lst.append(f"    leastsq: {self['method_lsq']}")
            doc_lst.append(' ')
            doc_lst.append("[Optimization Status]")
            doc_lst.append(f"    nfev: {self['nfev']}")
            doc_lst.append(f"    status: {self['status']}")
            doc_lst.append(f"    global_opt msg: {self['message_glb']}")
            doc_lst.append(f"    leastsq_opt msg: {self['message_lsq']}")
            doc_lst.append(' ')
            doc_lst.append("[Optimization Results]")
            doc_lst.append(f"    Data points: {self['num_pts']}")
            doc_lst.append(f"    Number of effective parameters: {self['n_param']}")
            doc_lst.append(f"    Degree of Freedom: {self['num_pts']-self['n_param']}")
            doc_lst.append(f"    Chi squared: {self['chi2']: .4f}".rstrip('0').rstrip('.'))
            doc_lst.append(f"    Reduced chi squared: {self['red_chi2']: .4f}".rstrip('0').rstrip('.'))
            doc_lst.append(f"    AIC (Akaike Information Criterion statistic): {self['aic']: .4f}".rstrip('0').rstrip('.'))
            doc_lst.append(f"    BIC (Bayesian Information Criterion statistic): {self['bic']: .4f}".rstrip('0').rstrip('.'))
            doc_lst.append(' ')
            doc_lst.append("[Parameters]")
            for pn, pv, p_err in zip(self['param_name'], self['x'], self['x_eps']):
                  doc_lst.append(f"    {pn}: {pv: .8f} +/- {p_err: .8f} ({100*np.abs(p_err/pv): .2f}%)".rstrip('0').rstrip('.'))
            doc_lst.append(' ')
            doc_lst.append("[Parameter Bound]")
            for pn, pv, bd in zip(self['param_name'], self['x'], self['bounds']):
                  doc_lst.append(f"    {pn}: {bd[0]: .8f}".rstrip('0').rstrip('.')+f" <= {pv: .8f} <= {bd[1]: .8f}".rstrip('0').rstrip('.'))
            doc_lst.append(' ')

            doc_lst.append("[Component Contribution]")
            doc_lst.append(f"    Static spectrum")
            if self['base_order'] is None:
                  coeff_abs = np.abs(self['c'])
                  coeff_contrib = 100*self['c']
            else:
                  coeff_abs = np.abs(self['c'][:-self['base_order']-1])
                  coeff_contrib = 100*self['c'][:-self['base_order']-1]
            coeff_sum = np.sum(coeff_abs)
            coeff_contrib = coeff_contrib/coeff_sum
            for v in range(self['n_voigt']):
                  row = [f"     {self['model']} {v+1}:"]
                  row.append(f'{coeff_contrib[v]: .2f}%')
                  doc_lst.append(' '.join(row))
            if self['edge'] is not None:
                  row = [f"     {self['edge']} type edge:"]
                  row.append(f"{coeff_contrib[self['n_voigt']]: .2f}%")
                  doc_lst.append(' '.join(row))
            doc_lst.append(' ')

            doc_lst.append("[Parameter Correlation]")
            doc_lst.append(f"    Parameter Correlations > {corr_tol: .3f}".rstrip('0').rstrip('.') + " are reported.")

            A = np.empty((len(self['x']), len(self['x'])), dtype=object)
            for i in range(len(self['x'])):
                  for j in range(len(self['x'])):
                        A[i,j] = (i,j)
            
            mask = (np.abs(self['corr']) > corr_tol)

            for pair in A[mask]:
                  if pair[0] > pair[1]:
                        tmp_str_lst = [f"    ({self['param_name'][pair[0]]},"]
                        tmp_str_lst.append(f"{self['param_name'][pair[1]]}")
                        tmp_str_lst.append('=')
                        tmp_str_lst.append(f"{self['corr'][pair]: .3f}".rstrip('0').rstrip('.'))
                        doc_lst.append(' '.join(tmp_str_lst))

            return '\n'.join(doc_lst)
      
      def __repr__(self):
            '''
            Print structure of StaticResult instance
            '''
            doc_lst = []
            print(self.items())
            for k,v in self.items():
                  if v is None:
                        doc_lst.append(f"{k}: None")
                  elif isinstance(v, (int, float, str)):
                        doc_lst.append(f"{k}: {v}")
                  elif len(v) == 1 and isinstance(v[0], (int, float, str)):
                        doc_lst.append(f"{k}: {v[0]}")
                  elif len(v) == 2 and isinstance(v[0], (int, float)) and isinstance(v[1], (int, float)):
                        doc_lst.append(f"{k}: ({v[0], v[1]})")
                  elif isinstance(v, np.ndarray):
                        doc_lst.append(f"{k}: numpy ndarray with datatype: {type(v[0,0])}, shape: {v.shape}")
            return '\n'.join(doc_lst)


def save_StaticResult(result: StaticResult, dirname: str):
      '''
      save static fitting result to the text file

      Args:
       result: static fitting result
       dirname: name of the directory in which text files for fitting result are saved.
       e: energy range of static spectrum
       eps: estimated error of static spectrum
      
      Returns:
       `fit_summary.txt`: Summary for the fitting result
       `weight.txt`: Weight of each voigt and edge component
       `fit.txt`: fitting, each voigt, edge and baseline curve for static spectrum
       `res.txt`: residual (fit-data) curve for static spectrum

      Note:
       If `dirname` directory is not exists, it creates `dirname` directory.
      '''
      if not (Path.cwd()/dirname).exists():
            os.mkdir(dirname)
      
      with open(f'{dirname}/fit_summary.txt', 'w') as f:
            f.write(str(result))

      tot_comp = result['n_voigt']
      if result['edge'] is not None:
            tot_comp = tot_comp+1
      if result['base_order'] is not None:
            tot_comp = tot_comp+1
      coeff_fmt = ['%.8e']
      fit_fmt = (2+tot_comp)*['%.8e']
      fit_header_lst = ['energy', 'fit']
      for i in range(result['n_voigt']):
            fit_header_lst.append(f'voigt_{i}')
      if result['edge'] is not None:
            fit_header_lst.append(f"{result['edge']}_type_edge")
      if result['base'] is not None:
            fit_header_lst.append("base")
            fit_save = np.vstack((result['e'], result['fit'], result['fit_comp'], result['base'])).T
      else:
            fit_save = np.vstack((result['e'], result['fit'], result['fit_comp'])).T
      res_save = np.vstack((result['e'], result['res'], result['eps'])).T
      np.savetxt(f'{dirname}/res.txt', res_save, fmt=['%.8e', '%.8e', '%.8e'], 
      header=f'energy \t res \t eps')
      fit_header = '\t'.join(fit_header_lst)
      coeff_header = 'static'

      np.savetxt(f'{dirname}/weight.txt', result['c'], fmt=coeff_fmt, header=coeff_header)
      np.savetxt(f'{dirname}/fit.txt', fit_save, fmt=fit_fmt, header=fit_header)
      
      return