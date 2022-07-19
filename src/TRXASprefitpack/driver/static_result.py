from typing import Optional, Sequence
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class StaticResult(dict):
      '''
      Represent results for fitting static driver routine

      Attributes:
       model ({'thy', 'voigt'}): model used for fitting

              `voigt`: sum of voigt function, edge function and base funtion

              `thy`: sum of voigt broadened theoretical lineshape spectrum, edge function and base function

       fit (np.ndarray): fitting curve for data (n,)
       base (np.ndaaray): fitting curve for baseline
       res (np.ndarray): residual curve (data-fit) for each data (n,)
       edge ({'none', 'gaussian', 'lorenzian'}): type of edge function

            'none': edge function is not included

            'gaussian': gaussian type edge function

            'lorenzian': lorenzian type edge function
       base_order (int): order of baseline function
                         if base_order is None then baseline is not included in the fitting model
       param_name (np.ndarray): name of parameter
       n_voigt (int): number of decay voigt (including baseline feature)
       x (np.ndarray): best parameter
       bounds (sequence of tuple): boundary of each parameter
       c (sequence of np.ndarray): best weight of each voigt component and edge of data
       chi2 (float): chi squared value of fitting
       aic (float): Akaike Information Criterion statistic: :math:`N\\log(\\chi^2/N)+2N_{parm}`
       bic (float): Bayesian Information Criterion statistic: :math:`N\\log(\\chi^2/N)+N_{parm}\log(N)`
       red_chi2 (float): total reduced chi squared value of fitting
       nfev (int): total number of function evaluation
       n_param (int): total number of effective parameter
       n_param_ind (int): number of parameter which affects fitting quality of indiviual time delay scan
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

def print_StaticResult(result: StaticResult, corr_tol: float =1e-1) -> str:
      '''
      print pretty docstring of StaticResult class

      Args:
       result: StaticResult class instance which has fitting result
       corr_tol: parameter correlation greather `corr_tol` would be reported
      
      Returns:
       string which reports fitting results
      '''

      doc_lst = []
      doc_lst.append("[Model information]")
      doc_lst.append(f"    model : {result['model']}")
      doc_lst.append(f"    edge: {result['edge']}")
      if result['base_order'] is not None:
            doc_lst.append(f"    base_order: {result['base_order']}")
      doc_lst.append(' ')
      doc_lst.append("[Optimization Method]")
      doc_lst.append(f"    global: {result['method_glb']}")
      doc_lst.append(f"    leastsq: {result['method_lsq']}")
      doc_lst.append(' ')
      doc_lst.append("[Optimization Status]")
      doc_lst.append(f"    nfev: {result['nfev']}")
      doc_lst.append(f"    status: {result['status']}")
      doc_lst.append(f"    global_opt msg: {result['message_glb']}")
      doc_lst.append(f"    leastsq_opt msg: {result['message_lsq']}")
      doc_lst.append(' ')
      doc_lst.append("[Optimization Results]")
      doc_lst.append(f"    Number of effective parameters: {result['n_param']}")
      doc_lst.append(f"    Degree of Freedom: {result['num_pts']-result['n_param']}")
      doc_lst.append(f"    Chi squared: {result['chi2']: .4f}".rstrip('0').rstrip('.'))
      doc_lst.append(f"    Reduced chi squared: {result['red_chi2']: .4f}".rstrip('0').rstrip('.'))
      doc_lst.append(f"    AIC (Akaike Information Criterion statistic): {result['aic']: .4f}".rstrip('0').rstrip('.'))
      doc_lst.append(f"    BIC (Bayesian Information Criterion statistic): {result['bic']: .4f}".rstrip('0').rstrip('.'))
      doc_lst.append(' ')
      doc_lst.append("[Parameters]")
      for pn, pv, p_err in zip(result['param_name'], result['x'], result['x_eps']):
            doc_lst.append(f"    {pn}: {pv: .8f} +/- {p_err: .8f} ({100*np.abs(p_err/pv): .2f}%)".rstrip('0').rstrip('.'))
      doc_lst.append(' ')
      doc_lst.append("[Parameter Bound]")
      for pn, pv, bd in zip(result['param_name'], result['x'], result['bounds']):
            doc_lst.append(f"    {pn}: {bd[0]: .8f}".rstrip('0').rstrip('.')+f" <= {pv: .8f} <= {bd[1]: .8f}".rstrip('0').rstrip('.'))
      doc_lst.append(' ')

      doc_lst.append("[Component Contribution]")
      doc_lst.append(f"    Static spectrum")
      if result['base_order'] == -1:
            coeff_abs = np.abs(result['c'])
      else:
            coeff_abs = np.abs(result['c'][:-result['base_order']-1])
      coeff_sum = np.sum(coeff_abs)
      coeff_contrib = 100*result['c'][:-result['base_order']-1]/coeff_sum
      for v in range(result['n_voigt']):
            row = [f"     voigt {v+1}"]
            row.append(f'{coeff_contrib[v]: .2f}%')
            doc_lst.append('\t'.join(row))
      if result['edge'] != 'none':
            row = [f"     {result['edge']}"]
            row.append(f"{coeff_contrib[result['n_voigt']]: .2f}%")
            doc_lst.append('\t'.join(row))
      doc_lst.append(' ')

      doc_lst.append("[Parameter Correlation]")
      doc_lst.append(f"    Parameter Correlations > {corr_tol: .3f}".rstrip('0').rstrip('.') + " are reported.")

      A = np.empty((len(result['x']), len(result['x'])), dtype=object)
      for i in range(len(result['x'])):
            for j in range(len(result['x'])):
                  A[i,j] = (i,j)
      mask = (np.abs(result['corr']) > corr_tol)

      for pair in A[mask]:
            if pair[0] > pair[1]:
                  doc_lst.append(f"    ({result['param_name'][pair[0]]}, {result['param_name'][pair[1]]}) = {result['corr'][pair] : .3f}".rstrip('0').rstrip('.'))

      return '\n'.join(doc_lst)

def plot_StaticResult(result: StaticResult,
                      x_min: Optional[float] = None, x_max: Optional[float] = None, save_fig: Optional[str] = None, 
                      e: Optional[np.ndarray] = None, 
                      data: Optional[np.ndarray] = None,
                      eps: Optional[np.ndarray] = None):
      '''
      plot static fitting Result

      Args:
       result: static fitting result
       name_of_dset: name of each dataset
       x_min: minimum x range
       x_max: maximum x range
       save_fig: prefix of saved png plots. If `save_fig` is `None`, plots are displayed istead of being saved.
       e: scan range of data
       data: static spectrum (it should not contain energy scan range)
       eps: estimated errors of static spectrum
      '''
      
      fig = plt.figure(0)
      title = 'Static Spectrum'
      subtitle = f"Chi squared: {result['red_chi2']: .2f}"
      plt.suptitle(title)
      sub1 = fig.add_subplot(211)
      sub1.set_title(subtitle)
      sub1.errorbar(e, data, eps, marker='o', mfc='none', label=f'expt {title}', linestyle='none')
      sub1.plot(e, result['fit'], label=f'fit {title}')
      sub1.plot(e, result['base'], label=f"base [order {result['base_order']}]", linestyle='dashed')
      sub1.legend()
      sub2 = fig.add_subplot(212)
      sub2.errorbar(e, result['res'], eps, 
      marker='o', mfc='none', label=f'res {title}', linestyle='none')
      sub2.legend()
      
      if x_min is not None and x_max is not None:
            sub1.set_xlim(x_min, x_max)
            sub2.set_xlim(x_min, x_max)
      if save_fig is not None:
            plt.savefig(f'{save_fig}_Static.png')
      if save_fig is None:
            plt.show()
      return

def save_StaticResult(result: StaticResult, dirname: str, name_of_dset: Optional[Sequence[str]] = None,
                      e: Optional[Sequence[np.ndarray]] = None,
                      eps: Optional[Sequence[np.ndarray]] = None):
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
       `fit.txt`: fitting curve and baseline curve for static spectrum
       `res.txt`: residual (fit-data) curve for static spectrum

      Note:
       If `dirname` directory is not exists, it creates `dirname` directory.
      '''
      if not (Path.cwd()/dirname).exists():
            os.mkdir(dirname)
      
      with open(f'{dirname}/fit_summary.txt', 'w') as f:
            f.write(print_StaticResult(result))
      

      coeff_fmt = ['%.8e']
      fit_fmt = 3*['%.8e']
      fit_header_lst = ['energy', 'fit', 'base']
      res_save = np.vstack((e, result['res'], eps)).T
      fit_save = np.vstack((e, result['fit'], result['base'])).T
      np.savetxt(f'{dirname}/res.txt', res_save, fmt=['%.8e', '%.8e', '%.8e'], 
      header=f'energy \t res \t eps')
      fit_header = '\t'.join(fit_header_lst)
      coeff_header = 'static'

      np.savetxt(f'{dirname}/weight.txt', result['c'], fmt=coeff_fmt, header=coeff_header)
      np.savetxt(f'{dirname}/fit.txt', fit_save, fmt=fit_fmt, header=fit_header)
      
      return