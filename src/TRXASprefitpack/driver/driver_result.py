from typing import Optional, Sequence
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class DriverResult(dict):
      '''
      Represent results for fitting driver routine

      Attributes:
       model ({'decay', 'dmp_osc', 'both'}): model used for fitting

              `decay`: sum of the convolution of exponential decay and instrumental response function

              `dmp_osc`: sum of the convolution of damped oscillation and instrumental response function

              `both`: sum of `decay` and `dmp_osc` model

       fit (sequence of np.ndarray): fitting curve for each data set
       res (sequence of np.ndarray): residual curve (data-fit) for each data set
       irf ({'g', 'c', 'pv'}): shape of instrument response function

            'g': gaussian instrumental response function

            'c': cauchy (lorenzian) instrumental response function

            'pv': pseudo voigt instrumental response function (linear combination of gaussian and lorenzian function)
       eta (float): mixing parameter ((1-eta)*g+eta*c)
       param_name (np.ndarray): name of parameter
       n_decay (int): number of decay components (including baseline feature)
       n_osc (int): number of damped oscillation components
       x (np.ndarray): best parameter
       bounds (sequence of tuple): boundary of each parameter
       base (bool): whether or not use baseline feature in fitting process
       c (sequence of np.ndarray): best weight of each component of each datasets 
       chi2 (float): total chi squared value of fitting
       aic (float): Akaike Information Criterion statistic: :math:`N\\log(\\chi^2/N)+2N_{parm}`
       bic (float): Bayesian Information Criterion statistic: :math:`N\\log(\\chi^2/N)+N_{parm}\log(N)`
       chi2_ind (np.ndarray): chi squared value of individual time delay scan
       red_chi2 (float): total reduced chi squared value of fitting
       red_chi2_ind (np.ndarray): reduced chi squared value of individul time delay scan
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
       method_glb ({'ampgo', 'basinhopping', 'dual_annealing'}): 
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

def print_DriverResult(result: DriverResult, name_of_dset: Optional[Sequence[str]] = None, corr_tol: float =1e-1) -> str:
      '''
      Pretty docstring of DriverResult class

      Args:
       result: DriverResult class instance which has fitting result
       name_of_dset: name of each data sets
       corr_tol: parameter correlation greather `corr_tol` would be reported
      
      Returns:
       string which reports fitting results
      '''

      if name_of_dset is None:
            name_of_dset = list(range(1, 1+len(result['red_chi2_ind'])))

      doc_lst = []
      doc_lst.append("[Model information]")
      doc_lst.append(f"    model : {result['model']}")
      doc_lst.append(f"    irf: {result['irf']}")
      doc_lst.append(f"    eta: {result['eta']}")
      doc_lst.append(f"    base: {result['base']}")
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
      for i in range(len(result['c'])):
            doc_lst.append(f"    DataSet {name_of_dset[i]}:")
            row = ['     #tscan']
            coeff_abs = np.abs(result['c'][i])
            coeff_sum = np.sum(coeff_abs, axis=0)
            coeff_contrib = np.einsum('j,ij->ij', 100/coeff_sum, result['c'][i])
            for j in range(coeff_contrib.shape[1]):
                  row.append(f'tscan_{j+1}')
            doc_lst.append('\t'.join(row))
            for d in range(result['n_decay']-1):
                  row = [f"     decay {d+1}"]
                  for l in range(coeff_contrib.shape[1]):
                        row.append(f'{coeff_contrib[d, l]: .2f}%')
                  doc_lst.append('\t'.join(row))
            if result['base']:
                  row = [f'     base']
            else:
                  row = [f"     decay {result['n_decay']}"]
            if result['n_decay'] > 0:
                  for l in range(coeff_contrib.shape[1]):
                        row.append(f"{coeff_contrib[result['n_decay']-1,l]: .2f}%")
                  doc_lst.append('\t'.join(row))
            for o in range(result['n_decay'], result['n_decay']+result['n_osc']):
                  row =[f"    dmp_osc {o+1-result['n_decay']}"]
                  for l in range(coeff_contrib.shape[1]):
                        row.append(f'{coeff_contrib[o, l]: .2f}%')
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

def plot_DriverResult(result: DriverResult, name_of_dset: Optional[Sequence[str]] = None,
                      x_min: Optional[float] = None, x_max: Optional[float] = None, save_fig: Optional[str] = None, 
                      t: Optional[Sequence[np.ndarray]] = None, 
                      data: Optional[Sequence[np.ndarray]] = None,
                      eps: Optional[Sequence[np.ndarray]] = None):
      '''
      plot fitting Result

      Args:
       result: fitting result
       name_of_dset: name of each dataset
       x_min: minimum x range
       x_max: maximum x range
       save_fig: prefix of saved png plots. If `save_fig` is `None`, plots are displayed istead of being saved.
       t: sequence of scan range of each dataset
       data: sequence of datasets for time delay scan (it should not contain time scan range)
       eps: sequence of estimated errors of each dataset
      '''
      if name_of_dset is None:
            name_of_dset = list(range(1, len(t)+1))
      
      start = 0
      for i in range(len(t)):
            for j in range(data[i].shape[1]):
                  fig = plt.figure(start+j)
                  title = f'{name_of_dset[i]} scan #{j+1}'
                  subtitle = f"Chi squared: {result['red_chi2_ind'][i][j]: .2f}"
                  plt.suptitle(title)
                  sub1 = fig.add_subplot(211)
                  sub1.set_title(subtitle)
                  sub1.errorbar(t[i], data[i][:, j], eps[i][:, j], marker='o', mfc='none',
                  label=f'expt {title}', linestyle='none')
                  sub1.plot(t[i], result['fit'][i][:, j], label=f'fit {title}')
                  sub1.legend()
                  sub2 = fig.add_subplot(212)
                  sub2.errorbar(t[i], result['res'][i][:, j], 
                  eps[i][:, j], marker='o', mfc='none', label=f'res {title}', linestyle='none')
                  sub2.legend()
                  if x_min is not None and x_max is not None:
                        sub1.set_xlim(x_min, x_max)
                        sub2.set_xlim(x_min, x_max)
                  if save_fig is not None:
                        plt.savefig(f'{save_fig}_{name_of_dset[i]}_{j+1}.png')
      if save_fig is None:
            plt.show()
      return

def save_DriverResult(result: DriverResult, dirname: str, name_of_dset: Optional[Sequence[str]] = None,
                      t: Optional[Sequence[np.ndarray]] = None,
                      eps: Optional[Sequence[np.ndarray]] = None):
      '''
      save fitting result to the text file

      Args:
       result: fitting result
       dirname: name of the directory in which text files for fitting result are saved.
       name_of_dset: name of each data sets. If `name_of_dset` is None then it is set to [1,2,3,....]
       t: sequence of scan range
       eps: sequence of estimated error of each datasets
      
      Returns:
       `fit_summary.txt`: Summary for the fitting result
       `weight_{name_of_dset[i]}.txt`: Weight of each model component of i th dataset
       `fit_{name_of_dset[i]}.txt`: fitting curve for i th dataset
       `res_{name_f_dset[i]}_j.txt`: residual (fit-data) curve for j th scan of i th data
                                     The format of text file is (t, res, eps)

      
      Note:
       If `dirname` directory is not exists, it creates `dirname` directory.
      '''
      if not (Path.cwd()/dirname).exists():
            os.mkdir(dirname)

      if name_of_dset is None:
            name_of_dset = list(range(1, 1+len(t)))
      
      with open(f'{dirname}/fit_summary.txt', 'w') as f:
            f.write(print_DriverResult(result, name_of_dset))
      
      for i in range(len(t)):
            coeff_fmt = eps[i].shape[1]*['%.8e']
            fit_fmt = (1+eps[i].shape[1])*['%.8e']
            coeff_header_lst = []
            fit_header_lst = ['time_delay']
            for j in range(eps[i].shape[1]):
                  res_save = np.vstack((t[i], result['res'][i][:,j], eps[i][:,j])).T
                  np.savetxt(f'{dirname}/res_{name_of_dset[i]}_{j+1}.txt', res_save,
                  fmt=['%.8e', '%.8e', '%.8e'], 
                  header=f'time_delay \t res_{name_of_dset[i]}_{j+1} \t eps')
                  fit_header_lst.append(f'fit_{name_of_dset[i]}_{j+1}')
                  coeff_header_lst.append(f'tscan_{name_of_dset[i]}_{j+1}')
            
            fit_header = '\t'.join(fit_header_lst)
            coeff_header = '\t'.join(coeff_header_lst)

            np.savetxt(f'{dirname}/weight_{name_of_dset[i]}.txt', result['c'][i], fmt=coeff_fmt,
            header=coeff_header)
            fit_save = np.vstack((t[i], result['fit'][i].T)).T
            np.savetxt(f'{dirname}/fit_{name_of_dset[i]}.txt', fit_save, fmt=fit_fmt, header=fit_header)
      
      return