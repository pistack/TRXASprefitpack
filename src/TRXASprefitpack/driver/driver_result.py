import os
from pathlib import Path
import numpy as np

class DriverResult(dict):
      '''
      Represent results for fitting driver routine

      Attributes:
       model ({'decay', 'dmp_osc', 'both'}): model used for fitting

              `decay`: sum of the convolution of exponential decay and instrumental response function

              `dmp_osc`: sum of the convolution of damped oscillation and instrumental response function

              `both`: sum of `decay` and `dmp_osc` model
       name_of_dset (sequence of str): name of each dataset
       t (sequence of np.ndarray): time range for each dataset
       data (sequence of np.ndarray): sequence of datasets for time delay scan (it should not contain time scan range)
       eps (sequence of np.ndarray): sequence of datasets for estimated error of time delay scan
       fit (sequence of np.ndarray): fitting curve for each data set
       fit_decay (sequence of np.ndarray): decay part of fitting curve for each data set [model = 'both']
       fit_osc (sequence of np.ndarray): oscillation part of fitting curve for each data set [model = 'both']
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
            Report DriverResult class

            Args:
            result: DriverResult class instance which has fitting result
            corr_tol: parameter correlation greather `corr_tol` would be reported
      
            Returns:
             string which reports fitting results
            '''

            doc_lst = []
            doc_lst.append("[Model information]")
            doc_lst.append(f"    model : {self['model']}")
            doc_lst.append(f"    irf: {self['irf']}")
            doc_lst.append(f"    eta: {self['eta']}")
            doc_lst.append(f"    base: {self['base']}")
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
            doc_lst.append(f"    Total Data points: {self['num_pts']}")
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
            for i in range(len(self['c'])):
                  doc_lst.append(f"    DataSet {self['name_of_dset'][i]}:")
                  row = ['     #tscan']
                  coeff_abs = np.abs(self['c'][i])
                  coeff_sum = np.sum(coeff_abs, axis=0)
                  coeff_contrib = np.einsum('j,ij->ij', 100/coeff_sum, self['c'][i])
                  for j in range(coeff_contrib.shape[1]):
                      row.append(f'tscan_{j+1}')
                  doc_lst.append('\t'.join(row))
                  for d in range(self['n_decay']-1):
                      row = [f"     decay {d+1}"]
                      for l in range(coeff_contrib.shape[1]):
                        row.append(f'{coeff_contrib[d, l]: .2f}%')
                      doc_lst.append('\t'.join(row))
                  if self['base']:
                        row = [f'     base']
                  else:
                        row = [f"     decay {self['n_decay']}"]
                  if self['n_decay'] > 0:
                        for l in range(coeff_contrib.shape[1]):
                              row.append(f"{coeff_contrib[self['n_decay']-1,l]: .2f}%")
                        doc_lst.append('\t'.join(row))
                  for o in range(self['n_decay'], self['n_decay']+self['n_osc']):
                        row =[f"    dmp_osc {o+1-self['n_decay']}"]
                        for l in range(coeff_contrib.shape[1]):
                              row.append(f'{coeff_contrib[o, l]: .2f}%')
                        doc_lst.append('\t'.join(row))
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
            Print structure of DriverResult instance
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

def save_DriverResult(result: DriverResult, dirname: str):
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
       `fit_osc_{name_of_dset[i]}.txt`: oscillation part of fitting curve for i th dataset [model = 'both']
       `fit_decay_{name_of_dset[i]}.txt`: decay part of fitting curve for i th dataset [model = 'both']
       `res_{name_f_dset[i]}_j.txt`: residual (fit-data) curve for j th scan of i th data
                                     The format of text file is (t, res, eps)

      
      Note:
       If `dirname` directory is not exists, it creates `dirname` directory.
      '''
      if not (Path.cwd()/dirname).exists():
            os.mkdir(dirname)
      
      with open(f'{dirname}/fit_summary.txt', 'w') as f:
            f.write(str(result))
      
      for i in range(len(result['t'])):
            coeff_fmt = result['eps'][i].shape[1]*['%.8e']
            fit_fmt = (1+result['eps'][i].shape[1])*['%.8e']
            coeff_header_lst = []
            fit_header_lst = ['time_delay']
            for j in range(result['eps'][i].shape[1]):
                  res_save = np.vstack((result['t'][i], result['res'][i][:,j], result['eps'][i][:,j])).T
                  np.savetxt(f"{dirname}/res_{result['name_of_dset'][i]}_{j+1}.txt", res_save,
                  fmt=['%.8e', '%.8e', '%.8e'], 
                  header=f"time_delay \t res_{result['name_of_dset'][i]}_{j+1} \t eps")
                  fit_header_lst.append(f"fit_{result['name_of_dset'][i]}_{j+1}")
                  coeff_header_lst.append(f"tscan_{result['name_of_dset'][i]}_{j+1}")
            
            fit_header = '\t'.join(fit_header_lst)
            coeff_header = '\t'.join(coeff_header_lst)

            np.savetxt(f"{dirname}/weight_{result['name_of_dset'][i]}.txt", result['c'][i], fmt=coeff_fmt,
            header=coeff_header)
            fit_save = np.vstack((result['t'][i], result['fit'][i].T)).T
            np.savetxt(f"{dirname}/fit_{result['name_of_dset'][i]}.txt", fit_save, fmt=fit_fmt, header=fit_header)
            if result['model'] == 'both':
                  fit_decay_save = np.vstack((result['t'][i], result['fit_decay'][i].T)).T
                  np.savetxt(f"{dirname}/fit_decay_{result['name_of_dset'][i]}.txt", fit_decay_save, fmt=fit_fmt, header=fit_header)
                  fit_osc_save = np.vstack((result['t'][i], result['fit_osc'][i].T)).T
                  np.savetxt(f"{dirname}/fit_osc_{result['name_of_dset'][i]}.txt", fit_osc_save, fmt=fit_fmt, header=fit_header)

      return