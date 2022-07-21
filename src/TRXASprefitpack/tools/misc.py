# misc
# submodule for miscellaneous function of
# tools subpackage
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ..driver import StaticResult, TransientResult

def read_data(prefix: str, num_scan: int, num_data_pts: int, default_SN: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Read data from prefix_i.txt (1 <= i <= num_scan)

    Args:
     prefix: prefix of scan_file
     num_scan: the number of scan file to read
     num_data_pts: the number of data points per each scan file
     default_SN: Default Signal/Noise
    
    Return:
     Tuple of data of eps
    '''
    data = np.zeros((num_data_pts, num_scan))
    eps = np.zeros((num_data_pts, num_scan))

    for i in range(num_scan):
        A = np.genfromtxt(f'{prefix}_{i+1}.txt')
        data[:, i] = A[:, 1]
        if A.shape[1] == 2:
            eps[:, i] = np.max(np.abs(data[:, i]))*np.ones(num_data_pts)/default_SN
        else:
            eps[:, i] = A[:, 2]
    return data, eps

def calc_param_rate_eq(mat_str: np.ndarray, tau_obs: np.ndarray) -> np.ndarray:
      '''
      Deduce rate equation parameter from observed time constants 
       Args:
        mat_str: user supplied rate equation
        tau_obs: observed time constants (time constant of each decay component)
       Returns:
        tau_rate: time constants for rate equation parameter 
       Note:
        Every entry in the rate equation matrix should be
        '0', '1*ki', '-x.xxx*ki', 'x.xxx*ki' or '-(x.xxx*ki+y.yyy*kj+...)'
        Number of non zero diagonal elements and size of tau should be same
        Number of parameter used to define rate equation and size of tau should 
        be same.
      '''
      mat_str_diag = np.diag(mat_str)
      mat_str_diag_reduced = mat_str_diag[mat_str_diag != '0']
      Eq = np.zeros((tau_obs.size, tau_obs.size), dtype=float)

      # Deduce rate equation parameter k
      for i in range(mat_str_diag_reduced.size):
            tmp = mat_str_diag_reduced[i]
            tmp.strip('-').strip('(').strip(')')
            k_lst = tmp.split('+')
            for k in k_lst:
                  k_pair = k.split('*')
                  Eq[i, int(k_pair[1][1:])-1] = float(k_pair[0])
      
      k_rate = np.linalg.solve(Eq, 1/tau_obs)
      tau_rate = 1/k_rate

      return tau_rate

def parse_matrix(mat_str: np.ndarray, tau_rate: np.ndarray) -> np.ndarray:
    '''
    Parse user supplied rate equation matrix

    Args:
     mat_str: user supplied rate equation (lower triangular matrix)
     tau_rate: time constants for rate equation parameter (1/ki)
    
    Return:
     parsed rate equation matrix.
     the value of lifetime 
     parameters (1/ki) used to define rate equation matrix
    
    Note:
     Every entry in the rate equation matrix should be
     '0', '1*ki', '-x.xxx*ki', 'x.xxx*ki' or '-(x.xxx*ki+y.yyy*kj+...)'
     Number of non zero diagonal elements and size of tau should be same
     Number of parameter used to define rate equation and size of tau should 
     be same.
    '''

    L = np.zeros_like(mat_str, dtype=float)
    mask = (mat_str != '0')
    red_mat_str = mat_str[mask]
    red_L = np.zeros_like(red_mat_str, dtype=float)

    for i in range(red_mat_str.size):
        tmp = red_mat_str[i]
        if '-' in tmp:
            tmp = tmp.strip('-')
            tmp = tmp.strip('('); tmp = tmp.strip(')')
            k_lst = tmp.split('+')
            for k in k_lst:
                k_pair = k.split('*')
                red_L[i] = red_L[i] - float(k_pair[0])/tau_rate[int(k_pair[1][1:])-1]
        else:
            tmp_pair = tmp.split('*')
            red_L[i] = float(tmp_pair[0])/tau_rate[int(tmp_pair[1][1:])-1]
    
    L[mask] = red_L
    
    return L

def save_StaticResult_txt(result: StaticResult, dirname: str):
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

def plot_StaticResult(result: StaticResult,
                      x_min: Optional[float] = None, x_max: Optional[float] = None, save_fig: Optional[str] = None):
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
      sub1.errorbar(result['e'], result['data'], result['eps'], 
      marker='o', mfc='none', label=f'expt {title}', linestyle='none')
      sub1.plot(result['e'], result['fit'], label=f'fit {title}')
      for i in range(result['n_voigt']):
            sub1.plot(result['e'], result['fit_comp'][i, :], label=f'{i}th voigt component', linestyle='dashed')
      if result['edge'] is not None:
            sub1.plot(result['e'], result['fit_comp'][-1, :], label=f"{result['edge']} type edge", linestyle='dashed')
      if result['base_order'] is not None:
        sub1.plot(result['e'], result['base'], label=f"base [order {result['base_order']}]", linestyle='dashed')
      sub1.legend()
      sub2 = fig.add_subplot(212)
      sub2.errorbar(result['e'], result['res'], result['eps'], 
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

def save_TransientResult_txt(result: TransientResult, dirname: str):
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

def plot_TransientResult(result: TransientResult,
                      x_min: Optional[float] = None, x_max: Optional[float] = None, save_fig: Optional[str] = None):
      '''
      plot fitting Result

      Args:
       result: fitting result
       x_min: minimum x range
       x_max: maximum x range
       save_fig: prefix of saved png plots. If `save_fig` is `None`, plots are displayed istead of being saved.
      '''
      
      start = 0
      for i in range(len(result['t'])):
            for j in range(result['data'][i].shape[1]):
                  fig = plt.figure(start+j)
                  title = f'{result["name_of_dset"][i]} scan #{j+1}'
                  subtitle = f"Chi squared: {result['red_chi2_ind'][i][j]: .2f}"
                  plt.suptitle(title)
                  sub1 = fig.add_subplot(211)
                  sub1.set_title(subtitle)
                  sub1.errorbar(result['t'][i], result['data'][i][:, j], result['eps'][i][:, j], marker='o', mfc='none',
                  label=f'expt {title}', linestyle='none')
                  sub1.plot(result['t'][i], result['fit'][i][:, j], label=f'fit {title}')
                  sub1.legend()
                  sub2 = fig.add_subplot(212)
                  if result['model'] in ['decay', 'osc']:
                        sub2.errorbar(result['t'][i], result['res'][i][:, j], 
                        result['eps'][i][:, j], marker='o', mfc='none', label=f'res {title}', linestyle='none')
                  else:
                        sub2.errorbar(result['t'][i], result['data'][i][:, j]-result['fit_decay'][i][:, j], 
                        result['eps'][i][:, j], marker='o', mfc='none', label=f'expt osc {title}', linestyle='none')
                        sub2.plot(result['t'][i], result['fit_osc'][i][:, j], label=f'fit osc {title}')
                  sub2.legend()
                  if x_min is not None and x_max is not None:
                        sub1.set_xlim(x_min, x_max)
                        sub2.set_xlim(x_min, x_max)
                  if save_fig is not None:
                        plt.savefig(f'{save_fig}_{result["name_of_dset"][i]}_{j+1}.png')
            start = start + result['data'][i].shape[1]
      if save_fig is None:
            plt.show()
      return