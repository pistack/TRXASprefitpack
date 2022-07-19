# misc
# submodule for miscellaneous function of
# tools subpackage

from typing import Optional, Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
from ..driver import StaticResult, DriverResult

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

def parse_matrix(mat_str: np.ndarray, tau: np.ndarray) -> np.ndarray:
    '''
    Parse user supplied rate equation matrix

    Args:
     mat_str: user supplied rate equation (lower triangular matrix)
     tau: life time constants (inverse of rate constant)
    
    Return:
     parsed rate equation matrix.
    
    Note:
     Every entry in the rate equation matrix should be
     '0', '1*ki', '-x.xxx*ki', 'x.xxx*ki' or '-(x.xxx*ki+y.yyy*kj+...)'
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
                red_L[i] = red_L[i] - float(k_pair[0])/tau[int(k_pair[1][1:])-1]
        else:
            tmp_pair = tmp.split('*')
            red_L[i] = float(tmp_pair[0])/tau[int(tmp_pair[1][1:])-1]
    
    L[mask] = red_L
    
    return L

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
      for i in range(result['n_voigt']):
            sub1.plot(e, result['fit_comp'][i, :], label=f'{i}th voigt component', linestyle='dashed')
      if result['edge'] is not None:
            sub1.plot(e, result['fit_comp'][-1, :], label=f"{result['edge']} type edge", linestyle='dashed')
      if result['base_order'] is not None:
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