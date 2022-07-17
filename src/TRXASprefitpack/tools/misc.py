# misc
# submodule for miscellaneous function of
# tools subpackage

from typing import Tuple, Union, Optional
import numpy as np
import matplotlib.pyplot as plt

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

def plot_result(scan_name: str, num_scan: int, chi2_ind: Union[list, np.ndarray],
data: np.ndarray, eps: np.ndarray, fit: np.ndarray, res: Optional[np.ndarray] = None):
    '''
    Plot fitting result

    Args:
     scan_name: name of scan
     num_scan: the number of scans
     chi2_ind: list or array of the chi squared value of each indivisual scan
     data: exprimental data
     eps: error or data quality of experimental data
     fit: fitting result
     res: residual of fitting (data-fit)

    Note:
     1. the first column of fit array should be time range
     2. data array should not contain time range
    '''
    t = fit[:, 0]
    if res is not None:
        for i in range(num_scan):
            fig = plt.figure(i+1)
            title = f'Chi squared: {chi2_ind[i]:.2f}'
            plt.suptitle(title)
            sub1 = fig.add_subplot(211)
            sub1.errorbar(t, data[:, i],
            eps[:, i], marker='o', mfc='none',
            label=f'expt {scan_name} {i+1}',
            linestyle='none')
            sub1.plot(t, fit[:, i+1],
            label=f'fit {scan_name} {i+1}')
            sub1.legend()
            sub2 = fig.add_subplot(212)
            sub2.errorbar(t, res[:, i+1],
            eps[:, i], marker='o', mfc='none',
            label=f'{scan_name} res {i+1}',
            linestyle='none')
            sub2.legend()

    else:
        for i in range(num_scan):
            plt.figure(i+1)
            title = f'Chi squared: {chi2_ind[i]:.2f}'
            plt.title(title)
            plt.errorbar(t, data[:, i],
            eps[:, i], marker='o', mfc='none',
            label=f'expt {scan_name} {i+1}',
            linestyle='none')
            plt.plot(t, fit[:, i+1],
            label=f'fit {scan_name} {i+1}')
            plt.legend()
    plt.show()
    return

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