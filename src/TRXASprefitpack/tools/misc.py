# misc
# submodule for miscellaneous function of
# tools subpackage

from typing import Tuple, Union, Optional
import numpy as np
import matplotlib.pyplot as plt

def set_bound_tau(tau: float):
    '''
    Setting bound for lifetime constant

    Args:
      tau: lifetime constant

    Returns:
     list of upper bound and lower bound of tau
    '''
    bound = [tau/2, 1]
    if 0.1 < tau <= 1:
        bound = [0.05, 10]
    elif 1 < tau <= 10:
        bound = [0.5, 50]
    elif 10 < tau <= 100:
        bound = [5, 500]
    elif 100 < tau <= 1000:
        bound = [50, 2000]
    elif 1000 < tau <= 5000:
        bound = [500, 10000]
    elif 5000 < tau <= 50000:
        bound = [2500, 100000]
    elif 50000 < tau <= 500000:
        bound = [25000, 1000000]
    elif 500000 < tau <= 1000000:
        bound = [250000, 2000000]
    elif 1000000 < tau:
        bound = [tau/4, 4*tau]
    return bound

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

def contribution_table(scan_name: str, table_title: str, num_scan: int, num_comp: int, coeff: np.ndarray) -> str:
    '''
    Draw contribution table (row: num_comp, col: num_scan)

    Args:
     scan_name: name of scan
     table_title: title of table
     num_scan: the number of scan
     num_comp: the number of component
     coeff: coefficient matrix
     
    Return:
     contribution table
    '''
    # calculate contribution
    coeff_abs = np.abs(coeff)
    coeff_sum = np.sum(coeff_abs, axis=0)
    coeff_contrib = np.zeros_like(coeff)
    for i in range(num_scan):
        coeff_contrib[:, i] = coeff[:, i]/coeff_sum[i]*100

    # draw table
    cont_table = '    '
    for i in range(num_scan):
        cont_table = cont_table + f'{scan_name} {i+1} |'
    cont_table = cont_table + '\n'
    for i in range(num_comp):
        cont_table = cont_table + '    '
        for j in range(num_scan):
            cont_table = cont_table + f'{coeff_contrib[i, j]:.2f} % |'
        cont_table = cont_table + '\n'
    
    cont_table = f'[[{table_title}]]' + '\n' + cont_table
    return cont_table

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
