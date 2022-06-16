# misc
# submodule for miscellaneous function of
# tools subpackage

from typing import Tuple, Union
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
    if 0.1 < tau <= 10:
        bound = [0.05, 100]
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
data: np.ndarray, eps: np.ndarray, fit: np.ndarray):
    '''
    Plot fitting result
    Args:
     scan_name: name of scan
     num_scan: the number of scans
     chi2_ind: list or array of the chi squared value of each indivisual scan
     data: exprimental data
     eps: error or data quality of experimental data
     fit: fitting result
    Note:
     1. the first column of fit array should be time range
     2. data array should not contain time range
    '''
    t = fit[:, 0]
    for i in range(num_scan):
        plt.figure(i+1)
        title = f'Chi squared: {chi2_ind[i]:.2f}'
        plt.title(title)
        plt.errorbar(t, data[:, i],
                     eps[:, i], marker='o', mfc='none',
                     label=f'{scan_name} expt {i+1}',
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