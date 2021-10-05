'''
scale:

submodule for data scaling

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Optional, Dict
import numpy as np
from scipy.interpolate import interp1d


def automate_scaling(A: np.ndarray, e_ref_index: int,
                     e: np.ndarray, t: np.ndarray,
                     escan_time: np.ndarray,
                     tscan_energy: np.ndarray,
                     time_zeros: np.ndarray,
                     escan_data: np.ndarray,
                     escan_data_eps: np.ndarray,
                     tscan_data: np.ndarray,
                     tscan_data_eps: np.ndarray,
                     warn: Optional[bool] = False
                     ) -> Dict[str, np.ndarray]:

    '''
    automate_scaling:
    Automate scale escan, tscan

    Args:
        A: array of parameter A for each escan
        e_ref: index of reference for which scaling escan and tscan
        e: array of eneriges in which we measured escan
        t: array of time delays in which we measured tscan
        escan_time: array of time delays at which
                    we measured escan
        tscan_energy: array of energies at which
                      we measured tscan
        time_zeros: array of time zero for
                    each tscan
        escan_data: data for escan
        escan_data_eps: standard error of
                        escan data
        tscan_data: data for tscan
        tscan_data_eps: standard error of
                        tscan_data
        warn: whether or not print warning message

    Returns:
        A dictionary of numpy.ndarray for
        scaled escan and tscan data

        ::

            {'escan' : scaled data of escan,
             'escan_eps' : scaled standard error of escan data,
             'tscan' : scaled data of tscan,
             'tscan_eps' : scaled standard error of tscan data
            }

    Note:
       1. escan_data should not contain energy range.
       2. tscan_data should not contain time delay range.
       3. If you set warn=True and escan_time[e_ref_index] < 10 ps then
          warning message will be printed.
    '''

    # init scaled_data
    scaled_data = dict()

    # scaling escan
    scaled_data['escan'] = np.zeros(escan_data.shape)
    scaled_data['escan_eps'] = np.zeros(escan_data_eps.shape)

    A_ref = A[e_ref_index]
    e_ref = escan_time[e_ref_index]

    cond1 = (e_ref < 10)

    for i in range(escan_data.shape[1]):
        scaled_data['escan'][:, i] = A_ref/A[i]*escan_data[:, i]
        scaled_data['escan_eps'][:, i] = A_ref/A[i]*escan_data_eps[:, i]

    # function for normal scaling procedure
    def do_procedure_normal():

        scaled_data['tscan'] = np.zeros(tscan_data.shape)
        scaled_data['tscan_eps'] = np.zeros(tscan_data_eps.shape)

        flu_ref = interp1d(e,
                           scaled_data['escan'][:, e_ref_index])(tscan_energy)

        for i in range(tscan_energy.shape[0]):

            flu_t = interp1d(t-time_zeros[i], tscan_data[:, i])(e_ref)

            scaled_data['tscan'][:, i] = flu_ref[i]/flu_t*tscan_data[:, i]
            scaled_data['tscan_eps'][:, i] = \
                flu_ref[i]/flu_t*tscan_data_eps[:, i]

        return

    def print_warn(sit):

        if not warn:
            return

        print('Warning !\n')

        if sit == 'fast_delay':
            print(f'Your e_ref: {e_ref} < 10 ps !\n')
            print('In such fast delay, scaling will depend on time zero.\n')
            print('Please watch change of time zeros during fitting !\n')

        return

    if cond1:
        do_procedure_normal()

        print_warn('fast_delay')

    else:
        do_procedure_normal()

    return scaled_data
