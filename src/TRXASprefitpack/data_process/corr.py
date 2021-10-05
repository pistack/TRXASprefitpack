'''
corr:

submodule for correction of scale using one time delay scan

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Dict
import numpy as np
from scipy.interpolate import interp1d


def corr_a_method(e_ref_index: int,
                  e: np.ndarray,
                  t: np.ndarray,
                  escan_time: np.ndarray,
                  ref_tscan_energy: float,
                  ref_time_zeros: float,
                  escan_data: np.ndarray,
                  escan_data_eps: np.ndarray,
                  ref_tscan_data: np.ndarray,
                  ref_tscan_data_eps: np.ndarray
                  ) -> Dict[str, np.ndarray]:

    '''
    corr_a_method:
    Corrects the scaling of escan using one time delay scan

    Args:
      e_ref: index of reference escab used for ``A-method``
      e: array of energies in which we measured escan
      t: array of time delays in which we measured tscan
      escan_time: array of time delays at which we measured escan
      ref_tscan_energy: reference energy for repairing scale of escan
      ref_time_zeros: time zero for reference tscan
      escan_data: data for escan
      escan_data_eps: standard error of escan data
      tscan_data: data for reference tscan
      tscan_data_eps: standard error of reference tscan

    Returns:
      A dictionary of numpy.ndarray for scale corrected escan data

      ::

          {'escan': scale corrected data for escan,
           'escan_eps': scale corrected standard error of escan
          }

    Note:
       1. escan_data should not contain energy range
       2. tscan_data should not contain time delay range
    '''

    # init scaled_data
    scaled_data = dict()

    # scaling escan
    scaled_data['escan'] = np.zeros(escan_data.shape)
    scaled_data['escan_eps'] = np.zeros(escan_data_eps.shape)

    ref_tscan_inv_sn = ref_tscan_data_eps/ref_tscan_data
    ref_tscan_inv_sn = interp1d(t-ref_time_zeros,
                                ref_tscan_inv_sn, kind='nearest')(escan_time)
    flu_ref = interp1d(t-ref_time_zeros, ref_tscan_data)(escan_time)
    flu_ref = flu_ref/flu_ref[e_ref_index]
    flu_e = interp1d(e, escan_data, axis=0)(ref_tscan_energy)
    flu_e = flu_e/flu_e[e_ref_index]

    #  Scaling factor obtained by tscan have N/S ambiguity
    for i in range(escan_data.shape[1]):
        cond = np.abs(flu_e[i]-flu_ref[i]) > \
               1.0*ref_tscan_inv_sn[i]*np.max([flu_ref[i], flu_e[i]])
        if cond:
            scaled_data['escan'][:, i] = flu_ref[i]/flu_e[i]*escan_data[:, i]
            scaled_data['escan_eps'][:, i] = \
                flu_ref[i]/flu_e[i]*escan_data_eps[:, i]
        else:
            scaled_data['escan'][:, i] = escan_data[:, i]
            scaled_data['escan_eps'][:, i] = escan_data_eps[:, i]

    return scaled_data
