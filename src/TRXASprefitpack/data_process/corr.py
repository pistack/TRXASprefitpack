'''
corr:

submodule for correction of scale using one time delay scan

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''
import numpy as np
from scipy.interpolate import interp1d


def corr_a_method(e_ref_index, e, t,
                  escan_time, ref_tscan_energy, ref_time_zeros,
                  escan_data=None, escan_data_eps=None,
                  ref_tscan_data=None, ref_tscan_data_eps=None,
                  warn=False):

    '''
    corr_a_method:
    Corrects the scaling of escan using one time delay scan 

    :param int e_ref: index of reference escab used for ``A-method``
    :param numpy_1d_array e: array of energies in which we measured escan
    :param numpy_1d_array t: array of time delays in which we measured tscan
    :param numpy_1d_array escan_time: 
     array of time delays at which we measured escan
    :param float ref_tscan_energy: 
     reference energy for repairing scale of escan
    :param float ref_time_zeros:
     time zero for reference tscan
    :param numpy_nd_array escan_data: data for escan
     (Note. escan data does not contains energy range)
    :param numpy_nd_array escan_data_eps: error for escan data 
    :param numpy_1d_array tscan_data: data for reference tscan
    :param numpy_1d_array tscan_data_eps: error for reference tscan 

    :return: corrected_data
    :rtype: dict
    :return: corrected_data['escan'] : corrected data for escan
    :rtype: numpy_nd_array
    :return: corrected_data['escan_eps'] : corrected error for escan
    :rtype: numpy_nd_array
    '''

    # init scaled_data
    scaled_data = dict()

    # scaling escan
    scaled_data['escan'] = np.zeros(escan_data.shape)
    scaled_data['escan_eps'] = np.zeros(escan_data_eps.shape)

    ref_tscan_inv_SN = ref_tscan_data_eps/ref_tscan_data 
    ref_tscan_inv_SN = interp1d(t-ref_time_zeros, 
                                ref_tscan_inv_SN, kind='nearest')(escan_time)
    flu_ref = interp1d(t-ref_time_zeros, ref_tscan_data)(escan_time)
    flu_ref = flu_ref/flu_ref[e_ref_index]
    flu_e = interp1d(e, escan_data, axis=0)(ref_tscan_energy)
    flu_e = flu_e/flu_e[e_ref_index]

    #  Scaling factor obtained by tscan have N/S ambiguity
    for i in range(escan_data.shape[1]):
        cond = np.abs(flu_e[i]-flu_ref[i]) > \
               1.0*ref_tscan_inv_SN[i]*np.max([flu_ref[i], flu_e[i]])
        if cond:
            scaled_data['escan'][:, i] = flu_ref[i]/flu_e[i]*escan_data[:, i]
            scaled_data['escan_eps'][:, i] = \
                flu_ref[i]/flu_e[i]*escan_data_eps[:, i]
        else:
            scaled_data['escan'][:, i] = escan_data[:, i]
            scaled_data['escan_eps'][:, i] = escan_data_eps[:, i]

    return scaled_data
