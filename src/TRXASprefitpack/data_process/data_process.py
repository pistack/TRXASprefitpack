# data_process.py
# Author: Pistack
# Date: 2021. 08. 29.
# Email: pistatex@yonsei.ac.kr
# Part of XAS-pre-fit-pack package
# Useful function for doing data processing

import numpy as np
from scipy.interpolate import interp1d


def automate_scaling(A, e_ref_index, e, t,
                     escan_time, tscan_energy, time_zeros=None,
                     escan_data=None, escan_data_eps=None,
                     tscan_data=None, tscan_data_eps=None,
                     warn=False):

    '''
    automate_scaling:
    Automate scale escan, tscan

    
    :param numpy_1d_array A: array of parameter A for each escan
    :param int e_ref: index for reference escan for scaling of escan and tscan
    :param numpy_1d_array e: array of energies in which we measured escan
    :param numpy_1d_array t: array of time delays in which we measured tscan 
    :param numpy_1d_array escan_time: 
     array of time delay at which we measure escan
    :param numpy_1d_array tscan_energy:
     array of energy at which we measure tscan
    :param numpy_1d_array time_zeros:
     array of time zero for every tscan
     (optional, mandatory escan_time[e_ref] < 10 ps)
    :param numpy_nd_array escan_data: data for escan
     (Note. escan data does not contains energy range)
    :param numpy_nd_array escan_data_eps: error for escan data
    :param numpy_nd_array tscan_data: data for tscan
     (Note. tscan data does not contains time delay range)
    :param numpy_nd_array tscan_data_eps: error for tscan
    :param bool warn: whether or not prints warning message [default: False]


    
    :return: scaled_data
     scaled_data['escan'] : scaled data for escan
     scaled_data['escan_eps'] : scaled error for escan
     scaled_data['tscan'] : scaled data for tscan
     scaled_data['tscan_eps'] : scaled error for tscan
    :rtype: dict
    '''

    # init scaled_data
    scaled_data = dict()

    # scaling escan
    scaled_data['escan'] = np.zeros(escan_data.shape)
    scaled_data['escan_eps'] = np.zeros(escan_data_eps.shape)

    A_ref = A[e_ref_index]
    e_ref = escan_time[e_ref_index]

    cond1 = (e_ref < 10)
    
    cond2 = (time_zeros is not None)

    for i in range(escan_data.shape[1]):
        scaled_data['escan'][:, i] = A_ref/A[i]*escan_data[:, i]
        scaled_data['escan_eps'][:, i] = A_ref/A[i]*escan_data_eps[:, i]

    # function for normal scaling procedure
    def do_procedure_normal(time_zeros=None):

        if time_zeros is None:
            time_zeros = np.zeros(tscan_energy.shape[0])

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

    def print_err(sit):

        print('Error !\n')

        if sit == 'fast delay without time zero':
            print(f'Your e_ref: {e_ref} < 10 ps !\n')
            print('In such fast delay, scaling will depend on time zero.\n')
            print('However you did not set timezero\n')
            print('Please set timezero first before proceed\n')

        print('Aborting !\n')

        return

    if cond1 and cond2:
        
        do_procedure_normal(time_zeros=time_zeros)

        print_warn('fast_delay')

    elif cond1 and (not cond2):

        print_err('fast delay without time zero')

    else:
        do_procedure_normal(time_zeros=time_zeros)

    return scaled_data


def corr_a_method(e_ref_index, e, t,
                  escan_time, ref_tscan_energy, ref_time_zeros,
                  escan_data=None, escan_data_eps=None,
                  ref_tscan_data=None, ref_tscan_data_eps=None,
                  warn=False):

    '''
    corr_a_method:
    Corrects the scaling of escan scaled with tscan 

    
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


    
    :return: corrected_data:
     corrected_data['escan'] : corrected data for escan
     corrected_data['escan_eps'] : corrected error for escan
    :rtype: dict
    '''

    # init scaled_data
    scaled_data = dict()

    # scaling escan
    scaled_data['escan'] = np.zeros(escan_data.shape)
    scaled_data['escan_eps'] = np.zeros(escan_data_eps.shape)

    ref_tscan_inv_SN = ref_tscan_data_eps/ref_tscan_data 
    ref_tscan_inv_SN = interp1d(t-ref_time_zeros, 
                                ref_tscan_data, kind='nearest')(escan_time)
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
