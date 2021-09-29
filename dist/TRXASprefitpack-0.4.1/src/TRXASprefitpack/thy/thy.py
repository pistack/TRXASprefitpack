# thy.py
# Author: Pistack
# Date: 2021. 08. 29.
# Email: pistatex@yonsei.ac.kr
# Part of XAS-pre-fit-pack package
# Useful function for broeadening spectrum

import numpy as np
from scipy.special import voigt_profile 


def gen_theory_data(e, peaks, A, fwhm_G, fwhm_L, peak_shift, out=None):

    '''
    voigt broadening theoretically calculated lineshape spectrum
    if out is not none:
    It will make
    out_thy.txt: txt file for rescaled and boroadend calc spectrum
    out_thy_stk.txt: txt file for rescaled and shifted calc peaks


    :param numpy_1d_array e: energy (unit: eV)
    :param float A: scaling parameter
    :param float fwhm_G: 
     full width at half maximum of gaussian shape (unit: eV)
    :param float fwhm_L: 
     full width at half maximum of lorenzian shape (unit: eV)
    :param float peak_shift: 
     discrepency of peak position between expt data and theoretically 
     broadened spectrum
    :param string out: prefix for output txt file [optional]

    
    :return: voigt broadened calc spectrum
    :rtype: numpy_1d_array
    '''

    num_e = e.shape[0]
    num_peaks = peaks.shape[0]
    V_matrix = np.zeros((num_e, num_peaks))

    for i in range(num_peaks):
        V_matrix[:, i] = voigt_profile(e-(peaks[i, 0]-peak_shift),
                                       fwhm_G/(2*np.sqrt(2*np.log(2))),
                                       fwhm_L/2)

    broadened_theory = A * V_matrix @ peaks[:, 1].reshape((num_peaks, 1))
    broadened_theory = broadened_theory.flatten()

    if out is not None:
        save_thy = np.vstack((e, broadened_theory))
        np.savetxt(out+'_thy'+'.txt', save_thy.T, fmt=['%.2f', '%.8e'],
                   header='energy \t abs_thy', newline='\n', delimiter='\t')
        np.savetxt(out+'_stk_rescaled'+'.txt',
                   np.vstack((peaks[:, 0]-peak_shift, A*peaks[:, 1])).T,
                   fmt=['%.2f', '%.8e'], header='energy \t abs_thy',
                   newline='\n', delimiter='\t')

    return broadened_theory
