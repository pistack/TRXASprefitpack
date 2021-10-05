'''
broad:
submodule for broading theoritical spectrum

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3
'''

from typing import Optional
import numpy as np
from scipy.special import voigt_profile


def gen_theory_data(e: np.ndarray,
                    peaks: np.ndarray,
                    A: float,
                    fwhm_G: float,
                    fwhm_L: float,
                    peak_shift: float,
                    out: Optional[str] = None) -> np.ndarray:

    '''
    voigt broadening theoretically calculated lineshape spectrum

    Args:
        e: energy (unit: eV)
        A: scaling parameter
        fwhm_G: full width at half maximum of gaussian shape (unit: eV)
        fwhm_L: full width at half maximum of lorenzian shape (unit: eV)
        peak_shift: discrepency of peak position 
                    between expt data and theoretically broadened spectrum
        out: prefix for output txt file [optional]

    Returns:
      numpy ndarray of voigt broadened theoritical lineshape spectrum

    Note:
      If out is not none, It will makes
       1. out_thy.txt: txt file for rescaled and boroadend calc spectrum
       2. out_thy_stk.txt: txt file for rescaled and shifted calc peaks
    '''

    num_e = e.shape[0]
    num_peaks = peaks.shape[0]
    v_matrix = np.zeros((num_e, num_peaks))

    for i in range(num_peaks):
        v_matrix[:, i] = voigt_profile(e-(peaks[i, 0]-peak_shift),
                                       fwhm_G/(2*np.sqrt(2*np.log(2))),
                                       fwhm_L/2)

    broadened_theory = A * v_matrix @ peaks[:, 1].reshape((num_peaks, 1))
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
