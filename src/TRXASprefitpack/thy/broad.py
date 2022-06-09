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
                    peak_shift: float) -> np.ndarray:

    '''
    voigt broadening theoretically calculated lineshape spectrum

    Args:
        e: energy 
        A: scaling parameter
        fwhm_G: full width at half maximum of gaussian shape (unit: same as energy)
        fwhm_L: full width at half maximum of lorenzian shape (unit: same as energy)
        peak_shift: discrepency of peak position 
                    between expt data and theoretically broadened spectrum

    Returns:
      numpy ndarray of voigt broadened theoritical lineshape spectrum
    '''

    sigma = fwhm_G/(2*np.sqrt(2*np.log(2)))
    gamma = fwhm_L/2

    num_e = e.shape[0]
    num_peaks = peaks.shape[0]
    v_matrix = np.zeros((num_e, num_peaks))

    for i in range(num_peaks):
        v_matrix[:, i] = voigt_profile(e-(peaks[i, 0]-peak_shift),
                                       sigma, gamma)

    broadened_theory = A * v_matrix @ peaks[:, 1].reshape((num_peaks, 1))
    broadened_theory = broadened_theory.flatten()

    return broadened_theory
