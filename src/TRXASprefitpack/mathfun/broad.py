'''
broad:
submodule for broading theoritical spectrum

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3
'''

from typing import Optional
import numpy as np
from .peak_shape import voigt


def gen_theory_data(e: np.ndarray,
                    peaks: np.ndarray,
                    A: float,
                    fwhm_G: float,
                    fwhm_L: float,
                    peak_factor: float,
                    policy: Optional[str] = 'shift') -> np.ndarray:

    '''
    voigt broadening theoretically calculated lineshape spectrum

    Args:
        e: energy 
        A: scaling parameter
        fwhm_G: full width at half maximum of gaussian shape (unit: same as energy)
        fwhm_L: full width at half maximum of lorenzian shape (unit: same as energy)
        peak_factor: Peak factor, its behavior depends on policy.
        policy: Policy to match discrepency between experimental data and theoretical
                spectrum.

                * 'shift' : Default option, shift peak position by peak_factor
                * 'scale' : scale peak position by peak_factor

    Returns:
      numpy ndarray of voigt broadened theoritical lineshape spectrum
    '''

    num_e = e.size
    num_peaks = peaks.shape[0]
    v_matrix = np.zeros((num_e, num_peaks))
    peak_copy = np.copy(peaks[:, 0])
    if policy == 'shift':
      peak_copy = peak_copy - peak_factor
    else:
      peak_copy = peak_factor*peak_copy 
    for i in range(num_peaks):
        v_matrix[:, i] = voigt(e-peak_copy[i], fwhm_G, fwhm_L)

    broadened_theory = A * v_matrix @ peaks[:, 1].reshape((num_peaks, 1))
    broadened_theory = broadened_theory.flatten()

    return broadened_theory
