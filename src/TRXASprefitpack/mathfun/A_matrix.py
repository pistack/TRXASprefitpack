'''
A_matrix:
submodule for evaluation of A_matrix

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''
import numpy as np
from .exp_conv_irf import exp_conv_gau, exp_conv_cauchy


def make_A_matrix(t: np.ndarray, k: float) -> np.ndarray:

    A = np.zeros((k.shape[0], t.shape[0]))
    for i in range(k.shape[0]):
        A[i, :] = np.exp(-k[i]*t)
    A = A*np.heaviside(t, 1)
    A[np.isnan(A)] = 0

    return A


def make_A_matrix_gau(t: np.ndarray, fwhm: float,
                      k: float) -> np.ndarray:

    A = np.zeros((k.shape[0], t.shape[0]))
    for i in range(k.shape[0]):
        A[i, :] = exp_conv_gau(t, fwhm, k[i])

    return A


def make_A_matrix_cauchy(t: np.ndarray, fwhm: float,
                         k: float) -> np.ndarray:

    A = np.zeros((k.shape[0], t.shape[0]))
    for i in range(k.shape[0]):
        A[i, :] = exp_conv_cauchy(t, fwhm, k[i])

    return A


def make_A_matrix_pvoigt(t: np.ndarray,
                         fwhm_G: float,
                         fwhm_L: float,
                         eta: float,
                         k: float) -> np.ndarray:

    return (1-eta)*make_A_matrix_gau(t, fwhm_G, k) + \
        eta*make_A_matrix_cauchy(t, fwhm_L, k)
