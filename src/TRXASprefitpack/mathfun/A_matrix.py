'''
A_matrix:
submodule for evaluation of A_matrix

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
import numpy as np
from .exp_conv_irf import exp_conv_gau, exp_conv_cauchy
from .exp_conv_irf import dmp_osc_conv_gau, dmp_osc_conv_cauchy


def make_A_matrix(t: np.ndarray, k: np.ndarray) -> np.ndarray:

    A = np.zeros((k.size, t.size))
    for i in range(k.size):
        A[i, :] = np.exp(-k[i]*t)
    A = A*np.heaviside(t, 1)
    A[np.isnan(A)] = 0

    return A


def make_A_matrix_gau(t: np.ndarray, fwhm: float,
                      k: np.ndarray) -> np.ndarray:

    A = np.zeros((k.size, t.size))
    for i in range(k.size):
        A[i, :] = exp_conv_gau(t, fwhm, k[i])

    return A


def make_A_matrix_cauchy(t: np.ndarray, fwhm: float,
                         k: np.ndarray) -> np.ndarray:

    A = np.zeros((k.size, t.size))
    for i in range(k.size):
        A[i, :] = exp_conv_cauchy(t, fwhm, k[i])

    return A


def make_A_matrix_pvoigt(t: np.ndarray,
                         fwhm_G: float,
                         fwhm_L: float,
                         eta: float,
                         k: np.ndarray) -> np.ndarray:
    
    u = make_A_matrix_gau(t, fwhm_G, k)
    v = make_A_matrix_cauchy(t, fwhm_L, k)

    return u + eta*(v-u)

def make_A_matrix_gau_osc(t: np.ndarray, fwhm: float,
k: np.ndarray, T: np.ndarray, phase: np.ndarray) -> np.ndarray:

    A = np.zeros((k.size, t.size))
    for i in range(k.size):
        A[i, :] = dmp_osc_conv_gau(t, fwhm, k[i], T[i], phase[i])

    return A


def make_A_matrix_cauchy_osc(t: np.ndarray, fwhm: float,
k: np.ndarray, T: np.ndarray, phase: np.ndarray) -> np.ndarray:

    A = np.zeros((k.size, t.size))
    for i in range(k.size):
        A[i, :] = dmp_osc_conv_cauchy(t, fwhm, k[i], T[i], phase[i])

    return A


def make_A_matrix_pvoigt_osc(t: np.ndarray, fwhm_G: float, fwhm_L: float, eta: float,
k: np.ndarray, T: np.ndarray, phase: np.ndarray) -> np.ndarray:
    
    u = make_A_matrix_gau_osc(t, fwhm_G, k, T, phase)
    v = make_A_matrix_cauchy_osc(t, fwhm_L, k, T, phase)

    return u + eta*(v-u)
