'''
rate_eq:
submodule which solves 1st order rate equation and computes
the solution and signal

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Tuple
import numpy as np
import scipy.linalg as LA  # replace numpy.linalg to scipy.linalg
from .A_matrix import make_A_matrix, make_A_matrix_cauchy
from .A_matrix import make_A_matrix_gau, make_A_matrix_pvoigt


def solve_model(equation: np.ndarray,
                y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''
    Solve system of first order rate equation

    Args:
      equation: matrix corresponding to model
      y0: initial condition

    Returns:
       1. eigenvalues of equation
       2. eigenvectors for equation
       3. coefficient where y0 = Vc
    '''

    eigval, V = LA.eig(equation)
    c = LA.solve(V, y0)

    return eigval.real, V, c


def compute_model(t: np.ndarray,
                  eigval: np.ndarray,
                  V: np.ndarray,
                  c: np.ndarray) -> np.ndarray:

    '''
    Compute solution of the system of rate equations solved by solve_model
    Note: eigval, V, c should be obtained from solve_model

    Args:
     t: time
     eigval: eigenvalue for equation
     V: eigenvectors for equation
     c: coefficient

    Returns:
      solution of rate equation

    Note:
      eigval, V, c should be obtained from solve_model.
    '''

    A = make_A_matrix(t, -eigval)
    y = (c * V) @ A
    return y


def compute_signal_gau(t: np.ndarray,
                       fwhm: float,
                       eigval: np.ndarray,
                       V: np.ndarray,
                       c: np.ndarray) -> np.ndarray:

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized gaussian distribution

    Args:
     t: time
     fwhm: full width at half maximum of normalized gaussian distribution
     eigval: eigenvalue for equation
     V: eigenvectors for equation
     c: coefficient

    Returns:
      Convolution of solution of rate equation and normalized gaussian
      distribution

    Note:
      eigval, V, c should be obtained from solve_model.
    '''

    A = make_A_matrix_gau(t, fwhm, -eigval)
    y_signal = (c * V) @ A
    return y_signal


def compute_signal_cauchy(t: np.ndarray,
                          fwhm: float,
                          eigval: np.ndarray,
                          V: np.ndarray,
                          c: np.ndarray) -> np.ndarray:

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized cauchy distribution

    Args:
     t: time
     fwhm: full width at half maximum of normalized cauchy distribution
     eigval: eigenvalue for equation
     V: eigenvectors for equation
     c: coefficient

    Returns:
      Convolution of solution of rate equation and normalized cauchy
      distribution

    Note:
      eigval, V, c should be obtained from solve_model.
    '''

    A = make_A_matrix_cauchy(t, fwhm, -eigval)
    y_signal = (c * V) @ A
    return y_signal


def compute_signal_pvoigt(t: np.ndarray,
                          fwhm_G: float,
                          fwhm_L: float,
                          eta: float,
                          eigval: np.ndarray,
                          V: np.ndarray,
                          c: np.ndarray) -> np.ndarray:

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized pseudo voigt profile

    .. math::

       \\mathrm{pvoigt}(t) = (1-\\eta) G(t) + \\eta L(t),

    G(t) stands for normalized gaussian,
    L(t) stands for normalized cauchy(lorenzian) distribution

    Args:
     t: time
     fwhm_G: full width at half maximum of gaussian part
     fwhm_L: full width at half maximum of cauchy part
     eta: mixing parameter
     eigval: eigenvalue for equation
     V: eigenvectors for equation
     c: coefficient

    Returns:
      Convolution of solution of rate equation and normalized pseudo
      voigt profile.

    Note:
      eigval, V, c should be obtained from solve_model.
    '''

    A = make_A_matrix_pvoigt(t, fwhm_G, fwhm_L, eta, -eigval)
    y_signal = (c * V) @ A
    return y_signal
