'''
rate_eq:
submodule which solves 1st order rate equation and computes
the solution and signal 

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

import scipy.linalg as LA
from .A_matrix import make_A_matrix, make_A_matrix_cauchy
from .A_matrix import make_A_matrix_gau, make_A_matrix_pvoigt


def solve_model(equation, y0):

    '''
    Solve system of first order rate equation

    
    :param equation: matrix corresponding to model
    :type equation: numpy_nd_array
    :param y0: initial condition
    :type y0: numpy_1d_array

    
    :return: eigenvalue of equation
    :rtype: numpy_1d_array
    :return: eigenvectors for equation
    :rtype: numpy_nd_array
    :return: coefficient where y_0 = Vc
    :rtype: numpy_1d_array
    '''

    eigval, V = LA.eig(equation)
    c = LA.solve(V, y0)

    return eigval.real, V, c


def compute_model(t, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    Note: eigval, V, c should be obtained from solve_model

    
    :param t: time
    :type t: numpy_1d_array
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array

    
    :return: solution of rate equation
    :rtype: numpy_nd_array
    '''

    A = make_A_matrix(t, -eigval)
    y = (c * V) @ A
    return y


def compute_signal_gau(t, fwhm, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized gaussian distribution
    Note: eigval, V, c should be obtained from solve_model

    
    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array
  

    :return: 
     solution of rate equation convolved with normalized gaussian distribution
    :rtype: numpy_nd_array 
    '''

    A = make_A_matrix_gau(t, fwhm, -eigval)
    y_signal = (c * V) @ A
    return y_signal


def compute_signal_cauchy(t, fwhm, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized cauchy distribution
    Note: eigval, V, c should be obtained from solve_model

    
    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array
  

    :return: 
     solution of rate equation convolved with normalized cachy distribution
    :rtype: numpy_nd_array 
    '''

    A = make_A_matrix_cauchy(t, fwhm, -eigval)
    y_signal = (c * V) @ A
    return y_signal


def compute_signal_pvoigt(t, fwhm_G, fwhm_L, eta, eigval, V, c):

    '''
    Compute solution of the system of rate equations solved by solve_model
    convolved with normalized pseudo voigt profile
    (:math:`pvoigt = (1-\\eta) G(t) + \\eta L(t)`,
    G(t): stands for normalized gaussian
    L(t): stands for normalized cauchy(lorenzian) distribution)

    Note: eigval, V, c should be obtained from solve_model

    :param t: time
    :type t: numpy_1d_array
    :param fwhm_G: 
     full width at half maximum of x-ray temporal pulse (gaussian part)
    :type fwhm_G: float
    :param fwhm_L:
     full width at half maximum of x-ray temporal pulse (lorenzian part)
    :type fwhm_L: float
    :param float eta: mixing parameter :math:`(0 < \\eta < 1)`

    :type eta: float
    :param eigval: eigenvalue for equation
    :type eigval: numpy_1d_array
    :param V: eigenvectors for equation
    :type V: numpy_nd_array
    :param c: coefficient
    :type c: numpy_1d_array

    :return: 
     solution of rate equation convolved with normalized pseudo voigt profile
    :rtype: numpy_nd_array
    '''

    A = make_A_matrix_pvoigt(t, fwhm_G, fwhm_L, eta, -eigval)
    y_signal = (c * V) @ A
    return y_signal

