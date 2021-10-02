'''
exp_conv_irf:
submodule for the mathematical functions for 
exponential decay convolved with irf

:copyright: 2021 by pistack (Junho Lee).
:license: LGPL3.
'''

import numpy as np
from scipy.special import erfc, exp1


def exp_conv_gau(t, fwhm, k):

    '''
    Compute exponential function convolved with normalized gaussian 
    distribution

    Note.
    We assume temporal pulse of x-ray is normalized gaussian distribution
    
    .. math::

       \sigma = \\frac{fwhm}{2\sqrt{2\log{2}}}


    .. math::

       IRF(t) = \\frac{1}{\\sigma \\sqrt{2\pi}}\\exp\\left(-\\frac{t^2}{2\\sigma^2}\\right)

    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param k: rate constant (inverse of life time)
    :type k: float

    
    :return: convolution of normalized gaussian distribution and exp(-kt)

    .. math::

       \\frac{1}{2}\exp\\left(\\frac{k^2}{2\sigma^2}-kt\\right){erfc}\\left(\\frac{1}{\sqrt{2}}\\left(k\sigma-\\frac{t}{\sigma}\\right)\\right)


    :rtype: numpy_1d_array
    '''

    sigma = fwhm/(2*np.sqrt(2*np.log(2)))

    return 1/2*np.exp((k*sigma)**2/2.0-k*t) * \
        erfc((k*sigma-t/sigma)/np.sqrt(2.0))


def exp_conv_cauchy(t, fwhm, k):

    '''
    Compute exponential function convolved with normalized cauchy 
    distribution

    Note.
    We assume temporal pulse of x-ray is normalized cauchy distribution

    .. math::
       \\gamma = \\frac{fwhm}{2}

    .. math::
       IRF(t) = \\frac{\\gamma}{\\pi}\\frac{1}{(x-t)^2+\\gamma^2}
    

    :param t: time
    :type t: numpy_1d_array
    :param fwhm: full width at half maximum of x-ray temporal pulse
    :type fwhm: float
    :param k: rate constant (inverse of life time)
    :type k: float

   
    :return: convolution of normalized cauchy distribution and exp(-kt)

    .. math::
       \\frac{\\exp(-kt)}{\\pi}\\Im\\left(\\exp(-ik\\gamma)\cdot{E1}(-kt-ik\\gamma)\\right)


    :rtype: numpy_1d_array 
    '''

    gamma = fwhm/2
    if k == 0:
        ans = 0.5+1/np.pi*np.arctan(t/gamma)
    else:
        ikgamma = complex(0, k*gamma)
        kt = k*t
        ans = np.exp(-ikgamma)*exp1(-kt-ikgamma)
        ans = np.exp(-kt)*ans.imag/np.pi
    return ans
