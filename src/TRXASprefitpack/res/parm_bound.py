'''
param_bound:
submodule for setting default parameter boundary of
irf parameter, time zero and lifetime constant tau
:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''
from typing import Tuple, Union
import numpy as np


def set_bound_t0(t0: float, fwhm: Union[float, np.ndarray]) -> Tuple[float, float]:
    '''
    Setting bound for time zero

    Args:
     t0: initial guess for time zero
     fwhm: initial guess for full width at half maximum of instrumental response function
           `float` for gaussian and cauchy shape, `np.ndarray` with two element `(fwhm_G, fwhm_L)` for pseudo voigt shape.
    
    Returns:
     Tuple of upper and lower bound of time zero
    '''
    if not isinstance(fwhm, np.ndarray):
        bound = (t0-2*fwhm, t0+2*fwhm)
    else:
        fwhm_eff = 0.5346*fwhm[1] + \
                np.sqrt(0.2166*fwhm[1]**2+fwhm[0]**2)
        bound = (t0-2*fwhm_eff, t0+2*fwhm_eff)
    
    return bound


def set_bound_tau(tau: float, fwhm: Union[float, np.ndarray]) -> Tuple[float, float]:
    '''
    Setting bound for lifetime constant

    Args:
      tau: initial guess for lifetime constant
      fwhm: initial guess for full width at half maximum of instrumental response function
           `float` for gaussian and cauchy shape, `np.ndarray` with two element `(fwhm_G, fwhm_L)` for pseudo voigt shape.

    Returns:
     Tuple of upper bound and lower bound of tau
    '''
    if not isinstance(fwhm, np.ndarray):
        fwhm_eff = fwhm
    else:
        fwhm_eff = 0.5346*fwhm[1] + \
                np.sqrt(0.2166*fwhm[1]**2+fwhm[0]**2)

    if tau <= fwhm_eff:
        bound = (tau/2, 5*fwhm_eff)
    elif fwhm_eff < tau <= 5*fwhm_eff:
        bound = (fwhm_eff/2, 25*fwhm_eff)
    elif 5*fwhm_eff < tau <= 50*fwhm_eff:
        bound = (2.5*fwhm_eff, 80*fwhm_eff)
    elif 50*fwhm_eff < tau <= 160*fwhm_eff:
        bound = (25*fwhm_eff, 256*fwhm_eff)
    elif 160*fwhm_eff < tau <= 512*fwhm_eff:
        bound = (80*fwhm_eff, 1024*fwhm_eff)
    elif 512*fwhm_eff < tau <= 2048*fwhm_eff:
        bound = (256*fwhm_eff, 4096*fwhm_eff)
    elif 2048*fwhm_eff < tau <= 8192*fwhm_eff:
        bound = (1024*fwhm_eff, 16384*fwhm_eff)
    elif 8192*fwhm_eff < tau <= 32768*fwhm_eff:
        bound = (4096*fwhm_eff, 65536*fwhm_eff)
    else:
        bound = (16384*fwhm_eff, 2*tau)
    return bound