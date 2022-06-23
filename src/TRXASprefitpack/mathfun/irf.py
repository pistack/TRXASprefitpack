'''
exp_conv_irf:
submodule for the mathematical functions for
irf (instrumental response function)

:copyright: 2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Union
import numpy as np

def gau_irf(t: Union[float, np.ndarray], fwhm: float) -> Union[float, np.ndarray]:

    '''
    Compute gaussian shape irf function

    Args:
      t: time
      fwhm: full width at half maximum of gaussian distribution

    Returns:
     normalized gaussian function.
    '''

    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    return np.exp(-t**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def cauchy_irf(t: Union[float, np.ndarray], fwhm: float) -> Union[float, np.ndarray]:

    '''
    Compute lorenzian shape irf function

    Args:
      t: time
      fwhm: full width at half maximum of cauchy distribution

    Returns:
     normalized lorenzian function.
    '''

    gamma = fwhm/2
    return gamma/np.pi/(t**2+gamma**2)

def pvoigt_irf(t: Union[float, np.ndarray], fwhm_G: float, fwhm_L: float, eta: float) -> Union[float, np.ndarray]:
    '''
    Compute pseudo voight shape irf function
    (i.e. linear combination of gaussian and lorenzian function)

    Args:
      t: time
      fwhm_G: full width at half maximum of gaussian part
      fwhm_L: full width at half maximum of lorenzian part
      eta: mixing parameter
    Returns:
     linear combination of gaussian and lorenzian function with mixing parameter eta.
    '''

    u = gau_irf(t, fwhm_G)
    v = cauchy_irf(t, fwhm_L)

    return u + eta*(v-u)
  
def calc_eta(fwhm_G: float, fwhm_L: float) -> float:
  '''
  Calculate eta of pseudo voigt profile with fwhm_G, fwhm_L based on 
  Journal of Applied Crystallography. 33 (6): 1311–1316.

  Args:
    fwhm_G: full width at half maximum of gaussian part
    fwhm_L: full width at half maximum of lorenzian part
  Returns:
   maxing parameter eta
  '''
  f = fwhm_G**5+2.69269*fwhm_G**4*fwhm_L + \
    2.42843*fwhm_G**3*fwhm_L**2 + \
    4.47163*fwhm_G**2*fwhm_L**3 + \
    0.07842*fwhm_G*fwhm_L**4 + \
    fwhm_L**5
  f = f**(1/5)
  x = fwhm_L/f
  eta = 1.36603*x-0.47719*x**2+0.11116*x**3
  return eta
