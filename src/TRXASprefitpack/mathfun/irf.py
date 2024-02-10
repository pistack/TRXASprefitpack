'''
exp_conv_irf:
submodule for the mathematical functions for
irf (instrumental response function)

:copyright: 2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Union, Tuple
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


def calc_fwhm(fwhm_G: float, fwhm_L: float) -> float:
    '''
    Calculate uniform fwhm of both gaussian and cauchy component of
    pseudo voigt profile with fwhm_G, fwhm_L based on
    Journal of Applied Crystallography. 33 (6): 1311–1316.

    Args:
      fwhm_G: full width at half maximum of gaussian part
      fwhm_L: full width at half maximum of lorenzian part
    Returns:
     uniform fwhm parameter of both gaussian and cauchy component of
     pseudo voigt profile
    '''
    f = fwhm_G**5+2.69269*fwhm_G**4*fwhm_L + \
        2.42843*fwhm_G**3*fwhm_L**2 + \
        4.47163*fwhm_G**2*fwhm_L**3 + \
        0.07842*fwhm_G*fwhm_L**4 + \
        fwhm_L**5
    return f**(1/5)


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


def pvoigt_irf(t: Union[float, np.ndarray], fwhm: float, eta: float) -> Union[float, np.ndarray]:
    '''
    Compute pseudo voight shape irf function
    (i.e. linear combination of gaussian and lorenzian function)

    Args:
      t: time
      fwhm: uniform full width at half paramter
      eta: mixing paramter
    Returns:
     pseudo voigt profile
    '''
    u = gau_irf(t, fwhm)
    v = cauchy_irf(t, fwhm)

    return u + eta*(v-u)


def deriv_fwhm(fwhm_G: float, fwhm_L: float) -> Tuple[float, float]:
    '''
    Computes gradient of uniform fwhm parameter of pseudo voigt approximation
    based on Journal of Applied Crystallography. 33 (6): 1311-1316.

    Args:
     fwhm_G: full width at half maximum of gaussian part
     fwhm_L: full width at half maximum of lorenzian part

    Returns:
     gradient of fwhm(fwhm_G, fwhm_L)
    '''
    f = fwhm_G**5+2.69269*fwhm_G**4*fwhm_L + \
        2.42843*fwhm_G**3*fwhm_L**2 + \
        4.47163*fwhm_G**2*fwhm_L**3 + \
        0.07842*fwhm_G*fwhm_L**4 + \
        fwhm_L**5
    df_fwhm_G = 5*fwhm_G**4+10.77076*fwhm_G**3*fwhm_L + \
        7.28529*fwhm_G**2*fwhm_L**2+8.94326*fwhm_G*fwhm_L**3 + \
        0.07842*fwhm_L**4
    df_fwhm_L = 5*fwhm_L**4 + 0.31368*fwhm_L**3*fwhm_G + \
        13.41489*fwhm_G**2*fwhm_L**2 + 4.85686*fwhm_L*fwhm_G**3 + \
        2.69269*fwhm_G**4
    return df_fwhm_G/f**(4/5)/5, df_fwhm_L/f**(4/5)/5


def deriv_eta(fwhm_G: float, fwhm_L: float) -> Tuple[float, float]:
    '''
    Calculate gradient of eta of pseudo voigt profile with fwhm_G, fwhm_L based on
    Journal of Applied Crystallography. 33 (6): 1311–1316.

    Args:
      fwhm_G: full width at half maximum of gaussian part
      fwhm_L: full width at half maximum of lorenzian part
    Returns:
     gradient of eta(fwhm_G, fwhm_L)
    '''
    f = fwhm_G**5+2.69269*fwhm_G**4*fwhm_L + \
        2.42843*fwhm_G**3*fwhm_L**2 + \
        4.47163*fwhm_G**2*fwhm_L**3 + \
        0.07842*fwhm_G*fwhm_L**4 + \
        fwhm_L**5
    g = f**(-1/5)
    x = fwhm_L*g
    df_fwhm_G = 5*fwhm_G**4+10.77076*fwhm_G**3*fwhm_L + \
        7.28529*fwhm_G**2*fwhm_L**2+8.94326*fwhm_G*fwhm_L**3 + \
        0.07842*fwhm_L**4
    df_fwhm_L = 5*fwhm_L**4 + 0.31368*fwhm_L**3*fwhm_G + \
        13.41489*fwhm_G**2*fwhm_L**2 + 4.85686*fwhm_L*fwhm_G**3 + \
        2.69269*fwhm_G**4
    dx_fwhm_G = -fwhm_L*df_fwhm_G*g/f/5
    dx_fwhm_L = g - fwhm_L*df_fwhm_L*g/f/5
    deta_x = 0.33348*x**2-0.95438*x+1.36603
    return deta_x*dx_fwhm_G, deta_x*dx_fwhm_L

def hess_fwhm_eta(fwhm_G: float, fwhm_L: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate hessian of fwhm and eta of pseudo voigt profile with fwhm_G, fwhm_L based on
    Journal of Applied Crystallography. 33 (6): 1311–1316.

    Args:
      fwhm_G: full width at half maximum of gaussian part
      fwhm_L: full width at half maximum of lorenzian part
    Returns:
     hessian of fwhm(fwhm_G, fwhm_L) and hessian of eta(fwhm_G, fwhm_L)
    
    Note:
     1st column: d^2 f / d(fwhm_G)^2
     2nd column: d^2 f / d(fwhm_G)d(fwhm_L)
     3rd column: d^2 f / d(fwhm_L)^2
    '''
    Hess_fwhm = np.zeros(3)
    Hess_eta = np.zeros(3)

    f = fwhm_G**5+2.69269*fwhm_G**4*fwhm_L + \
        2.42843*fwhm_G**3*fwhm_L**2 + \
        4.47163*fwhm_G**2*fwhm_L**3 + \
        0.07842*fwhm_G*fwhm_L**4 + \
        fwhm_L**5
    
    fwhm = f**(1/5)
    x = fwhm_L/fwhm
    d_eta_x = 1.36603-0.95438*x+0.33348*x**2
    d2_eta_x_2 = 0.66696*x-0.95438

    dfwhm_fwhm_G, dfwhm_fwhm_L = deriv_fwhm(fwhm_G, fwhm_L)

    dx_fwhm_G = -x*(dfwhm_fwhm_G/fwhm)
    dx_fwhm_L = 1/fwhm-x*(dfwhm_fwhm_L/fwhm)

    d2f_fwhm_G_2 = 20*fwhm_G**3+32.31228*fwhm_G**2*fwhm_L+\
    14.57058*fwhm_G*fwhm_L**2+8.94326*fwhm_L**3
    d2f_fwhm_G_fwhm_L = 10.77076*fwhm_G**3 + 14.57058*fwhm_G**2*fwhm_L+\
    26.82978*fwhm_G*fwhm_L**2+0.31368*fwhm_L**3
    d2f_fwhm_L_2 = 20*fwhm_L**3+0.94104*fwhm_L**2*fwhm_G + \
    26.82978*fwhm_G**2*fwhm_L+4.85686*fwhm_G**3

    Hess_fwhm[0] = d2f_fwhm_G_2*fwhm/f/5-\
        4*dfwhm_fwhm_G**2/fwhm
    Hess_fwhm[1] = d2f_fwhm_G_fwhm_L*fwhm/f/5-\
        4*dfwhm_fwhm_G*dfwhm_fwhm_L/fwhm
    Hess_fwhm[2] = d2f_fwhm_L_2*fwhm/f/5-\
        4*dfwhm_fwhm_L**2/fwhm

    d2x_fwhm_G_2 = x*(2*(dfwhm_fwhm_G/fwhm)**2-(Hess_fwhm[0]/fwhm))
    d2x_fwhm_G_fwhm_L = 2*x*(dfwhm_fwhm_G/fwhm)*(dfwhm_fwhm_L/fwhm)-\
    (dfwhm_fwhm_G/fwhm+x*Hess_fwhm[1])/fwhm
    d2x_fwhm_L_2 = 2*x*(dfwhm_fwhm_L/fwhm)**2 - \
    (x*Hess_fwhm[2]+2*dfwhm_fwhm_L/fwhm)/fwhm

    Hess_eta[0] = d2_eta_x_2*(dx_fwhm_G)**2+\
        d_eta_x*(d2x_fwhm_G_2)
    Hess_eta[1] = d2_eta_x_2*(dx_fwhm_G*dx_fwhm_L)+\
        d_eta_x*(d2x_fwhm_G_fwhm_L)
    Hess_eta[2] = d2_eta_x_2*(dx_fwhm_L)**2+\
        d_eta_x*(d2x_fwhm_L_2)

    return Hess_fwhm, Hess_eta
