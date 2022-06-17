'''
exp_conv_irf:
submodule for the mathematical functions for
exponential decay convolved with irf

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Union
import numpy as np
from scipy.special import erfc, erfcx, wofz, exp1


def exp_conv_gau(t: Union[float, np.ndarray], fwhm: float,
                 k: float) -> Union[float, np.ndarray]:

    '''
    Compute exponential function convolved with normalized gaussian
    distribution

    Args:
      t: time
      fwhm: full width at half maximum of gaussian distribution
      k: rate constant (inverse of life time)

    Returns:
     Convolution of normalized gaussian distribution and exponential
     decay :math:`(\\exp(-kt))`.
    '''

    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    ksigma = k*sigma
    z = ksigma - t/sigma

    if not isinstance(t, np.ndarray):
        if z > 0:
            ans = 1/2*np.exp(-(t/sigma)**2/2) * \
                erfcx(z/np.sqrt(2))
        else:
            ans = 1/2*np.exp(ksigma*z-ksigma**2/2) * \
                erfc(z/np.sqrt(2))
    else:
        mask = z > 0
        inv_mask = np.invert(mask)
        ans = np.zeros(t.shape[0], dtype=np.float64)
        ans[mask] = 1/2*np.exp(-(t[mask]/sigma)**2/2) * \
            erfcx(z[mask]/np.sqrt(2))
        ans[inv_mask] = 1/2*np.exp(ksigma*z[inv_mask]-ksigma**2/2) * \
            erfc(z[inv_mask]/np.sqrt(2))

    return ans


def exp_conv_cauchy(t: Union[float, np.ndarray],
                    fwhm: float,
                    k: float) -> Union[float, np.ndarray]:

    '''
    Compute exponential function convolved with normalized cauchy
    distribution

    Args:
       t: time
       fwhm: full width at half maximum of cauchy distribution
       k: rate constant (inverse of life time)

    Returns:
      Convolution of normalized cauchy distribution and
      exponential decay :math:`(\\exp(-kt))`.
    '''

    if k == 0:
        ans = 0.5+1/np.pi*np.arctan(2*t/fwhm)
    else:
        kgamma = k*fwhm/2
        ikgamma = complex(0, kgamma)
        kt = k*t
        if not isinstance(t, np.ndarray):
            if np.abs(kt) < 200:
                ans = complex(np.cos(kgamma), -np.sin(kgamma))*exp1(-kt-ikgamma)
                ans = np.exp(-kt)*ans.imag/np.pi
            else:
                inv_z = 1/(kt+ikgamma)
                ans = 1+10*inv_z; ans = 1+9*inv_z*ans
                ans = 1+8*inv_z*ans; ans = 1+7*inv_z*ans
                ans = 1+6*inv_z*ans; ans = 1+5*inv_z*ans
                ans = 1+4*inv_z*ans; ans = 1+3*inv_z*ans
                ans = 1+2*inv_z*ans; ans = 1+inv_z*ans
                ans = -inv_z*ans
                ans = ans.imag/np.pi
        else:
            mask = np.abs(kt) < 200
            inv_mask = np.invert(mask)

            ans = np.zeros(t.shape[0], dtype=np.float64)
            inv_z = 1/(kt[inv_mask]+ikgamma)

            # abs(kt) < 200
            ans1 = complex(np.cos(kgamma), -np.sin(kgamma))*exp1(-kt[mask]-ikgamma)
            ans[mask] = np.exp(-kt[mask])*ans1.imag/np.pi

            # abs(kt) > 200, use asymptotic series
            ans2 = 1+10*inv_z; ans2 = 1+9*inv_z*ans2
            ans2 = 1+8*inv_z*ans2; ans2 = 1+7*inv_z*ans2
            ans2 = 1+6*inv_z*ans2; ans2 = 1+5*inv_z*ans2
            ans2 = 1+4*inv_z*ans2; ans2 = 1+3*inv_z*ans2
            ans2 = 1+2*inv_z*ans2; ans2 = 1+inv_z*ans2
            ans2 = -inv_z*ans2
            ans[inv_mask] = ans2.imag/np.pi
    return ans


def exp_conv_pvoigt(t: Union[float, np.ndarray],
                    fwhm_G: float,
                    fwhm_L: float,
                    eta: float,
                    k: float) -> Union[float, np.ndarray]:

    '''
    Compute exponential function convolved with normalized pseudo
    voigt profile (i.e. linear combination of normalized gaussian and
    cauchy distribution)

    :math:`\\eta C(\\mathrm{fwhm}_L, t) + (1-\\eta)G(\\mathrm{fwhm}_G, t)`

    Args:
       t: time
       fwhm_G: full width at half maximum of gaussian part of
               pseudo voigt profile
       fwhm_L: full width at half maximum of cauchy part of
               pseudo voigt profile
       eta: mixing parameter
       k: rate constant (inverse of life time)

    Returns:
       Convolution of normalized pseudo voigt profile and
       exponential decay :math:`(\\exp(-kt))`.
    '''

    return eta*exp_conv_cauchy(t, fwhm_L, k) + \
        (1-eta)*exp_conv_gau(t, fwhm_G, k)

def dmp_osc_conv_gau(t: Union[float, np.ndarray], fwhm: float,
k: float, T: float, phase: float) -> Union[float, np.ndarray]:

    '''
    Compute damped oscillation convolved with normalized gaussian
    distribution

    Args:
      t: time
      fwhm: full width at half maximum of gaussian distribution
      k: damping constant (inverse of life time)
      T: period of vibration 
      phase: phase factor

    Returns:
     Convolution of normalized gaussian distribution and 
     damped oscillation :math:`(\\exp(-kt)cos(2\\pi t/T+phase))`.
    '''

    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    sigmak_cplx = complex(k*sigma, -2*np.pi*sigma/T)
    z = sigmak_cplx - t/sigma

    if not isinstance(t, np.ndarray):
        if z.real > 0:
            ans = 1/2*np.exp(-(t/sigma)**2/2) * \
                wofz(z/complex(0, -np.sqrt(2)))
        else:
            ans = np.exp(sigmak_cplx*z-sigmak_cplx**2/2) - \
                1/2*np.exp(-(t/sigma)**2/2)*wofz(z/complex(0, np.sqrt(2))) 
    else:
        mask = z.real > 0
        inv_mask = np.invert(mask)
        ans = np.zeros(t.shape[0], dtype=np.complex128)
        ans[mask] = 1/2*np.exp(-(t[mask]/sigma)**2/2) * \
            wofz(z[mask]/complex(0, -np.sqrt(2)))
        ans[inv_mask] = np.exp(sigmak_cplx*z[inv_mask]-sigmak_cplx**2/2) - \
            1/2*np.exp(-(t[inv_mask]/sigma)**2/2)*wofz(z[inv_mask]/complex(0, np.sqrt(2)))
    
    ans = (complex(np.cos(phase), np.sin(phase)))*ans

    return ans.real


'''
[TODO]
Implement analytic dmp_osc_conv_cauchy
'''