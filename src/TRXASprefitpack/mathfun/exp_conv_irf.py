'''
exp_conv_irf:
submodule for the mathematical functions for
exponential decay convolved with irf

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from typing import Union, Optional
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
        ans = np.empty(t.size, dtype=np.float64)
        ans[mask] = 1/2*np.exp(-(t[mask]/sigma)**2/2) * \
            erfcx(z[mask]/np.sqrt(2))
        ans[inv_mask] = 1/2*np.exp(ksigma*z[inv_mask]-ksigma**2/2) * \
            erfc(z[inv_mask]/np.sqrt(2))

    return ans

def deriv_exp_conv_gau(t: Union[float, np.ndarray], fwhm: float,
                 k: float) -> np.ndarray:

    '''
    Compute derivative of exponential function convolved with normalized gaussian
    distribution

    Args:
      t: time
      fwhm: full width at half maximum of gaussian distribution
      k: rate constant (inverse of life time)

    Returns:
     Derivative of Convolution of normalized gaussian distribution and exponential
     decay :math:`(\\exp(-kt))`.
    Note:
     1st row: df/dt
     2nd row: df/d(fwhm)
     3rd row: df/dk
    '''

    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    f = exp_conv_gau(t, fwhm, k)
    g = np.exp(-(t/sigma)**2/2)/np.sqrt(2*np.pi)

    if not isinstance(t, np.ndarray):
        grad = np.empty(3)
        grad[0] = g/sigma - k*f
        grad[1] = ((k**2*sigma)*f - (k+t/sigma**2)*g)/(2*np.sqrt(2*np.log(2)))
        grad[2] = (sigma**2*k-t)*f - sigma*g
    else:
        grad = np.empty((3, t.size))
        grad[0, :] = g/sigma - k*f
        grad[1, :] = ((k**2*sigma)*f - (k+t/sigma**2)*g)/(2*np.sqrt(2*np.log(2)))
        grad[2, :] = (sigma**2*k-t)*f - sigma*g

    return grad

def deriv_exp_sum_conv_gau(t: np.ndarray, fwhm: float,
                 k: np.ndarray, c: np.ndarray, base: Optional[bool] = True) -> np.ndarray:

    '''
    Compute derivative of sum of exponential function convolved with normalized gaussian
    distribution

    Args:
      t: time
      fwhm: full width at half maximum of gaussian distribution
      k: rate constant (inverse of life time)
      c: coefficient
      base: include baseline (i.e. k=0)

    Returns:
     Derivative of Convolution of normalized gaussian distribution and sum of exponential
     decay :math:`(\\exp(-kt))`.
    Note:
     1st row: df/dt
     2nd row: df/d(fwhm)
     2+i th row: df/dk_i
    '''
    grad = np.zeros((2+k.size, t.size))
    for i in range(k.size):
        grad_i = deriv_exp_conv_gau(t, fwhm, k[i])
        grad[0, :] = grad[0, :] + c[i]*grad_i[0, :]
        grad[1, :] = grad[1, :] + c[i]*grad_i[1, :]
        grad[2+i, :] = c[i]*grad_i[2, :]
    if base:
        grad_i = deriv_exp_conv_gau(t, fwhm, 0)
        grad[0, :] = grad[0, :] + c[-1]*grad_i[0, :]
        grad[1, :] = grad[1, :] + c[-1]*grad_i[1, :]

    return grad

def exp1x_asymp(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    return z*(1+z*(1+2*z*(1+3*z*(1+4*z*(1+5*z*(1+6*z*(1+7*z*(1+8*z*(1+9*z*(1+10*z))))))))))

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
                ans = exp1x_asymp(inv_z)
                ans = -ans.imag/np.pi
        else:
            mask = np.abs(kt) < 200
            inv_mask = np.invert(mask)

            ans = np.empty(t.size, dtype=np.float64)
            inv_z = 1/(kt[inv_mask]+ikgamma)

            # abs(kt) < 200
            ans1 = complex(np.cos(kgamma), -np.sin(kgamma))*exp1(-kt[mask]-ikgamma)
            ans[mask] = np.exp(-kt[mask])*ans1.imag/np.pi

            # abs(kt) > 200, use asymptotic series
            ans2 = exp1x_asymp(inv_z)
            ans[inv_mask] = -ans2.imag/np.pi
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

    u = exp_conv_gau(t, fwhm_G, k)
    v = exp_conv_cauchy(t, fwhm_L, k)

    return u + eta*(v-u)

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
        ans = np.empty(t.size, dtype=np.complex128)
        ans[mask] = 1/2*np.exp(-(t[mask]/sigma)**2/2) * \
            wofz(z[mask]/complex(0, -np.sqrt(2)))
        ans[inv_mask] = np.exp(sigmak_cplx*z[inv_mask]-sigmak_cplx**2/2) - \
            1/2*np.exp(-(t[inv_mask]/sigma)**2/2)*wofz(z[inv_mask]/complex(0, np.sqrt(2)))
    
    ans = (complex(np.cos(phase), np.sin(phase)))*ans

    return ans.real

def dmp_osc_conv_cauchy(t: Union[float, np.ndarray], fwhm: float,
k: float, T: float, phase: float) -> Union[float, np.ndarray]:

    '''
    Compute damped oscillation convolved with normalized cauchy
    distribution

    Args:
      t: time
      fwhm: full width at half maximum of cauchy distribution
      k: damping constant (inverse of life time)
      T: period of vibration 
      phase: phase factor

    Returns:
     Convolution of normalized cauchy distribution and 
     damped oscillation :math:`(\\exp(-kt)cos(2\\pi t/T+phase))`.
    '''

    gamma = fwhm/2; omega = 2*np.pi/T 
    z1 = (-k*t-omega*gamma) + complex(0,1)*(-k*gamma+omega*t)
    z2 = (-k*t+omega*gamma) + complex(0,-1)*(k*gamma+omega*t)
    
    if not isinstance(t, np.ndarray):
        if np.abs(z1.real) < 200:
            ans1 = np.exp(z1.real)*complex(np.cos(z1.imag+phase), np.sin(z1.imag+phase))*\
                (exp1(z1)+complex(0, 2*np.pi*np.heaviside(z1.imag, 1)))
        else:
            ans1 = exp1x_asymp(-1/z1)*complex(-np.cos(phase), -np.sin(phase))
        if np.abs(z2.real) < 200:
            ans2 =  np.exp(z2.real)*complex(np.cos(z2.imag-phase), np.sin(z1.imag-phase))*\
                exp1(z2)
        else:
            ans2 = exp1x_asymp(-1/z2)*complex(-np.cos(phase), np.sin(phase))
    
    else:
        ans1 = np.empty(t.size, dtype=np.complex128)
        ans2 = np.empty(t.size, dtype=np.complex128)
        mask1 = np.abs(z1.real) < 200; inv_mask1 = np.invert(mask1)
        mask2 = np.abs(z2.real) < 200; inv_mask2 = np.invert(mask2)

        ans1[mask1] = np.exp(z1[mask1].real)*(np.cos(z1[mask1].imag+phase)+complex(0, 1)*np.sin(z1[mask1].imag+phase))*\
            (exp1(z1[mask1])+complex(0, 2*np.pi)*np.heaviside(z1[mask1].imag, 1))
        ans1[inv_mask1] = exp1x_asymp(-1/z1[inv_mask1])*complex(-np.cos(phase), -np.sin(phase))

        ans2[mask2] = np.exp(z2[mask2].real)*(np.cos(z2[mask2].imag-phase)+complex(0, 1)*np.sin(z2[mask2].imag-phase))*\
            exp1(z2[mask2])
        ans2[inv_mask2] = exp1x_asymp(-1/z2[inv_mask2])*complex(-np.cos(phase), np.sin(phase))
        
    ans = (ans1.imag + ans2.imag)/(2*np.pi)

    return ans

def dmp_osc_conv_pvoigt(t: Union[float, np.ndarray], fwhm_G: float, fwhm_L: float, eta: float,
k: float, T: float, phase: float) -> Union[float, np.ndarray]:

    '''
    Compute damped oscillation convolved with normalized pseudo
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
       k: damping constant (inverse of life time)
       T: period of vibration 
       phase: phase factor

    Returns:
     Convolution of normalized pseudo voigt profile and
     damped oscillation :math:`(\\exp(-kt)cos(2\\pi t/T+phase))`.
    '''

    u = dmp_osc_conv_gau(t, fwhm_G, k, T, phase)
    v = dmp_osc_conv_cauchy(t, fwhm_L, k, T, phase)

    return u + eta*(v-u)


