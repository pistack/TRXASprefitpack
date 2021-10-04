# introduce asymptotic series at abs(kt) > 700

import numpy as np
from TRXASprefitpack import exp_conv_cauchy


def asymp(t, fwhm, k, N):
    z_inv = 1/(k*t+complex(0, k*fwhm/2))
    i = N
    ans = 1
    while i > 0:
        ans = 1+i*z_inv*ans
        i = i-1
    ans = -z_inv*ans
    return ans.imag/np.pi


t1 = np.arange(-700, -600, 1)
t2 = np.arange(600, 700, 1)
t = np.concatenate((t1, t2))

Z = np.allclose(exp_conv_cauchy(t, 0.01, 1), asymp(t, 0.01, 1, 10), rtol=1e-8,
                atol=1e-16)
A = np.allclose(exp_conv_cauchy(t, 0.1, 1), asymp(t, 0.1, 1, 10), rtol=1e-8,
                atol=1e-16)
B = np.allclose(exp_conv_cauchy(t, 1, 1), asymp(t, 1, 1, 10), rtol=1e-8,
                atol=1e-16)
C = np.allclose(exp_conv_cauchy(t, 10, 1), asymp(t, 10, 1, 10), rtol=1e-8,
                atol=1e-16)
D = np.allclose(exp_conv_cauchy(t, 100, 1), asymp(t, 100, 1, 10), rtol=1e-8,
                atol=1e-16)
E = np.allclose(exp_conv_cauchy(t, 1000, 1), asymp(t, 1000, 1, 10), rtol=1e-8,
                atol=1e-16)

print('fwhm: 0.01', np.all(Z))
print('fwhm: 0.1', np.all(A))
print('fwhm: 1', np.all(B))
print('fwhm: 10', np.all(C))
print('fwhm: 100', np.all(D))
print('fwhm: 1000', np.all(E))



