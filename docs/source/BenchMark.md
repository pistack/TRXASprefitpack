# BenchMark

Tests ``exp_conv_gau``, ``exp_conv_cauchy``, ``dmp_osc_conv_gau``, ``dmp_osc_conv_cauchy`` routine


```python
import numpy as np
from scipy.signal import convolve
from TRXASprefitpack import gau_irf, cauchy_irf, exp_conv_gau, exp_conv_cauchy, dmp_osc_conv_gau, dmp_osc_conv_cauchy
import matplotlib.pyplot as plt
```


```python
def decay(t, k):
    return np.exp(-k*t)*np.heaviside(t, 1)

def dmp_osc(t, k, T, phi):
    return np.exp(-k*t)*np.cos(2*np.pi*t/T+phi)*np.heaviside(t, 1)
```

## ``exp_conv_gau``

Test condition.

1. fwhm = 0.15, tau = 0.05, time range [-1, 1]

2. fwhm = 0.15, tau = 0.15, time range [-1, 1]

3. fwhm = 0.15, tau = 1.5, time range [-10, 10]
    


```python
fwhm = 0.15
tau_1 = 0.05
tau_2 = 0.15
tau_3 = 1.5

t_1 = np.linspace(-1, 1, 2001)
t_2 = np.linspace(-1, 1, 2001)
t_3 = np.linspace(-10, 10, 20001)

# To test exp_conv_gau routine, calculates convolution numerically
gau_irf_1 = gau_irf(t_1, fwhm)
gau_irf_2 = gau_irf(t_2, fwhm)
gau_irf_3 = gau_irf(t_3, fwhm)

decay_1 = decay(t_1, 1/tau_1)
decay_2 = decay(t_2, 1/tau_2)
decay_3 = decay(t_3, 1/tau_3)

gau_signal_1_ref = 0.001*convolve(gau_irf_1, decay_1, mode='same')
gau_signal_2_ref = 0.001*convolve(gau_irf_2, decay_2, mode='same')
gau_signal_3_ref = 0.001*convolve(gau_irf_3, decay_3, mode='same')
```

### Test -1-



```python
t_1_sample = np.hstack((np.arange(-1, -0.5, 0.1), np.arange(-0.5, 0.5, 0.05), np.arange(0.5, 1.1, 0.1)))
gau_signal_1_tst = exp_conv_gau(t_1_sample, fwhm, 1/tau_1)

plt.plot(t_1, gau_signal_1_ref, label='ref test 1')
plt.plot(t_1_sample, gau_signal_1_tst, 'ro', label='analytic expression')
plt.legend()
plt.show()
```


    
![png](BenchMark_files/BenchMark_6_0.png)
    



```python
%timeit gau_signal_1_ref = 0.001*convolve(gau_irf_1, decay_1, mode='same')
```

    379 µs ± 1.26 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)



```python
%timeit gau_signal_1_tst = exp_conv_gau(t_1_sample, fwhm, 1/tau_1)
```

    24 µs ± 106 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


Analytic implementation matches to numerical one. Moreover it saves much more computation time (about ~18x), since numerial implementation need to compute convolution of 0.001 ps step data to achieve similiar accuracy level of analytic one.

### Test -2-


```python
t_2_sample = np.hstack((np.arange(-1, -0.5, 0.1), np.arange(-0.5, 0.5, 0.05), np.arange(0.5, 1.1, 0.1)))
gau_signal_2_tst = exp_conv_gau(t_1_sample, fwhm, 1/tau_2)

plt.plot(t_2, gau_signal_2_ref, label='ref test 2')
plt.plot(t_2_sample, gau_signal_2_tst, 'ro', label='analytic expression')
plt.legend()
plt.show()
```


    
![png](BenchMark_files/BenchMark_11_0.png)
    


### Test -3-


```python
t_3_sample = np.hstack((np.arange(-1, 1, 0.1), np.arange(1, 3, 0.2), np.arange(3, 7, 0.4), np.arange(7, 11, 1)))
gau_signal_3_tst = exp_conv_gau(t_3_sample, fwhm, 1/tau_3)

plt.plot(t_3, gau_signal_3_ref, label='ref test 3')
plt.plot(t_3_sample, gau_signal_3_tst, 'ro', label='analytic expression')
plt.legend()
plt.show()
```


    
![png](BenchMark_files/BenchMark_13_0.png)
    


## ``exp_conv_cauchy``

Test condition.

1. fwhm = 0.15, tau = 0.05, time range [-1, 1]

2. fwhm = 0.15, tau = 0.15, time range [-1, 1]

3. fwhm = 0.15, tau = 1.5, time range [-10, 10]


```python
fwhm = 0.15
tau_1 = 0.05
tau_2 = 0.15
tau_3 = 1.5

t_1 = np.linspace(-1, 1, 2001)
t_2 = np.linspace(-1, 1, 2001)
t_3 = np.linspace(-10, 10, 20001)

# To test exp_conv_cauchy routine, calculates convolution numerically
cauchy_irf_1 = cauchy_irf(t_1, fwhm)
cauchy_irf_2 = cauchy_irf(t_2, fwhm)
cauchy_irf_3 = cauchy_irf(t_3, fwhm)

cauchy_signal_1_ref = 0.001*convolve(cauchy_irf_1, decay_1, mode='same')
cauchy_signal_2_ref = 0.001*convolve(cauchy_irf_2, decay_2, mode='same')
cauchy_signal_3_ref = 0.001*convolve(cauchy_irf_3, decay_3, mode='same')
```

### Test -1-


```python
t_1_sample = np.hstack((np.arange(-1, -0.5, 0.1), np.arange(-0.5, 0.5, 0.05), np.arange(0.5, 1.1, 0.1)))
cauchy_signal_1_tst = exp_conv_cauchy(t_1_sample, fwhm, 1/tau_1)

plt.plot(t_1, cauchy_signal_1_ref, label='ref test 1')
plt.plot(t_1_sample, cauchy_signal_1_tst, 'ro', label='analytic expression')
plt.legend()
plt.show()
```


    
![png](BenchMark_files/BenchMark_17_0.png)
    



```python
%timeit cauchy_signal_1_ref = 0.001*convolve(cauchy_irf_1, decay_1, mode='same')
```

    338 µs ± 9.16 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)



```python
%timeit cauchy_signal_1_tst = exp_conv_cauchy(t_1_sample, fwhm, 1/tau_1)
```

    68.4 µs ± 1.53 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


Computation time for ``exp_conv_cauchy`` routine is about 2-3 times longer than ``exp_conv_gau`` routine. Because ``exp_conv_cauchy`` routine requires expansive exponential integral function

### Test -2-


```python
t_2_sample = np.hstack((np.arange(-1, -0.5, 0.1), np.arange(-0.5, 0.5, 0.05), np.arange(0.5, 1.1, 0.1)))
cauchy_signal_2_tst = exp_conv_cauchy(t_1_sample, fwhm, 1/tau_2)

plt.plot(t_2, cauchy_signal_2_ref, label='ref test 2')
plt.plot(t_2_sample, cauchy_signal_2_tst, 'ro', label='analytic expression')
plt.legend()
plt.show()
```


    
![png](BenchMark_files/BenchMark_22_0.png)
    


### Test -3-


```python
t_3_sample = np.hstack((np.arange(-1, 1, 0.1), np.arange(1, 3, 0.2), np.arange(3, 7, 0.4), np.arange(7, 11, 1)))
cauchy_signal_3_tst = exp_conv_cauchy(t_3_sample, fwhm, 1/tau_3)

plt.plot(t_3, cauchy_signal_3_ref, label='ref test 3')
plt.plot(t_3_sample, cauchy_signal_3_tst, 'ro', label='analytic expression')
plt.legend()
plt.show()
```


    
![png](BenchMark_files/BenchMark_24_0.png)
    


## ``dmp_osc_conv_gau``

Test condition.

1. fwhm = 0.15, tau = 10, T = 0.1, phase: 0, time range [-50, 50]
2. fwhm = 0.15, tau = 10, T = 0.5, phase: $\pi/4$, time range [-50, 50]
2. fwhm = 0.15, tau = 10, T = 3, phase: $\pi/2$, time range [-50, 50]


```python
fwhm = 0.15
tau = 10
T_1 = 0.1
T_2 = 0.5
T_3 = 3
phi_1 = 0
phi_2 = np.pi/4
phi_3 = np.pi/2

t = np.linspace(-50, 50, 1000001)
t_sample = np.hstack((np.arange(-1, -0.5, 0.1), np.arange(-0.5, 0.5, 0.05), np.arange(0.5, 1, 0.1),
                     np.arange(1, 3, 0.2), np.arange(3, 10, 0.4), np.arange(10, 20, 1), np.arange(20, 55, 5)))

# To test exp_conv_gau routine, calculates convolution numerically
gau_irf_0 = gau_irf(t, fwhm)

dmp_osc_1 = dmp_osc(t, 1/tau, T_1, phi_1)
dmp_osc_2 = dmp_osc(t, 1/tau, T_2, phi_2)
dmp_osc_3 = dmp_osc(t, 1/tau, T_3, phi_3)


dmp_osc_signal_1_ref = (t[1]-t[0])*convolve(gau_irf_0, dmp_osc_1, mode='same')
dmp_osc_signal_2_ref = (t[1]-t[0])*convolve(gau_irf_0, dmp_osc_2, mode='same')
dmp_osc_signal_3_ref = (t[1]-t[0])*convolve(gau_irf_0, dmp_osc_3, mode='same')

dmp_osc_signal_1_tst = dmp_osc_conv_gau(t_sample, fwhm, 1/tau, T_1, phi_1)
dmp_osc_signal_2_tst = dmp_osc_conv_gau(t_sample, fwhm, 1/tau, T_2, phi_2)
dmp_osc_signal_3_tst = dmp_osc_conv_gau(t_sample, fwhm, 1/tau, T_3, phi_3)

```

### Test -1-


```python
plt.plot(t, dmp_osc_signal_1_ref, label='ref test 1')
plt.plot(t_sample, dmp_osc_signal_1_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 1)
plt.show()
```


    
![png](BenchMark_files/BenchMark_28_0.png)
    



```python
%timeit dmp_osc_signal_1_ref = (t[1]-t[0])*convolve(gau_irf_0, dmp_osc_1, mode='same')
```

    111 ms ± 1.91 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit dmp_osc_signal_1_tst = dmp_osc_conv_gau(t_sample, fwhm, 1/tau, T_1, phi_1)
```

    43.8 µs ± 110 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


Analytic implementation matches to numerical one. Moreover it saves much more computation time (about 1000~3000x), since numerial implementation need to compute convolution of 0.0001 ps step data to achieve similiar accuracy level of analytic one

### Test -2-


```python
plt.plot(t, dmp_osc_signal_2_ref, label='ref test 2')
plt.plot(t_sample, dmp_osc_signal_2_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 1)
plt.show()
```


    
![png](BenchMark_files/BenchMark_33_0.png)
    



```python
plt.plot(t, dmp_osc_signal_2_ref, label='ref test 2')
plt.plot(t_sample, dmp_osc_signal_2_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 10)
plt.show()
```


    
![png](BenchMark_files/BenchMark_34_0.png)
    


### Test -3-


```python
plt.plot(t, dmp_osc_signal_3_ref, label='ref test 3')
plt.plot(t_sample, dmp_osc_signal_3_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 2)
plt.show()
```


    
![png](BenchMark_files/BenchMark_36_0.png)
    



```python
plt.plot(t, dmp_osc_signal_3_ref, label='ref test 3')
plt.plot(t_sample, dmp_osc_signal_3_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 20)
plt.show()
```


    
![png](BenchMark_files/BenchMark_37_0.png)
    


## ``dmp_osc_conv_cauchy``

Test condition.

1. fwhm = 0.15, tau = 10, T = 0.1, phase: 0, time range [-50, 50]
2. fwhm = 0.15, tau = 10, T = 0.5, phase: $\pi/4$, time range [-50, 50]
2. fwhm = 0.15, tau = 10, T = 3, phase: $\pi/2$, time range [-50, 50]


```python
# To test exp_conv_cauchy routine, calculates convolution numerically
cauchy_irf_0 = cauchy_irf(t, fwhm)


dmp_osc_cauchy_signal_1_ref = (t[1]-t[0])*convolve(cauchy_irf_0, dmp_osc_1, mode='same')
dmp_osc_cauchy_signal_2_ref = (t[1]-t[0])*convolve(cauchy_irf_0, dmp_osc_2, mode='same')
dmp_osc_cauchy_signal_3_ref = (t[1]-t[0])*convolve(cauchy_irf_0, dmp_osc_3, mode='same')

dmp_osc_cauchy_signal_1_tst = dmp_osc_conv_cauchy(t_sample, fwhm, 1/tau, T_1, phi_1)
dmp_osc_cauchy_signal_2_tst = dmp_osc_conv_cauchy(t_sample, fwhm, 1/tau, T_2, phi_2)
dmp_osc_cauchy_signal_3_tst = dmp_osc_conv_cauchy(t_sample, fwhm, 1/tau, T_3, phi_3)
```

### Test -1-


```python
plt.plot(t, dmp_osc_cauchy_signal_1_ref, label='ref test 1')
plt.plot(t_sample, dmp_osc_cauchy_signal_1_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 1)
plt.show()
```


    
![png](BenchMark_files/BenchMark_41_0.png)
    



```python
%timeit dmp_osc_cauchy_signal_1_ref = (t[1]-t[0])*convolve(cauchy_irf_0, dmp_osc_1, mode='same')
```

    106 ms ± 1.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit dmp_osc_cauchy_signal_1_tst = dmp_osc_conv_cauchy(t_sample, fwhm, 1/tau, T_1, phi_1)
```

    274 µs ± 1.52 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


``dmp_osc_cauchy`` routine is 5-6 times slower than ``dmp_osc_gau`` routine due to the evaluation of expansive exponential integral function

### Test -2-


```python
plt.plot(t, dmp_osc_cauchy_signal_2_ref, label='ref test 2')
plt.plot(t_sample, dmp_osc_cauchy_signal_2_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 1)
plt.show()
```


    
![png](BenchMark_files/BenchMark_46_0.png)
    



```python
plt.plot(t, dmp_osc_cauchy_signal_2_ref, label='ref test 2')
plt.plot(t_sample, dmp_osc_cauchy_signal_2_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 10)
plt.show()
```


    
![png](BenchMark_files/BenchMark_47_0.png)
    


### Test -3-


```python
plt.plot(t, dmp_osc_cauchy_signal_3_ref, label='ref test 3')
plt.plot(t_sample, dmp_osc_cauchy_signal_3_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 2)
plt.show()
```


    
![png](BenchMark_files/BenchMark_49_0.png)
    



```python
plt.plot(t, dmp_osc_cauchy_signal_3_ref, label='ref test 3')
plt.plot(t_sample, dmp_osc_cauchy_signal_3_tst, 'ro', label='analytic expression')
plt.legend()
plt.xlim(-1, 20)
plt.show()
```


    
![png](BenchMark_files/BenchMark_50_0.png)
    

