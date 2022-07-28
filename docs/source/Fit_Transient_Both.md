# Fitting with time delay scan (model: sum of exponential decay and damped oscillation)
## Objective
1. Fitting with sum of exponential decay model and damped oscillation model
2. Save and Load fitting result
3. Calculates species associated coefficent from fitting result
4. Evaluates F-test based confidence interval


In this example, we only deal with gaussian irf 


```python
# import needed module
import numpy as np
import matplotlib.pyplot as plt
import TRXASprefitpack
from TRXASprefitpack import solve_seq_model, rate_eq_conv, dmp_osc_conv_gau 
plt.rcParams["figure.figsize"] = (12,9)
```

## Version information


```python
print(TRXASprefitpack.__version__)
```

    0.6.0
    

## Detecting oscillation feature


```python
# Generates fake experiment data
# Model: 1 -> 2 -> 3 -> GS
# lifetime tau1: 500 fs, tau2: 10 ps, tau3: 1000 ps
# oscillation: tau_osc: 1 ps, period_osc: 300 fs, phase: pi/4
# fwhm paramter of gaussian IRF: 100 fs

tau_1 = 0.5
tau_2 = 10
tau_3 = 1000
fwhm = 0.100
tau_osc = 1
period_osc = 0.3
phase = np.pi/4

# initial condition
y0 = np.array([1, 0, 0, 0])

# set time range (mixed step)
t_seq1 = np.arange(-2, -1, 0.2)
t_seq2 = np.arange(-1, 1, 0.02)
t_seq3 = np.arange(1, 5, 0.2)
t_seq4 = np.arange(5, 10, 1)
t_seq5 = np.arange(10, 100, 10)
t_seq6 = np.arange(100, 1000, 100)
t_seq7 = np.linspace(1000, 2000, 2)

t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5, t_seq6, t_seq7))

eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

# Now generates measured transient signal
# Last element is ground state

abs_1 = [1, 1, 1, 0]; abs_1_osc = 0.05
abs_2 = [0.5, 0.8, 0.2, 0]; abs_2_osc = 0.001
abs_3 = [-0.5, 0.7, 0.9, 0]; abs_3_osc = -0.002
abs_4 = [0.6, 0.3, -1, 0]; abs_4_osc = 0.0018

t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan

# generate measured data

y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')+\
    abs_1_osc*dmp_osc_conv_gau(t_seq-t0[0], fwhm, 1/tau_osc, period_osc, phase)
y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')+\
    abs_2_osc*dmp_osc_conv_gau(t_seq-t0[1], fwhm, 1/tau_osc, period_osc, phase)
y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')+\
    abs_3_osc*dmp_osc_conv_gau(t_seq-t0[2], fwhm, 1/tau_osc, period_osc, phase)
y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')+\
    abs_4_osc*dmp_osc_conv_gau(t_seq-t0[3], fwhm, 1/tau_osc, period_osc, phase)

# generate random noise with (S/N = 200)

# Define noise level (S/N=200) w.r.t peak
eps_obs_1 = np.max(np.abs(y_obs_1))/200*np.ones_like(y_obs_1)
eps_obs_2 = np.max(np.abs(y_obs_2))/200*np.ones_like(y_obs_2)
eps_obs_3 = np.max(np.abs(y_obs_3))/200*np.ones_like(y_obs_3)
eps_obs_4 = np.max(np.abs(y_obs_4))/200*np.ones_like(y_obs_4)

# generate random noise
noise_1 = np.random.normal(0, eps_obs_1, t_seq.size)
noise_2 = np.random.normal(0, eps_obs_2, t_seq.size)
noise_3 = np.random.normal(0, eps_obs_3, t_seq.size)
noise_4 = np.random.normal(0, eps_obs_4, t_seq.size)


# generate measured intensity
i_obs_1 = y_obs_1 + noise_1
i_obs_2 = y_obs_2 + noise_2
i_obs_3 = y_obs_3 + noise_3
i_obs_4 = y_obs_4 + noise_4

# print real values

print('-'*24)
print(f'fwhm: {fwhm}')
print(f'tau_1: {tau_1}')
print(f'tau_2: {tau_2}')
print(f'tau_3: {tau_3}')
print(f'tau_osc: {tau_osc}')
print(f'period_osc: {period_osc}')
print(f'phase_osc: {phase}')
for i in range(4):
    print(f't_0_{i+1}: {t0[i]}')
print('-'*24)
print('Excited Species contribution')
print(f'scan 1: {abs_1[0]} \t {abs_1[1]} \t {abs_1[2]}')
print(f'scan 2: {abs_2[0]} \t {abs_2[1]} \t {abs_2[2]}')
print(f'scan 3: {abs_3[0]} \t {abs_3[1]} \t {abs_3[2]}')
print(f'scan 4: {abs_4[0]} \t {abs_4[1]} \t {abs_4[2]}')

param_exact = [fwhm, t0[0], t0[1], t0[2], t0[3], tau_1, tau_2, tau_3, tau_osc, period_osc, phase]
```

    ------------------------
    fwhm: 0.1
    tau_1: 0.5
    tau_2: 10
    tau_3: 1000
    tau_osc: 1
    period_osc: 0.3
    phase_osc: 0.7853981633974483
    t_0_1: -0.06280648634000692
    t_0_2: 0.11250332258063539
    t_0_3: 0.015015489438238317
    t_0_4: 0.038270496972757376
    ------------------------
    Excited Species contribution
    scan 1: 1 	 1 	 1
    scan 2: 0.5 	 0.8 	 0.2
    scan 3: -0.5 	 0.7 	 0.9
    scan 4: 0.6 	 0.3 	 -1
    


```python
# plot model experimental data

plt.errorbar(t_seq, i_obs_1, eps_obs_1, label='1')
plt.errorbar(t_seq, i_obs_2, eps_obs_2, label='2')
plt.errorbar(t_seq, i_obs_3, eps_obs_3, label='3')
plt.errorbar(t_seq, i_obs_4, eps_obs_4, label='4')
plt.legend()
plt.show()
```


    
![png](Fit_Transient_Both_files/Fit_Transient_Both_6_0.png)
    



```python
plt.errorbar(t_seq, i_obs_1, eps_obs_1, label='1')
plt.errorbar(t_seq, i_obs_2, eps_obs_2, label='2')
plt.errorbar(t_seq, i_obs_3, eps_obs_3, label='3')
plt.errorbar(t_seq, i_obs_4, eps_obs_4, label='4')
plt.legend()
plt.xlim(-10*fwhm, 20*fwhm)
plt.show()
```


    
![png](Fit_Transient_Both_files/Fit_Transient_Both_7_0.png)
    


We can show oscillation feature at scan 1. First try fitting without oscillation.


```python
# import needed module for fitting
from TRXASprefitpack import fit_transient_exp

# time, intensity, eps should be sequence of numpy.ndarray
t = [t_seq] 
intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

# set initial guess
irf = 'g' # shape of irf function
fwhm_init = 0.15
t0_init = np.array([0, 0, 0, 0])
# test with one decay module
tau_init = np.array([0.2, 20, 1500])

fit_result_decay = fit_transient_exp(irf, fwhm_init, t0_init, tau_init, False, do_glb=True, t=t, intensity=intensity, eps=eps)

```


```python
# print fitting result
print(fit_result_decay)
```

    [Model information]
        model : decay
        irf: g
        fwhm:  0.1007
        eta:  0.0000
        base: False
     
    [Optimization Method]
        global: basinhopping
        leastsq: trf
     
    [Optimization Status]
        nfev: 3641
        status: 0
        global_opt msg: requested number of basinhopping iterations completed successfully
        leastsq_opt msg: `ftol` termination condition is satisfied.
     
    [Optimization Results]
        Total Data points: 600
        Number of effective parameters: 20
        Degree of Freedom: 580
        Chi squared:  1068.2021
        Reduced chi squared:  1.8417
        AIC (Akaike Information Criterion statistic):  386.0816
        BIC (Bayesian Information Criterion statistic):  474.0202
     
    [Parameters]
        fwhm_G:  0.10072518 +/-  0.00091828 ( 0.91%)
        t_0_1_1: -0.06304187 +/-  0.00042964 ( 0.68%)
        t_0_1_2:  0.11256677 +/-  0.00061425 ( 0.55%)
        t_0_1_3:  0.01526769 +/-  0.00077009 ( 5.04%)
        t_0_1_4:  0.03850668 +/-  0.00066916 ( 1.74%)
        tau_1:  0.49712192 +/-  0.00270823 ( 0.54%)
        tau_2:  9.97269461 +/-  0.05581337 ( 0.56%)
        tau_3:  1002.01450914 +/-  4.77270882 ( 0.48%)
     
    [Parameter Bound]
        fwhm_G:  0.075 <=  0.10072518 <=  0.3
        t_0_1_1: -0.3 <= -0.06304187 <=  0.3
        t_0_1_2: -0.3 <=  0.11256677 <=  0.3
        t_0_1_3: -0.3 <=  0.01526769 <=  0.3
        t_0_1_4: -0.3 <=  0.03850668 <=  0.3
        tau_1:  0.075 <=  0.49712192 <=  1.2
        tau_2:  4.8 <=  9.97269461 <=  76.8
        tau_3:  307.2 <=  1002.01450914 <=  4915.2
     
    [Component Contribution]
        DataSet dataset_1:
         #tscan	tscan_1	tscan_2	tscan_3	tscan_4
         decay 1	-1.07%	-28.49%	-51.17%	 8.56%
         decay 2	-0.63%	 54.12%	-9.61%	 52.81%
         decay 3	 98.30%	 17.39%	 39.23%	-38.62%
     
    [Parameter Correlation]
        Parameter Correlations >  0.1 are reported.
        (t_0_1_1, fwhm_G) =  0.106
        (t_0_1_2, fwhm_G) =  0.104
        (tau_1, t_0_1_2) =  0.122
        (tau_1, t_0_1_3) = -0.35
        (tau_2, t_0_1_2) =  0.105
        (tau_2, t_0_1_4) =  0.141
        (tau_3, tau_2) = -0.163
    


```python
# plot fitting result and experimental data

color_lst = ['red', 'blue', 'green', 'black']

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.show()
```


    
![png](Fit_Transient_Both_files/Fit_Transient_Both_11_0.png)
    



```python
# plot with shorter time range

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.xlim(-10*fwhm_init, 20*fwhm_init)
plt.show()

```


    
![png](Fit_Transient_Both_files/Fit_Transient_Both_12_0.png)
    


There may exists oscillation in experimental scan 1. To show oscillation feature plot residual (expt-fit)


```python
# To show oscillation feature plot residual
for i in range(4):
    plt.errorbar(t[0], fit_result_decay['res'][0][:, i], eps[0][:, i], label=f'res {i+1}', color=color_lst[i])

plt.legend()
plt.xlim(-10*fwhm_init, 20*fwhm_init)
plt.show()
```


    
![png](Fit_Transient_Both_files/Fit_Transient_Both_14_0.png)
    


Only residual for experimental scan 1 shows clear oscillation feature, Now add oscillation feature.


```python
from TRXASprefitpack import fit_transient_both
tau_osc_init = np.array([1.5])
period_osc_init = np.array([0.5])
phase_osc_init = np.array([0])

fit_result_decay_osc = fit_transient_both(irf, fwhm_init, t0_init, tau_init, 
tau_osc_init, period_osc_init, phase_osc_init, 
False, do_glb=True, t=t, intensity=intensity, eps=eps)

```


```python
# print fitting result
print(fit_result_decay_osc)
```

    [Model information]
        model : both
        irf: g
        fwhm:  0.0984
        eta:  0.0000
        base: False
     
    [Optimization Method]
        global: basinhopping
        leastsq: trf
     
    [Optimization Status]
        nfev: 14325
        status: 0
        global_opt msg: requested number of basinhopping iterations completed successfully
        leastsq_opt msg: `ftol` termination condition is satisfied.
     
    [Optimization Results]
        Total Data points: 600
        Number of effective parameters: 27
        Degree of Freedom: 573
        Chi squared:  592.5995
        Reduced chi squared:  1.0342
        AIC (Akaike Information Criterion statistic):  46.5535
        BIC (Bayesian Information Criterion statistic):  165.2706
     
    [Parameters]
        fwhm_G:  0.09842813 +/-  0.00073106 ( 0.74%)
        t_0_1_1: -0.06331250 +/-  0.00040055 ( 0.63%)
        t_0_1_2:  0.11238801 +/-  0.00045914 ( 0.41%)
        t_0_1_3:  0.01530694 +/-  0.00057183 ( 3.74%)
        t_0_1_4:  0.03841284 +/-  0.00050090 ( 1.30%)
        tau_1:  0.49762521 +/-  0.00208563 ( 0.42%)
        tau_2:  9.97873954 +/-  0.04213786 ( 0.42%)
        tau_3:  1001.51104548 +/-  3.57436348 ( 0.36%)
        tau_osc_1:  0.87427349 +/-  0.12665021 ( 14.49%)
        period_osc_1:  0.30222734 +/-  0.00260978 ( 0.86%)
        phase_osc_1: -2.18925019 +/-  0.08572030 ( 3.92%)
     
    [Parameter Bound]
        fwhm_G:  0.075 <=  0.09842813 <=  0.3
        t_0_1_1: -0.3 <= -0.06331250 <=  0.3
        t_0_1_2: -0.3 <=  0.11238801 <=  0.3
        t_0_1_3: -0.3 <=  0.01530694 <=  0.3
        t_0_1_4: -0.3 <=  0.03841284 <=  0.3
        tau_1:  0.075 <=  0.49762521 <=  1.2
        tau_2:  4.8 <=  9.97873954 <=  76.8
        tau_3:  307.2 <=  1001.51104548 <=  4915.2
        tau_osc_1:  0.3 <=  0.87427349 <=  4.8
        period_osc_1:  0.075 <=  0.30222734 <=  1.2
        phase_osc_1: -3.14159265 <= -2.18925019 <=  3.14159265
     
    [Component Contribution]
        DataSet dataset_1:
         #tscan	tscan_1	tscan_2	tscan_3	tscan_4
         decay 1	-0.10%	-28.40%	-51.12%	 8.57%
         decay 2	-0.83%	 53.93%	-9.60%	 52.66%
         decay 3	 94.01%	 17.33%	 39.22%	-38.52%
        dmp_osc 1	-5.06%	-0.34%	 0.05%	-0.25%
     
    [Parameter Correlation]
        Parameter Correlations >  0.1 are reported.
        (t_0_1_1, fwhm_G) =  0.275
        (t_0_1_2, fwhm_G) =  0.135
        (tau_1, t_0_1_2) =  0.121
        (tau_1, t_0_1_3) = -0.353
        (tau_2, t_0_1_2) =  0.106
        (tau_2, t_0_1_4) =  0.142
        (tau_3, tau_2) = -0.163
        (tau_osc_1, fwhm_G) =  0.137
        (period_osc_1, fwhm_G) = -0.265
        (period_osc_1, t_0_1_1) = -0.52
        (phase_osc_1, fwhm_G) = -0.336
        (phase_osc_1, t_0_1_1) = -0.59
        (phase_osc_1, t_0_1_2) = -0.114
        (phase_osc_1, t_0_1_4) = -0.119
        (phase_osc_1, period_osc_1) =  0.839
    


```python
# plot residual and oscilation fit

for i in range(1):
    plt.errorbar(t[0], intensity[0][:, i]-fit_result_decay_osc['fit_decay'][0][:, i], eps[0][:, i], label=f'res {i+1}', color='black')
    plt.errorbar(t[0], fit_result_decay_osc['fit_osc'][0][:, i], label=f'osc {i+1}', color='red')

plt.legend()
plt.xlim(-10*fwhm_init, 20*fwhm_init)
plt.show()

print()


```


    
![png](Fit_Transient_Both_files/Fit_Transient_Both_18_0.png)
    


    
    


```python
# Compare fitting value and exact value
for i in range(len(fit_result_decay_osc['x'])):
    print(f"{fit_result_decay_osc['param_name'][i]}: {fit_result_decay_osc['x'][i]} (fit) \t {param_exact[i]} (exact)")
```

    fwhm_G: 0.09842813455266139 (fit) 	 0.1 (exact)
    t_0_1_1: -0.06331249974079364 (fit) 	 -0.06280648634000692 (exact)
    t_0_1_2: 0.11238801254036403 (fit) 	 0.11250332258063539 (exact)
    t_0_1_3: 0.01530694015919008 (fit) 	 0.015015489438238317 (exact)
    t_0_1_4: 0.03841284222007836 (fit) 	 0.038270496972757376 (exact)
    tau_1: 0.4976252099062054 (fit) 	 0.5 (exact)
    tau_2: 9.978739537439782 (fit) 	 10 (exact)
    tau_3: 1001.511045476416 (fit) 	 1000 (exact)
    tau_osc_1: 0.8742734928953225 (fit) 	 1 (exact)
    period_osc_1: 0.30222734002594587 (fit) 	 0.3 (exact)
    phase_osc_1: -2.1892501947045324 (fit) 	 0.7853981633974483 (exact)
    

Since $\pi$ is corresponding to sign inversion ($e^{i \pi} = -1$), one can assume ($\pi+\phi = \phi$).


```python
# save fitting result to file
from TRXASprefitpack import save_TransientResult, load_TransientResult

save_TransientResult(fit_result_decay_osc, 'example_decay_osc') # save fitting result to example_decay_2.h5
loaded_result = load_TransientResult('example_decay_osc') # load fitting result from example_decay_2.h5
```

Now deduce species associated difference coefficient from sequential decay model


```python
y0 = np.array([1, 0, 0, 0]) # initial cond
eigval, V, c = solve_seq_model(loaded_result['x'][5:-3], y0)

# compute scaled V matrix
V_scale = np.einsum('j,ij->ij', c, V)
diff_abs_fit = np.linalg.solve(V_scale[:-1, :-1].T, loaded_result['c'][0][:-1,:]) 
# slice last column and row corresponding to ground state
# exclude oscillation factor

# compare with exact result
print('-'*24)
print('[Species Associated Difference Coefficent]')
print('scan # \t ex 1 (fit) \t ex 1 (exact) \t ex 2 (fit) \t ex 2 (exact) \t ex 3 (exact)')
print(f'1 \t {diff_abs_fit[0,0]} \t {abs_1[0]}  \t {diff_abs_fit[1,0]} \t {abs_1[1]} \t {diff_abs_fit[2,0]} \t {abs_1[2]}')
print(f'2 \t {diff_abs_fit[0,1]} \t {abs_2[0]}  \t {diff_abs_fit[1,1]} \t {abs_2[1]} \t {diff_abs_fit[2,1]} \t {abs_2[2]}')
print(f'3 \t {diff_abs_fit[0,2]} \t {abs_3[0]}  \t {diff_abs_fit[1,2]} \t {abs_3[1]} \t {diff_abs_fit[2,2]} \t {abs_3[2]}')
print(f'4 \t {diff_abs_fit[0,3]} \t {abs_4[0]}  \t {diff_abs_fit[1,3]} \t {abs_4[1]} \t {diff_abs_fit[2,3]} \t {abs_4[2]}')

```

    ------------------------
    [Species Associated Difference Coefficent]
    scan # 	 ex 1 (fit) 	 ex 1 (exact) 	 ex 2 (fit) 	 ex 2 (exact) 	 ex 3 (exact)
    1 	 0.9995083798993012 	 1  	 1.0005360582896203 	 1 	 0.9989966322803481 	 1
    2 	 0.4999209905462423 	 0.5  	 0.7997166372266675 	 0.8 	 0.1999859729039946 	 0.2
    3 	 -0.4988723489328367 	 -0.5  	 0.6978242442653088 	 0.7 	 0.9004651149775883 	 0.9
    4 	 0.5961170990381723 	 0.6  	 0.30268305082478103 	 0.3 	 -1.0008808735870494 	 -1
    

It also matches well, as expected.

Now calculates `F-test` based confidence interval.


```python
from TRXASprefitpack import confidence_interval

ci_result = confidence_interval(loaded_result, 0.05) # set significant level: 0.05 -> 95% confidence level
print(ci_result) # report confidence interval
```

    [Report for Confidence Interval]
        Method: f
        Significance level:  5.000000e-02
     
    [Confidence interval]
        0.09842813 -  0.00144482 <= b'fwhm_G' <=  0.09842813 +  0.00145726
        0.49762521 -  0.00407056 <= b'tau_1' <=  0.49762521 +  0.00409321
        9.97873954 -  0.08101581 <= b'tau_2' <=  9.97873954 +  0.08201876
        1001.51104548 -  6.98048697 <= b'tau_3' <=  1001.51104548 +  7.04980524
        0.87427349 -  0.18303831 <= b'tau_osc_1' <=  0.87427349 +  0.26468352
        0.30222734 -  0.00648109 <= b'period_osc_1' <=  0.30222734 +  0.00679835
        -2.18925019 -  0.20367644 <= b'phase_osc_1' <= -2.18925019 +  0.20843286
     
    *Note*
    The confidence interval for non shared parameter especially time zeros are not calculated.
    


```python
# compare with ase
from scipy.stats import norm

factor = norm.ppf(1-0.05/2)

print('[Confidence interval (from ASE)]')
print(f"{loaded_result['x'][0]} - {factor*loaded_result['x_eps'][0]}", 
f"<= {loaded_result['param_name'][0]} <=", f"{loaded_result['x'][0]} + {factor*loaded_result['x_eps'][0]}")
print(f"{loaded_result['x'][5]} - {factor*loaded_result['x_eps'][5]}", 
f"<= {loaded_result['param_name'][5]} <=", f"{loaded_result['x'][5]} + {factor*loaded_result['x_eps'][5]}")
print(f"{loaded_result['x'][6]} - {factor*loaded_result['x_eps'][6]}", 
f"<= {loaded_result['param_name'][6]} <=", f"{loaded_result['x'][6]} + {factor*loaded_result['x_eps'][6]}")
print(f"{loaded_result['x'][7]} - {factor*loaded_result['x_eps'][7]}", 
f"<= {loaded_result['param_name'][7]} <=", f"{loaded_result['x'][7]} + {factor*loaded_result['x_eps'][7]}")
print(f"{loaded_result['x'][8]} - {factor*loaded_result['x_eps'][8]}", 
f"<= {loaded_result['param_name'][8]} <=", f"{loaded_result['x'][8]} + {factor*loaded_result['x_eps'][8]}")
print(f"{loaded_result['x'][9]} - {factor*loaded_result['x_eps'][9]}", 
f"<= {loaded_result['param_name'][9]} <=", f"{loaded_result['x'][9]} + {factor*loaded_result['x_eps'][9]}")
print(f"{loaded_result['x'][10]} - {factor*loaded_result['x_eps'][10]}", 
f"<= {loaded_result['param_name'][10]} <=", f"{loaded_result['x'][10]} + {factor*loaded_result['x_eps'][10]}")
```

    [Confidence interval (from ASE)]
    0.09842813455266139 - 0.0014328577530255604 <= b'fwhm_G' <= 0.09842813455266139 + 0.0014328577530255604
    0.4976252099062054 - 0.004087756795018895 <= b'tau_1' <= 0.4976252099062054 + 0.004087756795018895
    9.978739537439782 - 0.0825886883979751 <= b'tau_2' <= 9.978739537439782 + 0.0825886883979751
    1001.511045476416 - 7.005623680242285 <= b'tau_3' <= 1001.511045476416 + 7.005623680242285
    0.8742734928953225 - 0.24822984925225894 <= b'tau_osc_1' <= 0.8742734928953225 + 0.24822984925225894
    0.30222734002594587 - 0.005115080453947856 <= b'period_osc_1' <= 0.30222734002594587 + 0.005115080453947856
    -2.1892501947045324 - 0.168008697965791 <= b'phase_osc_1' <= -2.1892501947045324 + 0.168008697965791
    

However, as you can see, in many case, ASE does not much different from more sophisticated `f-test` based error estimation.
