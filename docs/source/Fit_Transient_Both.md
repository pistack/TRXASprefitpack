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
    t_0_1: 0.023319938080473612
    t_0_2: 0.022808896228266443
    t_0_3: -0.023474178102756708
    t_0_4: -0.06616291848757526
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
        fwhm:  0.1001
        eta:  0.0000
        base: False
     
    [Optimization Method]
        global: basinhopping
        leastsq: trf
     
    [Optimization Status]
        nfev: 3516
        status: 0
        global_opt msg: requested number of basinhopping iterations completed successfully
        leastsq_opt msg: `ftol` termination condition is satisfied.
     
    [Optimization Results]
        Total Data points: 600
        Number of effective parameters: 20
        Degree of Freedom: 580
        Chi squared:  1026.0937
        Reduced chi squared:  1.7691
        AIC (Akaike Information Criterion statistic):  361.9508
        BIC (Bayesian Information Criterion statistic):  449.8894
     
    [Parameters]
        fwhm_G:  0.10007463 +/-  0.00089776 ( 0.90%)
        t_0_1_1:  0.02231415 +/-  0.00042144 ( 1.89%)
        t_0_1_2:  0.02226138 +/-  0.00060401 ( 2.71%)
        t_0_1_3: -0.02382116 +/-  0.00074505 ( 3.13%)
        t_0_1_4: -0.06663829 +/-  0.00064808 ( 0.97%)
        tau_1:  0.49965492 +/-  0.00262923 ( 0.53%)
        tau_2:  9.99814026 +/-  0.05342504 ( 0.53%)
        tau_3:  1004.29529145 +/-  4.69763846 ( 0.47%)
     
    [Parameter Bound]
        fwhm_G:  0.075 <=  0.10007463 <=  0.3
        t_0_1_1: -0.3 <=  0.02231415 <=  0.3
        t_0_1_2: -0.3 <=  0.02226138 <=  0.3
        t_0_1_3: -0.3 <= -0.02382116 <=  0.3
        t_0_1_4: -0.3 <= -0.06663829 <=  0.3
        tau_1:  0.075 <=  0.49965492 <=  1.2
        tau_2:  4.8 <=  9.99814026 <=  76.8
        tau_3:  307.2 <=  1004.29529145 <=  4915.2
     
    [Component Contribution]
        DataSet dataset_1:
         #tscan	tscan_1	tscan_2	tscan_3	tscan_4
         decay 1	-1.74%	-28.82%	-51.39%	 8.75%
         decay 2	-0.20%	 53.96%	-9.45%	 52.68%
         decay 3	 98.06%	 17.22%	 39.16%	-38.57%
     
    [Parameter Correlation]
        Parameter Correlations >  0.1 are reported.
        (t_0_1_1, fwhm_G) =  0.105
        (t_0_1_2, fwhm_G) =  0.102
        (tau_1, t_0_1_2) =  0.121
        (tau_1, t_0_1_3) = -0.345
        (tau_2, t_0_1_2) =  0.1
        (tau_2, t_0_1_4) =  0.133
        (tau_3, tau_2) = -0.164



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
        fwhm:  0.0982
        eta:  0.0000
        base: False
     
    [Optimization Method]
        global: basinhopping
        leastsq: trf
     
    [Optimization Status]
        nfev: 16200
        status: 0
        global_opt msg: requested number of basinhopping iterations completed successfully
        leastsq_opt msg: `ftol` termination condition is satisfied.
     
    [Optimization Results]
        Total Data points: 600
        Number of effective parameters: 27
        Degree of Freedom: 573
        Chi squared:  568.9827
        Reduced chi squared:  0.993
        AIC (Akaike Information Criterion statistic):  22.1522
        BIC (Bayesian Information Criterion statistic):  140.8693
     
    [Parameters]
        fwhm_G:  0.09817267 +/-  0.00070595 ( 0.72%)
        t_0_1_1:  0.02213832 +/-  0.00039378 ( 1.78%)
        t_0_1_2:  0.02212862 +/-  0.00044881 ( 2.03%)
        t_0_1_3: -0.02382946 +/-  0.00055378 ( 2.32%)
        t_0_1_4: -0.06671371 +/-  0.00048032 ( 0.72%)
        tau_1:  0.50024992 +/-  0.00202120 ( 0.40%)
        tau_2:  9.99079667 +/-  0.04010556 ( 0.40%)
        tau_3:  1003.42985657 +/-  3.51461165 ( 0.35%)
        tau_osc_1:  0.86388391 +/-  0.13013780 ( 15.06%)
        period_osc_1:  0.30468441 +/-  0.00268993 ( 0.88%)
        phase_osc_1: -2.18880168 +/-  0.08419389 ( 3.85%)
     
    [Parameter Bound]
        fwhm_G:  0.075 <=  0.09817267 <=  0.3
        t_0_1_1: -0.3 <=  0.02213832 <=  0.3
        t_0_1_2: -0.3 <=  0.02212862 <=  0.3
        t_0_1_3: -0.3 <= -0.02382946 <=  0.3
        t_0_1_4: -0.3 <= -0.06671371 <=  0.3
        tau_1:  0.075 <=  0.50024992 <=  1.2
        tau_2:  4.8 <=  9.99079667 <=  76.8
        tau_3:  307.2 <=  1003.42985657 <=  4915.2
        tau_osc_1:  0.3 <=  0.86388391 <=  4.8
        period_osc_1:  0.075 <=  0.30468441 <=  1.2
        phase_osc_1: -3.14159265 <= -2.18880168 <=  3.14159265
     
    [Component Contribution]
        DataSet dataset_1:
         #tscan	tscan_1	tscan_2	tscan_3	tscan_4
         decay 1	-0.40%	-28.85%	-51.36%	 8.69%
         decay 2	-0.68%	 53.89%	-9.45%	 52.70%
         decay 3	 93.83%	 17.19%	 39.18%	-38.57%
        dmp_osc 1	-5.09%	-0.06%	 0.01%	 0.04%
     
    [Parameter Correlation]
        Parameter Correlations >  0.1 are reported.
        (t_0_1_1, fwhm_G) =  0.254
        (t_0_1_2, fwhm_G) =  0.111
        (tau_1, t_0_1_2) =  0.123
        (tau_1, t_0_1_3) = -0.348
        (tau_2, t_0_1_4) =  0.131
        (tau_3, tau_2) = -0.164
        (tau_osc_1, fwhm_G) =  0.123
        (period_osc_1, fwhm_G) = -0.235
        (period_osc_1, t_0_1_1) = -0.52
        (phase_osc_1, fwhm_G) = -0.298
        (phase_osc_1, t_0_1_1) = -0.592
        (phase_osc_1, period_osc_1) =  0.838



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

    fwhm_G: 0.09817267448566006 (fit) 	 0.1 (exact)
    t_0_1_1: 0.02213831760155054 (fit) 	 0.023319938080473612 (exact)
    t_0_1_2: 0.02212861717160382 (fit) 	 0.022808896228266443 (exact)
    t_0_1_3: -0.023829456447629297 (fit) 	 -0.023474178102756708 (exact)
    t_0_1_4: -0.06671371415887084 (fit) 	 -0.06616291848757526 (exact)
    tau_1: 0.5002499230390485 (fit) 	 0.5 (exact)
    tau_2: 9.990796671170218 (fit) 	 10 (exact)
    tau_3: 1003.4298565703303 (fit) 	 1000 (exact)
    tau_osc_1: 0.8638839106790563 (fit) 	 1 (exact)
    period_osc_1: 0.3046844059272981 (fit) 	 0.3 (exact)
    phase_osc_1: -2.1888016822078504 (fit) 	 0.7853981633974483 (exact)


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
    1 	 0.9964197294868353 	 1  	 1.0006264175714779 	 1 	 0.9974852126182684 	 1
    2 	 0.4952242997097968 	 0.5  	 0.8017524559863427 	 0.8 	 0.19948994139714366 	 0.2
    3 	 -0.5020454807241927 	 -0.5  	 0.7002431770151698 	 0.7 	 0.8994776774561238 	 0.9
    4 	 0.5979286958144661 	 0.6  	 0.30152861842772394 	 0.3 	 -1.0003053038966574 	 -1


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
        0.09817267 -  0.00137419 <= b'fwhm_G' <=  0.09817267 +  0.00138019
        0.02213832 -  0.00077537 <= b't_0_1_1' <=  0.02213832 +  0.00077968
        0.02212862 -  0.00088784 <= b't_0_1_2' <=  0.02212862 +  0.00088411
        -0.02382946 -  0.00109041 <= b't_0_1_3' <= -0.02382946 +  0.00109126
        -0.06671371 -  0.00094465 <= b't_0_1_4' <= -0.06671371 +  0.00094325
        0.50024992 -  0.00393884 <= b'tau_1' <=  0.50024992 +  0.00396466
        9.99079667 -  0.07734601 <= b'tau_2' <=  9.99079667 +  0.078199
        1003.42985657 -  6.87202608 <= b'tau_3' <=  1003.42985657 +  6.93904522
        0.86388391 -  0.21114474 <= b'tau_osc_1' <=  0.86388391 +  0.32329336
        0.30468441 -  0.00550132 <= b'period_osc_1' <=  0.30468441 +  0.00552368
        -2.18880168 -  0.16792602 <= b'phase_osc_1' <= -2.18880168 +  0.17068111



```python
# compare with ase
from scipy.stats import norm

factor = norm.ppf(1-0.05/2)

print('[Confidence interval (from ASE)]')
for i in range(loaded_result['param_name'].size):
    print(f"{loaded_result['x'][i]: .8f} - {factor*loaded_result['x_eps'][i] :.8f}", 
          f"<= {loaded_result['param_name'][i]} <=", f"{loaded_result['x'][i] :.8f} + {factor*loaded_result['x_eps'][i]: .8f}")
```

    [Confidence interval (from ASE)]
     0.09817267 - 0.00138365 <= b'fwhm_G' <= 0.09817267 +  0.00138365
     0.02213832 - 0.00077180 <= b't_0_1_1' <= 0.02213832 +  0.00077180
     0.02212862 - 0.00087965 <= b't_0_1_2' <= 0.02212862 +  0.00087965
    -0.02382946 - 0.00108539 <= b't_0_1_3' <= -0.02382946 +  0.00108539
    -0.06671371 - 0.00094141 <= b't_0_1_4' <= -0.06671371 +  0.00094141
     0.50024992 - 0.00396148 <= b'tau_1' <= 0.50024992 +  0.00396148
     9.99079667 - 0.07860544 <= b'tau_2' <= 9.99079667 +  0.07860544
     1003.42985657 - 6.88851226 <= b'tau_3' <= 1003.42985657 +  6.88851226
     0.86388391 - 0.25506541 <= b'tau_osc_1' <= 0.86388391 +  0.25506541
     0.30468441 - 0.00527217 <= b'period_osc_1' <= 0.30468441 +  0.00527217
    -2.18880168 - 0.16501699 <= b'phase_osc_1' <= -2.18880168 +  0.16501699


However, as you can see, in many case, ASE does not much different from more sophisticated `f-test` based error estimation.
