# Fitting with time delay scan (model: exponential decay)
## Objective
1. Fitting with exponential decay model
2. Save and Load fitting result
3. Calculates species associated coefficent from fitting result
4. Evaluates F-test based confidence interval


In this example, we only deal with gaussian irf 


```python
# import needed module
import numpy as np
import matplotlib.pyplot as plt
import TRXASprefitpack
from TRXASprefitpack import solve_seq_model, solve_l_model, rate_eq_conv 
plt.rcParams["figure.figsize"] = (12,9)
```

## Version information


```python
print(TRXASprefitpack.__version__)
```

    0.6.0
    

## Fitting with exponential decay model


```python
# Generates fake experiment data
# Model: 1 -> 2 -> GS
# lifetime tau1: 500 ps, tau2: 10 ns
# fwhm paramter of gaussian IRF: 100 ps

tau_1 = 500
tau_2 = 10000
fwhm = 100

# initial condition
y0 = np.array([1, 0, 0])

# set time range (mixed step)
t_seq1 = np.arange(-2500, -500, 100)
t_seq2 = np.arange(-500, 1500, 50)
t_seq3 = np.arange(1500, 5000, 250)
t_seq4 = np.arange(5000, 50000, 2500)

t_seq = np.hstack((t_seq1, t_seq2, t_seq3, t_seq4))

eigval_seq, V_seq, c_seq = solve_seq_model(np.array([tau_1, tau_2]), y0)

# Now generates measured transient signal
# Last element is ground state

abs_1 = [1, 1, 0]
abs_2 = [0.5, 0.8, 0]
abs_3 = [-0.5, 0.7, 0]
abs_4 = [0.6, 0.3, 0]

t0 = np.random.normal(0, fwhm, 4) # perturb time zero of each scan

# generate measured data

y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')
y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')
y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')
y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')

# generate random noise with (S/N = 20)

# Define noise level (S/N=20) w.r.t peak
eps_obs_1 = np.max(np.abs(y_obs_1))/20*np.ones_like(y_obs_1)
eps_obs_2 = np.max(np.abs(y_obs_2))/20*np.ones_like(y_obs_2)
eps_obs_3 = np.max(np.abs(y_obs_3))/20*np.ones_like(y_obs_3)
eps_obs_4 = np.max(np.abs(y_obs_4))/20*np.ones_like(y_obs_4)

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
for i in range(4):
    print(f't_0_{i+1}: {t0[i]}')
print('-'*24)
print('Excited Species contribution')
print(f'scan 1: {abs_1[0]} \t {abs_1[1]}')
print(f'scan 2: {abs_2[0]} \t {abs_2[1]}')
print(f'scan 3: {abs_3[0]} \t {abs_3[1]}')
print(f'scan 4: {abs_4[0]} \t {abs_4[1]}')

param_exact = [fwhm, t0[0], t0[1], t0[2], t0[3], tau_1, tau_2]
```

    ------------------------
    fwhm: 100
    tau_1: 500
    tau_2: 10000
    t_0_1: -97.60456847864637
    t_0_2: -13.268888371466762
    t_0_3: 40.97789609363932
    t_0_4: -9.653684954801385
    ------------------------
    Excited Species contribution
    scan 1: 1 	 1
    scan 2: 0.5 	 0.8
    scan 3: -0.5 	 0.7
    scan 4: 0.6 	 0.3
    


```python
# plot model experimental data

plt.errorbar(t_seq, i_obs_1, eps_obs_1, label='1')
plt.errorbar(t_seq, i_obs_2, eps_obs_2, label='2')
plt.errorbar(t_seq, i_obs_3, eps_obs_3, label='3')
plt.errorbar(t_seq, i_obs_4, eps_obs_4, label='4')
plt.legend()
plt.show()
```


    
![png](Fit_Transient_Exp_files/Fit_Transient_Exp_6_0.png)
    



```python
# import needed module for fitting
from TRXASprefitpack import fit_transient_exp

# time, intensity, eps should be sequence of numpy.ndarray
t = [t_seq] 
intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

# set initial guess
irf = 'g' # shape of irf function
fwhm_init = 100
t0_init = np.array([0, 0, 0, 0])
# test with one decay module
tau_init = np.array([15000])

fit_result_decay_1 = fit_transient_exp(irf, fwhm_init, t0_init, tau_init, False, do_glb=True, t=t, intensity=intensity, eps=eps)

```


```python
# print fitting result
print(fit_result_decay_1)
```

    [Model information]
        model : decay
        irf: g
        fwhm:  168.0649
        eta:  0.0000
        base: False
     
    [Optimization Method]
        global: basinhopping
        leastsq: trf
     
    [Optimization Status]
        nfev: 756
        status: 0
        global_opt msg: requested number of basinhopping iterations completed successfully
        leastsq_opt msg: `ftol` termination condition is satisfied.
     
    [Optimization Results]
        Total Data points: 368
        Number of effective parameters: 10
        Degree of Freedom: 358
        Chi squared:  2990.0676
        Reduced chi squared:  8.3521
        AIC (Akaike Information Criterion statistic):  790.9484
        BIC (Bayesian Information Criterion statistic):  830.0292
     
    [Parameters]
        fwhm_G:  168.06492275 +/-  30.75123682 ( 18.30%)
        t_0_1_1: -98.56399616 +/-  16.35037089 ( 16.59%)
        t_0_1_2:  39.93477544 +/-  16.00856019 ( 40.09%)
        t_0_1_3:  200.00000000 +/-  19.62321338 ( 9.81%)
        t_0_1_4: -53.88607467 +/-  24.55942610 ( 45.58%)
        tau_1:  13081.76170043 +/-  944.37072913 ( 7.22%)
     
    [Parameter Bound]
        fwhm_G:  50 <=  168.06492275 <=  200
        t_0_1_1: -200 <= -98.56399616 <=  200
        t_0_1_2: -200 <=  39.93477544 <=  200
        t_0_1_3: -200 <=  200.00000000 <=  200
        t_0_1_4: -200 <= -53.88607467 <=  200
        tau_1:  3200 <=  13081.76170043 <=  51200
     
    [Component Contribution]
        DataSet dataset_1:
         #tscan	tscan_1	tscan_2	tscan_3	tscan_4
         decay 1	 100.00%	 100.00%	 100.00%	 100.00%
     
    [Parameter Correlation]
        Parameter Correlations >  0.1 are reported.
        (tau_1, fwhm_G) = -0.1
    


```python
# plot fitting result and experimental data

color_lst = ['red', 'blue', 'green', 'black']

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay_1['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.show()
```


    
![png](Fit_Transient_Exp_files/Fit_Transient_Exp_9_0.png)
    


For scan 1 and 2, experimental data and fitting data match well. However for scan 3 and 4, they do not match at shorter time region (< 10000).


```python
# plot with shorter time range

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay_1['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.xlim(-10*fwhm_init, 20*fwhm_init)
plt.show()

```


    
![png](Fit_Transient_Exp_files/Fit_Transient_Exp_11_0.png)
    


There may exists shorter lifetime component.


```python
# try with double exponential decay
# set initial guess
irf = 'g' # shape of irf function
fwhm_init = 100
t0_init = np.array([0, 0, 0, 0])
# test with two decay module
tau_init = np.array([300, 15000])

fit_result_decay_2 = fit_transient_exp(irf, fwhm_init, t0_init, tau_init, False, do_glb=True, t=t, intensity=intensity, eps=eps)

```


```python
# print fitting result
print(fit_result_decay_2)
```

    [Model information]
        model : decay
        irf: g
        fwhm:  99.4390
        eta:  0.0000
        base: False
     
    [Optimization Method]
        global: basinhopping
        leastsq: trf
     
    [Optimization Status]
        nfev: 1062
        status: 0
        global_opt msg: requested number of basinhopping iterations completed successfully
        leastsq_opt msg: `ftol` termination condition is satisfied.
     
    [Optimization Results]
        Total Data points: 368
        Number of effective parameters: 15
        Degree of Freedom: 353
        Chi squared:  387.9992
        Reduced chi squared:  1.0991
        AIC (Akaike Information Criterion statistic):  49.4747
        BIC (Bayesian Information Criterion statistic):  108.096
     
    [Parameters]
        fwhm_G:  99.43896773 +/-  8.93158710 ( 8.98%)
        t_0_1_1: -104.67574624 +/-  5.14656575 ( 4.92%)
        t_0_1_2: -12.02608173 +/-  7.73683384 ( 64.33%)
        t_0_1_3:  41.85381825 +/-  5.93597577 ( 14.18%)
        t_0_1_4: -12.98139437 +/-  4.63080999 ( 35.67%)
        tau_1:  460.33899330 +/-  18.11965013 ( 3.94%)
        tau_2:  10472.66755999 +/-  303.65911732 ( 2.90%)
     
    [Parameter Bound]
        fwhm_G:  50 <=  99.43896773 <=  200
        t_0_1_1: -200 <= -104.67574624 <=  200
        t_0_1_2: -200 <= -12.02608173 <=  200
        t_0_1_3: -200 <=  41.85381825 <=  200
        t_0_1_4: -200 <= -12.98139437 <=  200
        tau_1:  50 <=  460.33899330 <=  800
        tau_2:  3200 <=  10472.66755999 <=  51200
     
    [Component Contribution]
        DataSet dataset_1:
         #tscan	tscan_1	tscan_2	tscan_3	tscan_4
         decay 1	-7.66%	-30.84%	-63.50%	 46.66%
         decay 2	 92.34%	 69.16%	 36.50%	 53.34%
     
    [Parameter Correlation]
        Parameter Correlations >  0.1 are reported.
        (tau_1, fwhm_G) = -0.175
        (tau_1, t_0_1_3) = -0.354
        (tau_1, t_0_1_4) = -0.119
        (tau_2, tau_1) = -0.351
    


```python
# plot fitting result and experimental data

color_lst = ['red', 'blue', 'green', 'black']

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay_2['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.show()


```


    
![png](Fit_Transient_Exp_files/Fit_Transient_Exp_15_0.png)
    



```python
# plot with shorter time range

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay_2['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.xlim(-10*fwhm_init, 20*fwhm_init)
plt.show()
```


    
![png](Fit_Transient_Exp_files/Fit_Transient_Exp_16_0.png)
    


Two decay model fits well


```python
# Compare fitting value and exact value
for i in range(len(fit_result_decay_2['x'])):
    print(f"{fit_result_decay_2['param_name'][i]}: {fit_result_decay_2['x'][i]} (fit) \t {param_exact[i]} (exact)")
```

    fwhm_G: 99.43896773041614 (fit) 	 100 (exact)
    t_0_1_1: -104.67574623898551 (fit) 	 -97.60456847864637 (exact)
    t_0_1_2: -12.026081726022252 (fit) 	 -13.268888371466762 (exact)
    t_0_1_3: 41.853818252949026 (fit) 	 40.97789609363932 (exact)
    t_0_1_4: -12.981394369806363 (fit) 	 -9.653684954801385 (exact)
    tau_1: 460.33899329544414 (fit) 	 500 (exact)
    tau_2: 10472.667559988056 (fit) 	 10000 (exact)
    

Fitting result and exact result are match well.
For future use or transfer your fitting result to your collaborator or superviser, you want to save or load fitting result from file.


```python
# save fitting result to file
from TRXASprefitpack import save_TransientResult, load_TransientResult

save_TransientResult(fit_result_decay_2, 'example_decay_2') # save fitting result to example_decay_2.h5
loaded_result = load_TransientResult('example_decay_2') # load fitting result from example_decay_2.h5
```

Now deduce species associated difference coefficient from sequential decay model


```python
y0 = np.array([1, 0, 0]) # initial cond
eigval, V, c = solve_seq_model(loaded_result['x'][5:], y0)

# compute scaled V matrix
V_scale = np.einsum('j,ij->ij', c, V)
diff_abs_fit = np.linalg.solve(V_scale[:-1, :-1].T, loaded_result['c'][0]) # slice last column and row corresponding to ground state

# compare with exact result
print('-'*24)
print('[Species Associated Difference Coefficent]')
print('scan # \t ex 1 (fit) \t ex 1 (exact) \t ex 2 (fit) \t ex 2 (exact)')
print(f'1 \t {diff_abs_fit[0,0]} \t {abs_1[0]}  \t {diff_abs_fit[1,0]} \t {abs_1[1]}')
print(f'2 \t {diff_abs_fit[0,1]} \t {abs_2[0]}  \t {diff_abs_fit[1,1]} \t {abs_2[1]}')
print(f'3 \t {diff_abs_fit[0,2]} \t {abs_3[0]}  \t {diff_abs_fit[1,2]} \t {abs_3[1]}')
print(f'4 \t {diff_abs_fit[0,3]} \t {abs_4[0]}  \t {diff_abs_fit[1,3]} \t {abs_4[1]}')

```

    ------------------------
    [Species Associated Difference Coefficent]
    scan # 	 ex 1 (fit) 	 ex 1 (exact) 	 ex 2 (fit) 	 ex 2 (exact)
    1 	 0.969670748534202 	 1  	 1.010961092288663 	 1
    2 	 0.464994539130646 	 0.5  	 0.8022361205798151 	 0.8
    3 	 -0.5240334986515041 	 -0.5  	 0.6770218389372332 	 0.7
    4 	 0.5847140147893795 	 0.6  	 0.2981901320860125 	 0.3
    

It also matches well, as expected.

The error of paramter reported from `Transient` Driver is based on Asymptotic Standard Error.
However, strictly, ASE cannot be used in non-linear regression.
TRXASprefitpack provides alternative error estimation based on `F-test`.


```python
from TRXASprefitpack import confidence_interval

ci_result = confidence_interval(loaded_result, 0.05) # set significant level: 0.05 -> 95% confidence level
print(ci_result) # report confidence interval
```

    [Report for Confidence Interval]
        Method: f
        Significance level:  5.000000e-02
     
    [Confidence interval]
        99.43896773 -  19.2786607 <= b'fwhm_G' <=  99.43896773 +  20.51533343
        460.3389933 -  35.11919538 <= b'tau_1' <=  460.3389933 +  37.45901052
        10472.66755999 -  581.33251985 <= b'tau_2' <=  10472.66755999 +  605.17046763
     
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
```

    [Confidence interval (from ASE)]
    99.43896773041614 - 17.505589041046978 <= b'fwhm_G' <= 99.43896773041614 + 17.505589041046978
    460.33899329544414 - 35.513861671045596 <= b'tau_1' <= 460.33899329544414 + 35.513861671045596
    10472.667559988056 - 595.1609335161114 <= b'tau_2' <= 10472.667559988056 + 595.1609335161114
    

However, as you can see, in many case, ASE does not much different from more sophisticated `f-test` based error estimation.
