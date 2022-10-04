# Fitting with time delay scan -Raise-
## Objective
1. Understand why raise model is needed
2. Fitting with raise model
3. Save and Load fitting result
4. Calculates species associated coefficent from fitting result
5. Evaluates F-test based confidence interval

In this example, we only deal with gaussian irf 


```python
# import needed module
import numpy as np
import matplotlib.pyplot as plt
import TRXASprefitpack
from TRXASprefitpack import solve_seq_model, rate_eq_conv 
plt.rcParams["figure.figsize"] = (12,9)
```

## Version information


```python
print(TRXASprefitpack.__version__)
```

    0.7.1


## Seqential Decay Dynamics with finite raise time


```python
# Generates fake experiment data
# Model: GS -> 1 -> 2 -> GS
# lifetime tau1: 200 fs (raise)
# tau2: 800 fs (decay), tau2: 10 ps (decay)
# fwhm paramter of gaussian IRF: 100 fs

tau_1 = 0.2
tau_2 = 0.8
tau_3 = 10
fwhm = 0.1

# initial condition
y0 = np.array([1, 0, 0, 0])

# set time range (mixed step)
t_seq1 = np.arange(-1, -0.5, 0.1)
t_seq2 = np.arange(-0.5, 0.5, 0.05)
t_seq3 = np.arange(0.5, 1, 0.1)
t_seq4 = np.arange(1, 10, 0.5)
t_seq5 = np.arange(10, 110, 10)

t_seq = \
    np.hstack((t_seq1, t_seq2, t_seq3, t_seq4, t_seq5))

eigval_seq, V_seq, c_seq = \
    solve_seq_model(np.array([tau_1, tau_2, tau_3]), y0)

# Now generates measured transient signal
# First and Last element is ground state

abs_1 = [0, 1, 1, 0]
abs_2 = [0, 0.5, 0.8, 0]
abs_3 = [0, -0.5, 0.7, 0]
abs_4 = [0, 0.6, 0.3, 0]

t0 = np.array([0, 0, 0, 0]) # set time zero of each scan = 0

# generate measured data

y_obs_1 = rate_eq_conv(t_seq-t0[0], fwhm, abs_1, eigval_seq, V_seq, c_seq, irf='g')
y_obs_2 = rate_eq_conv(t_seq-t0[1], fwhm, abs_2, eigval_seq, V_seq, c_seq, irf='g')
y_obs_3 = rate_eq_conv(t_seq-t0[2], fwhm, abs_3, eigval_seq, V_seq, c_seq, irf='g')
y_obs_4 = rate_eq_conv(t_seq-t0[3], fwhm, abs_4, eigval_seq, V_seq, c_seq, irf='g')

# generate random noise with (S/N = 5)

# Define noise level (S/N=5) w.r.t peak
eps_obs_1 = np.max(np.abs(y_obs_1))/5*np.ones_like(y_obs_1)
eps_obs_2 = np.max(np.abs(y_obs_2))/5*np.ones_like(y_obs_2)
eps_obs_3 = np.max(np.abs(y_obs_3))/5*np.ones_like(y_obs_3)
eps_obs_4 = np.max(np.abs(y_obs_4))/5*np.ones_like(y_obs_4)

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
for i in range(4):
    print(f't_0_{i+1}: {t0[i]}')
print('-'*24)
print('Excited Species contribution')
print(f'scan 1: {abs_1[1]} \t {abs_1[2]}')
print(f'scan 2: {abs_2[1]} \t {abs_2[2]}')
print(f'scan 3: {abs_3[1]} \t {abs_3[2]}')
print(f'scan 4: {abs_4[1]} \t {abs_4[2]}')

param_exact = [fwhm, t0[0], t0[1], t0[2], t0[3], tau_1, tau_2, tau_3]
```

    ------------------------
    fwhm: 0.1
    tau_1: 0.2
    tau_2: 0.8
    tau_3: 10
    t_0_1: 0
    t_0_2: 0
    t_0_3: 0
    t_0_4: 0
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


    
![png](Fit_Transient_Raise_files/Fit_Transient_Raise_6_0.png)
    



```python
plt.errorbar(t_seq, i_obs_1, eps_obs_1, label='1')
plt.errorbar(t_seq, i_obs_2, eps_obs_2, label='2')
plt.errorbar(t_seq, i_obs_3, eps_obs_3, label='3')
plt.errorbar(t_seq, i_obs_4, eps_obs_4, label='4')
plt.legend()
plt.xlim(-0.5, 2)
plt.show()
```


    
![png](Fit_Transient_Raise_files/Fit_Transient_Raise_7_0.png)
    


Eventhough actual time zero of each time scan are exactly zero, time zero seems to be shifted when raise time is finite (or > IRF)

## Fitting with Exponential Decay model
try to fit these curve with exponential decay model with three lifetime constant


```python
# import needed module for fitting
from TRXASprefitpack import fit_transient_exp

# time, intensity, eps should be sequence of numpy.ndarray
t = [t_seq] 
intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

# set initial guess
irf = 'g' # shape of irf function
fwhm_init = 0.2
t0_init = np.array([0, 0, 0, 0])
tau_init = np.array([0.1, 1, 8])

# use global optimization method: AMPGO
fit_result_decay = fit_transient_exp(irf, fwhm_init, t0_init, tau_init, False, 
method_glb='ampgo', t=t, intensity=intensity, eps=eps)

```


```python
print(fit_result_decay)
```


```python
# plot fitting result and experimental data

color_lst = ['red', 'blue', 'green', 'black']

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.show()
```


    
![png](Fit_Transient_Raise_files/Fit_Transient_Raise_12_0.png)
    



```python
# plot with shorter time range

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_decay['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.xlim(-1, 2)
plt.show()

```


    
![png](Fit_Transient_Raise_files/Fit_Transient_Raise_13_0.png)
    


Exponential decay model with two lifetime constant well represent modeled time scan curve with finite raise time.
However, it gives delayed time zero (actual time zero is $0$) and broadened irf (actual irf is 100 fs).
To deal with such problem we need to consider raise model.
$\exp(-t/\tau_{i+1}) - \exp(-t/\tau_1)$.

## Fitting with Raise Model
Now include, finite raise time


```python
# import needed module for fitting
from TRXASprefitpack import fit_transient_raise

# time, intensity, eps should be sequence of numpy.ndarray
t = [t_seq] 
intensity = [np.vstack((i_obs_1, i_obs_2, i_obs_3, i_obs_4)).T]
eps = [np.vstack((eps_obs_1, eps_obs_2, eps_obs_3, eps_obs_4)).T]

# set initial guess
irf = 'g' # shape of irf function
fwhm_init = 0.2
t0_init = np.array([0, 0, 0, 0])
tau_init = np.array([0.1, 1, 8])
# first tau is raise time constant

# use global optimization method: AMPGO
fit_result_raise = fit_transient_raise(irf, fwhm_init, t0_init, tau_init, False, 
method_glb='ampgo', t=t, intensity=intensity, eps=eps)
```


```python
# print fitting result
print(fit_result_raise)
```

    [Model information]
        model : raise
        irf: g
        fwhm:  0.1420
        eta:  0.0000
        base: False
     
    [Optimization Method]
        global: ampgo
        leastsq: trf
     
    [Optimization Status]
        nfev: 3423
        status: 0
        global_opt msg: Requested Number of global iteration is finished.
        leastsq_opt msg: `ftol` termination condition is satisfied.
     
    [Optimization Results]
        Total Data points: 232
        Number of effective parameters: 16
        Degree of Freedom: 216
        Chi squared:  224.0295
        Reduced chi squared:  1.0372
        AIC (Akaike Information Criterion statistic):  23.8893
        BIC (Bayesian Information Criterion statistic):  79.0371
     
    [Parameters]
        fwhm_G:  0.14200868 +/-  0.12332934 ( 86.85%)
        t_0_1_1:  0.05624516 +/-  0.05359544 ( 95.29%)
        t_0_1_2:  0.05160202 +/-  0.06867133 ( 133.08%)
        t_0_1_3: -0.02150996 +/-  0.05897826 ( 274.19%)
        t_0_1_4:  0.04161519 +/-  0.04504457 ( 108.24%)
        tau_1:  0.15228303 +/-  0.08250323 ( 54.18%)
        tau_2:  1.04184215 +/-  0.26000147 ( 24.96%)
        tau_3:  12.11109480 +/-  2.25185368 ( 18.59%)
     
    [Parameter Bound]
        fwhm_G:  0.1 <=  0.14200868 <=  0.4
        t_0_1_1: -0.4 <=  0.05624516 <=  0.4
        t_0_1_2: -0.4 <=  0.05160202 <=  0.4
        t_0_1_3: -0.4 <= -0.02150996 <=  0.4
        t_0_1_4: -0.4 <=  0.04161519 <=  0.4
        tau_1:  0.05 <=  0.15228303 <=  0.4
        tau_2:  0.4 <=  1.04184215 <=  6.4
        tau_3:  1.6 <=  12.11109480 <=  25.6
     
     
    [Component Contribution]
        DataSet dataset_1:
         #tscan	tscan_1	tscan_2	tscan_3	tscan_4
         decay 1	-10.16%	-36.02%	-64.87%	 62.83%
         decay 2	 89.84%	 63.98%	 35.13%	 37.17%
     
    [Parameter Correlation]
        Parameter Correlations >  0.1 are reported.
        (t_0_1_1, fwhm_G) =  0.528
        (t_0_1_2, fwhm_G) =  0.427
        (t_0_1_2, t_0_1_1) =  0.398
        (t_0_1_3, fwhm_G) =  0.425
        (t_0_1_3, t_0_1_1) =  0.414
        (t_0_1_3, t_0_1_2) =  0.299
        (t_0_1_4, fwhm_G) =  0.61
        (t_0_1_4, t_0_1_1) =  0.578
        (t_0_1_4, t_0_1_2) =  0.453
        (t_0_1_4, t_0_1_3) =  0.524
        (tau_1, fwhm_G) = -0.659
        (tau_1, t_0_1_1) = -0.695
        (tau_1, t_0_1_2) = -0.554
        (tau_1, t_0_1_3) = -0.591
        (tau_1, t_0_1_4) = -0.813
        (tau_2, fwhm_G) =  0.239
        (tau_2, t_0_1_1) =  0.366
        (tau_2, t_0_1_2) =  0.356
        (tau_2, t_0_1_4) =  0.346
        (tau_2, tau_1) = -0.572
        (tau_3, fwhm_G) = -0.122
        (tau_3, t_0_1_1) = -0.156
        (tau_3, t_0_1_2) = -0.124
        (tau_3, t_0_1_3) = -0.135
        (tau_3, t_0_1_4) = -0.184
        (tau_3, tau_1) =  0.338
        (tau_3, tau_2) = -0.55



```python
# plot fitting result and experimental data

color_lst = ['red', 'blue', 'green', 'black']

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_raise['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.show()


```


    
![png](Fit_Transient_Raise_files/Fit_Transient_Raise_18_0.png)
    



```python
# plot with shorter time range

for i in range(4):
    plt.errorbar(t[0], intensity[0][:, i], eps[0][:, i], label=f'expt {i+1}', color=color_lst[i])
    plt.errorbar(t[0], fit_result_raise['fit'][0][:, i], label=f'fit {i+1}', color=color_lst[i])

plt.legend()
plt.xlim(-1, 2)
plt.show()
```


    
![png](Fit_Transient_Raise_files/Fit_Transient_Raise_19_0.png)
    


1. Raise model narrowed IRF than decay model (but still much broader irf than real irf)
2. Raise model correctly estimate time zero
3. Raise model gives slightly better chi squared and AIC value


```python
# Compare fitting value and exact value
for i in range(len(fit_result_raise['x'])):
    print(f"{fit_result_raise['param_name'][i]}: {fit_result_raise['x'][i]} (fit) \t {param_exact[i]} (exact)")
```

    fwhm_G: 0.1420086781289063 (fit) 	 0.1 (exact)
    t_0_1_1: 0.05624515935759496 (fit) 	 0 (exact)
    t_0_1_2: 0.05160202235539871 (fit) 	 0 (exact)
    t_0_1_3: -0.02150995767989459 (fit) 	 0 (exact)
    t_0_1_4: 0.04161519488619747 (fit) 	 0 (exact)
    tau_1: 0.15228303252953107 (fit) 	 0.2 (exact)
    tau_2: 1.041842146894771 (fit) 	 0.8 (exact)
    tau_3: 12.111094803123384 (fit) 	 10 (exact)


Fitting result and exact result are match well.
For future use or transfer your fitting result to your collaborator or superviser, you want to save or load fitting result from file.


```python
# save fitting result to file
from TRXASprefitpack import save_TransientResult, load_TransientResult

save_TransientResult(fit_result_raise, 'example_raise') # save fitting result to example_decay_2.h5
loaded_result = load_TransientResult('example_raise') # load fitting result from example_decay_2.h5
```

Now deduce species associated difference coefficient from sequential decay model


```python
y0 = np.array([1, 0, 0, 0]) # initial cond
eigval, V, c = solve_seq_model(loaded_result['x'][5:], y0)

# compute scaled V matrix
V_scale = np.einsum('j,ij->ij', c, V)
diff_abs_fit = np.linalg.solve(V_scale[1:-1, 1:-1].T, loaded_result['c'][0]) 
# slice first last column and row corresponding to ground state

# compare with exact result
print('-'*24)
print('[Species Associated Difference Coefficent]')
print('scan # \t ex 1 (fit) \t ex 1 (exact) \t ex 2 (fit) \t ex 2 (exact)')
print(f'1 \t {diff_abs_fit[0,0]} \t {abs_1[1]}  \t {diff_abs_fit[1,0]} \t {abs_1[2]}')
print(f'2 \t {diff_abs_fit[0,1]} \t {abs_2[1]}  \t {diff_abs_fit[1,1]} \t {abs_2[2]}')
print(f'3 \t {diff_abs_fit[0,2]} \t {abs_3[1]}  \t {diff_abs_fit[1,2]} \t {abs_3[2]}')
print(f'4 \t {diff_abs_fit[0,3]} \t {abs_4[1]}  \t {diff_abs_fit[1,3]} \t {abs_4[2]}')

```

    ------------------------
    [Species Associated Difference Coefficent]
    scan # 	 ex 1 (fit) 	 ex 1 (exact) 	 ex 2 (fit) 	 ex 2 (exact)
    1 	 0.9433545132025304 	 1  	 0.9556289542374516 	 1
    2 	 0.4539620781341119 	 0.5  	 0.8085121499172329 	 0.8
    3 	 -0.4609098098236544 	 -0.5  	 0.7056147982734182 	 0.7
    4 	 0.6677648517000669 	 0.6  	 0.24794124777296256 	 0.3


It also matches well, as expected.

## Summary
Raise model would be useful when S/N of experimental data is not good.
