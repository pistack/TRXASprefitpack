# fit_eq

fit eq: fitting tscan data using the solution of lower triangular 1st order rate equation covolved with gaussian/cauchy(lorenzian)/pseudo voigt irf function.
It uses lmfit python module to fitting experimental time trace data to sequential decay module.
To find contribution of each excited state species, it solves linear least square problem via scipy lstsq module.

* usage: fit_eq.py 
                 [-h] [-re_mat RATE_EQ_MAT] [-gsi GS_INDEX] [--irf {g,c,pv}]
                 [--fwhm_G FWHM_G] [--fwhm_L FWHM_L]
                 [-t0 TIME_ZEROS [TIME_ZEROS ...]] [-t0f TIME_ZEROS_FILE]
                 [--tau [TAU [TAU ...]]] [--fix_irf] [--slow] [-o OUT]
                 prefix

```{Note}
In rate equation model, the ground state would be
1. ''first_and_last'' species
2. ''first'' species
3. ''last'' species
4. ground state is not included in the rate equation model
```

```{Note}
1. The number of time zero parameter should be same as the number of scan to fit.
2. The rate equation matrix shoule be lower triangular.
3. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
```

* positional arguments:
  * prefix                prefix for tscan files It will read prefix_i.txt

* optional arguments:
  * -h, --help            show this help message and exit
  * -re_mat RATE_EQ_MAT, --rate_eq_mat RATE_EQ_MAT
                        Filename for user supplied rate equation matrix. 
                        ``i`` th rate constant should be denoted by ``ki`` in rate equation matrix file.
                        Moreover rate equation matrix should be lower triangular.
  * -gsi GS_INDEX, --gs_index GS_INDEX
    * Index of ground state species.
    1. ``first_and_last``, first and last species are both ground state
    2. ``first``, first species is ground state
    3. ``last``,  last species is ground state
    4. Did not set., There is no ground state species in model rate equation.
  * --irf {g,c,pv}        
    * shape of instrument response functon
    1. g: gaussian distribution
    2. c: cauchy distribution
    3. pv: pseudo voigt profile, linear combination of gaussian distribution and cauchy distribution pv = eta*c+(1-eta)*g 
       the mixing parameter is fixed according to Journal of Applied Crystallography. 33 (6): 1311–1316. 
  * --fwhm_G FWHM_G       
                        full width at half maximum for gaussian shape
                        It would not be used when you set cauchy irf function
  * --fwhm_L FWHM_L       
                        full width at half maximum for cauchy shape
                        It would not be used when you did not set irf or use gaussian irf function
  * -t0 TIME_ZEROS [TIME_ZEROS ...], --time_zeros TIME_ZEROS [TIME_ZEROS ...]
                        time zeros for each tscan
  * -t0f TIME_ZEROS_FILE, --time_zeros_file TIME_ZEROS_FILE
                        filename for time zeros of each tscan
  * --tau [TAU [TAU ...]]
                        lifetime of each decay path
  * --fix_irf             fix irf parameter (fwhm_G, fwhm_L) during fitting process
  * --slow                use slower but robust global optimization algorithm
  * -o OUT, --out OUT     prefix for output files


## Format of Rate Equation Matrix 

User supplied rate equation matrix should have following format.

### Example 1. Sequential Decay

```python
A -> B -> C -> D
```

\begin{matrix}
-1*k1 & 0 & 0 & 0 \\
1*k1 & -1*k2 & 0 & 0 \\
0 & 1*k2 & -1*k3 & 0 \\
0 & 0 & 1*k3 & 0
\end{matrix}

``k1, k2, k3`` mean rate constant of each decay, inverse of lifetime ``tau1, tau2, tau3``.

### Example 2. Branched Decay

```python
A -> B -> D
  -> C -> D
```

\begin{matrix}
-(1*k1+1*k2) & 0 & 0 & 0 \\
1*k1 & -1*k3 & 0 & 0 \\
1*k2 & 0 & -1*k4 & 0 \\
0 & 1*k3 & 1*k4 & 0
\end{matrix}

Note that the eigenvalues of Branched Decay are $-(k_1+k_2),\ -k_3,\ -k_4,\ 0$. That is the model signal deduced by branched decay is represented by sum of four exponential decay component $\{\exp\left(-(k_1+k_2)t\right), \exp\left(-k_3t\right), \exp\left(-k_4t\right), 1\}$. Therefore time scan fitting does not distinguish branched decay and sequential decay with rate constant $k'_1 = k_1+k_2$, $k'_2 = k_3$, $k'_3 = k_4$ and $k'_4 = 0$. Because of this, in time scan fitting, $k_1$ and $k_2$ in branched decay are highly correlated so that one cannot directly determine ratio between $k_1$ and $k_2$.

However if one knows difference absorption coefficient of each species or ratio of $k_1$ and $k_2$ from other experiments, they want to fix $k_2=r*k_1$ then user can give following rate equation matrix.

For example $r=0.8$,
\begin{matrix}
-(1*k1+0.8*k1) & 0 & 0 & 0 \\
1*k1 & -1*k2 & 0 & 0 \\
0.8*k1 & 0 & -1*k3 & 0 \\
0 & 1*k2 & 1*k3 & 0
\end{matrix}

