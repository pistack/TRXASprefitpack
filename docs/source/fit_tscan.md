# fit_tscan

fitting tscan data using sum of exponential decay covolved with gaussian/cauchy(lorenzian)/pseudo voigt irf function. It uses ``fact_anal_exp_conv`` to determine best c_i's when timezero, fwhm, and time constants are given. So, to use this script what you need to give are only timezero, fwhm, and time constants

```{Note}
* If you set shape of irf to pseudo voigt (pv), then
you should provide two full width at half maximum
value for gaussian and cauchy parts, respectively.
* If you did not set tau then it assume you finds the
timezero of this scan. So, --no_base option is discouraged.
```

* usage: fit_tscan    [-h] [--irf {g,c,pv}] [--fwhm_G FWHM_G] [--fwhm_L FWHM_L]
                     [-t0 TIME_ZEROS [TIME_ZEROS ...]] [-t0f TIME_ZEROS_FILE]
                     [--tau [TAU [TAU ...]]] [--no_base] [--slow] [-o OUT]
                     prefix

* positional arguments:
  * prefix                prefix for tscan files It will read prefix_i.txt

* optional arguments:
  * -h, --help            show this help message and exit
  * --irf {g,c,pv}        shape of instrument response function g: gaussian
                         distribution c: cauchy distribution pv: pseudo voigt
                         profile, linear combination of gaussian distribution
                         and cauchy distribution pv = eta*c+(1-eta)*g the
                         mixing parameter is guessed according to Journal of
                         Applied Crystallography. 33 (6): 1311–1316.
  * --fwhm_G FWHM_G       full width at half maximum for gaussian shape It
                         should not used when you set cauchy irf function
  * --fwhm_L FWHM_L       full width at half maximum for cauchy shape It should
                         not used when you did not set irf or use gaussian irf
                         function
  * -t0 TIME_ZEROS [TIME_ZEROS ...], --time_zeros TIME_ZEROS [TIME_ZEROS ...]
                         time zeros for each tscan
  * -t0f TIME_ZEROS_FILE, --time_zeros_file TIME_ZEROS_FILE
                         filename for time zeros of each tscan
  * --tau [TAU [TAU ...]]
                         lifetime of each component
  * --no_base             exclude baseline for fitting
  * --slow                use slower but robust global optimization algorithm
  * -o OUT, --out OUT     prefix for output files


## Parameter bound scheme

* fwhm: temporal width of x-ray pulse
  * lower bound: 0.5*fwhm_init
  * upper bound: 2*fwhm_init

* t_0: timezero for each scan
  * lower bound: t_0 - 2*fwhm_init
  * upper bound: t_0 + 2*fwhm_init

* tau: life_time of each component
  * if tau < 0.1
    * lower bound: tau/2
    * upper bound: 1

  * if 0.1 < tau < 10
    * lower bound: 0.05
    * upper bound: 100

  * if 10 < tau < 100
    * lower bound: 5
    * upper bound: 500

  * if 100 < tau < 1000
    * lower bound: 50
    * upper bound: 2000
	
  * if 1000 < tau < 5000 then
    * lower bound: 500
    * upper bound: 10000

  * if 5000 < tau < 50000 then
    * lower bound: 2500
    * upper bound: 100000

  * if 50000 < tau < 500000 then
    * lower bound: 25000
    * upper bound: 1000000

  * if 500000 < tau < 1000000 then
    * lower bound: 250000
    * upper bound: 2000000

  * if 1000000 < tau then
    * lower bound: tau/2
    * upper bound: np.inf
	 
## Mixing parameter eta

For pseudo voigt IRF function, mixing parameter eta is guessed to

\begin{equation*}
\eta = 1.36603({fwhm}_L/f)-0.47719({fwhm}_L/f)^2+0.11116({fwhm}_L/f)^3
\end{equation*}

where
\begin{align*}
f &= ({fwhm}_G^5+2.69269{fwhm}_G^4{fwhm}_L+2.42843{fwhm}_G^3{fwhm}_L^2 \\
  &+ 4.47163{fwhm}_G^2{fwhm}_L^3+0.07842{fwhm}_G{fwhm}_L^4 \\
  &+ {fwhm}_L^5)^{1/5}
\end{align*}

This guess is according to [J. Appl. Cryst. (2000). **33**, 1311-1316](https://doi.org/10.1107/S0021889800010219)


