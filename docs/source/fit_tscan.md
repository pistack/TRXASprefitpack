# fit_tscan

fit tscan: fitting experimental time trace spectrum data with the convolution of the sum of exponential decay and irf function
There are three types of irf function (gaussian, cauchy, pseudo voigt)
It uses lmfit python package to fitting time trace data and estimates error bound of irf parameter and lifetime constants.
To calculate the contribution of each life time component, it solve least linear square problem via scipy linalg lstsq module.

```{Note}
* The number of time zero parameter should be same as the
  number of scan to fit.
* If you set shape of irf to pseudo voigt (pv), then
you should provide two full width at half maximum
value for gaussian and cauchy parts, respectively.
* If you did not set tau then it assume you finds the
timezero of this scan. So, --no_base option is discouraged.
```

* usage: fit_tscan.py 
                      [-h] [--irf {g,c,pv}] [--fwhm_G FWHM_G] [--fwhm_L FWHM_L]
                      [-t0 TIME_ZEROS [TIME_ZEROS ...]] [-t0f TIME_ZEROS_FILE]
                      [--tau [TAU [TAU ...]]] [--no_base] [--fix_irf] [--slow]
                      [-o OUT]
                      prefix

* positional arguments:
  * prefix                prefix for tscan files It will read prefix_i.txt

* optional arguments:
  * -h, --help            show this help message and exit
  * --irf {g,c,pv}        shape of instrument response function 
   1. g: gaussian distribution 
   2. c: cauchy distribution 
   3. pv: pseudo voigt profile, linear combination of gaussian distribution
      and cauchy distribution pv = eta*c+(1-eta)*g the mixing parameter is fixed according to Journal of
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
  * --tau [TAU [TAU ...]] lifetime of each component
  * --no_base             exclude baseline for fitting
  * --fix_irf             fix irf parameter (fwhm_G, fwhm_L) during fitting
    process
  * --slow                use slower but robust global optimization algorithm
  * -o OUT, --out OUT     prefix for output files
