# calc_dads

calc dads: Calculate decay associated difference spectrum from experimental energy scan data and
the convolution of sum of exponential decay and instrumental response function

* usage: calc_dads 
                    [-h] [--irf {g,c,pv}] [--fwhm_G FWHM_G] [--fwhm_L FWHM_L] [-t0 TIME_ZERO] [--escan_time ESCAN_TIME [ESCAN_TIME ...]] [--tau TAU [TAU ...]] [--no_base] [-o OUT]
                    escan_file escan_err_file

```{Note}
1. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
```

* positional arguments:
  * escan_file            filename for scale corrected energy scan file
  * escan_err_file        filename for the scaled estimated experimental error of energy scan file

* optional arguments:
  * -h, --help            show this help message and exit
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
  * -t0 TIME_ZERO, --time_zero TIME_ZERO
                        time zero of energy scan
  * --escan_time ESCAN_TIME [ESCAN_TIME ...]
                        time delay for each energy scan
  * --tau TAU [TAU ...]   lifetime of each decay component
  * --no_base             Exclude baseline (i.e. very long lifetime component)
  * -o OUT, --out OUT     prefix for output files