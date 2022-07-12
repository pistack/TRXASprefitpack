# calc_sads

calc sads: Calculate species associated difference spectrum from experimental energy scan data and
the convolution of lower triangular 1st order rate equation model and instrumental response function

* usage: calc_sads
                    [-h] [-re_mat RATE_EQ_MAT] [--seq] [-gsi GS_INDEX] [--irf {g,c,pv}] [--fwhm_G FWHM_G] [--fwhm_L FWHM_L] [-t0 TIME_ZERO] [--escan_time ESCAN_TIME [ESCAN_TIME ...]]
                    [--tau TAU [TAU ...]] [-o OUT]
                    escan_file escan_err_file


```{Note}
In rate equation model, the ground state would be
1. ''first_and_last'' species
2. ''first'' species
3. ''last'' species
4. ground state is not included in the rate equation model
```

```{Note}
1. Associated difference spectrum of ground state species is zero.
2. The rate equation matrix shoule be lower triangular.
3. Rate equation matrix for sequential decay model is sparse, so if your model is sequential decay then use --seq option
4. if you set shape of irf to pseudo voigt (pv), then you should provide two full width at half maximum value for gaussian and cauchy parts, respectively.
```

* positional arguments:
  * escan_file            filename for scale corrected energy scan file
  * escan_err_file        filename for the scaled estimated experimental error of energy scan file

* optional arguments:
  * -h, --help            show this help message and exit
  * -re_mat RATE_EQ_MAT, --rate_eq_mat RATE_EQ_MAT
                        Filename for user supplied rate equation matrix. 
                        ``i`` th rate constant should be denoted by ``ki`` in rate equation matrix file.
                        Moreover rate equation matrix should be lower triangular.
  * --seq
                        Use sequential decay dynamics instead of more general lower triangular one.
                        If this option is turned on, it use following sequential decay dynamics
                        1 -> 2 -> 3 -> ... -> last
                        You can control the behavior of first and last species via --gsi option
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
  * -t0 TIME_ZERO, --time_zero TIME_ZERO
                        time zero of energy scan
  * --escan_time ESCAN_TIME [ESCAN_TIME ...]
                        time delay for each energy scan
  * --tau TAU [TAU ...]   lifetime of each decay path
  * -o OUT, --out OUT     prefix for output files
