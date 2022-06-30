# fit_seq

fit seq: fitting tscan data using the solution of sequtial decay equation covolved with gaussian/cauchy(lorenzian)/pseudo voigt irf function.
It uses lmfit python module to fitting experimental time trace data to sequential decay module.
To find contribution of each excited state species, it solves linear least square problem via scipy lstsq module.

It supports 4 types of sequential decay
* Type 0: both raising and decay
    ``GS -> 1 -> 2 -> ... -> n -> GS``
* Type 1: no raising
    ``1 -> 2 -> ... -> n -> GS``
* Type 2: no decay
    ``GS -> 1 -> 2 -> ... -> n``
* Type 3: Neither raising nor decay
    ``1 -> 2 -> ... -> n``

```{Note}
* The number of time zero parameter should be same as the
  number of scan to fit.
* If you set shape of irf to pseudo voigt (pv), then
you should provide two full width at half maximum
value for gaussian and cauchy parts, respectively.
* type 0 sequential decay needs ``n+1`` lifetime
* type 1, 2 sequential decay needs ``n`` lifetime
* type 3 sequential decay needs ``n-1`` lifetime 
```

* usage: fit_seq.py 
                  [-h] [-sdt SEQ_DECAY_TYPE] [--irf {g,c,pv}]
                  [--fwhm_G FWHM_G] [--fwhm_L FWHM_L]
                  [-t0 TIME_ZEROS [TIME_ZEROS ...]] [-t0f TIME_ZEROS_FILE]
                  [--tau [TAU [TAU ...]]] [--fix_irf] [--slow] [-o OUT]
                  prefix


* positional arguments:
  * prefix                prefix for tscan files It will read prefix_i.txt

* optional arguments:
  * -h, --help            show this help message and exit
  * -sdt SEQ_DECAY_TYPE, --seq_decay_type SEQ_DECAY_TYPE
    * type of sequential decay 
    1. type 0: GS -> 1 -> 2 -> ... -> n -> GS (both raising and decay) 
    2. type 1: 1 -> 2 -> ... -> n -> GS (No raising) 
    3. type 2: GS -> 1 -> 2 -> ... -> n (No decay) 
    4. type 3: 1 -> 2 -> ... -> n (Neither raising nor decay) 
   * Default option is type 0 (both raising and decay)
  * --irf {g,c,pv}        shape of instrument response function 
    1. g: gaussian distribution 
    2. c: cauchy distribution 
    3. pv: pseudo voigt profile, 
     linear combination of gaussian distribution
     and cauchy distribution pv = eta*c+(1-eta)*g the
     mixing parameter is fixed according to Journal of
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
  * --tau [TAU [TAU ...]] lifetime of each decay
  * --fix_irf             fix irf parameter (fwhm_G, fwhm_L) during fitting
    process
  * --slow                use slower but robust global optimization algorithm
  * -o OUT, --out OUT     prefix for output files
