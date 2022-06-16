# fit_static

fit static: fitting static spectrum with theoretically calculated line spectrum
broadened by spectral line shape
v: voigt profile,
g: gaussian,
l: lorenzian,
It uses lmfit python package to fit experimental spectrum and estimates the error bound of
broadening and peak parameter

```{Note}
1. Currently it uses linear line to correct baseline of experimental spectrum.
2. The unit between calculated line shape spectrum and experimental spectrum should be same.
```


* usage: fit_static.py 
                       [-h] [-ls {v,g,l}] [--fwhm_G FWHM_G] [--fwhm_L FWHM_L]
                       [--no_base] [--scale_energy] [-o OUT] [--fix_fwhm_G]
                       [--fix_fwhm_L]
                       prefix num_scan peak_file peak_factor



* positional arguments:
  * prefix                prefix for experimental spectrum files It will read
    prefix_i.txt files
  * num_scan              the number of static peak scan files
  * peak_file             filename for theoretical line shape spectrum
  * peak_factor           parameter to match descrepency between thoretical
    spectrum and experimental spectrum

* optional arguments:
  * -h, --help            show this help message and exit
  * -ls {v,g,l}, --line_shape {v,g,l} line shape of spectrum 
    1. v: voigt profile 
    2. g: gaussian shape 
    3. l: lorenzian shape
  * --fwhm_G FWHM_G       full width at half maximum for gaussian shape It would
    be not used when you set lorenzian line shape
  * --fwhm_L FWHM_L       full width at half maximum for lorenzian shape It
    would be not used when you use gaussian line shape
  * --no_base             Do not include linear base line during fitting process
  * --scale_energy        Scaling the energy of peak instead of shifting peak position to
    match experimental spectrum
  * -o OUT, --out OUT     prefix for output files
  * --fix_fwhm_G          fix gaussian fwhm value
  * --fix_fwhm_L          fix lorenzian fwhm value



 

