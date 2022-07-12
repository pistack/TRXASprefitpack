# calc_broad

calc_broad: calculates voigt broadened theoritical calc spectrum

* usage: calc_broad 
                       [-h] [--scale_energy] [-o OUT]
                       peak e_min e_max e_step A fwhm_G fwhm_L peak_factor



* positional arguments:
  * peak               filename for calculated line shape spectrum
  * e_min              minimum energy
  * e_max              maximum energy
  * e_step             energy step
  * A                  scale factor
  * fwhm_G             Full Width at Half Maximum of gaussian shape
  * fwhm_L             Full Width at Half Maximum of lorenzian shape
  * peak_factor        parameter to match descrepency between thoretical
    spectrum and experimental spectrum

* optional arguments:
  * -h, --help         show this help message and exit
  * --scale_energy     Scaling the energy of peak instead of shifting to match
    experimental spectrum
  * -o OUT, --out OUT  prefix for output files

