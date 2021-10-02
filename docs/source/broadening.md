# broadening

generates voigt broadened theoritical calculated lineshape spectrum

* usage: broadening [-h] [-o OUT] peak e_min e_max A fwhm_G fwhm_L peak_shift

* positional arguments:
  * peak               filename for calculated line shape spectrum
  * e_min              minimum energy
  * e_max              maximum energy
  * A                  scale factor
  * fwhm_G             Full Width at Half Maximum of gaussian shape
  * fwhm_L             Full Width at Half Maximum of lorenzian shape
  * peak_shift         discrepancy of peak position between theory and
                       experiment

* optional arguments:
  * -h, --help         show this help message and exit
  * -o OUT, --out OUT  prefix for output files

