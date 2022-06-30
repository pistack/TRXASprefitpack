# Parameter Bound scheme

## Static Spectrum fitting

* fwhm: broadening parameter
  * lower bound: 0.5*fwhm_init
  * upper bound: 2*fwhm_init

* peak_factor : parameter to shift or scale peak position
  * lower bound: 0.5*peak_factor_init
  * upper bound: 2*peak_factor_init

## Time Delay fitting

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

  * if 0.1 < tau < 1
    * lower bound: 0.05
    * upper bound: 10

  * if 1 < tau < 10
    * lowe bound: 0.5
    * upper bound: 50

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
    * lower bound: tau/4
    * upper bound: 4*tau