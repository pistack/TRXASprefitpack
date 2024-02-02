# TRXASprefitpack: package for TRXAS pre- and fitting process which aims for the first order dynamics

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

[![PyPI version](https://badge.fury.io/py/TRXASprefitpack.svg)](https://badge.fury.io/py/TRXASprefitpack)

[![Documentation Status](https://readthedocs.org/projects/trxasprefitpack/badge/?version=latest)](https://trxasprefitpack.readthedocs.io/en/latest/?badge=latest)

stable version:  0.7.3

current version: 0.7.3

current development version: 0.8.dev

Copyright: (C) 2021-2023  Junho Lee (@pistack) (Email: phistack@kaist.ac.kr)

Licence: LGPL3

## Features

### Utilites

* Match Utility
  1. match_scale: Match the scaling of each energy scan data to one reference time delay scan data
* Calc Utility
  1. calc_broad: broaden theoretically calculated line shape spectrum with voigt profile
  2. calc_dads: Calculates decay associated difference spectrum from experimental energy scan and sum of exponential decay model
  3. calc_sads: Calculates species associated difference spectrum frim experimental energy scan and 1st order rate equation model
  4. calc_dads_gui: GUI wrapper for calc_dads utility (New! in 0.7.2)
* Fit Utility
  1. fit_static: fitting sum of voigt component or voigt broadened experimental spectrum with experimental static spectrum
  2. fit_tscan: Find lifetime constants or oscillation period from experimental time delay spectrum
  3. fit_tscan_gui: GUI wrapper for fit_tscan utility (New! in 0.7.1)

### Libraries

* mathfun

  1. provides exact function for the convolution of exponential decay or exponentially damped oscillation and instrumental response function.
   There are three type of instrumental response function (gaussian, cauchy and pseudo voigt).
  2. provides factor analysis routine of time delay scan data, when time zero, lifetime constant and irf parameter (i.e. fwhm)
   are given.
  3. Solve diagonalizable 1st order rate equation exactly with arbitrary initial condition.
  4. Special fast solver for certain type (sequential decay and lower triangular rate equation) of 1st order rate equation

* res

 1. Provides scalar residual function and its gradient for 5 fitting model based on seperation scheme in least square regression.
  Such models are

    1. sum of voigt function, edge and polynomial baseline
    2. voigt broadened theoretical spectrum, edge and polynomial baseline
    3. Convolution of exponential decay and (gaussian, cauchy, pseudo voigt approximation) instrumental response function.
    4. Convolution of damped oscillation and (gaussian, cauchy, pseudo voigt approximation) instrumental response function.
    5. Sum of above two model.

* driver

 1. Provides driver routine to fit static spectrum with two model based on seperation scheme in least square regression.

    1. sum of voigt function, edge and polynomial baseline
    2. voigt broadened theoretical spectrum, edge and polynomial baseline
 
 2. Provides driver routine to fit a number of time delay scan data sets with shared lifetime paramter based on seperation scheme in least square regression.

    1. Convolution of exponential decay and (gaussian, cauchy, pseudo voigt approximation) instrumental response function.
    2. Convolution of damped oscillation and (gaussian, cauchy, pseudo voigt approximation) instrumental response function.
    3. Sum of above two model.
 
 3. Save and load fitting result through `hdf5` format

 4. Provides routine to evaluate confidence interval and compare two fit based on `f-test`.

* See source documents for stable version [Docs](https://trxasprefitpack.readthedocs.io/en/stable/)
* See source documents for latest version [Docs](https://trxasprefitpack.readthedocs.io/en/latest/)
  
## How to get documents for TRXASprefitpack package

* From www web
  * [Docs](https://trxasprefitpack.readthedocs.io/en/stable/) are hosted in readthedocs

* From source
  * go to docs directory and type
    * for windows: ``./make.bat``
    * for mac and linux: ``make``

## How to install TRXASprefitpack package

* Easy way
  * ``pip install TRXASprefitpack``
* Advanced way (from release tar archive)
  * Downloads release tar archive
  * unpack it
  * go to TRXASprefitpack-* directory
  * Now type ``pip install .``
* Advanced way (from repository)
  * ``git clone https://github.com/pistack/TRXASprefitpack.git``
  * ``git checkout v0.7.0.1``
  * ``cd TRXASprefitpack``
  * ``python3 -m build``
  * ``cd dist``
  * unpack tar gzip file
  * go to TRXASprefitpack-* directory
  * ``pip install .``

## Examples

Jupyter notebook examples for ``TRXASprefitpack`` are located in
[example](https://github.com/pistack/TRXASprefitpack-example/tree/v0.7.0)
