# TRXASprefitpack: package for TRXAS pre-fitting process which aims for the first order dynamics

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

[![PyPI version](https://badge.fury.io/py/TRXASprefitpack.svg)](https://badge.fury.io/py/TRXASprefitpack)

[![Total alerts](https://img.shields.io/lgtm/alerts/g/pistack/TRXASprefitpack.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/pistack/TRXASprefitpack/alerts/)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/pistack/TRXASprefitpack.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/pistack/TRXASprefitpack/context:python)

[![Documentation Status](https://readthedocs.org/projects/trxasprefitpack/badge/?version=latest)](https://trxasprefitpack.readthedocs.io/en/latest/?badge=latest)

stable version:  0.5.1

Copyright: (C) 2021-2022  Junho Lee (@pistack) (Email: pistack@yonsei.ac.kr)

Licence: LGPL3

## Features

* Utilites

1. broadening: broaden theoretically calculated line shape spectrum with voigt profile 
2. fit_static: fitting theoretically calculated line shape spectrum with experimental spectrum
3. fit_irf: Find irf parameter of experimental measured irf function
4. fit_tscan: Find lifetime constants of experimental time trace spectrum
5. fit_seq: fitting experimental time trace spectrum with 1st order sequential decay dynamics 
6. fit_eq: fitting experimental time trace spectrum by 1st order rate equation matrix supplied from user
7. fit_osc: fitting residual of experimental time trace spectrum with damped oscilliaton 

* libraries
  1. Compute convolution of exponential decay and gaussian, cauchy function analytically.
  2. Compute convolition of damped oscilliation and gaussian, cauchy function analytically.
  3. Voigt broadening line shape spectrum
  4. Solve first order rate equation (general, lower triangle, sequential decay) and analytically compute the convolution of rate equation model solution and gaussian, cauchy function.
  5. Find best coefficients for linear combination of exponential decay or damped oscilliation via solving linear least square problem.
  * See source documents [Docs](https://trxasprefitpack.readthedocs.io/en/stable/)
  
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
  * ``git checkout v0.5.1``
  * ``cd TRXASprefitpack``
  * ``python3 -m build``
  * ``cd dist``
  * unpack tar gzip file
  * go to TRXASprefitpack-* directory
  * ``pip install .``

## Examples

Jupyter notebook examples for ``TRXASprefitpack`` are located in
[example](https://github.com/pistack/TRXASprefitpack-example/tree/v0.5.1)
