# TRXASprefitpack: package for TRXAS pre-fitting process which aims for the first order dynamics

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

[![PyPI version](https://badge.fury.io/py/TRXASprefitpack.svg)](https://badge.fury.io/py/TRXASprefitpack)

[![Total alerts](https://img.shields.io/lgtm/alerts/g/pistack/TRXASprefitpack.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/pistack/TRXASprefitpack/alerts/)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/pistack/TRXASprefitpack.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/pistack/TRXASprefitpack/context:python)

[![Documentation Status](https://readthedocs.org/projects/trxasprefitpack/badge/?version=latest)](https://trxasprefitpack.readthedocs.io/en/latest/?badge=latest)

stable version:  0.5.1.1

Copyright: (C) 2021-2022  Junho Lee (@pistack) (Email: pistack@yonsei.ac.kr)

Licence: LGPL3

## Features

### Utilites

* Match Utility
  1. match_scale: Match the scaling of each energy scan data to one reference time delay scan data
* Calc Utility
  1. calc_broad: broaden theoretically calculated line shape spectrum with voigt profile
  2. calc_dads: Calculates decay associated difference spectrum from experimental energy scan and sum of exponential decay model
  3. calc_sads: Calculates species associated difference spectrum frim experimental energy scan and 1st order rate equation model
* Fit Utility
  1. fit_static: fitting sum of voigt component or voigt broadened experimental spectrum with experimental static spectrum
  2. fit_tscan: Find lifetime constants or oscillation period from experimental time delay spectrum

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
  * ``git checkout v0.5.1.1``
  * ``cd TRXASprefitpack``
  * ``python3 -m build``
  * ``cd dist``
  * unpack tar gzip file
  * go to TRXASprefitpack-* directory
  * ``pip install .``

## Examples

Jupyter notebook examples for ``TRXASprefitpack`` are located in
[example](https://github.com/pistack/TRXASprefitpack-example/tree/v0.5.1)
