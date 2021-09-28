## TRXASprefitpack: package for TRXAS pre-fitting process which aims for the first order dynamics

version:  0.4

Copyright: (C) 2021  Junho Lee (@pistack) (Email: pistatex@yonsei.ac.kr)

Licence: LGPL3

# Features
* Utilites
  * **auto_scale**: match the scaling of energy scan and time scan data
  * **broadenig**: voigt broadening your theoritical calculated line spectrum
  * **fit_static**: fitting experimental ground state spectrum using voigt broadened theoritical calculated line spectrum
  * **fit_tscan**: fitting time delay scan data with the sum of exponential decays convolved with gaussian, lorenzian(cauchy), pseudo voigt instrument response function

* libraries
  * See source documents [Docs](https://trxasprefitpack.readthedocs.io/)
  

# How to get documents for TRXASprefitpack package

* From www web
  * [Docs](https://trxasprefitpack.readthedocs.io/) are hosted in readthedocs

* From TRXASprefitpack_info utility
  * If you already installed TRXASprefitpack then just type ``TRXASprefitpack_info``
  * Otherwise, type ``python3 ./bin/TRXASprefitpack_info.py``

* From source
  * go to docs directory and type
    * for windows: ``./make.bat``
    * for mac and linux: ``make``

# How to install TRXASprefitpack package
* Easy way
  * ``pip install TRXASprefitpack``
* Advanced way (from release tar archive)
  * Downloads release tar archive
  * unpack it
  * go to TRXASprefitpack-* directory
  * Now type ``pip install .``
* Advanced way (from repository)
  * ``git clone https://github.com/pistack/TRXASprefitpack``
  * ``cd TRXASprefitpack``
  * ``python3 -m build``
  * ``cd dist``
  * unpack tar gzip file
  * go to TRXASprefitpack-* directory
  * ``pip install .``
