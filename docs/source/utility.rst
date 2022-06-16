Utilites
========

General Description
--------------------

Utilites of ``TRXASprefitpack`` package

1. broadening: broad theoretically calculated line shape spectrum with voigt profile 
2. fit_static: fitting theoretically calculated line shape spectrum with experimental spectrum
3. fit_irf: Find irf parameter of experimental measured irf function
4. fit_tscan: Find lifetime constants of experimental time trace spectrum
5. fit_seq: fitting experimental time trace spectrum with 1st order sequential decay dynamics 

* The utilites starting from ``fit`` use `lmfit <https://dx.doi.org/10.5281/zenodo.11813>`_ package to fit data and estimate parameter error bound.
* During optimization process it uses ``Nelder-Mead Algorithm`` to find least chi square solution and then ``Levenberg-Marquardt Algorithm`` to refine such solution and estimate parameter error bound
* When ``--slow`` option is turned, it uses global optimization algorithm ``Adaptive Memory Programming for Global Optimization`` to find least chi square solution.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   broadening
   fit_static
   fit_irf
   fit_tscan
   fit_seq