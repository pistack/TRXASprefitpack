Utilites
========

General Description
--------------------

Utilites of ``TRXASprefitpack`` package

* Match Utility
  1. match_scale: Match the scaling of each energy scan data to one reference time delay scan data

* Calc Utility
  1. calc_broad: broaden theoretically calculated line shape spectrum with voigt profile 
  2. calc_dads: Calculates decay associated difference spectrum from experimental energy scan and sum of exponential decay model
  3. calc_sads: Calculates species associated difference spectrum frim experimental energy scan and 1st order rate equation model

* Fit Utility
  1. fit_static: fitting theoretically calculated line shape spectrum with experimental spectrum
  2. fit_irf: Find irf parameter of experimental measured irf function
  3. fit_tscan: Find lifetime constants of experimental time trace spectrum
  4. fit_seq: fitting experimental time trace spectrum with 1st order sequential decay dynamics 
  5. fit_eq: fitting experimental time trace spectrum by 1st order rate equation matrix supplied from user
  6. fit_osc: fitting residual of experimental time trace spectrum with damped oscilliaton 

* The utilites starting from ``fit`` use `lmfit <https://dx.doi.org/10.5281/zenodo.11813>`_ package to fit data and estimate parameter error bound.
* During optimization process it uses ``Nelder-Mead Algorithm`` to find least chi square solution and then ``Levenberg-Marquardt Algorithm`` to refine such solution and estimate parameter error bound
* When ``--slow`` option is turned, it uses global optimization algorithm ``Adaptive Memory Programming for Global Optimization`` to find least chi square solution.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   match_scale
   calc_broad
   calc_dads
   calc_sads
   fit_static
   fit_irf
   fit_tscan
   fit_seq
   fit_eq
   fit_osc
   param_bound
   pseudo_voigt