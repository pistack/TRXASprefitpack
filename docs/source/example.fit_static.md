# fit_static Basic Example

Basic usage example ``fit_static`` utility.
Yon can find example file from [TRXASprefitpack-example](https://github.com/pistack/TRXASprefitpack-example/tree/v0.5.1) fit_static subdirectory.

## Peak position shifting

1. Go to shift subdirectory. In shift subdirectory, you can find ``example_calc_peak.txt`` and ``example_1.txt`` files. First one is simplified version of thoretically calculated x-ray absorption spectrum peak and its intensity. Last one is the fake experimental x-ray absorption spectrum.
2. Type ``fit_static -h``. Then it prints help message. You can find detailed description of arguments in the utility section of this document.
3. Type ``fit_staitc example 1 example_calc_peak.txt -1.25 -ls v --fwhm_G 0.5 --fwhm_L 2.0 -o example`` First positional argument is ``prefix`` of experimental spectrum file to fit and second positional argument is the number of scan to fit. In this example it reads ``example_1.txt`` file. If you set ``num_scan`` argument to 2, it will search ``example_2.txt`` file and read it if such file exists. Third positional argument is filename of theoretical line shape spectrum. In this example it reads ``example_calc_peak.txt`` to get peak position and intensity of theoretically calculated one. Last positional argument is ``peak_factor``. When ``--scale_energy`` option argument is not set, it left shift peak positions to match experimental one. In this example, it sets -1.25 as initial ``peak_factor``.
The fist optional argument ``-ls`` or ``--line_shape`` is line shape of spectrum. In this example it uses ``v`` voigt profile. The second and third optional argument ``--fwhm_G`` and ``--fwhm_L`` are initial broadening parameter of gaussian and lorenzian part respectively. In this example, It set ``0.5`` as initial broadening parameter of gaussian part and ``2.0`` as initial broadening parameter of lorenzian part.
4. After fitting process is finished, you can see both fitting result plot and report for fitting result in the console.

![png](fit_static_example_file/XAS_example_fit.png)

* Close all fitting result plot windows, after then you can find ``example_A.txt``, ``example_base.txt``, ``example_fit.txt`` and ``example_fit_report.txt``.

``example_A.txt`` contains peak intensity scaling factor of each spectrum.

``example_base.txt`` contains fitted base line feature of experimental spectrum.

``example_fit.txt`` contains fitted thoretically calculated spectrum.

``example_fit_report.txt`` contains fitting result report.

## Peak position scaling

1. Go to scale subdirectory. In scale subdirectory, you can find ``example_calc_peak.txt``  and ``example_1.txt`` files. First one contains thoretically calculated IR peak information and last one is fake experimental IR spectrum.
2. Type  ``fit_staitc example 1 example_calc_peak.txt 0.95 -ls g --fwhm_G 15 --scale_energy -o example``. Last positional argument is initial parameter of ``peak_factor``. Since we set ``--scale_energy``, it matchs peak position of thoretical spectrum to experimental one by uniformly scaling peak position instead of uniform shifting. 
In this example we use ``g`` gaussian peak shape, so we need not to set ``--fwhm_L`` initial lorenzian broadening parameter.
3.  After fitting process is finished, you can see both fitting result plot and report for fitting result in the console.

![png](fit_static_example_file/IR_example_fit.png)

* Close all fitting result plot windows, then fitting results files will be generated.