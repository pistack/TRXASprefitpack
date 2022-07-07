# fit_tscan Basic Example

Basic usage example for fit_tscan utility.
Yon can find example file from [TRXASprefitpack-example](https://github.com/pistack/TRXASprefitpack-example/tree/v0.5.1) fit_tscan subdirectory.

1. In fit_tscan sub directory you can find ``example_1.txt``, ``example_2.txt``, ``example_3.txt``, ``example_4.txt`` files.
These examples are generated from rate equation example section 2. branched decay with varying time zero of each scan around -0.15 ps.
2. Type ``fit_tscan -h`` Then it prints help message. You can find detailed description of arguments in the utility section of this document.
3. Type ``fit_tscan example --num_file 4 --irf g --fwhm_G 0.15 -t0 -0.15 -0.15 -0.15 -0.15 --tau 0.2 500 1000 --no_base`` The first and the only one positional argument is prefix of time delay scan file to read. In this example you set ``--num_file 4`` and four initial time zero parameter ``-t0 -0.15 -0.15 -0.15 -0.15``, so it searchs ``example_1.txt``,...,``example_4.txt`` files and read them all. Second optional argument ``--irf`` set temporal shape of probe pulse. In this example we set ``--irf`` to `g`, gaussian shape. Third optional argument ``--fwhm_G`` is initial full width and half maximum of temporal shape of probe pulse. Since we use gaussian shape irf, we need to set initial ``fwhm_G``. Fouth optional argument is ``--tau`` initial lifetime constant for each component. In this example we set three lifetime component with initial value 0.2, 500, 1000 respectively. If ``--no_base`` is not set, it will use infinite life time component to fit long lived spectral feature (eventhough it does not exist). Thus if you think there is not long lived spectral feature in your time delay scan result please set ``--no_base`` option to avoid over fitting.
4. After fitting process is finished, you can see both fitting result plot and report for fitting result in the console. Upper part of plot shows fitting curve and experimental data. Lower part of plot shows residual of fit (data-fit).

![png](fit_tscan_example_file/example_tscan_fit_1.png) ![png](fit_tscan_example_file/example_tscan_fit_2.png)
![png](fit_tscan_example_file/example_tscan_fit_3.png) ![png](fit_tscan_example_file/example_tscan_fit_4.png)

* In the component contribution section in the fitting result report, you can find contribution of each component in each time delay scan.

![png](fit_tscan_example_file/example_tscan_fit_comp_table.png)

* Close all fitting result plot windows then ``example_c.txt``, ``example_fit.txt``, ``example_fit_report.txt`` and ``example_res_i.txt`` files will be generated.

``example_c.txt`` contains coefficient of each component in each time delay scan

``example_fit.txt`` contains fitting curve of time delay scan

``example_fit_report.txt`` contains fitting result report.

``example_res_i.txt`` contains residual of time delay scan (data-fit)