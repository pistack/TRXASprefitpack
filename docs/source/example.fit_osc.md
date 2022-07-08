# fit_osc Basic Example

Basic usage example for fit_osc utility and advanced usage example for fit_tscan utility.
Yon can find example file from [TRXASprefitpack-example](https://github.com/pistack/TRXASprefitpack-example/tree/v0.5.1) fit_osc subdirectory.

1. In fit_osc sub directory you can find ``example_1.txt``, ``example_2.txt``, ``example_3.txt``, ``example_4.txt`` files.
These examples are generated from rate equation example section 2. branched decay with varying time zero of each scan around 0 ps. However lifetimes are different from rate equation section 2. brnached decay one.
2. Type ``fit_osc -h`` Then it prints help message. You can find detailed description of arguments in the utility section of this document.
3. Since ``fit_osc`` utility can not fit directly to experimental time delay scan, you should calcuate residual of experimental time delay scan from ``fit_tscan`` or ``fit_seq`` utility. In this example we use ``fit_tscan`` utility to calculate residual of experimental time delay scan.

## Calculation of residual of time delay scan 

1. Type ``fit_tscan example --irf g --fwhm_G 0.15 -t0 0 0 0 0 --tau 3 100 500 --no_base``, optionally you can turn on ``--slow`` option to use robust global optimization method ``ampgo`` instead of faster ``Nelder–Mead``.
2. After fitting process is finished, you can see both fitting result plot and report for fitting result in the console. Upper part of plot shows fitting curve and experimental data. Lower part of plot shows residual of fit (data-fit).

![png](fit_osc_example_file/example_osc_fit_1.png) ![png](fit_osc_example_file/example_osc_fit_2.png)
![png](fit_osc_example_file/example_osc_fit_3.png) ![png](fit_osc_example_file/example_osc_fit_4.png)

* Compare chi squared and residual plot of each scan, you can find that chi squared value of tscan 2 is much higher than others and residual seems to have oscillation feature. Below plot is the residual plot of time scan 2 with time range -2 to 10.

![png](fit_osc_example_file/example_res_2.png)

* Now close all fitting result plot windows 

## Fitting with residual 

```{Note}
To fitting with residual of time scan data, you should have high quality time scan.
```

![png](fit_osc_example_file/example_osc_fit_report.png)

1. From fitting report of previous time delay scan fitting, get full width and half maximum value of irf function and time zero of tscan 2.
2. Copy ``out_example_res_2.txt`` to ``example_osc_1.txt``, since we will do residual fitting with time scan 2.
3. Type ``fit_osc -h`` to get help message of ``fit_osc``. Detailed descriptions would be found in ``utility`` section.
4. Type ``fit_osc example_osc --irf g --fwhm_G 0.14 -t0 -0.06 --tau 3 --period 0.4 --phase 0 --fix_irf``. Since oscillation feature may distort time zero, we do not fix time zero value to the value from previous time scan fitting. However you can fix time zero via ``--fix_t0`` option. Optionally you can turn on ``ampgo`` method via ``--slow`` option.
5. After residual fitting is finished, it prints fitting result and draws plot.

![png](fit_osc_example_file/example_res_fit_2_report.png)

![png](fit_osc_example_file/example_res_fit_2.png)

* From residual fitting, we observe oscillation feature with lifetime 4 ps, vibrational period 0.4 ps or vibrational frequency 2.5 THz and phase factor $0.5 \approx \pi/6$.