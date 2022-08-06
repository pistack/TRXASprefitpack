# fit_static Basic Example

Basic usage example ``fit_static`` utility.
Yon can find example file from [TRXASprefitpack-example](https://github.com/pistack/TRXASprefitpack-example/tree/v0.6.1) fit_static subdirectory.

## Fitting with voigt profile

1. Go to `voigt` sub-directory. In  `voigt` sub directory,  you can find ``example_static_voigt.txt`` file.
This example is generated from Library example, fitting with static spectrum (model: voigt).
2. Type ``fit_static -h`` Then it prints help message. You can find detailed description of arguments in the utility section of this document.
3. First find edge feature. Type ``fit_static example_static.txt  --mode voigt --edge g --e0_edge 8992 --fwhm_edge 10 -o edge --do_glb``.

The first and the only one positional argument is the filename of static spectrum file to read.

Second optional argument ``--mode`` sets fitting model, we set ``--mode voigt`` that is fitting with sum of voigt component. 

Third optional argument ``--edge``, if it is not set, it does not include edge feature. In this example we set `--edge g`, that is gaussian type edge.

Fourth optional argument ``--e0_edge`` is initial edge position.

Fifth optional argument is ``--fwhm_edge`` initial guess for fwhm paramter of edge. 

Last optional argument is `-o` it sets name of `hdf5` file to save fitting result and directory to save text file format of fitting result.

4. After fitting process is finished, you can see both fitting result plot and report for fitting result in the console. Upper part of plot shows fitting curve and experimental data. Lower part of plot shows residual of fit (data-fit).

5. Inspecting residual panel, we can find two voigt component centered near 8985 and 9000

![png](fit_static_example_file/find_edge.png)

1. Based on this now add two voigt component.

2. Type ``fit_static example_static.txt  --mode voigt --e0_voigt 8985 9000 --fwhm_L_voigt 2 6  --edge g --e0_edge 8992 --fwhm_edge 10 -o fit --do_glb``.

First additional optional argument ``--e0_voigt`` sets initial peak position of voigt component

Second additional optional argument ``--fwhm_L_voigt`` sets initial fwhm parameter of voigt component. In this example we only set lorenzian part of voigt componnet, so our voigt component is indeed lorenzian component.

3. After fitting process is finished, you can see fitting result plot.

![png](fit_static_example_file/fit_voigt.png)

## Description for Output file in fit directory.

* ``fit.txt`` contains fitting and each component curve

* ``weight.txt`` weight of each fitting component

* ``fit_summary.txt`` Summary of fitting result.

* ``res.txt`` contains residual of fit

## Fitting with theoretical calculated line spectrum

1. Go to `thy` sub-directory. In  `thy` sub directory,  you can find ``example_static_thy.txt`` file.
This example is generated from Library example, fitting with static spectrum (model: thy).

### Check thoeretical Spectrum

1. Type ``calc_broad Ni_example_1.stk -10 20 0.25 0.3 0.5 --policy scale --peak_shift 0 -o Ni_tst_1``
2. Then ``calc_broad`` calculates voigt broadened thoeretical spectrum with fwhm_G 0.3 and fwhm_L 0.5 from -10 to 20 with 0.25 step.
3. Type ``calc_broad Ni_example_2.stk -10 20 0.25 0.3 0.5 --policy scale --peak_shift 0 -o Ni_tst_2``
4. Then ``calc_broad`` do the samething with ``Ni_example_2.stk`` file.

* Ni_example_1
![png](fit_static_example_file/Ni_gs_1.png)
* Ni_example_2
![png](fit_static_example_file/Ni_gs_2.png)

### fitting with theoretical Spectrum

1. First try with one theoretical Spectrum
2. Type ``fit_static example_static_thy.txt --mode thy --thy_file Ni_example_1.stk --fwhm_G_thy 0.3 --fwhm_L_thy 0.5 --policy shift --peak_shift 863 -o fit_thy_1 --do_glb``.

In this command, you set ``--mode thy`` and ``--thy_file Ni_example_1.stk``, so you use fitting static spectrum with voigt broadened thoeretical line shape spectrum and it reads thoretical peak position and intensity from ``Ni_example_1.stk``.
Moreover, you set uniform fwhm paramter for such voigt function through `--fwhm_G_thy` and `--fwhm_L_thy` option.
To resolve discrepency between thoretical peak position and peak position of static spectrum, you can set ``--policy``. In this example you set ``--policy shift``. So, it shift peak position of thoretical spectrum to match peak position. To do this you should set initial peak shift paramter via ``--peak_shift`` option. We set initial peak shift paramter to 863 (``--peak_shift 863``).

* fit_thy_1
![png](fit_static_example_file/fit_thy_1.png)

1. Next try with two theoretical Spectrum
2. Type ``fit_static example_static_thy.txt --mode thy --thy_file Ni_example_1.stk Ni_example_2.stk --fwhm_G_thy 0.3 --fwhm_L_thy 0.5 --policy shift --peak_shift 863 863 -o fit_thy_2 --do_glb``.

* fit_thy_2
![png](fit_static_example_file/fit_thy_2.png)

Look at the residual pannel, then you find gaussian type edge feature which is centered at `862` and its fwhm is about `2`.

1. Now add one gaussian type edge feature. Before adding one edge feature, you should refine your initial guess based on previous fitting result.
2. Type ``fit_static example_static_thy.txt --mode thy --thy_file Ni_example_1.stk Ni_example_2.stk --fwhm_G_thy 0.3 --fwhm_L_thy 0.5 --policy shift --peak_shift 862.5 863 --edge g --fwhm_edge 2 --e0_edge 862 -o fit_thy_2_edge_1 --do_glb``.

* fit_thy_2_edge_1
![png](fit_static_example_file/fit_thy_2_edge_1.png)

1. Add one more gaussian type edge feature.
2. Type ``fit_static example_static_thy.txt --mode thy --thy_file Ni_example_1.stk Ni_example_2.stk --fwhm_G_thy 0.3 --fwhm_L_thy 0.5 --policy shift --peak_shift 862.5 863 --edge g --fwhm_edge 1 2 --e0_edge 860.5 862 -o fit_thy_2_edge_2 --do_glb``.

* fit_thy_2_edge_2
![png](fit_static_example_file/fit_thy_2_edge_2.png)
