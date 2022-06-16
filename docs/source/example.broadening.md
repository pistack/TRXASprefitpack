# Broadening Basic Example

Basic usage example for broadening utility.
Yon can find example file from [TRXASprefitpack-example](https://github.com/pistack/TRXASprefitpack-example) broadening subdirectory.

## Peak position shifting

1. Go to shift subdirectory. In shift subdirectory, you can find ``example_calc_peak.txt`` file. It is simplified version of thoretically calculated x-ray absorption spectrum peak and its intensity.
2. Type ``broadening -h``. Then it prints help message. You can find detailed description of arguments in the utility section of this document.
3. Type ``broadening example_calc_peak.txt 2825 2845 0.25 0.003 0.5 2.0 -1.25 -o example`` It calculates voigt broadened spectrum with 0.5 eV gaussian and 2.0 eV lorenzian fwhm from 2825 eV to 2845 eV with 0.25 eV energy step. Then it scales peak intensity by 0.003. Finally it right shifts peak position by 1.25 eV. The last argument is ``peak_factor`` when ``--scale_energy`` is not set, it left shifts peak position by ``peak_factor``.
4. Then you can find ``example_thy_stk.txt`` and ``example_thy.txt`` files. ``example_thy_stk.txt`` contains peak infromation and ``example_thy.txt`` contains voigt broadened spectrum from 2825 eV to 2845 eV with 0.25 eV step. You can plot ``example_thy.txt`` file using common visualization software like gnuplot.

![png](broadening_example_file/XAS_example.png)

## Peak position scaling

1. Go to scale subdirectory. In scale subdirectory, you can find ``example_calc_peak.txt`` file. It contains thoretically calculated IR peak information.
2. Type ``broadening example_calc_peak.txt 500 3500 1 0.001 15 0 0.96 --scale_energy -o example``. It calculates voigt broadened spectrum with 15 cm-1 gausian and 0 eV lorenzian fwhm (i.e. gaussian function with 15 cm-1 fwhm) from 500 cm-1 to 3500 cm-1 with 1 cm-1 step. Then it scales peak intensity by 0.001. Since ``--scale_energy`` is set, it uniformly scales peak position by 0.96.
3. Then you can find ``example_thy_stk.txt`` and ``example_thy.txt`` files. ``example_thy_stk.txt`` contains peak infromation and ``example_thy.txt`` contains gaussian broadened spectrum from 500 cm-1 to 3500 cm-1 with 1 cm-1 step. You can plot ``example_thy.txt`` file using common visualization software like gnuplot.

![png](broadening_example_file/IR_example.png)
