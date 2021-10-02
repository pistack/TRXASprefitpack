# fit_static

fitting static spectrum with theoretically calculated line spectrum broadened by spectral line shape

```{Note}
Currently, it uses linear baseline.
```

```{Note}
energy unit of measured static spectrum must be KeV 
and energy unit of calc static spectrum must be eV
```

* usage: fit_static [-h] [-ls {v,g,l}] [-o OUT] prefix num_scan peak_name

* positional arguments:
  * prefix                prefix for experimental static peak files It will read
                         prefix_i.txt files
  * num_scan              the number of static peak scan files
  * peak_name             filename for theoretical line shape spectrum

* optional arguments:
  * -h, --help            show this help message and exit
  * -ls {v,g,l}, --line_shape {v,g,l}
                         line shape of spectrum v: voigt profile g: gaussian
                         shape l: lorenzian shape
  * -o OUT, --out OUT     prefix for output files

 

