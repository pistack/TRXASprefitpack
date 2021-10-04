# Basic

## import module


```python
import TRXASprefitpack
```

## Get general infomation of module


```python
help(TRXASprefitpack)
```

    Help on package TRXASprefitpack:
    
    NAME
        TRXASprefitpack
    
    DESCRIPTION
        TRXASprefitpack: 
        package for TRXAS pre fitting process which aims for the first order dynamics
        TRXAS stands for Time Resolved X-ray Absorption Spectroscopy
        
        :copyright: 2021 by pistack (Junho Lee)
        :license: LGPL3.
    
    PACKAGE CONTENTS
        data_process (package)
        doc (package)
        mathfun (package)
        thy (package)
        tools (package)
    
    DATA
        __info__ = {'TRXASprefitpack_info': '\nTRXASprefitpack_info\n*********...
    
    VERSION
        0.4.6
    
    FILE
        /home/lis1331/anaconda3/lib/python3.8/site-packages/TRXASprefitpack/__init__.py
    
    


## get version information


```python
TRXASprefitpack.__version__
```




    '0.4.6'



## get general information of subpackage

Note. doc subpackage is deprecated, will be removed in version 0.5


```python
help(TRXASprefitpack.doc)
```

    Help on package TRXASprefitpack.doc in TRXASprefitpack:
    
    NAME
        TRXASprefitpack.doc
    
    DESCRIPTION
        doc:
        Subpackage for the TRXASprefitpack documentation
        
        [deprecated]
        This subpackage will be removed in version 0.5
        
        :copyright: 2021 by pistack (Junho Lee).
        :license: LGPL3.
    
    PACKAGE CONTENTS
        info
    
    DATA
        __all__ = ['__info__']
        __info__ = {'TRXASprefitpack_info': '\nTRXASprefitpack_info\n*********...
    
    FILE
        /home/lis1331/anaconda3/lib/python3.8/site-packages/TRXASprefitpack/doc/__init__.py
    
    


## Get general information of function defined in TRXASprefitpack


```python
help(TRXASprefitpack.exp_conv_gau)
```

    Help on function exp_conv_gau in module TRXASprefitpack.mathfun.exp_conv_irf:
    
    exp_conv_gau(t, fwhm, k)
        Compute exponential function convolved with normalized gaussian 
        distribution
        
        Note.
        We assume temporal pulse of x-ray is normalized gaussian distribution
        
        .. math::
        
           \sigma = \frac{fwhm}{2\sqrt{2\log{2}}}
        
        
        .. math::
        
           IRF(t) = \frac{1}{\sigma \sqrt{2\pi}}\exp\left(-\frac{t^2}{2\sigma^2}\right)
        
        :param t: time
        :type t: float, numpy_1d_array
        :param fwhm: full width at half maximum of x-ray temporal pulse
        :type fwhm: float
        :param k: rate constant (inverse of life time)
        :type k: float
        
        
        :return: convolution of normalized gaussian distribution and exp(-kt)
        
        .. math::
        
           \frac{1}{2}\exp\left(\frac{k^2}{2\sigma^2}-kt\right){erfc}\left(\frac{1}{\sqrt{2}}\left(k\sigma-\frac{t}{\sigma}\right)\right)
        
        
        :rtype: numpy_1d_array
    

