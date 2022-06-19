# Basic

## import module


```python
import TRXASprefitpack
```

## Get general infomation of module


```python
help(TRXASprefitpack)
```

## get version information


```python
TRXASprefitpack.__version__
```




    '0.5.0'



## get general information of subpackage
Since 0.5.0 version, docs subpackage is removed


```python
help(TRXASprefitpack.docs)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_15875/1695297168.py in <module>
    ----> 1 help(TRXASprefitpack.docs)
    

    AttributeError: module 'TRXASprefitpack' has no attribute 'docs'



```python
help(TRXASprefitpack.mathfun)
```

## Get general information of function defined in TRXASprefitpack


```python
help(TRXASprefitpack.exp_conv_gau)
```

    Help on function exp_conv_gau in module TRXASprefitpack.mathfun.exp_conv_irf:
    
    exp_conv_gau(t: Union[float, numpy.ndarray], fwhm: float, k: float) -> Union[float, numpy.ndarray]
        Compute exponential function convolved with normalized gaussian
        distribution
        
        Args:
          t: time
          fwhm: full width at half maximum of gaussian distribution
          k: rate constant (inverse of life time)
        
        Returns:
         Convolution of normalized gaussian distribution and exponential
         decay :math:`(\exp(-kt))`.
    

